"""Declarative catalog: fields, tasks, evaluators, presets, manifests, architectures, models and GPU specs.

Everything here is data shipped as YAML in `core/data/`, except benchmarks (tasks), which are declared in
Python under `core/benchmarks/`. Neither imports execution code, so the server can validate and plan without
torch.
"""

from __future__ import annotations

import functools
import math
import re
from importlib import resources
from pathlib import PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import Field, computed_field, field_validator, model_validator

from effeval.core.fragments import ATTENTION_SITES, SITE_TENSORS
from effeval.core.graph import LayerGraph, ModelShape, merge_overrides
from effeval.core.listing import EvaluatorInfo, ListedMetric, ListedTask, TaskListing
from effeval.core.manifest import MethodManifest
from effeval.core.types import Artifact, Capabilities, Mode, StrictModel, utc_now

_SOURCE = re.compile(r"^(paper §.+|code: .+|assumed)$")
_MODEL_CODE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
Dtype = Literal["bfloat16", "float16", "float32"]


def model_code(checkpoint: str) -> str:
    """Global model code: the checkpoint's last path component, lower-cased; hyphens and dots are kept.

    'Qwen/Qwen3-0.6B-Base' -> 'qwen3-0.6b-base'. Raises ValueError if the result has other characters.
    """
    code = PurePosixPath(checkpoint.rstrip("/")).name.lower()
    if not _MODEL_CODE.match(code):
        raise ValueError(f"cannot derive a model code from {checkpoint!r}: got {code!r}")
    return code


class ArchitectureDef(StrictModel):
    """Module layout of one HF architecture class, extracted by ops.introspect (never hand-written)."""

    name: str = Field(description="HF class name from config.architectures, e.g. Qwen3ForCausalLM.")
    layers_path: str
    linears: dict[str, str] = Field(description="Linear module path in a decoder layer -> canonical name.")
    attention_path: str | None = Field(description="Attention module path in a decoder layer, if found.")
    supports_attention_backend: bool
    uncovered_weights: list[str] = Field(
        description="Matrix weights no linear site covers, e.g. fused experts."
    )
    extracted_with: dict[str, str]
    graph: LayerGraph | None = Field(
        default=None, description="One decoder layer as a graph (`core/graph.py`); None until extracted."
    )
    graph_overrides: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description="Hand fixes merged into `graph` by `effective_graph` (see `merge_overrides`).",
    )

    def effective_graph(self) -> LayerGraph | None:
        """`graph` with `graph_overrides` applied; None when no graph was extracted."""
        if self.graph is None or not self.graph_overrides:
            return self.graph
        return merge_overrides(self.graph, self.graph_overrides)

    def site_names(self) -> list[str]:
        names = [f"linear.{canon}.{t}" for canon in self.linears.values() for t in SITE_TENSORS]
        names += [f"lm_head.{t}" for t in SITE_TENSORS]
        if self.attention_path is not None and self.supports_attention_backend:
            names += list(ATTENTION_SITES)
        return sorted(names)


class ModelEntry(StrictModel):
    """One global model code. Registered once, never edited: the same code always means the same weights."""

    code: str
    checkpoint: str = Field(description="HF model id, or a path on compute nodes.")
    revision: str | None = Field(description="Pinned commit sha; None for local paths.")
    architecture: str
    dtype: Dtype = Field(description="Native dtype from the checkpoint config.")
    params: int
    tie_word_embeddings: bool
    shape: ModelShape | None = Field(
        default=None, description="Dimensions from the checkpoint's config.json; None for older entries."
    )
    description: str = ""

    @model_validator(mode="after")
    def _code_follows_the_rule(self) -> ModelEntry:
        if self.code != model_code(self.checkpoint):
            raise ValueError(f"model code must be {model_code(self.checkpoint)!r} for {self.checkpoint!r}")
        return self


class FieldDef(StrictModel):
    name: str
    description: str
    modes: list[Mode]
    reference: dict[str, Any] = Field(description="RunSpec overrides that derive the reference job.")
    generation: dict[str, Any] = Field(default_factory=dict, description="Required generation settings.")
    compare: list[str] = Field(description="Comparative metrics computed where applicable.")


class NativeTaskDef(StrictModel):
    protocol: Literal["gptq_ppl"]
    dataset: str
    dataset_config: str | None
    split: str
    text_column: str
    joiner: str
    use_fast_tokenizer: bool


SampleCount = int | float
"""A `limit`: an integer count (-1 for all) or a fraction strictly between 0 and 1."""


def check_limit(value: Any) -> int | float:
    """Return `value` if it is -1, an integer >= 1, or a float in (0, 1); raise ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("limit must be a number")
    if isinstance(value, int):
        if value == -1 or value >= 1:
            return value
        raise ValueError("limit must be -1 (all samples), an integer >= 1, or a fraction between 0 and 1")
    if 0.0 < value < 1.0:
        return value
    raise ValueError("a fractional limit must be strictly between 0 and 1; use an integer >= 1 for a count")


def sample_count(limit: int | float | None, total: int) -> int:
    """Number of samples a `limit` keeps out of `total`; a fraction rounds up."""
    if limit is None:
        return total
    if isinstance(limit, float):
        return min(total, math.ceil(total * limit))
    return min(total, limit)


class BenchmarkParams(StrictModel):
    """A benchmark's adjustable settings; every field needs a default (the paper's) and a description."""


class SampleLimit(BenchmarkParams):
    """Settings of benchmarks that can evaluate a subset of their samples."""

    limit: SampleCount = Field(
        default=-1,
        description="Samples per (sub)task: an integer >= 1 for a count, a number in (0, 1) for that fraction "
        "of the samples, -1 for all of them.",
    )

    @field_validator("limit", mode="before")
    @classmethod
    def _count_or_fraction(cls, value: Any) -> Any:
        return check_limit(value)


class LibrarySettings(SampleLimit):
    """Settings every library task shares."""

    num_fewshot: int | None = Field(
        default=None, ge=0, description="Few-shot examples per sample; empty keeps the task's default."
    )


DEFAULT_SAMPLING_TEMPERATURE = 0.7


class GenerationLibrarySettings(LibrarySettings):
    """Settings of library tasks that generate."""

    max_new_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of generated tokens for this task; overrides the run-level value.",
    )
    do_sample: bool | None = Field(
        default=None,
        description="Sample instead of decoding greedily for this task; empty keeps the run-level setting.",
    )
    temperature: float | None = Field(
        default=None,
        gt=0,
        examples=[DEFAULT_SAMPLING_TEMPERATURE],
        description="Sampling temperature for this task; only with do_sample. "
        f"Empty uses the run-level temperature, else {DEFAULT_SAMPLING_TEMPERATURE}.",
    )


class TaskSize(StrictModel):
    """Rough workload size for ETA priors only. Token averages are assumptions, not measurements."""

    samples: int
    subtasks: int = 1
    prompt_tokens: int = Field(description="Average prompt tokens per sample incl. few-shot (total for ppl).")
    output_tokens: int = 0
    choices: int = 1


class MetricFormat(StrictModel):
    style: Literal["fixed", "percent", "scientific", "integer", "duration"] = "fixed"
    decimals: int = Field(default=4, ge=0, le=12)


class MetricDef(StrictModel):
    """How a metric is named, read and shown wherever results appear; the single rendering contract."""

    name: str
    label: str
    kind: Literal["quality", "system"]
    description: str
    unit: str = ""
    format: MetricFormat = Field(default_factory=MetricFormat)
    higher_is_better: bool | None = Field(
        description="None when neither direction is better, e.g. an error bar."
    )
    placement: Literal["column", "detail"] = Field(
        default="column",
        description="column: a table column by default; detail: only on request and in details.",
    )


class CategoryDef(StrictModel):
    """A task category: one band of result columns, and a group of the task picker."""

    name: str
    label: str = Field(description="Band heading of the result columns.")
    order: int
    title: str | None = Field(
        default=None, description="What the tasks measure, for pickers; defaults to label."
    )
    rank: int = Field(
        default=1000, exclude=True, description="Precedence among categories whose rules match; lower wins."
    )
    metrics: list[str] = Field(
        default_factory=list,
        exclude=True,
        description="Metric names (filter suffix ignored) that place a task here.",
    )
    names: list[str] = Field(
        default_factory=list, exclude=True, description="Regexes searched in the lower-case task name."
    )
    tags: list[str] = Field(
        default_factory=list,
        exclude=True,
        description="Library task tags (case-insensitive) that place a task here.",
    )

    @field_validator("names")
    @classmethod
    def _compile(cls, names: list[str]) -> list[str]:
        for pattern in names:
            re.compile(pattern)
        return names


class AverageDef(StrictModel):
    """A derived column: the unweighted mean of `metric` over the tasks in `categories`."""

    label: str
    metric: str
    categories: list[str] = Field(min_length=1)


class KernelWeights(StrictModel):
    """Weight quantization a benchmarked kernel accepts; the method's linear weight rule must fit it."""

    type: Literal["int"]
    bits: list[int] = Field(min_length=1)
    symmetric: list[bool] = Field(min_length=1)
    group_sizes: list[int | None] = Field(min_length=1, description="None stands for per-channel scales.")


class BenchmarkDef(StrictModel):
    """Operator benchmark of a model's linear layers: shapes come from one decoder layer of the model."""

    kernel: str = Field(
        description="Name of the kernel implementation and of the Python package providing it."
    )
    activation_dtype: Literal["float16", "bfloat16"]
    baselines: list[Literal["float16", "bfloat16"]] = Field(min_length=1)
    weights: KernelWeights


class TaskMetric(StrictModel):
    name: str
    higher_is_better: bool | None = Field(description="None when the library declares no direction.")
    stderr_of: str | None = Field(default=None, description="The metric this one is the standard error of.")


@functools.cache
def _settings_schema(settings: type[BenchmarkParams]) -> dict[str, Any]:
    return settings.model_json_schema()


class TaskDef(StrictModel):
    name: str = Field(description="evaluator/target, the identity of the task everywhere.")
    evaluator: str
    target: str = Field(description="The evaluator's own name for the task.")
    version: str
    category: str
    description: str
    modes: list[Mode]
    default_mode: Mode
    produces: dict[Mode, frozenset[Artifact]] = Field(description="Artifacts the evaluator yields per mode.")
    num_fewshot: int | None = Field(description="Default few-shot count; None: subtasks keep their own.")
    fewshot_allowed: list[int] | None = Field(
        default=None, description="Accepted num_fewshot overrides; None accepts any, [] accepts none."
    )
    metrics: list[TaskMetric] = Field(
        min_length=1, description="Metrics the evaluator reports; the first is primary."
    )
    derived: list[str] = Field(
        default_factory=list, description="Derived metrics computed by default from per-sample outputs."
    )
    reference: bool = Field(
        default=True,
        description="Whether jobs get a reference job; benchmarks measure their baseline inline.",
    )
    benchmark: BenchmarkDef | None = None
    sample_scores: list[str] = Field(
        description="Per-sample score names every evaluator reports, e.g. correct."
    )
    size: TaskSize | None = None
    native: NativeTaskDef | None = None
    subsets: list[str] | None = None
    subtasks: list[str] = Field(default_factory=list, description="Namespaced names of direct members.")
    tags: list[str] = Field(default_factory=list)
    params: type[BenchmarkParams] = Field(
        default=BenchmarkParams, exclude=True, description="Adjustable settings; see params_schema."
    )

    @model_validator(mode="after")
    def _consistent(self) -> TaskDef:
        if self.name != f"{self.evaluator}/{self.target}":
            raise ValueError(f"task name {self.name!r} must be '<evaluator>/<target>'")
        if set(self.produces) != set(self.modes):
            raise ValueError(f"task {self.name!r}: produces must have exactly the modes {self.modes}")
        if self.default_mode not in self.modes:
            raise ValueError(f"task {self.name!r}: default mode {self.default_mode.value} is not a mode")
        for mode, artifacts in self.produces.items():
            if not artifacts:
                raise ValueError(f"task {self.name!r}: declares no artifacts in {mode.value} mode")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def params_schema(self) -> dict[str, Any]:
        """JSON schema of `params`; field descriptions are the UI's help texts."""
        return _settings_schema(self.params)

    def settings(self, overrides: dict[str, Any]) -> BenchmarkParams:
        """The full settings of a job: defaults with `overrides` applied. Raises pydantic.ValidationError."""
        return self.params.model_validate(overrides)


class EvaluatorDef(StrictModel):
    name: str
    version: str
    description: str
    capabilities: Capabilities
    source: Literal["builtin", "listing"] = "builtin"
    library: str | None = None
    library_version: str | None = None


class PaperRef(StrictModel):
    title: str
    arxiv: str
    official_repo: str


class PresetValue(StrictModel):
    value: Any
    source: str

    @field_validator("source")
    @classmethod
    def _valid_source(cls, value: str) -> str:
        if not _SOURCE.match(value):
            raise ValueError("source must be 'paper §<section>', 'code: <file>#L<line>' or 'assumed'")
        return value


class ReportedValue(StrictModel):
    model: str
    task: str
    metric: str
    value: str
    table: str
    tolerance: str


class PresetDef(StrictModel):
    method: str
    name: str
    description: str
    paper: PaperRef
    values: dict[str, PresetValue] = Field(description="Dotted RunSpec paths -> value with provenance.")
    reported: list[ReportedValue] = Field(default_factory=list)

    @field_validator("values")
    @classmethod
    def _paths_under_known_roots(cls, value: dict[str, PresetValue]) -> dict[str, PresetValue]:
        roots = {"method", "model", "tasks", "generation", "runtime"}
        for path in value:
            if path.split(".")[0] not in roots:
                raise ValueError(f"preset path {path!r} must start with one of {sorted(roots)}")
            if path.startswith("method.") and not path.startswith("method.config."):
                raise ValueError(f"preset path {path!r}: only method.config.* may be preset")
        return value


class GpuSpec(StrictModel):
    name: str
    aliases: list[str] = Field(description="Names users may type in hardware.gpu.")
    device_names: list[str] = Field(description="Exact torch.cuda.get_device_name() strings of this GPU.")
    memory_gb: float
    bf16_dense_tflops: float
    hbm_tb_per_s: float
    compute_capability: str
    source: str


class Catalog(StrictModel):
    fields: dict[str, FieldDef]
    metrics: dict[str, MetricDef]
    categories: dict[str, CategoryDef]
    averages: list[AverageDef]
    tasks: dict[str, TaskDef]
    evaluators: dict[str, EvaluatorDef]
    presets: dict[str, dict[str, PresetDef]]
    manifests: dict[str, MethodManifest]
    architectures: dict[str, ArchitectureDef]
    models: dict[str, ModelEntry]
    gpus: dict[str, GpuSpec]
    common_tasks: list[str] = Field(default_factory=list, description="Tasks the UI pins in its picker.")

    def metric_direction(self, task: str, metric: str) -> bool | None:
        """Direction of `metric` for `task`: the task's own declaration, else metrics.yaml, else None."""
        definition = self.tasks.get(task)
        for declared in definition.metrics if definition else []:
            if declared.name == metric and declared.higher_is_better is not None:
                return declared.higher_is_better
        rendering = self.metrics.get(metric)
        return None if rendering is None else rendering.higher_is_better

    def with_listed(self, evaluator: EvaluatorDef, tasks: dict[str, TaskDef]) -> Catalog:
        """Return a catalog where `evaluator` and exactly `tasks` replace all that evaluator had before.

        Raises ValueError if `evaluator` is a builtin one or a task is inconsistent with the catalog.
        """
        known = self.evaluators.get(evaluator.name)
        if known is not None and known.source == "builtin":
            raise ValueError(f"evaluator name {evaluator.name!r} is reserved by a builtin evaluator")
        merged = {n: t for n, t in self.tasks.items() if t.evaluator != evaluator.name}
        merged.update(tasks)
        result = self.model_copy(
            update={"tasks": merged, "evaluators": {**self.evaluators, evaluator.name: evaluator}}
        )
        for task in tasks.values():
            _check_task(task, result)
        return result

    def with_tasks(self, tasks: list[TaskDef]) -> Catalog:
        """Return a catalog that also has `tasks`. Raises ValueError on a name clash or an invalid task."""
        clash = sorted(task.name for task in tasks if task.name in self.tasks)
        if clash:
            raise ValueError(f"tasks {clash} already exist in the catalog")
        result = self.model_copy(update={"tasks": {**self.tasks, **{t.name: t for t in tasks}}})
        for task in tasks:
            _check_task(task, result)
        return result

    def listing_for(self, evaluator: str, names: list[str]) -> TaskListing:
        """A listing of `evaluator` holding just the tasks `names` and, transitively, their subtasks."""
        info = self.evaluators[evaluator]
        wanted: dict[str, TaskDef] = {}
        pending = list(names)
        while pending:
            name = pending.pop()
            if name not in wanted:
                wanted[name] = self.tasks[name]
                pending.extend(wanted[name].subtasks)
        prefix = f"{evaluator}/"
        tasks = [
            ListedTask(
                name=t.target,
                version=t.version,
                description=t.description,
                mode=t.default_mode,
                produces=t.produces[t.default_mode],
                num_fewshot=t.num_fewshot,
                fewshot_allowed=t.fewshot_allowed,
                metrics=[ListedMetric.model_validate(m.model_dump()) for m in t.metrics],
                sample_scores=t.sample_scores,
                subsets=t.subsets,
                subtasks=[sub.removeprefix(prefix) for sub in t.subtasks],
                tags=t.tags,
            )
            for _, t in sorted(wanted.items())
        ]
        return TaskListing(
            evaluator=EvaluatorInfo(
                name=info.name,
                version=info.version,
                description=info.description,
                capabilities=info.capabilities,
                library=info.library,
                library_version=info.library_version,
            ),
            generated_at=utc_now(),
            tasks=tasks,
        )

    def with_task_listings(self, listings: list[TaskListing]) -> Catalog:
        """Apply every listing with `with_listed`; a later listing of an evaluator replaces an earlier."""
        result = self
        for listing in listings:
            evaluator, tasks = task_defs_from_listing(listing, self.metrics, self.categories)
            result = result.with_listed(evaluator, tasks)
        return result

    def with_models(self, extra: list[ModelEntry]) -> Catalog:
        """Return a catalog that also knows registered models. Codes are global and cannot be redefined."""
        merged = dict(self.models)
        for entry in extra:
            if entry.code in merged and merged[entry.code] != entry:
                raise ValueError(f"model code {entry.code!r} already means {merged[entry.code].checkpoint}")
            if entry.architecture not in self.architectures:
                raise ValueError(f"model {entry.code!r}: unknown architecture {entry.architecture!r}")
            merged[entry.code] = entry
        return self.model_copy(update={"models": merged})

    def with_manifests(self, extra: list[MethodManifest]) -> Catalog:
        """Return a catalog that also knows user-registered manifests. Builtins cannot be shadowed."""
        merged = dict(self.manifests)
        for manifest in extra:
            if manifest.name in merged and merged[manifest.name].builtin:
                raise ValueError(f"method name {manifest.name!r} is reserved by a builtin method")
            merged[manifest.name] = manifest
        return self.model_copy(update={"manifests": merged})

    def gpu_for_device(self, device_name: str) -> GpuSpec | None:
        """The hwdb entry whose exact device name equals `device_name`, or None."""
        return next((g for g in self.gpus.values() if device_name in g.device_names), None)

    def gpu(self, name: str) -> GpuSpec | None:
        lowered = name.lower()
        for spec in self.gpus.values():
            if lowered == spec.name.lower() or lowered in (a.lower() for a in spec.aliases):
                return spec
        return None


def _load_dir(package: str) -> list[dict[str, Any]]:
    docs = []
    for entry in sorted(resources.files(package).iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".yaml"):
            docs.append(yaml.safe_load(entry.read_text(encoding="utf-8")))
    return docs


def builtin_catalog() -> Catalog:
    """The catalog shipped with the package plus a task `speed/<name>` for each user prompt set.

    User sets are rescanned on every call (cached while their files are unchanged). Raises
    pydantic.ValidationError on malformed data files and ValueError on a misnamed user set file.
    """
    from effeval.core.promptsets import user_set_stamp

    stamp = user_set_stamp()
    return _static_catalog() if not stamp else _with_user_sets(stamp)


@functools.lru_cache(maxsize=1)
def _with_user_sets(stamp: tuple[tuple[str, int, int], ...]) -> Catalog:
    from effeval.core.benchmarks.speed import user_tasks

    return _static_catalog().with_tasks(user_tasks())


@functools.lru_cache(maxsize=1)
def _static_catalog() -> Catalog:
    base = "effeval.core.data"
    presets: dict[str, dict[str, PresetDef]] = {}
    for method_dir in sorted(resources.files(f"{base}.presets").iterdir(), key=lambda p: p.name):
        if not method_dir.is_dir() or method_dir.name.startswith("_"):
            continue
        for doc in _load_dir(f"{base}.presets.{method_dir.name}"):
            preset = PresetDef.model_validate(doc)
            presets.setdefault(preset.method, {})[preset.name] = preset
    from effeval.core.benchmarks import discover

    tasks = {task.name: task for task in discover()}
    evaluators = {d["name"]: EvaluatorDef.model_validate(d) for d in _load_dir(f"{base}.evaluators")}
    gpus = yaml.safe_load(resources.files(base).joinpath("hwdb.yaml").read_text(encoding="utf-8"))
    rendering = yaml.safe_load(resources.files(base).joinpath("metrics.yaml").read_text(encoding="utf-8"))
    catalog = Catalog(
        metrics={d["name"]: MetricDef.model_validate(d) for d in rendering["metrics"]},
        categories=_load_categories(rendering["categories"], base),
        averages=[AverageDef.model_validate(d) for d in rendering["averages"]],
        fields={d["name"]: FieldDef.model_validate(d) for d in _load_dir(f"{base}.fields")},
        tasks=tasks,
        evaluators=evaluators,
        presets=presets,
        manifests={d["name"]: MethodManifest.model_validate(d) for d in _load_dir(f"{base}.methods")},
        architectures={
            d["name"]: ArchitectureDef.model_validate(d) for d in _load_dir(f"{base}.architectures")
        },
        models={
            m["code"]: ModelEntry.model_validate(m)
            for m in yaml.safe_load(
                resources.files(base).joinpath("models.yaml").read_text(encoding="utf-8")
            )["models"]
        },
        gpus={g["name"]: GpuSpec.model_validate(g) for g in gpus["gpus"]},
        common_tasks=yaml.safe_load(
            resources.files(base).joinpath("common_tasks.yaml").read_text(encoding="utf-8")
        )["tasks"],
    )
    _check_cross_references(catalog)
    snapshots = [
        TaskListing.model_validate_json(entry.read_text(encoding="utf-8"))
        for entry in sorted(resources.files(f"{base}.task_catalogs").iterdir(), key=lambda p: p.name)
        if entry.name.endswith(".json")
    ]
    return catalog.with_task_listings(snapshots)


OTHER_CATEGORY = "other"


def _load_categories(declared: list[dict[str, Any]], base: str) -> dict[str, CategoryDef]:
    """The categories of metrics.yaml with the rules of task_categories.yaml (file order ranks them)."""
    rules = yaml.safe_load(resources.files(base).joinpath("task_categories.yaml").read_text(encoding="utf-8"))
    by_name = {d["name"]: d for d in declared}
    unknown = [r["category"] for r in rules["rules"] if r["category"] not in by_name]
    if unknown:
        raise ValueError(f"task_categories.yaml: categories {unknown} are not declared in metrics.yaml")
    for rank, rule in enumerate(rules["rules"]):
        by_name[rule["category"]].update({**{k: v for k, v in rule.items() if k != "category"}, "rank": rank})
    return {name: CategoryDef.model_validate(d) for name, d in by_name.items()}


def _metric_key(name: str) -> str:
    """A library metric name without its filter suffix or module path: 'acc,none' -> 'acc'."""
    return name.split(",")[0].rsplit("/", 1)[-1].lower()


@functools.cache
def _alternation(patterns: tuple[str, ...]) -> re.Pattern[str] | None:
    return re.compile("|".join(f"(?:{p})" for p in patterns)) if patterns else None


def _own_category(listed: ListedTask, ranked: list[CategoryDef]) -> str | None:
    """The category the rules give `listed` itself: by metric names, then name patterns, then tags."""
    metrics = {_metric_key(m.name) for m in listed.metrics}
    for category in ranked:
        if metrics & set(category.metrics):
            return category.name
    name = listed.name.lower()
    for category in ranked:
        pattern = _alternation(tuple(category.names))
        if pattern and pattern.search(name):
            return category.name
    tags = {t.lower() for t in listed.tags}
    for category in ranked:
        if tags & {t.lower() for t in category.tags}:
            return category.name
    return None


def categorize(listing: TaskListing, categories: dict[str, CategoryDef]) -> dict[str, str]:
    """Category of every task in `listing`, by task name.

    A task takes the category its own metrics, name or tags give; otherwise that of the first group
    containing it, else OTHER_CATEGORY.
    """
    ranked = sorted(categories.values(), key=lambda c: c.rank)
    own = {t.name: _own_category(t, ranked) for t in listing.tasks}
    parents: dict[str, list[str]] = {}
    for task in listing.tasks:
        for member in task.subtasks:
            parents.setdefault(member, []).append(task.name)
    resolved: dict[str, str] = {}

    def resolve(name: str, seen: frozenset[str]) -> str:
        if name not in resolved:
            found = own[name]
            for parent in parents.get(name, []) if found is None else []:
                inherited = resolve(parent, seen | {name}) if parent not in seen else OTHER_CATEGORY
                if inherited != OTHER_CATEGORY:
                    found = inherited
                    break
            resolved[name] = found or OTHER_CATEGORY
        return resolved[name]

    return {t.name: resolve(t.name, frozenset()) for t in listing.tasks}


def task_defs_from_listing(
    listing: TaskListing, metrics: dict[str, MetricDef], categories: dict[str, CategoryDef]
) -> tuple[EvaluatorDef, dict[str, TaskDef]]:
    """Convert an evaluator's task listing into its EvaluatorDef and namespaced TaskDefs.

    Raises ValueError if a task is inconsistent with the metric and category registries.
    """
    info = listing.evaluator
    evaluator = EvaluatorDef(
        name=info.name,
        version=info.version,
        description=info.description,
        capabilities=info.capabilities,
        source="listing",
        library=info.library,
        library_version=info.library_version,
    )
    tasks: dict[str, TaskDef] = {}
    category_of = categorize(listing, categories)
    for listed in listing.tasks:
        task = TaskDef(
            name=f"{info.name}/{listed.name}",
            evaluator=info.name,
            target=listed.name,
            version=listed.version,
            category=category_of[listed.name],
            description=listed.description,
            modes=[listed.mode],
            default_mode=listed.mode,
            produces={listed.mode: listed.produces},
            num_fewshot=listed.num_fewshot,
            fewshot_allowed=listed.fewshot_allowed,
            metrics=[TaskMetric.model_validate(m.model_dump()) for m in listed.metrics],
            sample_scores=listed.sample_scores,
            subsets=listed.subsets,
            subtasks=[f"{info.name}/{s}" for s in listed.subtasks],
            tags=listed.tags,
            params=GenerationLibrarySettings if listed.mode == Mode.GENERATION else LibrarySettings,
        )
        tasks[task.name] = task
    scope = Catalog.model_construct(
        metrics=metrics, categories=categories, evaluators={info.name: evaluator}, tasks=tasks
    )
    for task in tasks.values():
        _check_task(task, scope)
    return evaluator, tasks


def _check_task(task: TaskDef, catalog: Catalog) -> None:
    if task.category not in catalog.categories:
        raise ValueError(f"task {task.name!r}: category {task.category!r} is not declared in metrics.yaml")
    evaluator = catalog.evaluators.get(task.evaluator)
    if evaluator is None:
        raise ValueError(f"task {task.name!r}: unknown evaluator {task.evaluator!r}")
    unsupported = set(task.modes) - set(evaluator.capabilities.modes)
    if unsupported:
        raise ValueError(
            f"task {task.name!r}: evaluator {evaluator.name!r} cannot run mode(s) "
            f"{sorted(m.value for m in unsupported)}"
        )
    unknown = [m for m in task.derived if m not in catalog.metrics]
    if unknown:
        raise ValueError(f"task {task.name!r}: derived metrics {unknown} are not declared in metrics.yaml")
    for metric in task.metrics:
        declared = catalog.metrics.get(metric.name)
        if (
            declared is not None
            and metric.higher_is_better is not None
            and declared.higher_is_better is not None
            and metric.higher_is_better != declared.higher_is_better
        ):
            raise ValueError(
                f"task {task.name!r}: metric {metric.name!r} says higher_is_better="
                f"{metric.higher_is_better} but metrics.yaml says {declared.higher_is_better}"
            )
    missing = [s for s in task.subtasks if s not in catalog.tasks]
    if missing:
        raise ValueError(f"task {task.name!r}: unknown subtasks {missing}")


def _check_cross_references(catalog: Catalog) -> None:
    for task in catalog.tasks.values():
        _check_task(task, catalog)
    for average in catalog.averages:
        if average.metric not in catalog.metrics:
            raise ValueError(f"average {average.label!r}: unknown metric {average.metric!r}")
        unknown = [c for c in average.categories if c not in catalog.categories]
        if unknown:
            raise ValueError(f"average {average.label!r}: unknown categories {unknown}")
    for entry in catalog.models.values():
        if entry.architecture not in catalog.architectures:
            raise ValueError(f"model {entry.code!r}: unknown architecture {entry.architecture!r}")
    for method, by_name in catalog.presets.items():
        manifest = catalog.manifests.get(method)
        if manifest is None:
            raise ValueError(f"preset for unknown method {method!r}")
        if manifest.default_preset is not None and manifest.default_preset not in by_name:
            raise ValueError(f"method {method!r}: default preset {manifest.default_preset!r} missing")
