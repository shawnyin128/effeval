"""Declarative validation of a RunSpec against the catalog (architecture.md §5.2).

`resolve` is shared with plan: it applies the preset, picks evaluators, and records every conflict as an
Issue.
Nothing here imports execution code, so the server, the CLI and MCP get identical answers.
"""

from __future__ import annotations

import copy
import difflib
import fnmatch
import itertools
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from effeval.core.canonical import canonical_json
from effeval.core.catalog import (
    ArchitectureDef,
    Catalog,
    EvaluatorDef,
    FieldDef,
    KernelWeights,
    ModelEntry,
    PresetDef,
    PresetValue,
    TaskDef,
)
from effeval.core.fragments import QuantRule, linear_weight_rules
from effeval.core.graph import attention_output_position
from effeval.core.lowrank import qkv_budget, transform_architecture
from effeval.core.manifest import MethodManifest, validate_method_config
from effeval.core.preprocess import (
    DATA_SOURCES,
    GLOBAL_POSITIONS,
    RESERVED_SOURCES,
    PreprocessPresetDef,
    builtin_preprocess_presets,
    position_dim,
    rotation_option_problem,
    rotation_size_problem,
    source_position_problem,
)
from effeval.core.spec import DEFAULT_PRESET, ResolvedModel, RunSpec
from effeval.core.types import Artifact, Deviation, Fidelity, HandleKind, Issue, Mode, Severity
from effeval.metrics.derived import DERIVED, applicable
from effeval.metrics.quality import COMPARATIVE, why_not_applicable

GIB = 1 << 30
BYTES_PER_PARAM = 2
WEIGHT_HEADROOM = 1.15
GPU_USABLE_BYTES = 44 * GIB
"""Weights in 16 bits plus 15% headroom must fit in 44 GiB per GPU (a 48 GB card)."""

GENERATION_ONLY_KEYS = ("do_sample", "temperature", "max_new_tokens", "top_p", "top_k")
GENERATION_TASK_KEYS = ("max_new_tokens", "do_sample", "temperature")


@dataclass
class TaskPlan:
    task: TaskDef
    mode: Mode
    evaluator: EvaluatorDef
    num_fewshot: int | None = None
    limit: int | float | None = None
    max_new_tokens: int | None = None
    do_sample: bool | None = None
    temperature: float | None = None
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def produces(self) -> frozenset[Artifact]:
        return self.task.produces[self.mode]


@dataclass
class Resolution:
    """A RunSpec after preset application, with its resolved tasks and all findings."""

    spec: RunSpec | None = None
    model: ResolvedModel | None = None
    model_entry: ModelEntry | None = None
    draft_model: ResolvedModel | None = None
    draft_manifest: MethodManifest | None = None
    architecture: ArchitectureDef | None = None
    manifest: MethodManifest | None = None
    field_def: FieldDef | None = None
    preset: PresetDef | None = None
    preprocess_preset: PreprocessPresetDef | None = None
    fidelity: Fidelity = field(default_factory=Fidelity)
    tasks: list[TaskPlan] = field(default_factory=list)
    compare: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    def error(self, code: str, message: str, paths: list[str] | None = None, hint: str = "") -> None:
        self.issues.append(
            Issue(code=code, severity=Severity.ERROR, paths=paths or [], message=message, hint=hint)
        )

    def note(self, code: str, message: str, paths: list[str] | None = None) -> None:
        self.issues.append(Issue(code=code, severity=Severity.INFO, paths=paths or [], message=message))


def validate(raw: dict[str, Any], catalog: Catalog) -> list[Issue]:
    """Validate a RunSpec document, expanding `sweep` so every combination is checked."""
    combos = expand_sweep(raw)
    if len(combos) == 1:
        return resolve(combos[0], catalog).issues
    issues: list[Issue] = []
    for index, combo in enumerate(combos):
        for issue in resolve(combo, catalog).issues:
            issues.append(issue.model_copy(update={"message": f"[sweep #{index}] {issue.message}"}))
    return issues


def expand_sweep(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Cartesian product over `sweep` paths, each without `sweep`; invalid sweeps are left to resolve."""
    sweep = raw.get("sweep") or {}
    if not isinstance(sweep, dict) or not sweep:
        return [{k: v for k, v in raw.items() if k != "sweep"}]
    paths = sorted(sweep)
    combos = []
    for values in itertools.product(*(sweep[p] for p in paths)):
        doc = copy.deepcopy({k: v for k, v in raw.items() if k != "sweep"})
        for path, value in zip(paths, values, strict=True):
            set_path(doc, path, value)
        combos.append(doc)
    return combos


def get_path(doc: dict[str, Any], path: str) -> tuple[bool, Any]:
    node: Any = doc
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def set_path(doc: dict[str, Any], path: str, value: Any) -> None:
    node = doc
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {path!r}: {part!r} is not a mapping")
    node[parts[-1]] = copy.deepcopy(value)


def _preset_for(
    raw: dict[str, Any], manifest: MethodManifest, catalog: Catalog, res: Resolution
) -> PresetDef | None:
    requested = (raw.get("method") or {}).get("preset", DEFAULT_PRESET)
    name = manifest.default_preset if requested == DEFAULT_PRESET else requested
    if name is None:
        return None
    preset = catalog.presets.get(manifest.name, {}).get(name)
    if preset is not None:
        return preset
    if name in manifest.custom_presets:
        return PresetDef(
            method=manifest.name,
            name=name,
            description=f"custom preset of {manifest.name}",
            paper={"title": "custom", "arxiv": "-", "official_repo": "-"},
            values={
                f"method.config.{k}": {"value": v, "source": "assumed"}
                for k, v in manifest.custom_presets[name].items()
            },
        )
    available = sorted(catalog.presets.get(manifest.name, {})) + sorted(manifest.custom_presets)
    res.error(
        "preset.unknown",
        f"method {manifest.name!r} has no preset {name!r}",
        ["method.preset"],
        hint=f"available presets: {available}" if available else "this method has no presets",
    )
    return None


def apply_preset(raw: dict[str, Any], preset: PresetDef) -> tuple[dict[str, Any], Fidelity]:
    """Fill preset values into paths the user left unset; explicit user values that differ are deviations."""
    fidelity = Fidelity(preset=f"{preset.method}/{preset.name}")
    return _fill_values(raw, preset.values, fidelity), fidelity


def _fill_values(raw: dict[str, Any], values: dict[str, PresetValue], fidelity: Fidelity) -> dict[str, Any]:
    """Fill `values` into unset paths of a copy of `raw`, recording provenance and deviations in `fidelity`."""
    merged = copy.deepcopy(raw)
    for path, item in values.items():
        present, user_value = get_path(raw, path)
        if path == "tasks" and present and user_value == []:
            present = False
        if not present:
            set_path(merged, path, item.value)
        elif canonical_json(normalize(path, user_value)) != canonical_json(normalize(path, item.value)):
            fidelity.deviations.append(Deviation(path=path, preset=item.value, actual=user_value))
            continue
        if item.source == "assumed":
            fidelity.assumed.append(path)
        else:
            fidelity.confirmed.append(path)
    return merged


def _check_paired_preset(
    raw: dict[str, Any],
    preset: PreprocessPresetDef | None,
    manifest: MethodManifest | None,
    catalog: Catalog,
    res: Resolution,
) -> None:
    """With a preprocess set, a quantizer method that has presets needs an explicit method preset.

    The implicit (default) method preset is an error, listing the preprocess preset's paired method presets
    when it declares any and the method's own presets otherwise; an explicit preset outside the paired list
    is a warning; `preset: null` (a full user config) is not checked.
    """
    method = raw.get("method") or {}
    paired = [] if preset is None else preset.quantizers.get(method.get("name"), [])
    requested = method.get("preset", DEFAULT_PRESET)
    if requested is None:
        return
    if requested == DEFAULT_PRESET:
        if paired:
            res.error(
                "preset.needs_choice",
                f"preprocess {preset.name!r} pairs with {method['name']} presets {paired}; set method.preset",
                ["method.preset"],
                hint=f"set method.preset to one of {paired}",
            )
            return
        if manifest is not None and manifest.field == "quantization":
            available = sorted(catalog.presets.get(manifest.name, {})) + sorted(manifest.custom_presets)
            if available:
                res.error(
                    "preset.needs_choice",
                    f"a preprocess is set; choose the {manifest.name} preset to run it with (method.preset)",
                    ["method.preset"],
                    hint=f"set method.preset to one of {available}, or null with an explicit config",
                )
        return
    if paired and requested not in paired:
        res.issues.append(
            Issue(
                code="preset.unpaired",
                severity=Severity.WARNING,
                paths=["method.preset"],
                message=f"preprocess {preset.name!r} is meant to run with {method['name']} presets {paired}, "
                f"not {requested!r}",
            )
        )


def _apply_preprocess_preset(
    raw: dict[str, Any], res: Resolution, manifest: MethodManifest | None, catalog: Catalog
) -> dict[str, Any]:
    """Check the method preset against a non-empty `preprocess` and fill the named preprocess preset's values
    into unset `preprocess.*` paths; unknown names are errors."""
    section = raw.get("preprocess")
    if not isinstance(section, dict) or not (section.get("preset") or section.get("steps")):
        return raw
    name = section.get("preset")
    preset = None
    if name is not None:
        preset = builtin_preprocess_presets().get(name)
        if preset is None:
            res.error(
                "preprocess.preset_unknown",
                f"no preprocess preset {name!r}",
                ["preprocess.preset"],
                hint=f"available presets: {sorted(builtin_preprocess_presets())}",
            )
            return raw
        res.preprocess_preset = preset
    _check_paired_preset(raw, preset, manifest, catalog, res)
    if preset is None:
        return raw
    fidelity = Fidelity()
    merged = _fill_values(raw, preset.values, fidelity)
    res.fidelity.confirmed += fidelity.confirmed
    res.fidelity.assumed += fidelity.assumed
    res.fidelity.deviations += fidelity.deviations
    return merged


def normalize(path: str, value: Any) -> Any:
    """Return ``value`` in the form preset comparison uses; ``tasks`` lists reduce to task names."""
    if path == "tasks" and isinstance(value, list):
        return [item if isinstance(item, str) else item.get("name") for item in value]
    return value


def _check_benchmarks(
    spec: RunSpec, manifest: MethodManifest, architecture: ArchitectureDef, catalog: Catalog, res: Resolution
) -> None:
    """Kernel benchmarks need a target GPU and a linear weight rule the benchmarked kernel accepts."""
    benchmarks = [p for p in res.tasks if p.task.benchmark is not None]
    if not benchmarks:
        return
    if spec.hardware.gpu is None:
        res.error(
            "kernel.hardware_required",
            "kernel benchmarks are only meaningful on a known GPU",
            ["hardware.gpu"],
            hint=f"set hardware.gpu to one of {sorted(catalog.gpus)}",
        )
    if "quant_rules" not in manifest.uses:
        res.error(
            "kernel.no_weights",
            f"{manifest.name} does not quantize weights; nothing to benchmark",
            ["method"],
        )
        return
    try:
        rules = [QuantRule.model_validate(r) for r in spec.method.config.get("rules", [])]
    except ValidationError:
        return  # reported by the method config check
    sites = sorted(s for s in architecture.site_names() if s.startswith("linear.") and s.endswith(".weight"))
    for plan in benchmarks:
        accepts = plan.task.benchmark.weights
        for site, rule in linear_weight_rules(rules, sites).items():
            why = _kernel_misfit(rule, accepts)
            if why is not None:
                res.error(
                    "kernel.unsupported_config",
                    f"{plan.task.name}: {site} {why}",
                    ["method.config.rules"],
                    hint=f"the kernel takes {accepts.type}{accepts.bits} weights, "
                    f"symmetric {accepts.symmetric}, group sizes {accepts.group_sizes} (null = per channel), "
                    "on every decoder layer",
                )
                break


def _kernel_misfit(rule: QuantRule | None, accepts: KernelWeights) -> str | None:
    if rule is None:
        return "is not quantized"
    if rule.layers != "all":
        return f"is only quantized on layers {rule.layers!r}"
    quant = rule.quant
    if quant.type != accepts.type or quant.bits not in accepts.bits:
        return f"uses {quant.type}{getattr(quant, 'bits', '')}"
    if quant.symmetric not in accepts.symmetric:
        return "uses asymmetric scales" if not quant.symmetric else "uses symmetric scales"
    group = (
        quant.group_size
        if quant.granularity == "per_group"
        else None
        if quant.granularity == "per_channel"
        else "x"
    )
    if group not in accepts.group_sizes:
        return f"uses {quant.granularity}{f' {quant.group_size}' if quant.group_size else ''} scales"
    return None


def _check_metrics(spec: RunSpec, catalog: Catalog, res: Resolution) -> None:
    for index, name in enumerate(spec.metrics):
        if name not in catalog.metrics or name not in DERIVED:
            res.error(
                "metric.unknown",
                f"{name!r} is not a derived metric",
                [f"metrics[{index}]"],
                hint=f"derived metrics: {sorted(DERIVED)}",
            )
    for plan in res.tasks:
        for name in plan.task.derived:
            if name not in DERIVED:
                res.error(
                    "metric.unimplemented",
                    f"task {plan.task.name!r} reports {name!r} by default but nothing computes it",
                    ["tasks"],
                    hint="register it with @derived in effeval.metrics.derived",
                )
        for name in spec.metrics:
            if name in DERIVED and not applicable(name, plan.produces, plan.task.sample_scores):
                res.note(
                    "metric.not_applicable",
                    f"{name} is not computed for {plan.task.name}: its outputs lack what {name} needs",
                    ["metrics"],
                )


def job_metrics(task: TaskDef, produces: frozenset[Artifact], extra: list[str]) -> list[str]:
    """Derived metrics a job computes: the task's defaults, then `extra`, where applicable."""
    names = dict.fromkeys([*task.derived, *extra])
    return [n for n in names if n in DERIVED and applicable(n, produces, task.sample_scores)]


def _task_settings(
    task: TaskDef, given: dict[str, Any], index: int, res: Resolution
) -> tuple[int | float | None, int | None, dict[str, Any], dict[str, Any]] | None:
    """(limit, effective num_fewshot, other settings that differ from the defaults, generation overrides).

    The generation overrides are the max_new_tokens, do_sample and temperature the task sets itself.
    None after an error."""
    for key in GENERATION_TASK_KEYS:
        if key in given and key not in task.params.model_fields:
            res.error(
                "task.params_invalid",
                f"{task.name}: {key} does not apply, this task never generates text",
                [f"tasks[{index}].params.{key}"],
                hint=f"settings of {task.name}: {sorted(task.params.model_fields)}",
            )
            return None
    try:
        values = task.settings(given).model_dump(mode="json")
    except ValidationError as exc:
        for err in exc.errors():
            path = ".".join(str(p) for p in err["loc"])
            res.error(
                "task.params_invalid",
                f"{task.name}: {err['msg']}",
                [f"tasks[{index}].params.{path}"],
                hint=f"settings of {task.name}: {sorted(task.params.model_fields)}",
            )
        return None
    defaults = task.settings({}).model_dump(mode="json")
    limit = values.pop("limit", -1)
    generation = {key: values.pop(key, None) for key in GENERATION_TASK_KEYS}
    if generation["temperature"] is not None and not generation["do_sample"]:
        res.error(
            "task.params_invalid",
            f"{task.name}: temperature needs do_sample on for this task",
            [f"tasks[{index}].params.temperature"],
            hint="turn on do_sample, or remove temperature",
        )
        return None
    num_fewshot = task.num_fewshot
    if "num_fewshot" in values:
        given_fewshot = values.pop("num_fewshot")
        if given_fewshot is not None:
            if task.fewshot_allowed is not None and given_fewshot not in task.fewshot_allowed:
                res.error(
                    "task.params_invalid",
                    f"{task.name}: num_fewshot {given_fewshot} is not accepted by this task",
                    [f"tasks[{index}].params.num_fewshot"],
                    hint=f"accepted values: {task.fewshot_allowed}"
                    if task.fewshot_allowed
                    else "this task takes no few-shot override",
                )
                return None
            num_fewshot = given_fewshot
    changed = {k: v for k, v in values.items() if v != defaults[k]}
    return (None if limit == -1 else limit), num_fewshot, changed, generation


def _evaluator_for(
    task: TaskDef,
    mode: Mode,
    manifest: MethodManifest,
    catalog: Catalog,
    res: Resolution,
    index: int,
    runtime: str,
) -> EvaluatorDef | None:
    """The task's evaluator if it can score `mode` under `runtime` on models of `manifest`.

    Otherwise records the error.

    On an engine runtime the model is not a handle produced by the method, so the evaluator must take engine
    handles instead.
    """
    evaluator = catalog.evaluators[task.evaluator]
    path = f"tasks[{index}]"
    if mode not in evaluator.capabilities.modes:
        res.error(
            "mode.evaluator_unsupported",
            f"{evaluator.name} cannot score {task.name!r} in {mode.value} mode",
            [f"{path}.mode"],
        )
        return None
    if runtime != "hf":
        if evaluator.name != "speed":
            res.error(
                "runtime.task_unsupported",
                f"{runtime} only runs speed tasks, not {task.name!r}",
                [f"{path}.name", "runtime.name"],
            )
            return None
        if HandleKind.ENGINE not in evaluator.capabilities.handles:
            res.error(
                "handle.unsupported",
                f"{evaluator.name} cannot run on the {runtime} engine",
                [f"{path}.name", "runtime.name"],
            )
            return None
    elif not (manifest.capabilities.handles & evaluator.capabilities.handles):
        res.error(
            "handle.unsupported",
            f"{evaluator.name} cannot run models produced by {manifest.name!r}",
            [f"{path}.name"],
        )
        return None
    return evaluator


def _check_sites(rules: list[Any], architecture: ArchitectureDef, entry: ModelEntry, res: Resolution) -> None:
    """Site patterns must exist on the model's architecture; tied lm_head weights stay untouched."""
    vocabulary = architecture.site_names()
    for r_index, rule in enumerate(rules):
        patterns = rule.get("sites", []) if isinstance(rule, dict) else []
        for s_index, pattern in enumerate(patterns):
            if not any(fnmatch.fnmatchcase(name, pattern) for name in vocabulary):
                res.error(
                    "site.unknown",
                    f"site pattern {pattern!r} matches nothing on {architecture.name}",
                    [f"method.config.rules[{r_index}].sites[{s_index}]"],
                    hint=f"{architecture.name} sites: {vocabulary}",
                )
        if entry.tie_word_embeddings and any(fnmatch.fnmatchcase("lm_head.weight", p) for p in patterns):
            res.error(
                "site.tied_weights",
                f"{entry.code} ties lm_head to the input embedding; editing lm_head.weight would also "
                "change the embedding",
                [f"method.config.rules[{r_index}].sites"],
                hint="quantize lm_head.input/output instead, or use an untied model",
            )


def _check_decomposition(spec: RunSpec, manifest: MethodManifest, entry: ModelEntry, res: Resolution) -> None:
    """Warn when the shared-down q/k/v factorisation at `decomposition.rank_ratio` saves no parameters."""
    decomposition = spec.method.config.get("decomposition")
    ratio = decomposition.get("rank_ratio") if isinstance(decomposition, dict) else None
    if manifest.architecture_transform != "qkv_shared" or entry.shape is None:
        return
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
        return
    shape = entry.shape
    budget = qkv_budget(shape, ratio)
    if budget.param_ratio < 1:
        return
    out_total = (shape.heads + 2 * shape.kv_heads) * shape.head_dim
    r_eq = 2 * (shape.hidden * out_total / (shape.hidden + out_total)) / shape.hidden
    res.issues.append(
        Issue(
            code="decomposition.no_saving",
            severity=Severity.WARNING,
            paths=["method.config.decomposition.rank_ratio"],
            message=f"rank_ratio {ratio} keeps on average {budget.mean_rank:.0f} of {shape.hidden} singular "
            f"values per layer: the factored q/k/v hold {budget.param_ratio:.1%} of the dense parameters and "
            f"the latent is {budget.latent_ratio:.1%} of the K/V width",
            hint=f"rank_ratio below {r_eq:.3f} saves parameters on this model",
        )
    )


def _check_selection_basis(spec: RunSpec, manifest: MethodManifest, res: Resolution) -> None:
    """Warn when the LSVD FFN channel order is computed in a basis the job's preprocessing does not apply."""
    ffn = spec.method.config.get("ffn")
    if manifest.architecture_transform != "lsvd" or not isinstance(ffn, dict):
        return
    if ffn.get("enabled") is not True or ffn.get("selection_rotation") != "random_hadamard":
        return
    seed = ffn.get("selection_rotation_seed", 0)
    applied = [
        entry.source
        for step in (spec.preprocess.steps if spec.preprocess is not None else [])
        if step.kind == "rotation"
        for entry in step.entries
        if entry.position == "residual"
    ]
    if any(s.type == "random_hadamard" and s.seed == seed for s in applied):
        return
    found = ", ".join(f"{s.type} rotation" for s in applied) or "no residual rotation"
    res.issues.append(
        Issue(
            code="decomposition.selection_basis",
            severity=Severity.WARNING,
            paths=["method.config.ffn.selection_rotation"],
            message=f"the FFN channel order is computed in the basis of a random Hadamard with seed {seed}, "
            f"but this job's preprocessing applies {found}",
            hint="set selection_rotation to none without a residual rotation, or match the seed",
        )
    )


def _position_table(
    architecture: ArchitectureDef, entry: ModelEntry
) -> dict[str, dict[str, dict[str, str]]] | None:
    """Position id -> kind -> mode -> "ok" or the reason it is unavailable; None without graph or shape."""
    from effeval.core.graph import positions

    if architecture.graph is None or entry.shape is None:
        return None
    return positions(architecture.graph, entry.shape)


def _check_preprocess(
    spec: RunSpec, architecture: ArchitectureDef, entry: ModelEntry, res: Resolution
) -> None:
    """Check each preprocess step and entry against the cross-field rules and the model's position table."""
    pre = spec.preprocess
    if pre is None:
        return
    if not pre.steps:
        res.error(
            "preprocess.empty",
            "preprocess has no steps",
            ["preprocess.steps"],
            hint="list at least one step, or name a preset that provides them",
        )
        return
    table = None
    table_loaded = False
    norm_scaling_at: int | None = None
    value_rotation_at: int | None = None
    output_position = None if architecture.graph is None else attention_output_position(architecture.graph)
    for i, step in enumerate(pre.steps):
        base = f"preprocess.steps[{i}]"
        if step.calibration is None and any(e.source.type in DATA_SOURCES for e in step.entries):
            res.error(
                "preprocess.calibration_required",
                f"a {step.kind} step with a data-driven source needs calibration data",
                [f"{base}.calibration"],
            )
        for j, item in enumerate(step.entries):
            path = f"{base}.entries[{j}]"
            if item.source.type in RESERVED_SOURCES:
                res.error(
                    "preprocess.source_unimplemented",
                    f"source type {item.source.type!r} is not implemented yet",
                    [f"{path}.source.type"],
                )
            if item.position in GLOBAL_POSITIONS and item.layers != "all":
                res.error(
                    "preprocess.layers_global",
                    f"{item.position} is one matrix for the whole model; layers must be 'all'",
                    [f"{path}.layers"],
                )
            if step.kind == "rotation" and item.position == "residual":
                if norm_scaling_at is not None:
                    res.error(
                        "preprocess.order",
                        "the residual rotation folds the norm gains into the linears, which cancels the "
                        f"scaling folded into a norm gain in step {norm_scaling_at}; the rotation must come first",
                        ["preprocess.steps"],
                        hint="put the rotation step before the scaling step",
                    )
                if entry.tie_word_embeddings and item.options.get("untie_embeddings") is not True:
                    res.error(
                        "preprocess.tied_embeddings",
                        f"{entry.code} ties lm_head to the input embedding; rotating the residual stream "
                        "needs them untied",
                        [f"{path}.options.untie_embeddings"],
                        hint="set options.untie_embeddings to true",
                    )
            if (
                step.kind == "scaling"
                and item.mode == "offline"
                and item.position.startswith("norm.")
                and norm_scaling_at is None
            ):
                norm_scaling_at = i
            if step.kind == "scaling" and item.position == "attn.v_head" and value_rotation_at is not None:
                res.error(
                    "preprocess.order",
                    "an earlier rotation step in step "
                    f"{value_rotation_at} acts on the value path (attn.v_head or the o_proj input); an "
                    "offline fold made after it would not see that rotation "
                    "and would fold in the wrong basis",
                    ["preprocess.steps"],
                    hint="put the scaling step first",
                )
            if step.kind == "rotation" and item.position in ("attn.v_head", output_position):
                value_rotation_at = i
            if not table_loaded:
                table, table_loaded = _position_table(architecture, entry), True
                if table is None:
                    res.error(
                        "preprocess.architecture_unsupported",
                        f"{architecture.name} has no inspected architecture graph or model shape",
                        ["model.name"],
                    )
            if table is None:
                continue
            kinds = table.get(item.position)
            if kinds is None:
                res.error(
                    "preprocess.position_unknown",
                    f"{architecture.name} has no position {item.position!r}",
                    [f"{path}.position"],
                    hint=f"positions: {sorted(table)}",
                )
                continue
            verdict = kinds[step.kind][item.mode]
            if verdict != "ok":
                res.error(
                    "preprocess.mode_unavailable",
                    f"{step.kind} {item.mode} at {item.position} is not available: {verdict}",
                    [f"{path}.mode"],
                )
                continue
            if step.kind == "scaling":
                problem = source_position_problem(step.kind, item.source.type, item.position)
                if problem is not None:
                    res.error("preprocess.source_position", problem, [f"{path}.source.type"])
            if step.kind == "rotation":
                problem = rotation_option_problem(item.position, item.source, item.options)
                if problem is not None:
                    res.error("preprocess.option", problem, [f"{path}.options"])
                width = position_dim(architecture.graph, entry.shape, item.position)
                problem = rotation_size_problem(item.source, item.position, width)
                if problem is not None:
                    res.error("preprocess.hadamard_size", problem, [f"{path}.source"])


def _resolved_model(entry: ModelEntry, dtype: str | None) -> ResolvedModel:
    return ResolvedModel(
        name=entry.code,
        checkpoint=entry.checkpoint,
        revision=entry.revision,
        architecture=entry.architecture,
        dtype=dtype or entry.dtype,
    )


def _check_decoding(
    raw: dict[str, Any], spec: RunSpec, entry: ModelEntry, catalog: Catalog, res: Resolution
) -> None:
    """Rules of `decoding`, the draft role, the runtime provider and the timed (speed) mode."""
    decoding = spec.decoding
    speculative = decoding is not None and decoding.mode == "speculative"
    if decoding is not None and not speculative:
        given = {"draft": decoding.draft, "num_speculative_tokens": decoding.num_speculative_tokens}
        for name, value in given.items():
            if value is not None:
                res.error(
                    "decoding.ineffective",
                    f"decoding.{name} has no effect: decoding.mode is autoregressive",
                    [f"decoding.{name}"],
                    hint="set decoding.mode to speculative, or remove it",
                )
    if speculative:
        _check_speculative(spec, entry, catalog, res)

    timed = any(p.mode == Mode.SPEED for p in res.tasks)
    if timed and spec.hardware.gpu is None:
        res.error(
            "speed.hardware_required",
            "speed tasks are only meaningful on a known GPU",
            ["hardware.gpu"],
            hint=f"set hardware.gpu to one of {sorted(catalog.gpus)}",
        )
    if (speculative or timed) and spec.generation.batch_size != 1:
        res.error(
            "decoding.batch_size",
            f"speculative and speed runs decode one sequence at a time, got batch_size "
            f"{spec.generation.batch_size}",
            ["generation.batch_size"],
        )
    if spec.runtime.name != "hf":
        _check_engine_runtime(raw, spec, catalog, res)
        _check_static_cache_capacity(spec, res, speculative, hint="raise runtime.static_cache_tokens")
    elif (speculative or timed) and spec.runtime.cuda_graphs:
        _check_static_cache_rules(spec, res)
        _check_static_cache_capacity(spec, res, speculative)


def _check_static_cache_rules(spec: RunSpec, res: Resolution) -> None:
    """CUDA-graph decoding with the static KV cache cannot serve some rules.

    Rejected: attn.k/attn.v sequence options, attn.scores/attn.probs sites, MX scale e4m3 on activation sites
    (a host sync during capture) and zero_point float over the key axis of attn.k/attn.v (the static cache's
    zero rows change the min/max).
    """
    rules = spec.method.config.get("rules", []) if isinstance(spec.method.config, dict) else []
    for r_index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        quant = rule.get("quant")
        quant = quant if isinstance(quant, dict) else {}
        sites = [p for p in rule.get("sites", []) if isinstance(p, str)]
        options = [k for k in ("residual", "group_axis") if quant.get(k) is not None]
        unsupported = ("attn.scores", "attn.probs")
        inside = [p for p in sites if any(fnmatch.fnmatchcase(n, p) for n in unsupported)]
        runtime_sites = [p for p in sites if p.rsplit(".", 1)[-1] != "weight"]
        host_sync = (
            [f"MX scale e4m3 on {runtime_sites}"]
            if quant.get("type") == "mx" and quant.get("scale") == "e4m3" and runtime_sites
            else []
        )
        key_axis = (
            quant.get("type") == "int"
            and quant.get("zero_point") == "float"
            and quant.get("granularity") in ("per_tensor", "per_channel")
            and any(fnmatch.fnmatchcase(n, p) for p in sites for n in ("attn.k", "attn.v"))
        )
        float_zero = ["zero_point float over the key axis of attn.k/attn.v"] if key_axis else []
        if options or inside or host_sync or float_zero:
            what = (
                f"quant options {options}"
                if options
                else f"sites {inside}"
                if inside
                else (host_sync or float_zero)[0]
            )
            res.error(
                "runtime.static_cache_unsupported",
                f"rule {r_index} uses {what}, which the static KV cache of speed and speculative jobs "
                "does not support",
                [f"method.config.rules[{r_index}]"],
                hint="set runtime.cuda_graphs: false (both arms then decode with the dynamic cache)",
            )


def _check_static_cache_capacity(
    spec: RunSpec,
    res: Resolution,
    speculative: bool,
    hint: str = "raise runtime.static_cache_tokens, or set runtime.cuda_graphs: false",
) -> None:
    """A generation must fit the static KV cache (an engine's context length): its prompt plus
    max_new_tokens - 1 positions.

    The prompt length is unknown here, so the check uses one prompt token: it rejects only tasks that cannot
    fit whatever the prompt is. The runtime fails a longer generation with the same hint.
    """
    capacity = spec.runtime.static_cache_tokens
    runtime = spec.runtime.name
    where = "the static KV cache" if runtime == "hf" else f"the {runtime} engine's context"
    for index, plan in enumerate(res.tasks):
        if plan.mode != Mode.SPEED and (not speculative or plan.mode == Mode.LOGLIKELIHOOD):
            continue
        budget = spec.generation.max_new_tokens if plan.max_new_tokens is None else plan.max_new_tokens
        if budget > capacity:
            res.error(
                "runtime.static_cache_capacity",
                f"{plan.task.name} generates up to {budget} tokens after its prompt, more than the "
                f"{capacity} positions of {where}",
                [f"tasks[{index}].params.max_new_tokens", "runtime.static_cache_tokens"],
                hint=hint,
            )


def _samples(spec: RunSpec, plan: TaskPlan) -> bool:
    """Whether the task samples: its own do_sample, else the run-level one."""
    return spec.generation.do_sample if plan.do_sample is None else plan.do_sample


def _check_sampling(spec: RunSpec, res: Resolution) -> None:
    """top_p / top_k need sampling and only speed tasks pass them; repeats of a sampled run share a seed."""
    sampling = [(index, plan) for index, plan in enumerate(res.tasks) if _samples(spec, plan)]
    for key in ("top_p", "top_k"):
        if getattr(spec.generation, key) is None:
            continue
        if not sampling:
            res.error(
                "generation.filter_needs_sampling",
                f"generation.{key} needs do_sample on",
                [f"generation.{key}"],
                hint="turn on generation.do_sample, or remove it",
            )
    if spec.generation.top_p is not None or spec.generation.top_k is not None:
        for index, plan in sampling:
            if plan.mode != Mode.SPEED:
                res.error(
                    "generation.filter_speed_only",
                    f"{plan.task.name}: top_p/top_k apply to speed tasks only; "
                    f"{plan.evaluator.name} does not pass them",
                    ["generation.top_p", "generation.top_k", f"tasks[{index}]"],
                )
    repeats = spec.repeats
    if sampling and repeats.seeds is None and repeats.count is not None and repeats.count > 1:
        res.note(
            "repeats.sampled_same_seed",
            f"every repeat uses seed {spec.determinism.seed}, so sampled runs repeat the same tokens and "
            "differ only in timing; use repeats.seeds for independent samples",
            ["repeats.count"],
        )


def _check_speculative(spec: RunSpec, entry: ModelEntry, catalog: Catalog, res: Resolution) -> None:
    decoding = spec.decoding
    if decoding.draft is None:
        res.error("decoding.draft_required", "speculative decoding needs a draft model", ["decoding.draft"])
    if decoding.num_speculative_tokens is None:
        res.error(
            "decoding.k_required",
            "speculative decoding needs num_speculative_tokens",
            ["decoding.num_speculative_tokens"],
            hint="the best value depends on the draft; there is no default",
        )
    if spec.field != "speculative_decoding":
        res.error(
            "decoding.field_required",
            "speculative decoding is evaluated in the speculative_decoding field",
            ["field", "decoding.mode"],
        )
    for index, plan in enumerate(res.tasks):
        if plan.mode == Mode.LOGLIKELIHOOD:
            res.error(
                "decoding.no_effect",
                f"{plan.task.name} is scored by loglikelihood and never decodes, "
                "so speculative decoding has no effect",
                [f"tasks[{index}]", "decoding.mode"],
            )
            continue
        if plan.mode != Mode.SPEED and _samples(spec, plan):
            res.error(
                "decoding.sampling_unavailable",
                "speculative sampling is available on speed tasks (speed/<prompt set>) only; "
                f"{plan.task.name} decodes greedily under speculative decoding",
                ["generation.do_sample", f"tasks[{index}].params.do_sample"],
                hint="set do_sample false for this task",
            )
    draft = decoding.draft
    if draft is None:
        return
    draft_entry = catalog.models.get(draft.model.name)
    if draft_entry is not None:
        res.draft_model = _resolved_model(draft_entry, draft.model.dtype)
    res.draft_manifest = catalog.manifests.get(draft.method.name)
    if draft_entry is None:
        res.error(
            "decoding.draft_unknown_model",
            f"unknown draft model code {draft.model.name!r}",
            ["decoding.draft.model.name"],
            hint="register it once with `effeval model add <checkpoint>`",
        )
    elif entry.shape is None or draft_entry.shape is None:
        res.error(
            "decoding.vocab_mismatch",
            "cannot check that draft and target share a vocabulary: a model entry has no shape",
            ["decoding.draft.model.name", "model.name"],
        )
    elif entry.shape.vocab != draft_entry.shape.vocab:
        res.error(
            "decoding.vocab_mismatch",
            f"draft {draft_entry.code} has vocab {draft_entry.shape.vocab}, "
            f"target {entry.code} has {entry.shape.vocab}",
            ["decoding.draft.model.name", "model.name"],
        )
    if draft.method.name not in catalog.manifests:
        res.error(
            "method.unknown",
            f"unknown draft method {draft.method.name!r}",
            ["decoding.draft.method.name"],
            hint=f"known methods: {sorted(catalog.manifests)}",
        )
    elif draft.method.name != "base" or draft.preprocess is not None:
        res.error(
            "decoding.draft_method_later",
            "methods and preprocessing on the draft model are not supported yet: "
            "the draft runs as a base model",
            ["decoding.draft.method", "decoding.draft.preprocess"],
        )


def _check_engine_runtime(raw: dict[str, Any], spec: RunSpec, catalog: Catalog, res: Resolution) -> None:
    """vLLM and SGLang load plain checkpoints: no PyTorch-hook methods, no transformers settings."""
    runtime = spec.runtime.name
    if isinstance(raw.get("runtime"), dict) and "attn_implementation" in raw["runtime"]:
        res.error(
            "runtime.attn_unsupported",
            f"runtime.attn_implementation only applies to the hf runtime, not {runtime}",
            ["runtime.attn_implementation"],
        )
    if (
        runtime == "sglang"
        and spec.generation.top_k is not None
        and spec.generation.top_p is not None
        and any(_samples(spec, plan) for plan in res.tasks)
    ):
        res.error(
            "runtime.sampling_filters",
            "SGLang filters top_k and top_p jointly for plain decoding but one after the other when "
            "verifying drafts, so the two runs would sample different distributions; "
            "set only one of them",
            ["generation.top_k", "generation.top_p"],
        )
    if res.draft_model is not None and res.model is not None and res.draft_model.dtype != res.model.dtype:
        res.error(
            "runtime.draft_dtype",
            f"{runtime} runs the draft model in the target's dtype, but the draft is "
            f"{res.draft_model.dtype} and the target {res.model.dtype}",
            ["decoding.draft.model.dtype", "model.dtype"],
            hint="give the draft the target's dtype",
        )
    if spec.hardware.count != 1:
        res.error(
            "runtime.single_gpu",
            f"the {runtime} runtime runs on one GPU, got hardware.count {spec.hardware.count}",
            ["hardware.count"],
        )
    roles = [("", spec.method.name, spec.preprocess)]
    if spec.decoding is not None and spec.decoding.draft is not None:
        draft = spec.decoding.draft
        roles.append(("decoding.draft.", draft.method.name, draft.preprocess))
    for prefix, method, preprocess in roles:
        manifest = catalog.manifests.get(method)
        if (manifest is not None and manifest.entry is not None) or preprocess is not None:
            res.error(
                "runtime.method_unsupported",
                f"methods run as PyTorch hooks in the transformers runtime; {runtime} cannot run them",
                [f"{prefix}method", f"{prefix}preprocess"],
            )


def resolve(raw: dict[str, Any], catalog: Catalog) -> Resolution:
    """Resolve one RunSpec document (no sweep). Never raises for user errors; they become Issues."""
    res = Resolution()
    if not isinstance(raw, dict):
        res.error("spec.invalid", "a RunSpec must be a mapping")
        return res
    if raw.get("sweep"):
        res.error("spec.invalid", "resolve() takes one combination; expand `sweep` first", ["sweep"])
        return res

    method_name = (raw.get("method") or {}).get("name")
    manifest = catalog.manifests.get(method_name) if isinstance(method_name, str) else None
    merged = raw
    if manifest is not None:
        res.manifest = manifest
        res.preset = _preset_for(raw, manifest, catalog, res)
        if not res.ok:
            return res
        if res.preset is not None:
            merged, res.fidelity = apply_preset(raw, res.preset)
    merged = _apply_preprocess_preset(merged, res, manifest, catalog)
    if not res.ok:
        return res

    try:
        spec = RunSpec.model_validate(merged)
    except ValidationError as exc:
        for err in exc.errors():
            res.error("spec.invalid", err["msg"], [".".join(str(p) for p in err["loc"])])
        return res
    res.spec = spec

    if manifest is None:
        res.error(
            "method.unknown",
            f"unknown method {spec.method.name!r}",
            ["method.name"],
            hint=f"known methods: {sorted(catalog.manifests)}; register custom methods with a manifest",
        )
        return res
    field_def = catalog.fields.get(spec.field)
    if field_def is None:
        res.error(
            "field.unknown",
            f"unknown field {spec.field!r}",
            ["field"],
            hint=f"known: {sorted(catalog.fields)}",
        )
        return res
    res.field_def = field_def
    if manifest.field not in {spec.field, "any"}:
        res.error(
            "method.field_mismatch",
            f"method {manifest.name!r} belongs to field {manifest.field!r}",
            ["field", "method.name"],
        )

    entry = catalog.models.get(spec.model.name)
    if entry is None:
        res.error(
            "model.unknown",
            f"unknown model code {spec.model.name!r}",
            ["model.name"],
            hint="register it once with `effeval model add <checkpoint>`; "
            "codes are the lower-cased repo name",
        )
        return res
    architecture = catalog.architectures[entry.architecture]
    res.model_entry, res.architecture = entry, architecture
    res.model = _resolved_model(entry, spec.model.dtype)
    weight_bytes = entry.params * BYTES_PER_PARAM * WEIGHT_HEADROOM
    if weight_bytes > spec.hardware.count * GPU_USABLE_BYTES:
        res.error(
            "hardware.too_small",
            f"{entry.code} needs about {weight_bytes / GIB:.0f} GiB of weights and activations, "
            f"{spec.hardware.count} GPU(s) offer {spec.hardware.count * GPU_USABLE_BYTES / GIB:.0f} GiB",
            ["hardware.count"],
            hint="set hardware.count to split the model across more GPUs",
        )
    if entry.revision is None:
        res.issues.append(
            Issue(
                code="model.unpinned",
                severity=Severity.WARNING,
                paths=["model.name"],
                message=f"{entry.code} has no pinned revision: its weights can change underneath "
                "the same code",
            )
        )
    if manifest.architectures is not None and entry.architecture not in manifest.architectures:
        res.error(
            "method.architecture_unsupported",
            f"{manifest.name} supports {manifest.architectures}, not {entry.architecture} ({entry.code})",
            ["method.name", "model.name"],
        )

    res.issues.extend(validate_method_config(manifest, spec.method.config, "method.config"))
    layout = architecture
    if manifest.architecture_transform is not None:
        try:
            layout = transform_architecture(architecture, manifest.architecture_transform, spec.method.config)
        except ValueError as exc:
            res.error("method.architecture_unsupported", str(exc), ["method.name", "model.name"])
            return res
        _check_decomposition(spec, manifest, entry, res)
        _check_selection_basis(spec, manifest, res)
    if "quant_rules" in manifest.uses and isinstance(spec.method.config.get("rules"), list):
        _check_sites(spec.method.config["rules"], layout, entry, res)
    _check_preprocess(spec, layout, entry, res)

    if not spec.tasks:
        res.error("tasks.required", "no tasks given and the method has no preset tasks", ["tasks"])
    for name in spec.evaluator.passthrough:
        if name not in catalog.evaluators:
            res.error(
                "evaluator.unknown",
                f"passthrough for unknown evaluator {name!r}",
                [f"evaluator.passthrough.{name}"],
                hint=f"known evaluators: {sorted(catalog.evaluators)}",
            )
    for index, ref in enumerate(spec.tasks):
        task = catalog.tasks.get(ref.name)
        if task is None:
            close = difflib.get_close_matches(ref.name, catalog.tasks, n=20, cutoff=0.5)
            res.error(
                "task.unknown",
                f"unknown task {ref.name!r}",
                [f"tasks[{index}].name"],
                hint=f"closest known tasks: {close}" if close else "tasks are named <evaluator>/<task>",
            )
            continue
        mode = ref.mode or task.default_mode
        if mode not in task.modes:
            res.error(
                "mode.task_unsupported",
                f"task {task.name!r} does not support {mode.value} mode",
                [f"tasks[{index}].mode"],
                hint=f"supported: {[m.value for m in task.modes]}",
            )
            continue
        if mode not in field_def.modes:
            res.error(
                "mode.field_conflict",
                f"field {field_def.name!r} cannot be evaluated in {mode.value} mode",
                [f"tasks[{index}].mode", "field"],
                hint=f"field supports: {[m.value for m in field_def.modes]}",
            )
            continue
        settings = _task_settings(task, ref.params, index, res)
        if settings is None:
            continue
        evaluator = _evaluator_for(task, mode, manifest, catalog, res, index, spec.runtime.name)
        if evaluator is not None:
            res.tasks.append(
                TaskPlan(
                    task=task,
                    mode=mode,
                    evaluator=evaluator,
                    limit=settings[0],
                    num_fewshot=settings[1],
                    params=settings[2],
                    **settings[3],
                )
            )

    _check_metrics(spec, catalog, res)
    _check_benchmarks(spec, manifest, architecture, catalog, res)

    explicit_generation = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
    if res.tasks and all(t.mode == Mode.LOGLIKELIHOOD for t in res.tasks):
        for key in GENERATION_ONLY_KEYS:
            if key in explicit_generation:
                res.error(
                    "mode.ineffective_param",
                    f"generation.{key} has no effect: every task is loglikelihood",
                    [f"generation.{key}"],
                    hint="remove it, or add a generation-mode task",
                )
    for key, required in field_def.generation.items():
        if key in explicit_generation and explicit_generation[key] != required:
            res.error(
                "field.generation_conflict",
                f"field {field_def.name!r} requires generation.{key} = {required!r}",
                [f"generation.{key}"],
            )
    given_generation = merged.get("generation") if isinstance(merged.get("generation"), dict) else {}
    defaults = {k: v for k, v in field_def.generation.items() if k not in given_generation}
    if defaults:
        spec = res.spec = spec.model_copy(update={"generation": spec.generation.model_copy(update=defaults)})
    _check_decoding(raw, spec, entry, catalog, res)
    _check_sampling(spec, res)

    if spec.hardware.gpu is not None and catalog.gpu(spec.hardware.gpu) is None:
        res.error(
            "hardware.unknown",
            f"GPU {spec.hardware.gpu!r} is not in hwdb",
            ["hardware.gpu"],
            hint=f"known: {sorted(catalog.gpus)}",
        )

    explicit_compare = spec.compare is not None
    for name in spec.compare if explicit_compare else field_def.compare:
        if name not in COMPARATIVE:
            res.error(
                "metric.unknown",
                f"unknown comparative metric {name!r}",
                ["compare"],
                hint=f"known: {sorted(COMPARATIVE)}",
            )
            continue
        for plan in res.tasks:
            why = why_not_applicable(name, plan.produces, plan.task.sample_scores)
            if why is not None and explicit_compare:
                res.error(
                    "metric.missing_artifact",
                    f"{name} cannot be computed for {plan.task.name} with {plan.evaluator.name}: {why}",
                    ["compare"],
                )
            elif why is not None:
                res.note("metric.skipped", f"{name} is not computed for {plan.task.name}: {why}")
        if name not in res.compare:
            res.compare.append(name)
    return res
