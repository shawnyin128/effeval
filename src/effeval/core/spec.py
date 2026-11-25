"""RunSpec (user input), JobSpec (one frozen job), Plan (what a compute node executes) and RunKey."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from effeval.core.canonical import canonical_json, sha256_hex
from effeval.core.catalog import ModelEntry
from effeval.core.listing import TaskListing
from effeval.core.manifest import MethodManifest
from effeval.core.preprocess import Preprocess, ResolvedPreprocess
from effeval.core.types import Fidelity, Issue, Mode, StrictModel

DEFAULT_PRESET = "default"
"""Sentinel for `method.preset`: use the method's default preset (if it has one). `null` disables presets."""


Dtype = Literal["bfloat16", "float16", "float32"]


class ModelRef(StrictModel):
    name: str = Field(
        min_length=1, description="Global model code from the model registry, e.g. qwen3-8b-base."
    )
    dtype: Dtype | None = Field(default=None, description="Defaults to the checkpoint's native dtype.")


class ResolvedModel(StrictModel):
    name: str
    checkpoint: str
    revision: str | None
    architecture: str
    dtype: Dtype


class MethodRef(StrictModel):
    name: str
    version: str | None = Field(default=None, description="Filled from the method manifest when frozen.")
    preset: str | None = Field(
        default=DEFAULT_PRESET, description="Preset name, 'default', or null for none."
    )
    config: dict[str, Any] = Field(default_factory=dict)


class RuntimeSpec(StrictModel):
    name: Literal["hf", "vllm", "sglang"] = Field(
        default="hf", description="Inference provider; hf is the transformers provider."
    )
    attn_implementation: Literal["sdpa", "eager"] = Field(
        default="sdpa",
        description="Base attention backend when no attention-internal site is quantized; only used by hf.",
    )
    cuda_graphs: bool = Field(
        default=True,
        description="hf: static KV cache and CUDA graphs for effeval's decoders; vllm: false sets "
        "enforce_eager (also disables torch.compile); sglang: false sets disable_cuda_graph.",
    )
    static_cache_tokens: int = Field(
        default=4096,
        ge=256,
        description="Cache positions per generation (prompt + max_new_tokens - 1). hf: static cache size "
        "with cuda_graphs; vllm/sglang: the engine context length is this plus one.",
    )


class RoleSpec(StrictModel):
    """A model role besides the target, e.g. the draft of speculative decoding."""

    model: ModelRef
    method: MethodRef = Field(default_factory=lambda: MethodRef(name="base", preset=None))
    preprocess: Preprocess | None = None

    @field_validator("model", mode="before")
    @classmethod
    def _model_shorthand(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value


class DecodingSpec(StrictModel):
    mode: Literal["autoregressive", "speculative"] = "autoregressive"
    draft: RoleSpec | None = Field(default=None, description="The draft model; required iff speculative.")
    num_speculative_tokens: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description="Tokens the draft proposes per round; required iff speculative.",
    )


class TaskRef(StrictModel):
    name: str
    mode: Mode | None = Field(default=None, description="Defaults to the task's default mode.")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark settings, e.g. limit; unset ones keep their defaults."
    )


class EvaluatorRef(StrictModel):
    passthrough: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Evaluator name -> library-specific options; part of the RunKey and flagged in tables.",
    )


class GenerationSpec(StrictModel):
    do_sample: bool = False
    temperature: float | None = Field(default=None, ge=0.0)
    max_new_tokens: int = Field(default=512, ge=1)
    batch_size: int = Field(default=8, ge=1)


class DeterminismSpec(StrictModel):
    seed: int = 0
    deterministic_algorithms: bool = True
    tf32: bool = False


class RepeatSpec(StrictModel):
    count: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Run each job this many times with determinism.seed; omitted: once. With `seeds`, must equal "
        "its length.",
    )
    seeds: list[int] | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Run each job once per listed seed (determinism.seed is then unused); no duplicates.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> RepeatSpec:
        if self.seeds is None:
            return self
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("repeats.seeds must not contain duplicate seeds")
        if self.count is not None and self.count != len(self.seeds):
            raise ValueError(f"repeats.count is {self.count} but repeats.seeds lists {len(self.seeds)} seeds")
        return self


class HardwareSpec(StrictModel):
    gpu: str | None = Field(
        default=None, description="Target GPU name from hwdb; checked on the compute node."
    )
    count: int = Field(
        default=1, ge=1, le=4, description="GPUs one run uses; the model is split across them.",
    )


ReusePolicy = Literal["auto", "same_hardware", "never"]


class RunSpec(StrictModel):
    """What a user (CLI, UI, MCP) asks for. Validated, then frozen into a Plan."""

    model: ModelRef
    method: MethodRef
    preprocess: Preprocess | None = Field(
        default=None, description="Rotation / scaling applied to the model before the method runs."
    )
    field: str
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    decoding: DecodingSpec | None = Field(
        default=None, description="How tokens are generated; omitted: plain autoregressive decoding."
    )
    tasks: list[TaskRef] = Field(default_factory=list, description="Empty: use the preset's paper tasks.")
    evaluator: EvaluatorRef = Field(default_factory=EvaluatorRef)
    generation: GenerationSpec = Field(default_factory=GenerationSpec)
    determinism: DeterminismSpec = Field(default_factory=DeterminismSpec)
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    repeats: RepeatSpec = Field(default_factory=RepeatSpec)
    reuse: ReusePolicy = "auto"
    compare: list[str] | None = Field(
        default=None,
        description="Override the field's comparative metrics; explicit entries must be computable.",
    )
    sweep: dict[str, list[Any]] = Field(
        default_factory=dict, description="Dotted path -> values (cartesian)."
    )
    metrics: list[str] = Field(
        default_factory=list, description="Derived metrics to compute on top of each task's default metrics."
    )

    @field_validator("model", mode="before")
    @classmethod
    def _model_shorthand(cls, value: Any) -> Any:
        return {"name": value} if isinstance(value, str) else value

    @field_validator("tasks", mode="before")
    @classmethod
    def _task_shorthand(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value


class ResolvedMethod(StrictModel):
    name: str
    version: str
    preset: str | None
    config: dict[str, Any]


class ResolvedRole(StrictModel):
    model: ResolvedModel
    method: ResolvedMethod
    preprocess: ResolvedPreprocess | None = None


class ResolvedDecoding(StrictModel):
    mode: Literal["speculative"]
    draft: ResolvedRole
    num_speculative_tokens: int
    verification: Literal["greedy", "sampling"]


class ResolvedTask(StrictModel):
    name: str
    version: str
    mode: Mode
    num_fewshot: int | None = Field(description="Effective few-shot count; None keeps each subtask's own.")
    limit: int | float | None = Field(
        description="Samples per (sub)task: a count, or a fraction in (0, 1); None evaluates all of them."
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Benchmark settings other than limit that differ from the defaults."
    )


class ResolvedEvaluator(StrictModel):
    name: str
    version: str
    library: str | None = Field(
        description="Pinned library, e.g. lm-eval==0.4.13; None for builtin evaluators."
    )
    passthrough: dict[str, Any]


class JobSpec(StrictModel):
    """Everything that determines one job's result. Produced by plan.freeze; never edited by hand."""

    model: ResolvedModel
    method: ResolvedMethod
    preprocess: ResolvedPreprocess | None = None
    field: str
    runtime: RuntimeSpec
    runtime_library: str | None = Field(
        default=None, description="Pinned engine library, e.g. vllm==0.32.0; None for hf."
    )
    decoding: ResolvedDecoding | None = Field(
        default=None, description="None means autoregressive decoding."
    )
    task: ResolvedTask
    evaluator: ResolvedEvaluator
    generation: GenerationSpec
    determinism: DeterminismSpec
    hardware: HardwareSpec
    metrics: list[str] = Field(
        default_factory=list,
        description="Derived metrics to compute from per-sample outputs; not in the RunKey.",
    )
    repeat: int = Field(
        default=0, ge=0, description="Index of this run among repeats of the same job; part of the RunKey when > 0."
    )


class Job(StrictModel):
    job_id: str
    run_key: str
    role: Literal["main", "reference"]
    reference_key: str | None
    compare: list[str]
    fidelity: Fidelity
    spec: JobSpec


class ReusedJob(StrictModel):
    run_key: str
    role: Literal["main", "reference"]
    record_id: str
    reason: str


class Plan(StrictModel):
    plan_version: Literal[1] = 1
    plan_id: str
    created_at: str
    owner: str | None
    server: str | None
    jobs: list[Job]
    reused: list[ReusedJob] = Field(default_factory=list)
    notes: list[Issue] = Field(default_factory=list)
    manifests: dict[str, MethodManifest] = Field(
        default_factory=dict,
        description="Non-builtin manifests the jobs use, so offline nodes can check them.",
    )
    models: dict[str, ModelEntry] = Field(
        default_factory=dict, description="Non-builtin model entries the jobs use, for the same reason."
    )
    listings: dict[str, TaskListing] = Field(
        default_factory=dict,
        description="Listed evaluators' tasks the jobs use (with subtasks), for nodes whose own catalog "
        "lacks them.",
    )


def runtime_excluded_fields(spec: JobSpec) -> set[str] | None:
    """RuntimeSpec fields left out of a job's identity.

    The decoder settings (cuda_graphs, static_cache_tokens) count only for speed and speculative jobs, the
    jobs that run effeval's own decoders.
    """
    if spec.task.mode == Mode.SPEED or spec.decoding is not None:
        return None
    return {"cuda_graphs", "static_cache_tokens"}


def runtime_identity(spec: JobSpec) -> dict[str, Any]:
    """The runtime part of a job's identity.

    hf: the RuntimeSpec less excluded fields. Engines (always speed jobs): name, cuda_graphs and
    static_cache_tokens.
    """
    if spec.runtime.name == "hf":
        return spec.runtime.model_dump(mode="json", exclude=runtime_excluded_fields(spec))
    return spec.runtime.model_dump(mode="json", include={"name", "cuda_graphs", "static_cache_tokens"})


def run_key(spec: JobSpec) -> str:
    """Identity of a job's result for comparison and reuse (architecture.md §9.1).

    Quality jobs exclude hardware so results group across GPUs; benchmark and speed jobs include it, since
    speed is a property of the GPU. Generation parameters other than batch size only count in generation and
    speed modes, since loglikelihood scoring never decodes.
    """
    generation: dict[str, Any] = {"batch_size": spec.generation.batch_size}
    if spec.task.mode in (Mode.GENERATION, Mode.SPEED):
        generation = spec.generation.model_dump(mode="json")
    identity = {
        "model": spec.model.model_dump(mode="json"),
        "method": {"name": spec.method.name, "version": spec.method.version, "config": spec.method.config},
        "runtime": runtime_identity(spec),
        "task": spec.task.model_dump(mode="json", exclude={"params"} if not spec.task.params else None),
        "evaluator": spec.evaluator.model_dump(mode="json"),
        "generation": generation,
        "determinism": spec.determinism.model_dump(mode="json"),
    }
    if spec.runtime_library is not None:
        identity["runtime_library"] = spec.runtime_library
    if spec.preprocess is not None:
        identity["preprocess"] = spec.preprocess.model_dump(mode="json", include={"steps", "versions"})
    if spec.decoding is not None:
        identity["decoding"] = decoding_identity(spec.decoding)
    if spec.repeat:
        identity["repeat"] = spec.repeat
    if spec.task.mode in (Mode.BENCHMARK, Mode.SPEED):
        identity["hardware"] = spec.hardware.model_dump(mode="json")
    return "rk_" + sha256_hex(canonical_json(identity))


def decoding_identity(decoding: ResolvedDecoding) -> dict[str, Any]:
    """The decoding part of a job identity: mode, k, verification and the draft model, method, preprocess."""
    draft = decoding.draft
    identity: dict[str, Any] = {
        "model": draft.model.model_dump(mode="json"),
        "method": {"name": draft.method.name, "version": draft.method.version, "config": draft.method.config},
    }
    if draft.preprocess is not None:
        identity["preprocess"] = draft.preprocess.model_dump(mode="json", include={"steps", "versions"})
    return {
        "mode": decoding.mode,
        "num_speculative_tokens": decoding.num_speculative_tokens,
        "verification": decoding.verification,
        "draft": identity,
    }
