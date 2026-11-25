"""Model preprocessing (rotation / scaling applied before the quantizer stage): spec schema and presets.

Torch-free, so the server, the CLI and MCP validate and freeze preprocessing without compute libraries.
"""

from __future__ import annotations

import fnmatch
import functools
from importlib import resources
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from effeval.core.catalog import PaperRef, PresetValue
from effeval.core.fragments import Calibration, IntFormat, LayerSelector, TextBatches, sequence_options
from effeval.core.graph import LayerGraph, ModelShape, hadamard_size_ok, input_width
from effeval.core.types import StrictModel

Kind = Literal["rotation", "scaling"]
Mode = Literal["offline", "online"]

GLOBAL_POSITIONS = frozenset({"residual"})
"""Positions with one matrix shared by the whole model; their entries only accept layers = all."""


class Hadamard(StrictModel):
    type: Literal["hadamard"] = "hadamard"


class RandomHadamard(StrictModel):
    type: Literal["random_hadamard"] = "random_hadamard"
    seed: int = 0
    fallback: Literal["block_hadamard"] | None = Field(
        default=None, description="Used when the dimension has no Hadamard matrix of its own size."
    )


class RandomOrthogonal(StrictModel):
    type: Literal["random_orthogonal"] = "random_orthogonal"
    seed: int = 0


class DuQuantRotation(StrictModel):
    type: Literal["duquant"] = "duquant"
    block_size: int = Field(default=128, ge=1)
    max_steps: int = Field(default=256, ge=0)
    permutation_times: int = Field(default=1, ge=0)
    seed: int = 2


def default_trained_act() -> IntFormat:
    """The activation quantizer the trained sources minimise the error of: 4-bit per-token min-max."""
    return IntFormat(type="int", bits=4, symmetric=False, granularity="per_token", zero_point="float")


def check_trained_act(act: IntFormat) -> IntFormat:
    """`act` as is when it is a per-token asymmetric min-max grid with a float zero point and no sequence
    options (the grid `act_quant_mse` computes). Raises ValueError otherwise."""
    if act.symmetric or act.granularity != "per_token" or act.zero_point != "float" or sequence_options(act):
        raise ValueError(
            "a trained source minimises the error of a per-token asymmetric grid with zero_point float "
            "and no group_axis or residual"
        )
    return act


class TrainedRotation(StrictModel):
    """R1 trained by Adam on a Cayley parametrization, minimising the activation quantization error of the
    rotated normalized outputs of every residual-stream norm (CodeQuant's AOS)."""

    type: Literal["trained"] = "trained"
    objective: Literal["act_quant_mse"] = "act_quant_mse"
    act: IntFormat = Field(default_factory=default_trained_act)
    optimizer: Literal["adam"] = "adam"
    lr: float = Field(gt=0.0, description="Adam learning rate.")
    init_seed: int = Field(
        default=0, description="Seed of the float32 nn.Linear whose weight initialises the rotation."
    )
    calibration: TextBatches

    @field_validator("act")
    @classmethod
    def _act_grid(cls, act: IntFormat) -> IntFormat:
        return check_trained_act(act)


class SmoothQuantScaling(StrictModel):
    type: Literal["smoothquant"] = "smoothquant"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    weight_reduce: Literal["group_max", "first_consumer"] = "group_max"
    act_stat: Literal["absmax"] = "absmax"


class SmoothAttentionScaling(StrictModel):
    """QServe SmoothAttention: per-channel scale of the K cache, undone on the query side."""

    type: Literal["smoothattention"] = "smoothattention"
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    granularity: Literal["per_head", "shared"] = Field(
        default="per_head", description="One scale per KV head, or one per channel shared by all KV heads."
    )
    act_stat: Literal["absmax"] = "absmax"


class ClipSearch(StrictModel):
    n_grid: int = Field(default=20, ge=1)
    max_shrink: float = Field(default=0.5, gt=0.0, lt=1.0)
    n_tokens: int = Field(default=512, ge=1)
    skip: list[str] = Field(default_factory=lambda: ["q_proj", "k_proj"])


class AwqScaling(StrictModel):
    type: Literal["awq"] = "awq"
    n_grid: int = Field(default=20, ge=1)
    act_stat: Literal["absmean"] = "absmean"
    clip_search: ClipSearch | None = None


class TrainedScaling(StrictModel):
    """One scaling vector per norm output, trained by Adam to minimise the activation quantization error of
    the scaled output (CodeQuant2); initialised as 1 + 0.01 * N(0, 1)."""

    type: Literal["trained"] = "trained"
    objective: Literal["act_quant_mse"] = "act_quant_mse"
    act: IntFormat = Field(default_factory=default_trained_act)
    optimizer: Literal["adam"] = "adam"
    lr: float = Field(gt=0.0, description="Adam learning rate.")
    init_seed: int = Field(default=0, description="Seed of the generator drawing the initial vectors.")
    calibration: TextBatches

    @field_validator("act")
    @classmethod
    def _act_grid(cls, act: IntFormat) -> IntFormat:
        return check_trained_act(act)


RotationSource = Annotated[
    Hadamard | RandomHadamard | RandomOrthogonal | DuQuantRotation | TrainedRotation,
    Field(discriminator="type"),
]
ScalingSource = Annotated[
    SmoothQuantScaling | AwqScaling | SmoothAttentionScaling | TrainedScaling, Field(discriminator="type")
]

SOURCE_VERSIONS: dict[str, dict[str, str]] = {
    "rotation": {
        "hadamard": "1",
        "random_hadamard": "1",
        "random_orthogonal": "1",
        "duquant": "1",
        "trained": "1",
    },
    "scaling": {"smoothquant": "1", "awq": "1", "smoothattention": "1", "trained": "1"},
}
"""Algorithm version of each source type, by kind; bump it when a source's numerics change."""

RESERVED_SOURCES: frozenset[str] = frozenset()
"""Source types declared but not implemented; validation rejects them."""
DATA_SOURCES = frozenset({"smoothquant", "awq", "smoothattention", "duquant"})
"""Source types whose result depends on the calibration data of their step; `trained` carries its own."""

SOURCE_POSITIONS: dict[str, dict[str, list[str]]] = {
    "scaling": {
        "smoothattention": ["attn.qk_pre_rope", "attn.qk_rope", "attn.v_head"],
        "smoothquant": ["norm.*.out", "linear.*.input"],
        "awq": ["norm.*.out", "linear.*.input"],
    },
}
"""Kind -> source type -> fnmatch patterns of the positions it can be used at; a source not listed fits
any."""

NO_PAPER: dict[str, dict[str, list[str]]] = {
    "scaling": {"smoothattention": ["attn.qk_rope", "attn.v_head"]},
}
"""Kind -> source type -> positions (within SOURCE_POSITIONS) where no published method exists:
SmoothAttention is published for K before RoPE only, so its online post-RoPE scaling and its V-cache scaling
are generalizations."""

KV_ROTATION_OPTIONS: dict[str, frozenset[str]] = {
    "attn.qk_rope": frozenset({"per_kv_head"}),
    "attn.v_head": frozenset({"per_kv_head"}),
}
"""Position -> option keys a rotation entry there accepts."""


class Entry(StrictModel):
    position: str = Field(min_length=1, description="Position id from the architecture graph.")
    layers: LayerSelector = Field(
        default="all", description="Decoder layers; only 'all' for global positions."
    )
    mode: Mode = Field(
        description="offline: folded into weights and norms; online: runtime op plus weight half."
    )
    options: dict[str, Any] = Field(default_factory=dict, description="Position-specific options.")


class RotationEntry(Entry):
    source: RotationSource

    @model_validator(mode="after")
    def _trained_is_offline_residual(self) -> RotationEntry:
        if self.source.type == "trained" and (self.position != "residual" or self.mode != "offline"):
            raise ValueError("a trained rotation is available at the residual position, offline, only")
        return self


class ScalingEntry(Entry):
    source: ScalingSource

    @model_validator(mode="after")
    def _trained_is_offline_norm(self) -> ScalingEntry:
        if self.source.type == "trained" and (
            not fnmatch.fnmatchcase(self.position, "norm.*.out") or self.mode != "offline"
        ):
            raise ValueError("a trained scaling is available at norm.*.out positions, offline, only")
        return self


class RotationStep(StrictModel):
    kind: Literal["rotation"] = "rotation"
    calibration: Calibration | None = None
    entries: list[RotationEntry] = Field(min_length=1)


class ScalingStep(StrictModel):
    kind: Literal["scaling"] = "scaling"
    calibration: Calibration | None = None
    entries: list[ScalingEntry] = Field(min_length=1)


Step = Annotated[RotationStep | ScalingStep, Field(discriminator="kind")]


class Preprocess(StrictModel):
    """What a user asks for; steps are applied in order, one kind each (`both` is two steps)."""

    preset: str | None = Field(
        default=None, description="Preset whose steps fill an unset `steps`; not part of the RunKey."
    )
    steps: list[Step] = Field(default_factory=list)


class ResolvedPreprocess(StrictModel):
    """The frozen form held by JobSpec; `steps` and `versions` are the RunKey's preprocess identity."""

    preset: str | None
    steps: list[Step]
    versions: dict[str, str] = Field(
        description="'<kind>.<source type>' -> algorithm version, per source used."
    )


def source_versions(preprocess: Preprocess) -> dict[str, str]:
    """Versions of every source type a preprocess uses, keyed '<kind>.<source type>'."""
    return {
        f"{step.kind}.{entry.source.type}": SOURCE_VERSIONS[step.kind][entry.source.type]
        for step in preprocess.steps
        for entry in step.entries
    }


def resolve_preprocess(preprocess: Preprocess) -> ResolvedPreprocess:
    return ResolvedPreprocess(
        preset=preprocess.preset, steps=preprocess.steps, versions=source_versions(preprocess)
    )


class PreprocessPresetDef(StrictModel):
    """A paper's preprocessing recipe; `values` maps dotted RunSpec paths under `preprocess.` to values.

    `quantizers` maps a method name to the names of its presets this recipe is meant to run with.
    """

    name: str
    title: str
    description: str
    paper: PaperRef
    values: dict[str, PresetValue]
    quantizers: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _paths_under_preprocess(self) -> PreprocessPresetDef:
        for path in self.values:
            if not path.startswith("preprocess."):
                raise ValueError(f"preprocess preset path {path!r} must start with 'preprocess.'")
        return self


@functools.lru_cache(maxsize=1)
def builtin_preprocess_presets() -> dict[str, PreprocessPresetDef]:
    """Preprocess presets shipped in `core/data/preprocess_presets/*.yaml`, by name."""
    presets: dict[str, PreprocessPresetDef] = {}
    for item in sorted(
        resources.files("effeval.core.data.preprocess_presets").iterdir(), key=lambda p: p.name
    ):
        if item.name.endswith(".yaml"):
            preset = PreprocessPresetDef.model_validate(yaml.safe_load(item.read_text(encoding="utf-8")))
            presets[preset.name] = preset
    return presets


def preprocess_title(preprocess: Preprocess | ResolvedPreprocess) -> str:
    """Display title of a preprocess: the preset's title, `<title>*` when its steps differ from the preset's.

    A preprocess without a (known) preset is "Custom preprocessing". Presets are compared on the whole
    `preprocess.steps` value, the only path a preprocess preset may set.
    """
    preset = None if preprocess.preset is None else builtin_preprocess_presets().get(preprocess.preset)
    if preset is None:
        return "Custom preprocessing"
    unsupported = [path for path in preset.values if path != "preprocess.steps"]
    if unsupported:
        raise ValueError(f"preprocess preset {preset.name!r} sets unsupported paths {unsupported}")
    expected = Preprocess.model_validate({"steps": preset.values["preprocess.steps"].value}).steps
    same = [s.model_dump(mode="json") for s in expected] == [
        s.model_dump(mode="json") for s in preprocess.steps
    ]
    return preset.title if same else f"{preset.title}*"


def preprocess_label(
    preprocess: Preprocess | ResolvedPreprocess | None, method_title: str, method_preset: str | None = None
) -> str:
    """Run label `<preprocess title>[*] + <method title>[ (<method preset>)]`.

    Example: "QuaRot + GPTQ (quarot_w4a4kv4)".

    The method title alone if there is no preprocess; the preset is shown only when it names a preset
    (not `None` and not the `default` sentinel).
    """
    from effeval.core.spec import DEFAULT_PRESET  # spec imports this module

    if preprocess is None:
        return method_title
    if method_preset is not None and method_preset != DEFAULT_PRESET:
        method_title = f"{method_title} ({method_preset})"
    return f"{preprocess_title(preprocess)} + {method_title}"


def position_dim(graph: LayerGraph, shape: ModelShape, position: str) -> int:
    """Width of the channel a position rotates or scales.

    `residual` and `norm.*.out` are `hidden`; `attn.v_head`, `attn.qk_pre_rope` and `attn.qk_rope` are
    `head_dim`; a linear's input is `hidden` after a norm, the MLP width
    (`intermediate`, `moe_intermediate` or `shared_intermediate`) after the gating product, and
    `heads * head_dim` after attention. Raises ValueError for an unknown position.
    """
    if position == "residual" or (position.startswith("norm.") and position.endswith(".out")):
        return shape.hidden
    if position in ("attn.v_head", "attn.qk_pre_rope", "attn.qk_rope"):
        return shape.head_dim
    if position.startswith("linear.") and position.endswith(".input"):
        canon = position[len("linear.") : -len(".input")]
        for node in graph.of_kind("linear"):
            if node.canon == canon:
                return input_width(graph, shape, node)
    raise ValueError(f"unknown position {position!r}")


def plain_hadamard_ok(position: str, dim: int) -> bool:
    """Whether a plain Hadamard matrix of width `dim` may be used at `position`.

    It needs `dim` = K * 2^m with K in QuaRot's list; online Q/K rotation after RoPE (`attn.qk_rope`)
    additionally needs a power-of-two `dim`.
    """
    return hadamard_size_ok(dim) and (position != "attn.qk_rope" or dim & (dim - 1) == 0)


def rotation_size_problem(source: Any, position: str, dim: int) -> str | None:
    """Why a Hadamard-family rotation `source` cannot be built for a channel of width `dim`, or None.

    `hadamard` needs `plain_hadamard_ok`; `random_hadamard` needs it or its `block_hadamard` fallback (largest
    power-of-two divisor of `dim` at least 64). Other sources have no size rule.
    """
    if source.type not in ("hadamard", "random_hadamard") or plain_hadamard_ok(position, dim):
        return None
    if source.type == "random_hadamard" and source.fallback == "block_hadamard":
        if dim & -dim >= 64:
            return None
        return f"{dim} has no power-of-two divisor >= 64 for the block_hadamard fallback"
    return (
        f"no usable Hadamard matrix of size {dim} at {position} (K * 2^m with K in QuaRot's list, "
        "power of two after RoPE); set fallback: block_hadamard or use random_orthogonal"
    )


def source_position_problem(kind: str, source_type: str, position: str) -> str | None:
    """Why a source of `source_type` cannot be used at `position`, or None (see SOURCE_POSITIONS)."""
    patterns = SOURCE_POSITIONS.get(kind, {}).get(source_type)
    if patterns is None or any(fnmatch.fnmatchcase(position, pattern) for pattern in patterns):
        return None
    if source_type == "smoothattention":
        return "smoothattention scales the K/V cache: use attn.qk_pre_rope, attn.qk_rope or attn.v_head"
    if position.startswith("attn."):
        return f"{source_type} scales a linear input; at {position} use smoothattention"
    return f"{source_type} scales a norm output or a linear input; {position} is neither"


def rotation_option_problem(position: str, source: Any, options: dict[str, Any]) -> str | None:
    """Why a rotation entry's `options` are invalid at a KV position, or None.

    Only `attn.qk_rope` and `attn.v_head` are checked (KV_ROTATION_OPTIONS): unknown keys are rejected,
    `per_kv_head` must be a bool and needs a seeded source (not the deterministic `hadamard`).
    """
    allowed = KV_ROTATION_OPTIONS.get(position)
    if allowed is None:
        return None
    unknown = sorted(set(options) - allowed)
    if unknown:
        return f"unknown options {unknown} at {position}; accepted: {sorted(allowed)}"
    per_kv_head = options.get("per_kv_head", False)
    if not isinstance(per_kv_head, bool):
        return "per_kv_head must be true or false"
    if per_kv_head and source.type == "hadamard":
        return "hadamard is deterministic; per_kv_head needs random_hadamard or random_orthogonal"
    return None
