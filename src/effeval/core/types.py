"""Data types exchanged between layers: handles, issues, capabilities, results and records."""

from __future__ import annotations

import datetime as dt
import enum
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class StrictModel(BaseModel):
    """Base for every user-facing configuration model: unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class Mode(str, enum.Enum):
    LOGLIKELIHOOD = "loglikelihood"
    GENERATION = "generation"
    BENCHMARK = "benchmark"
    SPEED = "speed"


class HandleKind(str, enum.Enum):
    IN_PROCESS = "in_process"
    ENGINE = "engine"


class Artifact(str, enum.Enum):
    PER_SAMPLE = "per_sample"
    CHOICE_LOGLIKELIHOODS = "choice_loglikelihoods"


class Severity(str, enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Visibility(str, enum.Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    RETRACTED = "retracted"


@dataclass(frozen=True)
class InProcess:
    """A model object living in the current Python process."""

    model: Any
    tokenizer: Any
    preprocess_applied: dict[str, Any] | None = None
    """Summary of the model preprocessing applied before the method; goes into the record's protocol."""
    decoder: Any = None
    """A speculative `Decoder` wrapping `model`; None for plain autoregressive decoding."""
    kind: ClassVar[HandleKind] = HandleKind.IN_PROCESS


@dataclass(frozen=True)
class EngineHandle:
    """A client of an inference engine (vLLM, SGLang) running in a separate process."""

    client: Any
    tokenizer: Any
    runtime: str
    device: str
    kind: ClassVar[HandleKind] = HandleKind.ENGINE


Handle = InProcess | EngineHandle


class Issue(BaseModel):
    """One validation finding. Rendered as a CLI error, a UI modal entry, or MCP JSON."""

    code: str
    severity: Severity
    paths: list[str] = Field(default_factory=list)
    message: str
    hint: str = ""


class Capabilities(StrictModel):
    modes: frozenset[Mode] = frozenset()
    handles: frozenset[HandleKind] = frozenset()

    @field_serializer("modes", "handles")
    def _sorted(self, value: frozenset[enum.Enum]) -> list[str]:
        return sorted(item.value for item in value)


class Sample(BaseModel):
    """Per-sample outcome; `scores` holds per-sample metric values, e.g. {"correct": 1.0} or {"nll": 2.3}."""

    id: str
    target: Any = None
    prediction: Any = None
    scores: dict[str, float] = Field(default_factory=dict)
    choice_loglikelihoods: list[float] | None = None
    output_ids: list[list[int]] | None = None
    """Generated token ids, one list per turn, for tasks that compare outputs token by token."""
    category: str | None = None
    """The sample's category in its prompt set, for tasks that report metrics per category."""


class QualityResult(BaseModel):
    kind: Literal["quality"] = "quality"
    metrics: dict[str, str]
    samples: list[Sample]
    protocol: dict[str, Any]
    produces: frozenset[Artifact]


class EnvFingerprint(BaseModel):
    hostname: str
    python: str
    gpu: str | None
    gpu_count: int
    driver: str | None
    cuda: str | None
    packages: dict[str, str]
    effeval_git_commit: str | None
    attn_implementation: str | None


class Deviation(BaseModel):
    path: str
    preset: Any
    actual: Any


class Fidelity(BaseModel):
    """How a run relates to its paper preset; see architecture.md §10."""

    preset: str | None = None
    confirmed: list[str] = Field(default_factory=list)
    assumed: list[str] = Field(default_factory=list)
    deviations: list[Deviation] = Field(default_factory=list)
    repro: Literal["pass", "fail", "untested"] = "untested"


SubmittedVia = Literal["cli", "paste", "mcp", "api", "paper"]


class Record(BaseModel):
    """One executed job. Immutable once stored; annotations live elsewhere."""

    record_id: str
    job_id: str
    run_key: str
    reference_key: str | None
    role: Literal["main", "reference"]
    compare: list[str] = Field(
        default_factory=list, description="Comparative metrics to compute vs reference."
    )
    spec: dict[str, Any]
    metrics: dict[str, str]
    protocol: dict[str, Any]
    env: EnvFingerprint | None
    fidelity: Fidelity
    artifacts: dict[str, str] = Field(default_factory=dict)
    produces: list[Artifact] = Field(default_factory=list)
    timings: dict[str, float] = Field(
        default_factory=dict, description="prepare_s, measure_s on this hardware."
    )
    reusable: bool = True
    probe: bool = False
    owner: str | None = None
    visibility: Visibility = Visibility.PRIVATE
    submitted_via: SubmittedVia = "cli"
    created_at: str


def metric_str(value: str | float | int | Decimal) -> str:
    """Return the full-precision text form of a metric value.

    Strings pass through unchanged, floats use the shortest round-trip `repr`, ints and Decimals use `str`.
    Raises TypeError for anything else (including bool, which would silently become "True").
    """
    if isinstance(value, bool):
        raise TypeError("metric values must be numeric or numeric strings, got bool")
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (int, Decimal)):
        return str(value)
    raise TypeError(f"unsupported metric value type: {type(value).__name__}")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
