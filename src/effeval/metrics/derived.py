"""Metrics derived from one run's per-sample outputs.

Pure Python (no torch/numpy) so compute nodes add them to new records and the server backfills them on stored
records whose per-sample artifacts it holds. Register with `@derived(...)`; declare how the metric is shown in
core/data/metrics.yaml.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from effeval.core.types import Artifact, Sample, metric_str


@dataclass(frozen=True)
class Derived:
    name: str
    requires: frozenset[Artifact]
    sample_scores: frozenset[str]
    compute: Callable[[list[Sample]], str | None]


DERIVED: dict[str, Derived] = {}


def derived(
    name: str,
    *,
    requires: frozenset[Artifact] = frozenset({Artifact.PER_SAMPLE}),
    sample_scores: frozenset[str] = frozenset(),
) -> Callable[[Callable[[list[Sample]], str | None]], Callable[[list[Sample]], str | None]]:
    """Register `fn(samples) -> value text or None` as metric `name`. ValueError on a duplicate name."""

    def register(fn: Callable[[list[Sample]], str | None]) -> Callable[[list[Sample]], str | None]:
        if name in DERIVED:
            raise ValueError(f"derived metric {name!r} is already registered")
        DERIVED[name] = Derived(name, requires, sample_scores, fn)
        return fn

    return register


def applicable(name: str, produces: frozenset[Artifact], sample_scores: list[str]) -> bool:
    """Whether derived metric `name` can be computed from a run with these artifacts and per-sample scores.

    Raises KeyError for names that are not registered.
    """
    metric = DERIVED[name]
    return metric.requires <= produces and metric.sample_scores <= set(sample_scores)


def compute(names: list[str], samples: list[Sample]) -> dict[str, str]:
    """Values of the named derived metrics; metrics that yield None (too few samples) are left out."""
    out = {}
    for name in names:
        value = DERIVED[name].compute(samples)
        if value is not None:
            out[name] = value
    return out


@derived("accuracy", sample_scores=frozenset({"correct"}))
def accuracy(samples: list[Sample]) -> str | None:
    """Mean per-sample correctness (micro average); None without samples."""
    correct = [s.scores["correct"] for s in samples if "correct" in s.scores]
    return metric_str(math.fsum(correct) / len(correct)) if correct else None


@derived("ppl", sample_scores=frozenset({"nll"}))
def ppl(samples: list[Sample]) -> str | None:
    """exp(mean per-window NLL), exact for equal-length windows (GPTQ protocol); None without samples."""
    nll = [s.scores["nll"] for s in samples if "nll" in s.scores]
    return metric_str(math.exp(math.fsum(nll) / len(nll))) if nll else None


@derived("accuracy_stderr", sample_scores=frozenset({"correct"}))
def accuracy_stderr(samples: list[Sample]) -> str | None:
    """Standard error of the mean correctness, sqrt(p (1 - p) / (n - 1)); None below two samples."""
    correct = [s.scores["correct"] for s in samples if "correct" in s.scores]
    if len(correct) < 2:
        return None
    p = math.fsum(correct) / len(correct)
    return metric_str(math.sqrt(p * (1 - p) / (len(correct) - 1)))
