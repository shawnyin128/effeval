"""Comparative quality metrics between a main run and its reference run.

Pure Python (no torch/numpy) so the server computes them when both records are stored, regardless of which
evaluator, machine or upload path produced them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from effeval.core.types import Artifact, Sample, metric_str


@dataclass(frozen=True)
class Comparative:
    name: str
    requires: frozenset[Artifact]
    sample_scores: frozenset[str]
    description: str
    compute: Callable[[dict[str, str], dict[str, str], list[Sample], list[Sample]], dict[str, str]]


def _aligned(main: list[Sample], ref: list[Sample]) -> list[tuple[Sample, Sample]]:
    by_id = {s.id: s for s in ref}
    pairs = [(m, by_id[m.id]) for m in main if m.id in by_id]
    if not pairs:
        raise ValueError("main and reference share no sample ids")
    return pairs


def _flip_rate(
    _m: dict[str, str], _r: dict[str, str], main: list[Sample], ref: list[Sample]
) -> dict[str, str]:
    """Fraction of aligned samples whose correctness differs between main and reference."""
    pairs = [(m, r) for m, r in _aligned(main, ref) if "correct" in m.scores and "correct" in r.scores]
    if not pairs:
        return {}
    flips = sum(1 for m, r in pairs if (m.scores["correct"] > 0.5) != (r.scores["correct"] > 0.5))
    return {"flip_rate": metric_str(flips / len(pairs)), "flip_rate.n": str(len(pairs))}


def _log_softmax(values: list[float]) -> list[float]:
    peak = max(values)
    log_norm = peak + math.log(math.fsum(math.exp(v - peak) for v in values))
    return [v - log_norm for v in values]


def _kl(_m: dict[str, str], _r: dict[str, str], main: list[Sample], ref: list[Sample]) -> dict[str, str]:
    """Mean KL(reference || main) between answer-choice distributions softmax(choice loglikelihoods)."""
    terms = []
    for m, r in _aligned(main, ref):
        if m.choice_loglikelihoods is None or r.choice_loglikelihoods is None:
            continue
        if len(m.choice_loglikelihoods) != len(r.choice_loglikelihoods):
            raise ValueError(f"sample {m.id}: choice counts differ between main and reference")
        log_p = _log_softmax(r.choice_loglikelihoods)
        log_q = _log_softmax(m.choice_loglikelihoods)
        terms.append(math.fsum(math.exp(lp) * (lp - lq) for lp, lq in zip(log_p, log_q, strict=True)))
    if not terms:
        return {}
    return {"kl": metric_str(math.fsum(terms) / len(terms)), "kl.n": str(len(terms))}


COMPARATIVE: dict[str, Comparative] = {
    c.name: c
    for c in (
        Comparative(
            "flip_rate",
            frozenset({Artifact.PER_SAMPLE}),
            frozenset({"correct"}),
            "share of samples whose correctness flips",
            _flip_rate,
        ),
        Comparative(
            "kl",
            frozenset({Artifact.PER_SAMPLE, Artifact.CHOICE_LOGLIKELIHOODS}),
            frozenset(),
            "mean KL(reference || main) over answer-choice distributions",
            _kl,
        ),
    )
}


def why_not_applicable(name: str, produces: frozenset[Artifact], sample_scores: list[str]) -> str | None:
    """Reason `name` cannot be computed for a task, or None if it can. Raises KeyError for unknown metrics."""
    metric = COMPARATIVE[name]
    lacking = metric.requires - produces
    if lacking:
        return f"needs artifacts {sorted(a.value for a in lacking)}"
    missing_scores = metric.sample_scores - set(sample_scores)
    if missing_scores:
        return f"needs per-sample scores {sorted(missing_scores)}"
    return None
