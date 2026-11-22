"""Speed comparatives between a main run and its autoregressive reference run.

Pure Python, like `effeval.metrics.quality`, whose registry these are merged into.
"""

from __future__ import annotations

import math

from effeval.core.types import Artifact, Sample, metric_str
from effeval.metrics.quality import COMPARATIVE as REGISTRY
from effeval.metrics.quality import Comparative, _aligned


def _speedup(
    _m: dict[str, str], _r: dict[str, str], main: list[Sample], ref: list[Sample]
) -> dict[str, str]:
    """Mean per-sample tokens/s of main over that of the reference, overall and per sample category."""
    pairs = [
        (m, r) for m, r in _aligned(main, ref) if "tokens_per_s" in m.scores and "tokens_per_s" in r.scores
    ]
    if not pairs:
        return {}

    def ratio(selected: list[tuple[Sample, Sample]]) -> float:
        return math.fsum(m.scores["tokens_per_s"] for m, _ in selected) / math.fsum(
            r.scores["tokens_per_s"] for _, r in selected
        )

    out = {"speedup": metric_str(ratio(pairs)), "speedup.n": str(len(pairs))}
    for category in dict.fromkeys(m.category for m, _ in pairs if m.category is not None):
        out[f"speedup.{category}"] = metric_str(ratio([p for p in pairs if p[0].category == category]))
    return out


def _output_match(
    _m: dict[str, str], _r: dict[str, str], main: list[Sample], ref: list[Sample]
) -> dict[str, str]:
    """Fraction of aligned samples with identical generated tokens; their text when token ids are absent."""
    pairs = _aligned(main, ref)
    equal = 0
    for m, r in pairs:
        if m.output_ids is not None and r.output_ids is not None:
            equal += m.output_ids == r.output_ids
        else:
            equal += m.prediction == r.prediction
    return {"output_match": metric_str(equal / len(pairs)), "output_match.n": str(len(pairs))}


def _multi_turn_prompt_match(
    _m: dict[str, str], _r: dict[str, str], main: list[Sample], ref: list[Sample]
) -> dict[str, str]:
    """Share of aligned multi-turn samples whose later-turn prompts equal the reference's.

    A later turn's prompt contains the answers to the turns before it, so the prompts are identical exactly
    when all answers but the last are (compared as text, which is what the prompt is built from). Single-turn
    samples are not counted; nothing is returned when there is no multi-turn sample.
    """
    pairs = [
        (m, r)
        for m, r in _aligned(main, ref)
        if isinstance(m.prediction, list) and isinstance(r.prediction, list) and len(m.prediction) > 1
    ]
    if not pairs:
        return {}
    equal = sum(m.prediction[:-1] == r.prediction[:-1] for m, r in pairs)
    return {
        "multi_turn_prompt_match": metric_str(equal / len(pairs)),
        "multi_turn_prompt_match.n": str(len(pairs)),
    }


COMPARATIVE: dict[str, Comparative] = {
    c.name: c
    for c in (
        Comparative(
            "speedup",
            frozenset({Artifact.PER_SAMPLE}),
            frozenset({"tokens_per_s"}),
            "mean per-sample tokens/s over the reference's, overall and per category",
            _speedup,
        ),
        Comparative(
            "output_match",
            frozenset({Artifact.PER_SAMPLE}),
            frozenset(),
            "share of samples whose generated tokens equal the reference's",
            _output_match,
        ),
        Comparative(
            "multi_turn_prompt_match",
            frozenset({Artifact.PER_SAMPLE}),
            frozenset(),
            "share of multi-turn samples whose later-turn prompts equal the reference's",
            _multi_turn_prompt_match,
        ),
    )
}

REGISTRY.update(COMPARATIVE)
