"""EvalScope data shapes <-> effeval: task listing from benchmark metadata, scoring from review files.

Pure functions over plain dicts; never imports evalscope, so the logic is testable without the library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from effeval.core.listing import ListedMetric, ListedTask
from effeval.core.types import Artifact, Mode, Sample, metric_str

ACCURACY_METRICS = {"accuracy", "acc"}
"""Metrics whose per-sample value is 0 or 1, so a mean of them is an accuracy."""
SUBSET_SEPARATOR = ":"
DESCRIPTION_LIMIT = 200


def metric_names(metric_list: list[Any]) -> list[str]:
    """Distinct metric names of a benchmark's `metric_list`: strings, or the key of a one-entry dict."""
    names: list[str] = []
    for entry in metric_list:
        name = (
            entry if isinstance(entry, str) else next(iter(entry), None) if isinstance(entry, dict) else None
        )
        if name and name not in names:
            names.append(name)
    return names


def primary_name(meta: dict[str, Any]) -> str | None:
    """The benchmark's primary metric: `primary_metric` (a name or {"name": ...}) else the first metric."""
    primary = meta.get("primary_metric")
    if isinstance(primary, dict):
        primary = primary.get("name")
    names = metric_names(meta.get("metric_list") or [])
    return primary if isinstance(primary, str) else (names[0] if names else None)


def _fewshot(meta: dict[str, Any]) -> tuple[int, list[int] | None]:
    default = int(meta.get("few_shot_num") or 0)
    mode = meta.get("few_shot_mode", "auto")
    allowed = meta.get("allowed_few_shot_nums")
    if mode == "disabled":
        return 0, [0]
    if allowed:
        return default, sorted({*allowed, default})
    return default, None


def listed_tasks(metas: list[dict[str, Any]]) -> list[ListedTask]:
    """Listed tasks for benchmark metadata dicts: each generation benchmark, plus one child task per subset.

    A child `<benchmark>:<subset>` runs just that subset; the benchmark itself runs all subsets. Benchmarks
    without generation output, without metrics, or with whitespace in the name are left out.
    """
    tasks: list[ListedTask] = []
    for meta in sorted(metas, key=lambda m: m["name"]):
        name = meta["name"]
        primary = primary_name(meta)
        if (
            Mode.GENERATION.value not in (meta.get("output_types") or [])
            or primary is None
            or any(c.isspace() for c in name)
        ):
            continue
        names = metric_names(meta["metric_list"])
        ordered = [primary, *[n for n in names if n != primary]]
        shots, allowed = _fewshot(meta)
        description = (meta.get("description") or meta.get("pretty_name") or "").strip()[:DESCRIPTION_LIMIT]
        subsets = [
            s
            for s in meta.get("subset_list") or []
            if not any(c.isspace() for c in s) and SUBSET_SEPARATOR not in s
        ]
        has_subsets = len(meta.get("subset_list") or []) > 1
        common = {
            "version": str(meta.get("evaluation_version", "unversioned")),
            "mode": Mode.GENERATION,
            "produces": frozenset({Artifact.PER_SAMPLE}),
            "num_fewshot": shots,
            "fewshot_allowed": allowed,
            "metrics": [ListedMetric(name=n, higher_is_better=None) for n in ordered],
            "sample_scores": [*ordered, *(["correct"] if primary in ACCURACY_METRICS else [])],
            "tags": list(meta.get("tags") or []),
        }
        children = [f"{name}{SUBSET_SEPARATOR}{s}" for s in subsets] if has_subsets else []
        tasks.append(
            ListedTask(
                name=name,
                description=description,
                subsets=meta.get("subset_list") if has_subsets else None,
                subtasks=children,
                **common,
            )
        )
        for subset, child in zip(subsets if has_subsets else [], children, strict=True):
            tasks.append(
                ListedTask(
                    name=child,
                    description=f"{description} Subset {subset}.".strip(),
                    subsets=[subset],
                    **common,
                )
            )
    return tasks


def benchmark_of(target: str) -> str:
    """The benchmark a listed name belongs to: `arc:ARC-Challenge` -> `arc`."""
    return target.split(SUBSET_SEPARATOR, 1)[0]


def review_files(review_dir: Path, benchmark: str, expected: list[str] | None) -> list[tuple[str, Path]]:
    """(subset, path) of every review file of `benchmark` under `review_dir`, sorted by subset.

    EvalScope writes `<benchmark>_<subset>.jsonl` relative to `review_dir`, so a subset containing "/" is
    a file in a subdirectory. Raises FileNotFoundError when no file exists or when a subset in `expected`
    (None: any) has none.
    """
    prefix = f"{benchmark}_"
    found: dict[str, Path] = {}
    for path in review_dir.rglob("*.jsonl"):
        relative = path.relative_to(review_dir).as_posix()[: -len(".jsonl")]
        if relative.startswith(prefix):
            found[relative[len(prefix) :]] = path
    missing = [s for s in expected or [] if s not in found]
    if not found or missing:
        raise FileNotFoundError(
            f"EvalScope wrote no review file for {missing or benchmark!r} under {review_dir}; "
            f"found subsets: {sorted(found)}"
        )
    return sorted(found.items())


def metrics_from_reviews(
    reviews: list[tuple[str, dict[str, Any]]], primary: str, correct: bool
) -> tuple[dict[str, str], list[Sample]]:
    """Micro-mean of every metric over per-sample review rows, plus the per-sample outcomes.

    `reviews` are (subset, review row) pairs; EvalScope rounds its own aggregates to 4 decimals, so they are
    recomputed here. Raises ValueError if a row's main score is not `primary` or no row scored `primary`.
    """
    sums: dict[str, list[float]] = {}
    samples: list[Sample] = []
    for subset, review in reviews:
        score = review["sample_score"]["score"]
        if score.get("main_score_name") != primary:
            raise ValueError(
                f"EvalScope main score is {score.get('main_score_name')!r}, the listing expects {primary!r}"
            )
        values = {k: float(v) for k, v in score["value"].items() if isinstance(v, (int, float))}
        if primary not in values:
            continue  # EvalScope drops unusable scores from aggregation; do the same
        for key, value in values.items():
            sums.setdefault(key, []).append(value)
        samples.append(
            Sample(
                id=f"{subset}:{review['index']}",
                target=review.get("target"),
                prediction=score.get("extracted_prediction"),
                scores={**values, **({"correct": values[primary]} if correct else {})},
            )
        )
    if primary not in sums:
        raise ValueError(f"EvalScope produced no usable {primary!r} scores")
    return {k: metric_str(sum(v) / len(v)) for k, v in sums.items()}, samples
