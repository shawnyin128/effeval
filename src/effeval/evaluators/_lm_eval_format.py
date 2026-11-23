"""lm-evaluation-harness data shapes <-> effeval: task listing from the harness's task index, result parsing.

Pure functions over plain dicts; never imports lm_eval, so the logic is testable without the library.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from effeval.core.listing import ListedMetric, ListedTask
from effeval.core.types import Artifact, Mode, Sample, metric_str

LOGLIKELIHOOD_OUTPUTS = {"multiple_choice", "loglikelihood", "loglikelihood_rolling"}
ZERO_ONE_METRICS = {"acc", "acc_norm", "exact_match", "acc_mutual_info"}
"""Metrics whose per-sample value is 0 or 1, so a mean of them is an accuracy."""
DESCRIPTION_LIMIT = 200


def metric_name(metric: str, filter_name: str) -> str:
    """effeval's name of a harness metric: bare for the default filter, `metric,filter` otherwise."""
    return metric if filter_name == "none" else f"{metric},{filter_name}"


def harness_metric(metric: str) -> str:
    """The name the harness reports for a `metric_list` entry.

    A `!function module.fn` metric loads as the string `/abs/path/module.fn`; the harness reports it by
    the function's name, `fn`.
    """
    return metric.rsplit(".", 1)[1] if "/" in metric else metric


def stderr_name(metric: str, filter_name: str) -> str:
    return metric_name(f"{metric}_stderr", filter_name)


def _filters(cfg: dict[str, Any]) -> list[str]:
    names = [f.get("name") for f in cfg.get("filter_list") or [] if isinstance(f, dict) and f.get("name")]
    return names or ["none"]


def _metrics(
    entries: list[Any], filters: list[str], direction: Callable[[str], bool | None], fallback: dict[str, bool]
) -> list[ListedMetric]:
    """Listed metrics (each followed by its stderr) for every metric entry under every filter."""
    listed: dict[str, ListedMetric] = {}
    for filter_name in filters:
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("metric"), str):
                continue
            metric = harness_metric(entry["metric"])
            declared = entry.get("higher_is_better")
            better = declared if isinstance(declared, bool) else fallback.get(metric, direction(metric))
            name = metric_name(metric, filter_name)
            listed.setdefault(name, ListedMetric(name=name, higher_is_better=better))
            err = stderr_name(metric, filter_name)
            listed.setdefault(err, ListedMetric(name=err, higher_is_better=None, stderr_of=name))
    return list(listed.values())


def _description(cfg: dict[str, Any]) -> str:
    text = cfg.get("description")
    return text.strip()[:DESCRIPTION_LIMIT] if isinstance(text, str) else ""


def _version(cfg: dict[str, Any]) -> str:
    metadata = cfg.get("metadata")
    return str(metadata.get("version", "unversioned")) if isinstance(metadata, dict) else "unversioned"


class _Leaf:
    def __init__(self, task: ListedTask, output_type: str, fixed: bool):
        self.task = task
        self.output_type = output_type
        self.fixed = fixed


def _leaf(name: str, entry: dict[str, Any], direction: Callable[[str], bool | None]) -> _Leaf | None:
    cfg = entry.get("cfg") or {}
    output_type = cfg.get("output_type")
    entries = cfg.get("metric_list")
    if not isinstance(output_type, str) or not isinstance(entries, list):
        return None
    mode = Mode.LOGLIKELIHOOD if output_type in LOGLIKELIHOOD_OUTPUTS else Mode.GENERATION
    if output_type not in LOGLIKELIHOOD_OUTPUTS and output_type != "generate_until":
        return None
    metrics = _metrics(entries, _filters(cfg), direction, {})
    if not metrics:
        return None
    produces = {Artifact.PER_SAMPLE}
    if output_type == "multiple_choice":
        produces.add(Artifact.CHOICE_LOGLIKELIHOODS)
    fixed = "num_fewshot" in cfg and cfg["num_fewshot"] == 0
    shots = cfg.get("num_fewshot", 0)
    primary = next(m for m in metrics if m.stderr_of is None)
    base = primary.name.split(",")[0]
    sample_scores = sorted(
        {harness_metric(e["metric"]) for e in entries if isinstance(e, dict) and isinstance(e.get("metric"), str)}
    )
    if base in ZERO_ONE_METRICS and primary.higher_is_better:
        sample_scores.append("correct")
    task = ListedTask(
        name=name,
        version=_version(cfg),
        description=_description(cfg),
        mode=mode,
        produces=frozenset(produces),
        num_fewshot=shots if isinstance(shots, int) else 0,
        fewshot_allowed=[0] if fixed else None,
        metrics=metrics,
        sample_scores=sample_scores,
        tags=sorted(entry.get("tags") or []),
    )
    return _Leaf(task, output_type, fixed)


def listed_tasks(
    index: dict[str, dict[str, Any]], direction: Callable[[str], bool | None]
) -> tuple[list[ListedTask], dict[str, int]]:
    """Convert the harness's task index into listed tasks, plus counts of what was left out.

    `index` maps a name to {"kind": "task" | "py_task" | "group" | "tag", "cfg": dict | None, "tags": [...]};
    for a tag, "tags" holds the names of its members. Leaf tasks need an `output_type` and a `metric_list`.
    Groups are listed when they declare `aggregate_metric_list` and all their leaves share one mode; a group
    with an inline (dict) member, an unlisted member or no aggregate is left out, and its listed members stay
    runnable. `direction` gives the harness's own direction of a metric by name.
    """
    leaves: dict[str, _Leaf] = {}
    skipped = {"tasks_without_output_type": 0, "groups_without_aggregate": 0, "groups_unusable": 0}
    for name, entry in sorted(index.items()):
        if entry["kind"] in {"task", "py_task"}:
            leaf = _leaf(name, entry, direction) if not any(c.isspace() for c in name) else None
            if leaf is None:
                skipped["tasks_without_output_type"] += 1
            else:
                leaves[name] = leaf

    members_cache: dict[str, list[str] | None] = {}

    def members(group: str, stack: tuple[str, ...]) -> list[str] | None:
        """Direct members of a group with tags expanded; None if any member is not a plain name."""
        if group in members_cache:
            return members_cache[group]
        result: list[str] | None = []
        for item in (index[group]["cfg"] or {}).get("task") or []:
            if not isinstance(item, str) or item not in index:
                result = None
                break
            if index[item]["kind"] == "tag":
                result.extend(sorted(index[item]["tags"]))
            else:
                result.append(item)
        members_cache[group] = result
        return result

    done: dict[str, ListedTask | None] = {}
    leaf_names: dict[str, list[str] | None] = {}

    def convert(group: str, stack: tuple[str, ...]) -> ListedTask | None:
        if group in done:
            return done[group]
        done[group] = None
        direct = members(group, stack)
        cfg = index[group]["cfg"] or {}
        aggregate = cfg.get("aggregate_metric_list")
        if direct is None or group in stack or not direct or any(c.isspace() for c in group):
            skipped["groups_unusable"] += 1
            return None
        if not isinstance(aggregate, list) or not aggregate:
            skipped["groups_without_aggregate"] += 1
            for member in direct:
                if index[member]["kind"] == "group":
                    convert(member, (*stack, group))
            return None
        underlying: list[str] = []
        listed_members: list[str] = []
        for member in direct:
            if index[member]["kind"] == "group":
                child = convert(member, (*stack, group))
                below = leaf_names.get(member)
                if below is None:
                    skipped["groups_unusable"] += 1
                    return None
                underlying.extend(below)
                listed_members.extend([member] if child is not None else _listed_below(member))
            elif member in leaves:
                underlying.append(member)
                listed_members.append(member)
            else:
                skipped["groups_unusable"] += 1
                return None
        leaf_names[group] = underlying
        found = [leaves[n] for n in underlying]
        modes = {leaf.task.mode for leaf in found}
        if len(modes) != 1:
            skipped["groups_unusable"] += 1
            return None
        by_metric: dict[str, bool] = {}
        for leaf in found:
            for declared in leaf.task.metrics:
                if declared.higher_is_better is not None:
                    by_metric.setdefault(declared.name.split(",")[0], declared.higher_is_better)
        filters: list[str] = []
        for entry in aggregate:
            if isinstance(entry, dict):
                wanted = entry.get("filter_list")
                for name in [wanted] if isinstance(wanted, str) else wanted or ["none"]:
                    if name not in filters:
                        filters.append(name)
        metrics = _metrics(aggregate, filters, direction, by_metric)
        if not metrics:
            skipped["groups_unusable"] += 1
            return None
        shots = {leaf.task.num_fewshot for leaf in found}
        common = shots.pop() if len(shots) == 1 else None
        fixed = any(leaf.fixed for leaf in found)
        produces = {Artifact.PER_SAMPLE}
        if all(Artifact.CHOICE_LOGLIKELIHOODS in leaf.task.produces for leaf in found):
            produces.add(Artifact.CHOICE_LOGLIKELIHOODS)
        primary = found[0].task
        tags = index[group].get("tags") or []
        task = ListedTask(
            name=group,
            version=_version(cfg),
            description=_description(cfg),
            mode=modes.pop(),
            produces=frozenset(produces),
            num_fewshot=common,
            fewshot_allowed=([] if common is None else [common]) if fixed else None,
            metrics=metrics,
            sample_scores=primary.sample_scores,
            subtasks=listed_members,
            tags=sorted(tags),
        )
        done[group] = task
        return task

    def _listed_below(group: str) -> list[str]:
        found = []
        for member in members(group, ()) or []:
            if index[member]["kind"] == "group":
                task = done.get(member)
                found.extend([member] if task is not None else _listed_below(member))
            elif member in leaves:
                found.append(member)
        return found

    for name, entry in sorted(index.items()):
        if entry["kind"] == "group":
            convert(name, ())

    tasks = [leaf.task for leaf in leaves.values()] + [t for t in done.values() if t is not None]
    listed_names = {t.name for t in tasks}
    tasks = [t.model_copy(update={"subtasks": [s for s in t.subtasks if s in listed_names]}) for t in tasks]
    return sorted(tasks, key=lambda t: t.name), skipped


def metrics_from_results(task_results: dict[str, Any]) -> dict[str, str]:
    """effeval metric values from one entry of the harness's `results["results"]`.

    Keys are `<metric>,<filter>`; keys without a comma (such as the alias) are not metrics, and "N/A" values
    (a standard error that could not be computed) are left out.
    """
    out: dict[str, str] = {}
    for key, value in task_results.items():
        if "," not in key or value == "N/A":
            continue
        metric, filter_name = key.split(",", 1)
        out[metric_name(metric, filter_name)] = metric_str(float(value))
    return out


def samples_from_rows(
    rows_by_task: dict[str, list[dict[str, Any]]],
    primary_filter: str,
    multiple_choice: bool,
    score_names: list[str],
    correct_metric: str | None,
    keep_prediction: bool,
) -> list[Sample]:
    """Per-sample outcomes from the harness's logged samples, one per document.

    The harness repeats each document once per filter; only rows of `primary_filter` are kept. `score_names`
    are the per-sample metrics to record; `correct_metric` (when not None) is also recorded as `correct`.
    """
    samples = []
    for subtask, rows in rows_by_task.items():
        for row in rows:
            if row.get("filter", "none") != primary_filter:
                continue
            scores = {n: float(row[n]) for n in score_names if n != "correct" and n in row}
            if correct_metric is not None and correct_metric in row:
                scores["correct"] = float(row[correct_metric])
            samples.append(
                Sample(
                    id=f"{subtask}:{row['doc_id']}",
                    target=row.get("target"),
                    prediction=row["filtered_resps"] if keep_prediction else None,
                    scores=scores,
                    choice_loglikelihoods=(
                        [float(resp[0]) for resp in row["filtered_resps"]] if multiple_choice else None
                    ),
                )
            )
    return samples
