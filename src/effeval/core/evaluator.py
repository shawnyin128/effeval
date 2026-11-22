"""The evaluator interface: how effeval lists and runs the tasks of an evaluation library.

An evaluator is any object satisfying `Evaluator`, announced under the `effeval.evaluators` entry point group
as `name = "package.module:OBJECT"`. This module never imports an evaluation library.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Protocol, runtime_checkable

from effeval.core.catalog import TaskDef
from effeval.core.listing import TaskListing
from effeval.core.spec import JobSpec
from effeval.core.types import InProcess, QualityResult

ENTRY_POINT_GROUP = "effeval.evaluators"


@runtime_checkable
class Evaluator(Protocol):
    name: str

    def list_tasks(self) -> TaskListing | None:
        """The tasks this evaluator can run, or None when its tasks are declared in effeval's own data."""
        ...

    def run(self, handle: InProcess | None, job: JobSpec, task: TaskDef, workdir: Path) -> QualityResult:
        """Run one job's task on the model `handle` and return its metrics and per-sample outputs."""
        ...


def entry_points() -> dict[str, importlib.metadata.EntryPoint]:
    """Installed evaluator entry points by name. Raises ValueError when two share a name."""
    found: dict[str, importlib.metadata.EntryPoint] = {}
    for point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        if point.name in found:
            raise ValueError(
                f"two evaluators are announced as {point.name!r}: {found[point.name].value}, {point.value}"
            )
        found[point.name] = point
    return found


def load(name: str) -> Evaluator:
    """Load the evaluator announced as `name`.

    Raises LookupError for an unknown name, TypeError for an object that is not an Evaluator, and ValueError
    when the object's own name differs from its entry point name.
    """
    points = entry_points()
    if name not in points:
        raise LookupError(f"unknown evaluator {name!r}; known evaluators: {sorted(points)}")
    obj = points[name].load()
    if not isinstance(obj, Evaluator):
        raise TypeError(f"entry point {name!r} ({points[name].value}) is not an Evaluator")
    if obj.name != name:
        raise ValueError(f"entry point {name!r} loads an evaluator named {obj.name!r}")
    return obj


def load_all() -> list[Evaluator]:
    """Every announced evaluator, sorted by name."""
    return [load(name) for name in sorted(entry_points())]


def library_issue(library: str | None) -> str | None:
    """None when the installed version matches `library` ("lm-eval==0.4.13"); otherwise what differs."""
    if library is None:
        return None
    distribution, _, wanted = library.partition("==")
    try:
        installed = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return f"{distribution} is not installed"
    if installed != wanted:
        return f"{distribution} {installed} is installed but the plan was made with {wanted}"
    return None


def check_result_metrics(task: TaskDef, metrics: dict[str, str]) -> None:
    """Raise ValueError unless `metrics` holds the task's primary metric."""
    primary = task.metrics[0].name
    if primary not in metrics:
        raise ValueError(
            f"evaluator {task.evaluator!r} did not report the primary metric {primary!r} of {task.name!r}"
        )
