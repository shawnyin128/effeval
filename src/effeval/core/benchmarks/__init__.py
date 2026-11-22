"""Benchmarks (tasks), one module each: a module declaring `BENCHMARK = TaskDef(...)`, or
`BENCHMARKS = [TaskDef(...), ...]` for a family of tasks, is found automatically.

Modules here must not import execution code (torch, evaluators): the server loads them to validate and plan.
"""

from __future__ import annotations

import importlib
import pkgutil

from effeval.core.catalog import TaskDef


def discover() -> list[TaskDef]:
    """Every benchmark declared in this package, by module name.

    Raises ValueError if a module declares neither `BENCHMARK` nor `BENCHMARKS` or two tasks share a name.
    """
    found: dict[str, TaskDef] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{__name__}.{info.name}")
        declared = getattr(module, "BENCHMARKS", None)
        if declared is None:
            declared = [getattr(module, "BENCHMARK", None)]
        if not declared or not all(isinstance(task, TaskDef) for task in declared):
            raise ValueError(f"benchmark module {module.__name__} must define BENCHMARK = TaskDef(...)")
        for task in declared:
            if task.name in found:
                raise ValueError(f"benchmark {task.name!r} is declared twice")
            found[task.name] = task
    return list(found.values())
