"""Benchmarks (tasks), one module each: a module declaring `BENCHMARK = TaskDef(...)` is found automatically.

Modules here must not import execution code (torch, evaluators): the server loads them to validate and plan.
"""

from __future__ import annotations

import importlib
import pkgutil

from effeval.core.catalog import TaskDef


def discover() -> list[TaskDef]:
    """Every benchmark declared in this package, by module name.

    Raises ValueError if a module declares no `BENCHMARK` TaskDef or two modules declare the same name.
    """
    found: dict[str, TaskDef] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{__name__}.{info.name}")
        declared = getattr(module, "BENCHMARK", None)
        if not isinstance(declared, TaskDef):
            raise ValueError(f"benchmark module {module.__name__} must define BENCHMARK = TaskDef(...)")
        if declared.name in found:
            raise ValueError(f"benchmark {declared.name!r} is declared twice")
        found[declared.name] = declared
    return list(found.values())
