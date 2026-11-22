"""Task listings: what an evaluator reports about the tasks it can run, as data the server can store.

Imports only pydantic and core types, so the server handles listings without any evaluation library.
"""

from __future__ import annotations

from pydantic import Field, field_serializer, model_validator

from effeval.core.canonical import canonical_json, sha256_hex
from effeval.core.types import Artifact, Capabilities, Mode, StrictModel


class ListedMetric(StrictModel):
    name: str = Field(description="Library name, e.g. acc, exact_match,strict-match or acc_stderr.")
    higher_is_better: bool | None
    stderr_of: str | None = None


class ListedTask(StrictModel):
    name: str = Field(description="The evaluator's own name; no whitespace, unique in the listing.")
    version: str
    description: str = ""
    mode: Mode
    produces: frozenset[Artifact]
    num_fewshot: int | None = Field(description="None: subtasks keep their own defaults.")
    fewshot_allowed: list[int] | None = None
    metrics: list[ListedMetric] = Field(min_length=1, description="The first is primary.")
    sample_scores: list[str]
    subsets: list[str] | None = None
    subtasks: list[str] = Field(default_factory=list, description="Direct members, by name in this listing.")
    tags: list[str] = Field(default_factory=list)

    @field_serializer("produces")
    def _sorted_produces(self, value: frozenset[Artifact]) -> list[str]:
        return sorted(a.value for a in value)


class EvaluatorInfo(StrictModel):
    name: str
    version: str
    description: str
    capabilities: Capabilities
    library: str | None = Field(description="Distribution name of the evaluation library, e.g. lm-eval.")
    library_version: str | None


class TaskListing(StrictModel):
    evaluator: EvaluatorInfo
    generated_at: str
    tasks: list[ListedTask]

    @model_validator(mode="after")
    def _consistent(self) -> TaskListing:
        by_name: dict[str, ListedTask] = {}
        for task in self.tasks:
            if task.name in by_name:
                raise ValueError(f"task {task.name!r} is listed twice")
            if not task.name or any(c.isspace() for c in task.name):
                raise ValueError(f"task {task.name!r}: names must be non-empty and contain no whitespace")
            by_name[task.name] = task
        for task in self.tasks:
            if task.mode not in self.evaluator.capabilities.modes:
                raise ValueError(
                    f"task {task.name!r}: mode {task.mode.value} is not in the capabilities of "
                    f"{self.evaluator.name!r}"
                )
            if (
                task.fewshot_allowed is not None
                and task.num_fewshot is not None
                and task.num_fewshot not in task.fewshot_allowed
            ):
                raise ValueError(
                    f"task {task.name!r}: fewshot_allowed {task.fewshot_allowed} must contain "
                    f"its default {task.num_fewshot}"
                )
            metric_names = {m.name for m in task.metrics}
            for metric in task.metrics:
                if metric.stderr_of is not None and metric.stderr_of not in metric_names:
                    raise ValueError(
                        f"task {task.name!r}: metric {metric.name!r} is the stderr of unknown "
                        f"metric {metric.stderr_of!r}"
                    )
            for sub in task.subtasks:
                if sub not in by_name:
                    raise ValueError(f"task {task.name!r}: unknown subtask {sub!r}")
        state: dict[str, int] = {}

        def visit(name: str) -> None:
            if state.get(name) == 2:
                return
            if state.get(name) == 1:
                raise ValueError(f"task {name!r}: subtasks form a cycle")
            state[name] = 1
            for sub in by_name[name].subtasks:
                visit(sub)
            state[name] = 2

        for name in by_name:
            visit(name)
        return self

    def digest(self) -> str:
        """sha256 of the canonical JSON of the whole listing."""
        return sha256_hex(canonical_json(self.model_dump(mode="json")))
