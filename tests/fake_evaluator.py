"""A stand-in evaluator for discovery and task-command tests."""

from effeval.core.listing import EvaluatorInfo, ListedMetric, ListedTask, TaskListing
from effeval.core.types import Artifact, Capabilities, HandleKind, Mode, QualityResult, Sample


class FakeEvaluator:
    name = "fake"

    def list_tasks(self):
        return TaskListing(
            evaluator=EvaluatorInfo(
                name="fake",
                version="1",
                description="fake evaluator",
                capabilities=Capabilities(modes={Mode.GENERATION}, handles={HandleKind.IN_PROCESS}),
                library="fakelib",
                library_version="1.0",
            ),
            generated_at="2026-11-20T00:00:00+00:00",
            tasks=[
                ListedTask(
                    name="echo",
                    version="1",
                    description="echo task",
                    mode=Mode.GENERATION,
                    produces={Artifact.PER_SAMPLE},
                    num_fewshot=0,
                    metrics=[ListedMetric(name="score", higher_is_better=True)],
                    sample_scores=["correct"],
                )
            ],
        )

    def run(self, handle, job, task, workdir):
        return QualityResult(
            metrics={"score": "1.0"},
            samples=[Sample(id="0", scores={"correct": 1.0})],
            protocol={},
            produces=frozenset({Artifact.PER_SAMPLE}),
        )


FAKE = FakeEvaluator()
