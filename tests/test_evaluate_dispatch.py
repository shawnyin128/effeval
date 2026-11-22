"""execute.evaluate hands each job to the evaluator its spec names."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from effeval.api import execute  # noqa: E402
from effeval.core.catalog import ModelEntry, builtin_catalog  # noqa: E402
from effeval.core.types import Artifact, InProcess, QualityResult, Sample  # noqa: E402

pytestmark = pytest.mark.compute


class Recorder:
    def __init__(self, name, metric, value):
        self.name = name
        self.metric = metric
        self.value = value
        self.ran = []

    def list_tasks(self):
        return None

    def run(self, handle, job, task, workdir):
        self.ran.append((job.task.name, job.evaluator.name, task.name))
        return QualityResult(
            metrics={self.metric: self.value},
            samples=[Sample(id="0", scores={"correct": 1.0})],
            protocol={"ran_by": self.name},
            produces=frozenset({Artifact.PER_SAMPLE}),
        )


def test_each_task_runs_on_the_evaluator_its_job_names(monkeypatch, tmp_path):
    evaluators = {
        "lm_eval": Recorder("lm_eval", "acc", "0.5"),
        "evalscope": Recorder("evalscope", "accuracy", "0.25"),
        "native": Recorder("native", "ppl", "9.0"),
    }
    loaded = []

    def load(name):
        loaded.append(name)
        return evaluators[name]

    monkeypatch.setattr(execute.evaluators, "load", load)
    local = ModelEntry(
        code="my-ckpt",
        checkpoint="/data/My-Ckpt",
        revision=None,
        architecture="Qwen3ForCausalLM",
        dtype="float32",
        params=10,
        tie_word_embeddings=False,
    )
    catalog = builtin_catalog().with_models([local])
    monkeypatch.setattr(execute, "builtin_catalog", lambda: catalog)
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(_attn_implementation="eager")
    records = execute.evaluate(
        InProcess(model=model, tokenizer=None),
        model="my-ckpt",
        tasks=["lm_eval/arc_challenge", "evalscope/gsm8k"],
        field="quantization",
        out_dir=tmp_path,
    )
    assert sorted(loaded) == ["evalscope", "lm_eval"]
    assert evaluators["lm_eval"].ran == [("lm_eval/arc_challenge", "lm_eval", "lm_eval/arc_challenge")]
    assert evaluators["evalscope"].ran == [("evalscope/gsm8k", "evalscope", "evalscope/gsm8k")]
    assert evaluators["native"].ran == []
    assert sorted((r.spec["task"]["name"], r.metrics, r.protocol["ran_by"]) for r in records) == [
        ("evalscope/gsm8k", {"accuracy": "0.25"}, "evalscope"),
        ("lm_eval/arc_challenge", {"acc": "0.5"}, "lm_eval"),
    ]
