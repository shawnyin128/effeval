"""What the lm-eval and EvalScope adapters hand to their libraries, which are replaced by recorders."""

import json

import pytest

from catalogs import CATALOG
from effeval.core.spec import (
    DeterminismSpec,
    GenerationSpec,
    HardwareSpec,
    JobSpec,
    ResolvedEvaluator,
    ResolvedMethod,
    ResolvedModel,
    ResolvedTask,
    RuntimeSpec,
)
from effeval.core.types import InProcess, Mode


def job(task_name, num_fewshot, limit=7, mode=None):
    task = CATALOG.tasks[task_name]
    return JobSpec(
        model=ResolvedModel(
            name="m", checkpoint="unused", revision=None, architecture="Qwen3ForCausalLM", dtype="float32"
        ),
        method=ResolvedMethod(name="base", version="1", preset=None, config={}),
        field="quantization",
        runtime=RuntimeSpec(),
        task=ResolvedTask(
            name=task_name,
            version="1",
            mode=mode or task.default_mode,
            num_fewshot=num_fewshot,
            limit=limit,
        ),
        evaluator=ResolvedEvaluator(name=task.evaluator, version="1", library=None, passthrough={}),
        generation=GenerationSpec(max_new_tokens=4, batch_size=2),
        determinism=DeterminismSpec(seed=11),
        hardware=HardwareSpec(),
    )


HANDLE = InProcess(model=object(), tokenizer=object())


def run_lm_eval(monkeypatch, tmp_path, task_name, num_fewshot, results, limit=7):
    pytest.importorskip("lm_eval")
    from effeval.evaluators import lm_eval as adapter

    calls = {}

    class RecordingHFLM:
        def __init__(self, **kwargs):
            calls["hflm"] = kwargs

    def simple_evaluate(**kwargs):
        calls["simple_evaluate"] = kwargs
        return results

    monkeypatch.setattr(adapter, "HFLM", RecordingHFLM)
    monkeypatch.setattr(adapter, "simple_evaluate", simple_evaluate)
    result = adapter.EVALUATOR.run(HANDLE, job(task_name, num_fewshot, limit), CATALOG.tasks[task_name], tmp_path)
    return result, calls


@pytest.mark.parametrize("num_fewshot", [0, 3, None])
def test_lm_eval_receives_the_jobs_fewshot_limit_seed_and_task_target(monkeypatch, tmp_path, num_fewshot):
    results = {
        "results": {"arc_challenge": {"alias": "arc", "acc,none": 0.5, "acc_norm,none": 0.25}},
        "samples": {"arc_challenge": []},
    }
    _, calls = run_lm_eval(monkeypatch, tmp_path, "lm_eval/arc_challenge", num_fewshot, results)
    kwargs = calls["simple_evaluate"]
    assert kwargs["num_fewshot"] == num_fewshot
    assert (kwargs["tasks"], kwargs["limit"]) == (["arc_challenge"], 7)
    assert {kwargs[k] for k in ("random_seed", "numpy_random_seed", "torch_random_seed")} == {11}
    assert calls["hflm"]["batch_size"] == 2


def test_lm_eval_reports_the_groups_own_entry_not_the_first_one_in_results(monkeypatch, tmp_path):
    row = {
        "doc_id": 0,
        "target": 1,
        "filtered_resps": [(-1.0, False), (-2.0, True)],
        "filter": "none",
        "acc": 1.0,
    }
    results = {
        "results": {
            "mmlu_anatomy": {"alias": "anatomy", "acc,none": 0.9, "acc_stderr,none": 0.01},
            "mmlu": {"alias": "mmlu", "acc,none": 0.4, "acc_stderr,none": 0.02},
            "mmlu_stem": {"alias": "stem", "acc,none": 0.7, "acc_stderr,none": 0.03},
        },
        "samples": {"mmlu_anatomy": [row]},
        "n-samples": {},
    }
    result, _ = run_lm_eval(monkeypatch, tmp_path, "lm_eval/mmlu", 0, results)
    assert result.metrics == {"acc": "0.4", "acc_stderr": "0.02"}
    assert result.protocol["subtasks"] == {
        "mmlu_anatomy": {"acc": "0.9", "acc_stderr": "0.01"},
        "mmlu_stem": {"acc": "0.7", "acc_stderr": "0.03"},
    }
    assert [s.id for s in result.samples] == ["mmlu_anatomy:0"]


def review_row(index):
    return {
        "index": index,
        "target": "A",
        "sample_score": {
            "score": {"value": {"accuracy": 1.0}, "main_score_name": "accuracy", "extracted_prediction": "A"}
        },
    }


def run_evalscope(monkeypatch, tmp_path, task_name, num_fewshot, subsets_written, limit=7):
    pytest.importorskip("evalscope")
    from effeval.evaluators import evalscope as adapter

    calls = {}

    class RecordingTaskConfig:
        def __init__(self, **kwargs):
            calls["task_config"] = kwargs

    def run_task(config):
        reviews = tmp_path / "reviews" / adapter.MODEL_NAME
        for subset in subsets_written:
            path = reviews / f"{calls['task_config']['datasets'][0]}_{subset}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(review_row(0)) + "\n", encoding="utf-8")

    monkeypatch.setattr(adapter, "TaskConfig", RecordingTaskConfig)
    monkeypatch.setattr(adapter, "run_task", run_task)
    monkeypatch.setattr(adapter, "InProcessModelAPI", lambda model, tokenizer: "model-api")
    spec = job(task_name, num_fewshot, limit, mode=Mode.GENERATION)
    result = adapter.EVALUATOR.run(HANDLE, spec, CATALOG.tasks[task_name], tmp_path)
    return result, calls["task_config"]


def test_evalscope_receives_the_jobs_fewshot_subset_limit_and_seed(monkeypatch, tmp_path):
    result, config = run_evalscope(monkeypatch, tmp_path, "evalscope/mmlu:anatomy", 3, ["anatomy"])
    assert config["dataset_args"] == {"mmlu": {"few_shot_num": 3, "subset_list": ["anatomy"]}}
    assert (config["datasets"], config["limit"], config["seed"]) == (["mmlu"], 7, 11)
    assert result.protocol["subsets"] == ["anatomy"] and result.metrics == {"accuracy": "1.0"}


def test_evalscope_leaves_fewshot_to_the_benchmark_when_the_job_has_none(monkeypatch, tmp_path):
    _, config = run_evalscope(monkeypatch, tmp_path, "evalscope/mmlu:anatomy", None, ["anatomy"])
    assert config["dataset_args"] == {"mmlu": {"subset_list": ["anatomy"]}}


def test_evalscope_parent_task_runs_every_listed_subset(monkeypatch, tmp_path):
    subsets = CATALOG.tasks["evalscope/mmlu"].subsets
    _, config = run_evalscope(monkeypatch, tmp_path, "evalscope/mmlu", 0, subsets)
    assert config["dataset_args"]["mmlu"]["subset_list"] == subsets
    assert len(subsets) > 1


def test_evalscope_fails_when_a_requested_subset_wrote_no_reviews(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="anatomy"):
        run_evalscope(monkeypatch, tmp_path, "evalscope/mmlu:anatomy", 0, ["astronomy"])


def test_evalscopes_own_review_paths_for_slash_subsets_are_found(tmp_path):
    pytest.importorskip("evalscope")
    from pathlib import Path

    from evalscope.api.evaluator.cache import CacheManager
    from evalscope.utils.io_utils import OutputsStructure

    from effeval.evaluators._evalscope_format import review_files

    subsets = ["Chemistry", "Biology/Medicine", "Computer Science/AI", "F=MA_2024", "中华文化", "a & b, c"]
    cache = CacheManager(OutputsStructure(str(tmp_path)), "effeval_model", "hle")
    for subset in subsets:
        Path(cache.get_review_cache_path(subset)).write_text("{}\n", encoding="utf-8")
    found = review_files(tmp_path / "reviews" / "effeval_model", "hle", subsets)
    assert [subset for subset, _ in found] == sorted(subsets)


@pytest.mark.parametrize("limit", [0.1, 7])
def test_lm_eval_passes_the_limit_through_unchanged(monkeypatch, tmp_path, limit):
    results = {"results": {"arc_challenge": {"acc,none": 0.5}}, "samples": {"arc_challenge": []}}
    _, calls = run_lm_eval(monkeypatch, tmp_path, "lm_eval/arc_challenge", 0, results, limit)
    assert calls["simple_evaluate"]["limit"] == limit
    assert type(calls["simple_evaluate"]["limit"]) is type(limit)


@pytest.mark.parametrize("limit", [0.1, 7])
def test_evalscope_passes_the_limit_through_unchanged(monkeypatch, tmp_path, limit):
    _, config = run_evalscope(monkeypatch, tmp_path, "evalscope/mmlu:anatomy", 3, ["anatomy"], limit)
    assert config["limit"] == limit and type(config["limit"]) is type(limit)
