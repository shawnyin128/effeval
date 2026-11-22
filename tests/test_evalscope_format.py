"""EvalScope conversions on benchmark metadata dicts copied from evalscope 1.15.0's registered benchmarks."""

import pytest

from effeval.core.types import Artifact, Mode
from effeval.evaluators._evalscope_format import (
    benchmark_of,
    listed_tasks,
    metrics_from_reviews,
    review_files,
)

MMLU = {
    "name": "mmlu",
    "output_types": ["generation"],
    "subset_list": ["abstract_algebra", "anatomy", "astronomy"],
    "few_shot_num": 5,
    "few_shot_mode": "auto",
    "allowed_few_shot_nums": None,
    "metric_list": ["accuracy"],
    "primary_metric": None,
    "description": "MMLU.",
    "tags": ["Knowledge", "MCQ"],
    "evaluation_version": "v1.0",
}
GSM8K = {
    "name": "gsm8k",
    "output_types": ["generation"],
    "subset_list": ["main"],
    "few_shot_num": 4,
    "few_shot_mode": "auto",
    "allowed_few_shot_nums": None,
    "metric_list": [{"accuracy": {"numeric": True}}],
    "primary_metric": None,
    "pretty_name": "GSM8K",
    "tags": ["Math", "Reasoning"],
    "evaluation_version": "v1.2",
}


def by_name(*metas):
    return {t.name: t for t in listed_tasks(list(metas))}


def test_benchmarks_become_generation_tasks_with_their_metrics_and_shots():
    tasks = by_name(MMLU, GSM8K)
    mmlu, gsm8k = tasks["mmlu"], tasks["gsm8k"]
    assert (mmlu.mode, mmlu.produces, mmlu.num_fewshot) == (Mode.GENERATION, {Artifact.PER_SAMPLE}, 5)
    assert [m.name for m in gsm8k.metrics] == ["accuracy"] and gsm8k.metrics[0].higher_is_better is None
    assert (gsm8k.version, gsm8k.num_fewshot, gsm8k.description) == ("v1.2", 4, "GSM8K")
    assert "correct" in mmlu.sample_scores and mmlu.tags == ["Knowledge", "MCQ"]


def test_subsets_become_child_tasks_while_the_parent_runs_them_all():
    tasks = by_name(MMLU, GSM8K)
    assert tasks["mmlu"].subtasks == ["mmlu:abstract_algebra", "mmlu:anatomy", "mmlu:astronomy"]
    assert tasks["mmlu"].subsets == ["abstract_algebra", "anatomy", "astronomy"]
    assert tasks["mmlu:anatomy"].subsets == ["anatomy"] and tasks["mmlu:anatomy"].subtasks == []
    assert tasks["gsm8k"].subtasks == [] and tasks["gsm8k"].subsets is None
    assert benchmark_of("mmlu:anatomy") == "mmlu" and benchmark_of("gsm8k") == "gsm8k"


def test_primary_metric_comes_first_and_fewshot_rules_are_kept():
    meta = {
        **MMLU,
        "name": "x",
        "metric_list": ["f1", "accuracy"],
        "primary_metric": {"name": "accuracy"},
        "few_shot_mode": "fixed",
        "allowed_few_shot_nums": [0, 3],
        "few_shot_num": 3,
    }
    task = by_name(meta)["x"]
    assert [m.name for m in task.metrics] == ["accuracy", "f1"]
    assert (task.num_fewshot, task.fewshot_allowed) == (3, [0, 3])
    off = by_name({**MMLU, "name": "y", "few_shot_mode": "disabled", "few_shot_num": 0})["y"]
    assert off.fewshot_allowed == [0]


def test_benchmarks_without_generation_output_or_metrics_are_excluded():
    other = {**MMLU, "name": "vision", "output_types": ["image_editing"]}
    bare = {**MMLU, "name": "bare", "metric_list": []}
    assert set(by_name(MMLU, other, bare)) >= {"mmlu"} and not {"vision", "bare"} & set(
        by_name(MMLU, other, bare)
    )


def review(index, value, main="accuracy", extra=None):
    return {
        "index": index,
        "target": "A",
        "sample_score": {
            "score": {
                "value": {"accuracy": value, **(extra or {})},
                "main_score_name": main,
                "extracted_prediction": "A",
            }
        },
    }


def test_reviews_give_full_precision_means_of_every_metric():
    rows = [
        ("algebra", review(0, 1.0, extra={"f1": 0.5})),
        ("algebra", review(1, 0.0, extra={"f1": 0.25})),
        ("anatomy", review(0, 0.0)),
    ]
    metrics, samples = metrics_from_reviews(rows, "accuracy", True)
    assert metrics == {"accuracy": repr(1 / 3), "f1": "0.375"}
    assert [s.id for s in samples] == ["algebra:0", "algebra:1", "anatomy:0"]
    assert samples[0].scores == {"accuracy": 1.0, "f1": 0.5, "correct": 1.0}


def test_a_main_score_other_than_the_listed_primary_is_an_error():
    with pytest.raises(ValueError, match="main score"):
        metrics_from_reviews([("a", review(0, 1.0, main="f1"))], "accuracy", True)


def test_a_row_without_the_primary_score_is_skipped_from_samples_and_every_mean():
    rows = [
        ("a", review(0, 1.0, extra={"f1": 0.5})),
        ("a", {"index": 1, "sample_score": {"score": {"value": {"f1": 0.0}, "main_score_name": "accuracy"}}}),
        ("a", review(2, 0.0, extra={"f1": 1.0})),
    ]
    metrics, samples = metrics_from_reviews(rows, "accuracy", True)
    assert [s.id for s in samples] == ["a:0", "a:2"]
    assert metrics == {"accuracy": "0.5", "f1": "0.75"}


def test_review_files_are_found_in_subdirectories_and_named_by_subset(tmp_path):
    (tmp_path / "hle_Biology").mkdir()
    for relative in ["hle_Chemistry.jsonl", "hle_Biology/Medicine.jsonl", "other_x.jsonl"]:
        (tmp_path / relative).write_text("{}", encoding="utf-8")
    (tmp_path / "hle_Chemistry.jsonl.rerun-1").write_text("{}", encoding="utf-8")
    found = review_files(tmp_path, "hle", ["Biology/Medicine"])
    assert [(subset, path.relative_to(tmp_path).as_posix()) for subset, path in found] == [
        ("Biology/Medicine", "hle_Biology/Medicine.jsonl"),
        ("Chemistry", "hle_Chemistry.jsonl"),
    ]


def test_a_missing_review_file_for_an_expected_subset_is_an_error(tmp_path):
    (tmp_path / "hle_Chemistry.jsonl").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Biology/Medicine"):
        review_files(tmp_path, "hle", ["Chemistry", "Biology/Medicine"])
    with pytest.raises(FileNotFoundError, match="nothing"):
        review_files(tmp_path, "nothing", None)
