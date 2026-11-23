"""lm-eval conversions on dicts copied from the harness's task YAMLs (0.4.13) and result shapes."""

from effeval.core.types import Artifact, Mode
from effeval.evaluators._lm_eval_format import listed_tasks, metrics_from_results, samples_from_rows

MC_TEMPLATE = {
    "output_type": "multiple_choice",
    "metric_list": [{"metric": "acc", "aggregation": "mean", "higher_is_better": True}],
    "metadata": {"version": 1.0},
}
GSM8K = {
    "task": "gsm8k",
    "tag": ["math_word_problems"],
    "output_type": "generate_until",
    "metric_list": [{"metric": "exact_match", "aggregation": "mean", "higher_is_better": True}],
    "num_fewshot": 5,
    "filter_list": [{"name": "strict-match"}, {"name": "flexible-extract"}],
    "metadata": {"version": 3.0},
}
WIKITEXT = {
    "task": "wikitext",
    "output_type": "loglikelihood_rolling",
    "num_fewshot": 0,
    "metric_list": [
        {"metric": "word_perplexity"},
        {"metric": "byte_perplexity"},
        {"metric": "bits_per_byte"},
    ],
    "metadata": {"version": 2.0},
}


def index():
    return {
        "mmlu": {
            "kind": "group",
            "cfg": {
                "group": "mmlu",
                "task": ["mmlu_stem"],
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
                "metadata": {"version": 2},
            },
            "tags": [],
        },
        "mmlu_stem": {
            "kind": "group",
            "cfg": {
                "group": "mmlu_stem",
                "task": ["mmlu_anatomy", "mmlu_algebra"],
                "aggregate_metric_list": [{"metric": "acc", "weight_by_size": True}],
            },
            "tags": [],
        },
        "mmlu_anatomy": {"kind": "task", "cfg": {"task": "mmlu_anatomy", **MC_TEMPLATE}, "tags": []},
        "mmlu_algebra": {"kind": "task", "cfg": {"task": "mmlu_algebra", **MC_TEMPLATE}, "tags": []},
        "gsm8k": {"kind": "task", "cfg": GSM8K, "tags": ["math_word_problems"]},
        "wikitext": {"kind": "task", "cfg": WIKITEXT, "tags": []},
        "math_word_problems": {"kind": "tag", "cfg": None, "tags": ["gsm8k"]},
        "pytask": {"kind": "py_task", "cfg": {"task": "pytask", "class": "x.Y"}, "tags": []},
        "loose": {
            "kind": "group",
            "cfg": {"group": "loose", "task": ["gsm8k", "mmlu_anatomy"]},
            "tags": [],
        },
        "mixed": {
            "kind": "group",
            "cfg": {
                "group": "mixed",
                "task": ["gsm8k", "mmlu_anatomy"],
                "aggregate_metric_list": [{"metric": "acc"}],
            },
            "tags": [],
        },
    }


def listed():
    tasks, skipped = listed_tasks(index(), lambda m: False if m.endswith("perplexity") else None)
    return {t.name: t for t in tasks}, skipped


def test_group_with_aggregate_is_a_runnable_task_with_its_subtasks():
    tasks, _ = listed()
    mmlu = tasks["mmlu"]
    assert [m.name for m in mmlu.metrics] == ["acc", "acc_stderr"]
    assert mmlu.mode == Mode.LOGLIKELIHOOD
    assert Artifact.CHOICE_LOGLIKELIHOODS in mmlu.produces
    assert mmlu.subtasks == ["mmlu_stem"] and tasks["mmlu_stem"].subtasks == ["mmlu_anatomy", "mmlu_algebra"]
    assert mmlu.version == "2" and mmlu.metrics[0].higher_is_better is True
    assert mmlu.num_fewshot == 0


def test_gsm8k_has_one_metric_per_filter_and_its_default_shots():
    gsm8k = listed()[0]["gsm8k"]
    names = [m.name for m in gsm8k.metrics]
    assert names[0] == "exact_match,strict-match" and "exact_match,flexible-extract" in names
    assert "exact_match_stderr,strict-match" in names
    assert (gsm8k.mode, gsm8k.num_fewshot, gsm8k.fewshot_allowed) == (Mode.GENERATION, 5, None)
    assert gsm8k.tags == ["math_word_problems"] and "correct" in gsm8k.sample_scores


def test_direction_comes_from_the_declaration_then_the_harness():
    wikitext = listed()[0]["wikitext"]
    by_name = {m.name: m.higher_is_better for m in wikitext.metrics}
    assert by_name["word_perplexity"] is False and by_name["bits_per_byte"] is None
    assert "correct" not in wikitext.sample_scores


def test_a_task_configured_with_zero_shots_rejects_overrides():
    wikitext = listed()[0]["wikitext"]
    assert (wikitext.num_fewshot, wikitext.fewshot_allowed) == (0, [0])
    assert listed()[0]["mmlu"].fewshot_allowed is None


def test_unusable_groups_and_python_tasks_are_left_out_and_counted():
    tasks, skipped = listed()
    assert "pytask" not in tasks and "loose" not in tasks and "mixed" not in tasks
    assert skipped["tasks_without_output_type"] == 1
    assert skipped["groups_without_aggregate"] == 1 and skipped["groups_unusable"] == 1


def test_not_available_standard_errors_are_omitted():
    results = {"alias": "mmlu", "acc,none": 0.25, "acc_stderr,none": "N/A", "exact_match,strict-match": 0.5}
    assert metrics_from_results(results) == {"acc": "0.25", "exact_match,strict-match": "0.5"}


def test_rows_repeated_per_filter_give_one_sample_per_document():
    def row(doc_id, flt, value):
        return {
            "doc_id": doc_id,
            "target": "t",
            "filtered_resps": ["x"],
            "filter": flt,
            "exact_match": value,
        }

    rows = {
        "gsm8k": [row(0, "strict-match", 1.0), row(0, "flexible-extract", 0.0), row(1, "strict-match", 0.0)]
    }
    samples = samples_from_rows(rows, "strict-match", False, ["exact_match"], "exact_match", True)
    assert [s.id for s in samples] == ["gsm8k:0", "gsm8k:1"]
    assert samples[0].scores == {"exact_match": 1.0, "correct": 1.0}


def test_multiple_choice_rows_keep_choice_loglikelihoods():
    rows = {
        "arc": [
            {
                "doc_id": 3,
                "target": 1,
                "filtered_resps": [(-1.5, False), (-0.5, True)],
                "filter": "none",
                "acc": 1.0,
            }
        ]
    }
    [sample] = samples_from_rows(rows, "none", True, ["acc"], "acc", False)
    assert sample.choice_loglikelihoods == [-1.5, -0.5] and sample.scores["correct"] == 1.0


def group(name, members):
    cfg = {"group": name, "task": members, "aggregate_metric_list": [{"metric": "acc"}]}
    return {"kind": "group", "cfg": cfg, "tags": []}


def fewshot_index():
    def leaf(name, **extra):
        return {"kind": "task", "cfg": {"task": name, **MC_TEMPLATE, **extra}, "tags": []}

    return {
        "fixed_a": leaf("fixed_a", num_fewshot=0),
        "fixed_b": leaf("fixed_b", num_fewshot=0),
        "five": leaf("five", num_fewshot=5),
        "default": leaf("default"),
        "all_fixed": group("all_fixed", ["fixed_a", "fixed_b"]),
        "fixed_and_five": group("fixed_and_five", ["fixed_a", "five"]),
        "fixed_and_default": group("fixed_and_default", ["fixed_a", "default"]),
        "free": group("free", ["five", "default"]),
    }


def test_a_group_with_any_zero_shot_member_rejects_overrides():
    tasks, _ = listed_tasks(fewshot_index(), lambda m: None)
    by_name = {t.name: t for t in tasks}
    assert (by_name["all_fixed"].num_fewshot, by_name["all_fixed"].fewshot_allowed) == (0, [0])
    assert (by_name["fixed_and_default"].num_fewshot, by_name["fixed_and_default"].fewshot_allowed) == (
        0,
        [0],
    )
    assert (by_name["fixed_and_five"].num_fewshot, by_name["fixed_and_five"].fewshot_allowed) == (None, [])
    assert (by_name["free"].num_fewshot, by_name["free"].fewshot_allowed) == (None, None)


def test_standard_errors_link_to_the_metric_they_belong_to():
    tasks, _ = listed()
    by_name = {t.name: {m.name: m.stderr_of for m in t.metrics} for t in tasks.values()}
    assert by_name["mmlu"] == {"acc": None, "acc_stderr": "acc"}
    assert by_name["gsm8k"]["exact_match_stderr,strict-match"] == "exact_match,strict-match"
    assert by_name["gsm8k"]["exact_match_stderr,flexible-extract"] == "exact_match,flexible-extract"
    assert by_name["gsm8k"]["exact_match,strict-match"] is None
    assert by_name["wikitext"]["word_perplexity_stderr"] == "word_perplexity"


def test_only_multiple_choice_tasks_and_groups_of_them_produce_choice_loglikelihoods():
    ll = {
        "output_type": "loglikelihood",
        "metric_list": [{"metric": "acc", "higher_is_better": True}],
        "metadata": {"version": 1.0},
    }
    idx = {
        **index(),
        "lambada": {"kind": "task", "cfg": {"task": "lambada", **ll}, "tags": []},
        "only_ll": group("only_ll", ["lambada", "wikitext"]),
        "mc_and_ll": group("mc_and_ll", ["mmlu_anatomy", "lambada"]),
        "only_mc": group("only_mc", ["mmlu_anatomy", "mmlu_algebra"]),
    }
    tasks, _ = listed_tasks(idx, lambda m: None)
    produces = {t.name: Artifact.CHOICE_LOGLIKELIHOODS in t.produces for t in tasks}
    assert produces["mmlu_anatomy"] and produces["only_mc"]
    assert not produces["lambada"] and not produces["wikitext"]
    assert not produces["only_ll"] and not produces["mc_and_ll"]


def test_function_metrics_are_listed_by_the_name_the_harness_reports():
    cfg = {
        "task": "chartqa",
        "output_type": "generate_until",
        "metric_list": [
            {"metric": "/opt/env/site-packages/lm_eval/tasks/chartqa/utils.relaxed_accuracy", "higher_is_better": True}
        ],
        "filter_list": [{"name": "strict"}],
    }
    tasks, _ = listed_tasks({"chartqa": {"kind": "task", "cfg": cfg, "tags": []}}, lambda m: None)
    (task,) = tasks
    assert [m.name for m in task.metrics] == ["relaxed_accuracy,strict", "relaxed_accuracy_stderr,strict"]
    assert task.metrics[1].stderr_of == "relaxed_accuracy,strict"
    assert task.sample_scores == ["relaxed_accuracy"]
