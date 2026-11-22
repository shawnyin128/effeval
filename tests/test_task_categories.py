import json
from pathlib import Path

import pytest

from effeval.core.catalog import OTHER_CATEGORY, builtin_catalog, categorize
from effeval.core.listing import TaskListing

DATA = Path(__file__).parent / "data" / "listings"


def listing_of(*tasks: dict) -> TaskListing:
    """The lm_eval fixture's evaluator with `tasks`: dicts of name, metric names, subtasks and tags."""
    raw = json.loads((DATA / "lm_eval.json").read_text(encoding="utf-8"))
    template = raw["tasks"][0]
    raw["tasks"] = [
        {
            **template,
            "name": t["name"],
            "metrics": [
                {"name": m, "higher_is_better": True, "stderr_of": None} for m in t.get("metrics", ["acc"])
            ],
            "subtasks": t.get("subtasks", []),
            "tags": t.get("tags", []),
            "subsets": None,
        }
        for t in tasks
    ]
    return TaskListing.model_validate(raw)


def category_of(*tasks: dict) -> dict[str, str]:
    return categorize(listing_of(*tasks), builtin_catalog().categories)


def test_a_perplexity_metric_places_a_task_in_language_modeling_whatever_its_name():
    got = category_of({"name": "mystery_corpus", "metrics": ["word_perplexity", "byte_perplexity"]})
    assert got == {"mystery_corpus": "language_modeling"}


def test_the_filter_suffix_does_not_hide_a_metric_name_and_stderr_names_do_not_match():
    got = category_of(
        {"name": "alpha", "metrics": ["perplexity,none"]},
        {"name": "beta", "metrics": ["acc", "word_perplexity_stderr"]},
    )
    assert got == {"alpha": "language_modeling", "beta": OTHER_CATEGORY}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("hellaswag", "commonsense"),
        ("arc_easy", "commonsense"),
        ("humaneval", "code"),
        ("gsm8k", "math"),
        ("mmlu_anatomy", "qa"),
        ("global_mmlu_full_ar", "multilingual"),
        ("gpqa_main_zeroshot", "reasoning"),
        ("ifeval", "instruction_following"),
        ("longbench_qasper", "long_context"),
        ("wikitext", "language_modeling"),
        ("completely_unknown", OTHER_CATEGORY),
    ],
)
def test_name_rules(name, expected):
    assert category_of({"name": name}) == {name: expected}


def test_tags_place_a_task_without_a_name_or_metric_match():
    assert category_of({"name": "zzz", "tags": ["Math"]}) == {"zzz": "math"}


def test_a_task_without_a_match_takes_its_groups_category():
    got = category_of({"name": "mmlu", "subtasks": ["zzz_child"]}, {"name": "zzz_child"})
    assert got == {"mmlu": "qa", "zzz_child": "qa"}


def test_a_name_match_beats_the_groups_category():
    got = category_of({"name": "global_mmlu", "subtasks": ["humaneval_x"]}, {"name": "humaneval_x"})
    assert got["humaneval_x"] == "code"


def test_the_shipped_catalog_places_the_known_tasks():
    tasks = builtin_catalog().tasks
    assert tasks["native/wikitext2_ppl"].category == "language_modeling"
    assert tasks["lm_eval/wikitext"].category == "language_modeling"
    assert tasks["lm_eval/mmlu"].category == "qa"
    assert tasks["evalscope/humaneval"].category == "code"
    assert tasks["evalscope/gsm8k"].category == "math"
    other = sum(t.category == OTHER_CATEGORY for t in tasks.values())
    assert other / len(tasks) < 0.15


def test_changing_the_rules_changes_the_outcome():
    categories = builtin_catalog().categories
    task = {"name": "mystery_corpus", "metrics": ["word_perplexity"]}
    assert categorize(listing_of(task), categories) == {"mystery_corpus": "language_modeling"}
    no_metrics = categories["language_modeling"].model_copy(update={"metrics": []})
    broken = {**categories, "language_modeling": no_metrics}
    assert categorize(listing_of(task), broken) == {"mystery_corpus": OTHER_CATEGORY}
    swapped = {**categories, "math": categories["math"].model_copy(update={"names": ["gsm8k"], "rank": -1})}
    assert categorize(listing_of({"name": "hellaswag_gsm8k"}), swapped)["hellaswag_gsm8k"] == "math"
