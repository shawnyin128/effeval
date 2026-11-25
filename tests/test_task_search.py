"""Ranked task search: whole-word matching, aliases, categories and ordering, on the shipped snapshot and on small hand-made catalogs."""

import re

import pytest

from effeval.core.catalog import builtin_catalog
from effeval.core.task_search import search_tasks


@pytest.fixture(scope="module")
def snapshot():
    catalog = builtin_catalog()
    titles = {n: c.title or c.label for n, c in catalog.categories.items()}
    return {n: t.model_dump(mode="json") for n, t in catalog.tasks.items()}, titles


def task(name, *, category="qa", subtasks=(), tags=(), metrics=("acc",), description=""):
    return {
        "name": name,
        "target": name.split("/", 1)[1],
        "category": category,
        "subtasks": list(subtasks),
        "tags": list(tags),
        "metrics": [{"name": m} for m in metrics],
        "description": description,
    }


def catalog(*tasks):
    return {t["name"]: t for t in tasks}


def test_ppl_finds_perplexity_tasks_first_not_words_that_merely_contain_it(snapshot):
    tasks, titles = snapshot
    ranked = search_tasks(tasks, "ppl", titles)
    assert set(ranked[:3]) == {"native/c4_ppl", "native/wikitext2_ppl", "native/wikitext2_preprocessed_ppl"}
    assert ranked[3] == "lm_eval/wikitext"
    assert not [n for n in ranked[:10] if "blimp" in n or "include" in n]
    by_word = [n for n in ranked if {"ppl", "perplexity"} & set(re.split(r"[^a-z0-9]+", n.lower()))]
    substring_only = ranked.index("lm_eval/include_base_44_azerbaijani_applied_science")
    assert by_word and all(ranked.index(n) < substring_only for n in by_word)


def test_an_exact_task_name_comes_first(snapshot):
    tasks, titles = snapshot
    ranked = search_tasks(tasks, "gsm8k", titles)
    assert ranked[0] == "lm_eval/gsm8k"
    assert {"evalscope/gsm8k", "lm_eval/gsm8k_cot"} <= set(ranked[:10])


def test_a_category_name_lists_that_categorys_tasks_before_any_other(snapshot):
    tasks, titles = snapshot
    ranked = search_tasks(tasks, "code", titles)
    categories = [tasks[n]["category"] for n in ranked]
    in_category = categories.count("code")
    assert in_category > 10
    assert set(categories[:in_category]) == {"code"}
    assert "lm_eval/humaneval" in ranked[:in_category]


def test_aliases_find_tasks_the_query_does_not_name(snapshot):
    tasks, titles = snapshot
    assert set(search_tasks(tasks, "wiki2", titles)[:2]) == {"native/wikitext2_ppl", "lm_eval/wikitext"}
    assert "native/wikitext2_ppl" in search_tasks(tasks, "perplexity", titles)[:20]


def test_a_term_matches_whole_words_and_word_prefixes_before_bare_substrings():
    ranked = search_tasks(
        catalog(task("e/applied"), task("e/wikitext2_ppl"), task("e/ppl_long_name"), task("e/ppls")), "ppl"
    )
    assert ranked == ["e/ppls", "e/ppl_long_name", "e/wikitext2_ppl", "e/applied"]


def test_camel_case_and_digit_boundaries_start_words():
    tasks = catalog(task("e/WikiPpl"), task("e/wikitext2ppl"), task("e/applied"))
    assert search_tasks(tasks, "ppl") == ["e/WikiPpl", "e/wikitext2ppl", "e/applied"]


def test_every_term_must_match_and_the_worst_term_sets_the_rank():
    tasks = catalog(
        task("e/mmlu_anatomy"), task("e/mmlu_law"), task("e/anatomy", description="mmlu style"), task("e/zzz")
    )
    assert search_tasks(tasks, "mmlu anatomy") == ["e/mmlu_anatomy", "e/anatomy"]


def test_equal_rank_puts_groups_before_leaves_then_shorter_names():
    tasks = catalog(
        task("e/a_mmlu_longer"), task("e/a_mmlu"), task("e/b_mmlu_group", subtasks=["e/z"]), task("e/z")
    )
    assert search_tasks(tasks, "mmlu") == ["e/b_mmlu_group", "e/a_mmlu", "e/a_mmlu_longer"]


def test_subtasks_of_a_matching_group_are_left_out_but_kept_when_the_group_does_not_match():
    tasks = catalog(
        task("e/mmlu", subtasks=["e/mmlu_law"]),
        task("e/mmlu_law"),
        task("e/stem", subtasks=["e/mmlu_stem_part"]),
        task("e/mmlu_stem_part"),
    )
    assert search_tasks(tasks, "mmlu") == ["e/mmlu", "e/mmlu_stem_part"]


def test_tags_rank_above_metric_and_category_title_matches_which_rank_above_descriptions():
    tasks = catalog(
        task("e/by_description", description="reports zebra scores"),
        task("e/by_tag", tags=["zebra_tag"]),
        task("e/by_metric", metrics=("zebra,none",)),
        task("e/unrelated"),
    )
    assert search_tasks(tasks, "zebra") == ["e/by_tag", "e/by_metric", "e/by_description"]


def test_a_category_title_matches_only_when_titles_are_given():
    tasks = catalog(task("e/by_title", category="animals"), task("e/unrelated"))
    assert search_tasks(tasks, "zebra", {"animals": "Zebra"}) == ["e/by_title"]
    assert search_tasks(tasks, "zebra") == []


def test_a_description_word_ranks_above_a_bare_name_substring():
    tasks = catalog(task("e/xzebrax"), task("e/other", description="scores on a zebra"))
    assert search_tasks(tasks, "zebra") == ["e/other", "e/xzebrax"]


def test_a_name_prefix_ranks_above_a_word_inside_the_name_whatever_the_length():
    tasks = catalog(task("e/a_mmlu"), task("e/mmlu_with_a_very_long_name"))
    assert search_tasks(tasks, "mmlu") == ["e/mmlu_with_a_very_long_name", "e/a_mmlu"]


def test_empty_query_lists_every_task_including_subtasks():
    tasks = catalog(task("e/b", subtasks=["e/a"]), task("e/a"))
    assert search_tasks(tasks, "  ") == ["e/a", "e/b"]
