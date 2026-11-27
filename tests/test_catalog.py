import pytest

from catalogs import CATALOG, LISTINGS, raw_listing
from effeval.core.catalog import (
    TaskDef,
    TaskMetric,
    _check_task,
    _static_catalog,
    builtin_catalog,
    sample_count,
)
from effeval.core.listing import TaskListing
from effeval.core.preprocess import builtin_preprocess_presets

BUILTIN = builtin_catalog()


@pytest.mark.compute
@pytest.mark.parametrize("name", ["rtn", "gptq", "paroquant", "codequant", "codequant2", "qsvd"])
def test_builtin_manifest_files_match_the_code(name):
    pytest.importorskip("torch")
    from effeval.api.execute import export_manifest

    stored = BUILTIN.manifests[name]
    exported = export_manifest(stored.entry)
    assert exported.model_copy(update={"builtin": True}) == stored, (
        f"src/effeval/core/data/methods/{name}.yaml is stale: re-export it with `effeval method export`"
    )


def test_builtin_methods_carry_their_standard_display_titles():
    assert {n: m.title for n, m in BUILTIN.manifests.items()} == {
        "base": "Base",
        "rtn": "RTN",
        "gptq": "GPTQ",
        "paroquant": "ParoQuant",
        "codequant": "CodeQuant",
        "codequant2": "CodeQuant2",
        "qsvd": "QSVD",
    }


def test_every_preset_value_has_a_source_and_reported_rows_parse():
    for by_name in BUILTIN.presets.values():
        for preset in by_name.values():
            assert preset.values
            for reported in preset.reported:
                float(reported.value), float(reported.tolerance)
            for name in preset.values["tasks"].value if "tasks" in preset.values else []:
                assert name in BUILTIN.tasks, (
                    f"preset {preset.method}/{preset.name} names unknown task {name}"
                )


def test_every_task_declares_artifacts_for_each_mode():
    for task in CATALOG.tasks.values():
        assert set(task.produces) == set(task.modes)
        assert all(task.produces.values())


def test_metric_registry_covers_derived_metrics_and_task_categories():
    from effeval.metrics.derived import DERIVED

    for task in CATALOG.tasks.values():
        assert task.category in CATALOG.categories
    assert set(DERIVED) <= CATALOG.metrics.keys()


def test_builtin_tasks_are_discovered_without_a_snapshot():
    native = {n for n, t in BUILTIN.tasks.items() if t.evaluator in {"native", "kernel"}}
    assert native == {
        "native/wikitext2_ppl",
        "native/wikitext2_preprocessed_ppl",
        "native/c4_ppl",
        "kernel/gemm_w4a16",
    }
    assert {"native", "kernel"} <= set(BUILTIN.evaluators)
    schema = BUILTIN.tasks["native/wikitext2_ppl"].params_schema["properties"]
    assert (schema["limit"]["default"], schema["seqlen"]["default"]) == (-1, 2048)
    assert schema["seqlen"]["description"]


def test_a_task_metric_missing_from_metrics_yaml_is_accepted():
    ppl = BUILTIN.tasks["native/wikitext2_ppl"]
    extra = ppl.model_copy(update={"metrics": [*ppl.metrics, TaskMetric(name="bleu", higher_is_better=True)]})
    _check_task(extra, BUILTIN)


def test_a_task_metric_contradicting_metrics_yaml_is_rejected():
    ppl = BUILTIN.tasks["native/wikitext2_ppl"]
    wrong = ppl.model_copy(update={"metrics": [TaskMetric(name="ppl", higher_is_better=True)]})
    with pytest.raises(ValueError, match="ppl"):
        _check_task(wrong, BUILTIN)


def test_a_mode_the_evaluator_cannot_run_is_rejected():
    from effeval.core.types import Artifact, Mode

    ppl = BUILTIN.tasks["native/wikitext2_ppl"]
    wrong = ppl.model_copy(
        update={
            "modes": [Mode.GENERATION],
            "default_mode": Mode.GENERATION,
            "produces": {Mode.GENERATION: frozenset({Artifact.PER_SAMPLE})},
        }
    )
    with pytest.raises(ValueError, match="generation"):
        _check_task(wrong, BUILTIN)


def test_a_task_name_must_be_evaluator_slash_target():
    ppl = BUILTIN.tasks["native/wikitext2_ppl"]
    with pytest.raises(ValueError, match="<evaluator>/<target>"):
        TaskDef(**{**{k: getattr(ppl, k) for k in TaskDef.model_fields}, "name": "wikitext2_ppl"})


def test_tasks_sharing_a_settings_class_share_the_schema_object():
    assert CATALOG.tasks["lm_eval/mmlu"].params_schema is CATALOG.tasks["lm_eval/arc_challenge"].params_schema
    assert CATALOG.tasks["lm_eval/gsm8k"].params_schema is CATALOG.tasks["evalscope/gsm8k"].params_schema


def _benchmark_dir(tmp_path, monkeypatch, source):
    import effeval.core.benchmarks as package

    (tmp_path / "extra_bench.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(package, "__path__", [*package.__path__, str(tmp_path)])
    monkeypatch.delitem(__import__("sys").modules, "effeval.core.benchmarks.extra_bench", raising=False)
    return package


def test_a_new_benchmark_module_is_found_without_registration(tmp_path, monkeypatch):
    package = _benchmark_dir(
        tmp_path,
        monkeypatch,
        "from effeval.core.benchmarks.wikitext2_ppl import BENCHMARK as PPL\n"
        "BENCHMARK = PPL.model_copy(update={'name': 'native/ppl_copy', 'target': 'ppl_copy'})\n",
    )
    assert "native/ppl_copy" in {b.name for b in package.discover()}


def test_a_module_without_a_benchmark_is_rejected(tmp_path, monkeypatch):
    package = _benchmark_dir(tmp_path, monkeypatch, "VALUE = 1\n")
    with pytest.raises(ValueError, match="must define BENCHMARK"):
        package.discover()


def test_listed_tasks_are_namespaced_with_subtasks_and_categories():
    assert CATALOG.tasks["lm_eval/mmlu"].subtasks == ["lm_eval/mmlu_stem"]
    assert CATALOG.tasks["lm_eval/gsm8k"].category == "math"
    assert CATALOG.tasks["evalscope/mmlu"].category == "qa"
    assert CATALOG.tasks["evalscope/mmlu"].subtasks == ["evalscope/mmlu:anatomy"]
    assert CATALOG.tasks["lm_eval/arc_challenge"].category == "commonsense"
    assert CATALOG.evaluators["lm_eval"].source == "listing"
    assert CATALOG.evaluators["lm_eval"].library_version == "0.4.13"


def test_metric_directions_come_from_the_task_then_metrics_yaml():
    assert CATALOG.metric_direction("lm_eval/arc_challenge", "acc") is True
    assert CATALOG.metric_direction("evalscope/mmlu", "accuracy") is True  # from metrics.yaml
    assert CATALOG.metric_direction("evalscope/mmlu", "bleu") is None
    assert CATALOG.metric_direction("lm_eval/wikitext", "bits_per_byte_stderr") is None
    assert CATALOG.metric_direction("lm_eval/wikitext", "word_perplexity") is False
    assert CATALOG.metric_direction("lm_eval/wikitext", "bits_per_byte") is False
    assert CATALOG.metric_direction("native/wikitext2_ppl", "ppl") is False
    assert CATALOG.metric_direction("native/wikitext2_ppl", "accuracy") is True


def test_a_second_listing_of_an_evaluator_replaces_all_its_tasks():
    raw = raw_listing("lm_eval")
    raw["tasks"] = [t for t in raw["tasks"] if t["name"] == "gsm8k"]
    narrowed = CATALOG.with_task_listings([TaskListing.model_validate(raw)])
    assert {n for n in narrowed.tasks if n.startswith("lm_eval/")} == {"lm_eval/gsm8k"}
    assert "evalscope/mmlu" in narrowed.tasks
    assert "native/wikitext2_ppl" in narrowed.tasks


def test_a_listing_may_not_take_a_builtin_evaluator_name():
    raw = raw_listing("lm_eval")
    raw["evaluator"]["name"] = "native"
    with pytest.raises(ValueError, match="native"):
        BUILTIN.with_task_listings([TaskListing.model_validate(raw)])


def test_listings_fixture_is_applied_to_both_evaluators():
    assert {listing.evaluator.name for listing in LISTINGS} == {"lm_eval", "evalscope"}


def test_shipped_snapshot_loads():
    import time

    _static_catalog.cache_clear()
    start = time.perf_counter()
    catalog = builtin_catalog()
    elapsed = time.perf_counter() - start
    print(f"builtin_catalog() with the shipped snapshot: {elapsed:.2f}s, {len(catalog.tasks)} tasks")
    assert elapsed < 3
    for name, distribution in (("lm_eval", "lm-eval"), ("evalscope", "evalscope")):
        evaluator = catalog.evaluators[name]
        assert evaluator.source == "listing" and evaluator.library == distribution
    assert catalog.evaluators["lm_eval"].library_version == "0.4.13"
    assert catalog.evaluators["evalscope"].library_version == "1.15.0"
    for name in (
        "lm_eval/mmlu",
        "lm_eval/gsm8k",
        "lm_eval/arc_challenge",
        "lm_eval/hellaswag",
        "lm_eval/winogrande",
        "lm_eval/piqa",
        "lm_eval/wikitext",
        "evalscope/mmlu",
        "evalscope/gsm8k",
    ):
        assert name in catalog.tasks
    assert [n for n in catalog.common_tasks if n not in catalog.tasks] == []


def test_common_tasks_are_a_subset_of_the_catalog_in_the_fixture_catalog():
    assert "lm_eval/mmlu" in CATALOG.common_tasks

@pytest.mark.parametrize(
    ("limit", "total", "expected"),
    [(None, 10, 10), (3, 10, 3), (30, 10, 10), (0.25, 10, 3), (0.5, 10, 5), (0.01, 10, 1), (0.999, 10, 10)],
)
def test_sample_count_rounds_a_fraction_up(limit, total, expected):
    assert sample_count(limit, total) == expected


def test_limit_schema_lets_agents_see_that_a_fraction_is_accepted():
    for name in ("native/wikitext2_ppl", "lm_eval/mmlu"):
        task = CATALOG.tasks[name]
        limit = task.params_schema["properties"]["limit"]
        assert {"integer", "number"} <= {option["type"] for option in limit["anyOf"]}
        assert "fraction" in limit["description"]


def test_every_preset_a_preprocess_preset_pairs_with_exists():
    pairs = [
        (name, method, preset)
        for name, pre in builtin_preprocess_presets().items()
        for method, presets in pre.quantizers.items()
        for preset in presets
    ]
    assert pairs
    for name, method, preset in pairs:
        assert preset in BUILTIN.presets[method], (name, method, preset)
    quarot = builtin_preprocess_presets()["quarot"].quantizers
    assert quarot == {
        "gptq": ["quarot_w4a16", "quarot_w4a4kv4"],
        "rtn": ["quarot_w4a16", "quarot_w4a4kv4"],
    }

