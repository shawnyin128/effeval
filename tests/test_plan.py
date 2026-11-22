import itertools

import pytest

from catalogs import CATALOG
from effeval import validate as validate_module
from effeval.core.canonical import canonical_json, sha256_hex
from effeval.core.catalog import DEFAULT_SAMPLING_TEMPERATURE, PaperRef, PresetValue
from effeval.core.preprocess import SOURCE_VERSIONS, PreprocessPresetDef
from effeval.core.spec import Job, run_key
from effeval.core.types import Artifact, Fidelity, Record, Severity
from effeval.plan import freeze, model_group_key, shard

GPTQ = {"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}


class FakeLookup:
    def __init__(self, records):
        self.records = records
        self.calls = []

    def find_reusable(self, key, *, viewer, need, hardware):
        self.calls.append((key, need, hardware))
        record = self.records.get(key)
        if record is None or not need <= set(record.produces):
            return None
        return record


def _record(key, produces=()):
    return Record(
        record_id="rec_1",
        job_id="job_x",
        run_key=key,
        reference_key=None,
        role="reference",
        spec={},
        metrics={"ppl": "27.65"},
        protocol={},
        env=None,
        fidelity=Fidelity(),
        produces=list(produces),
        owner="alice",
        created_at="2026-11-20T00:00:00+00:00",
    )


def test_gptq_plan_has_main_and_matching_reference():
    plan, issues = freeze(GPTQ, CATALOG, owner="bob", server=None, lookup=None)
    assert plan is not None, issues
    main, ref = (next(j for j in plan.jobs if j.role == r) for r in ("main", "reference"))
    assert main.reference_key == ref.run_key
    assert ref.spec.method.name == "base" and ref.spec.method.config == {}
    assert ref.spec.model == main.spec.model and main.spec.model.dtype == "bfloat16"
    assert ref.spec.task == main.spec.task and ref.spec.evaluator == main.spec.evaluator
    assert main.compare == []  # flip_rate/kl need per-sample correctness or choice logliks
    assert main.fidelity.preset == "gptq/paper"
    assert all(j.job_id.startswith("job_") for j in plan.jobs)
    assert [i.code for i in issues if i.code == "reuse.unavailable"]


def test_run_keys_do_not_depend_on_method_titles():
    plan, _ = freeze(GPTQ, CATALOG, owner="bob", server=None, lookup=None)
    keys = {j.role: j.run_key for j in plan.jobs}
    assert keys == {
        "reference": "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b",
        "main": "rk_dc033ef0a9468ea5942dbd611125887f4b6488d85567eff4a5625bf0140c1d3c",
    }
    retitled = CATALOG.model_copy(
        update={"manifests": {n: m.model_copy(update={"title": "X " + n}) for n, m in CATALOG.manifests.items()}}
    )
    plan, _ = freeze(GPTQ, retitled, owner="bob", server=None, lookup=None)
    assert {j.role: j.run_key for j in plan.jobs} == keys


def test_sweep_shares_one_reference():
    raw = {
        "model": "qwen3-8b-base",
        "method": {
            "name": "rtn",
            "preset": None,
            "config": {
                "rules": [
                    {
                        "sites": ["linear.*.weight"],
                        "quant": {"type": "int", "bits": 4, "symmetric": False, "granularity": "per_channel"},
                    }
                ]
            },
        },
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
        "sweep": {
            "method.config.rules": [
                [
                    {
                        "sites": ["linear.*.weight"],
                        "quant": {"type": "int", "bits": b, "symmetric": False, "granularity": "per_channel"},
                    }
                ]
                for b in (3, 4)
            ]
        },
    }
    plan, issues = freeze(raw, CATALOG, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    assert sorted(j.role for j in plan.jobs) == ["main", "main", "reference"]
    main_jobs = [j for j in plan.jobs if j.role == "main"]
    assert {j.compare == ["flip_rate", "kl"] for j in main_jobs} == {True}
    assert plan.jobs[0].role == "reference" or plan.jobs[0].spec.method.name != "base"


def test_reference_is_reused_only_with_required_artifacts():
    plan, _ = freeze(GPTQ, CATALOG, owner="bob", server=None, lookup=None)
    ref_key = next(j for j in plan.jobs if j.role == "reference").run_key
    lookup = FakeLookup({ref_key: _record(ref_key)})
    plan, _ = freeze(GPTQ, CATALOG, owner="bob", server=None, lookup=lookup)
    assert [r.run_key for r in plan.reused] == [ref_key]
    assert [j.role for j in plan.jobs] == ["main"]

    mmlu = {**GPTQ, "tasks": ["lm_eval/mmlu"], "model": "qwen3-8b-base"}
    plan, _ = freeze(mmlu, CATALOG, owner="bob", server=None, lookup=None)
    ref_key = next(j for j in plan.jobs if j.role == "reference").run_key
    aggregate_only = FakeLookup({ref_key: _record(ref_key)})
    plan, _ = freeze(mmlu, CATALOG, owner="bob", server=None, lookup=aggregate_only)
    assert plan.reused == []  # flip_rate/kl need per-sample artifacts the stored record lacks
    assert aggregate_only.calls[-1][1] == frozenset(
        {Artifact.PER_SAMPLE, Artifact.CHOICE_LOGLIKELIHOODS}
    ) or any(
        call[1] == frozenset({Artifact.PER_SAMPLE, Artifact.CHOICE_LOGLIKELIHOODS})
        for call in aggregate_only.calls
    )


def test_reuse_never_skips_lookup_and_same_hardware_requires_gpu():
    lookup = FakeLookup({})
    freeze({**GPTQ, "reuse": "never"}, CATALOG, owner=None, server=None, lookup=lookup)
    assert lookup.calls == []
    _, issues = freeze({**GPTQ, "reuse": "same_hardware"}, CATALOG, owner=None, server=None, lookup=lookup)
    assert "reuse.no_hardware" in [i.code for i in issues]
    freeze(
        {**GPTQ, "reuse": "same_hardware", "hardware": {"gpu": "H100-SXM"}},
        CATALOG,
        owner=None,
        server=None,
        lookup=lookup,
    )
    assert lookup.calls[-1][2] == "H100-SXM"


def test_quality_run_key_ignores_hardware():
    a, _ = freeze({**GPTQ, "hardware": {"gpu": "H100-SXM"}}, CATALOG, owner=None, server=None, lookup=None)
    b, _ = freeze({**GPTQ, "hardware": {"gpu": "H200-SXM"}}, CATALOG, owner=None, server=None, lookup=None)
    assert {j.run_key for j in a.jobs} == {j.run_key for j in b.jobs}
    assert run_key(a.jobs[0].spec) == a.jobs[0].run_key


def _keys(tasks):
    plan, issues = freeze({**GPTQ, "tasks": tasks}, CATALOG, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return {j.role: j for j in plan.jobs}


def test_benchmark_settings_are_part_of_the_run_key_only_when_changed():
    default = _keys(["native/wikitext2_ppl"])
    spelled_out = _keys([{"name": "native/wikitext2_ppl", "params": {"limit": -1, "seqlen": 2048}}])
    assert {r: j.run_key for r, j in spelled_out.items()} == {r: j.run_key for r, j in default.items()}
    shorter = _keys([{"name": "native/wikitext2_ppl", "params": {"seqlen": 1024}}])
    fewer = _keys([{"name": "native/wikitext2_ppl", "params": {"limit": 10}}])
    keys = [{j.run_key for j in plan.values()} for plan in (default, shorter, fewer)]
    assert all(a.isdisjoint(b) for a, b in itertools.combinations(keys, 2))
    assert shorter["main"].spec.task.params == {"seqlen": 1024}
    assert (fewer["main"].spec.task.limit, fewer["main"].spec.task.params) == (10, {})
    assert shorter["reference"].spec.task == shorter["main"].spec.task


def test_default_settings_keep_the_run_key_of_specs_without_settings():
    spec = _keys(["native/wikitext2_ppl"])["main"].spec
    task = spec.task
    legacy = {
        "model": spec.model.model_dump(mode="json"),
        "method": {"name": spec.method.name, "version": spec.method.version, "config": spec.method.config},
        "runtime": spec.runtime.model_dump(mode="json", exclude={"cuda_graphs", "static_cache_tokens"}),
        "task": {
            "name": task.name,
            "version": task.version,
            "mode": task.mode.value,
            "num_fewshot": task.num_fewshot,
            "limit": task.limit,
        },
        "evaluator": spec.evaluator.model_dump(mode="json"),
        "generation": {"batch_size": spec.generation.batch_size},
        "determinism": spec.determinism.model_dump(mode="json"),
    }
    assert run_key(spec) == "rk_" + sha256_hex(canonical_json(legacy))


def test_invalid_benchmark_settings_are_reported_on_their_path():
    for params, path in (
        ({"seqlen": 4}, "tasks[0].params.seqlen"),
        ({"limit": 0}, "tasks[0].params.limit"),
        ({"window": 3}, "tasks[0].params.window"),
    ):
        plan, issues = freeze(
            {**GPTQ, "tasks": [{"name": "native/wikitext2_ppl", "params": params}]},
            CATALOG,
            owner=None,
            server=None,
            lookup=None,
        )
        assert plan is None
        assert [(i.code, i.paths) for i in issues if i.severity == Severity.ERROR] == [
            ("task.params_invalid", [path])
        ]


def test_changed_benchmark_settings_are_deviations_from_the_paper():
    main = _keys([{"name": "native/wikitext2_ppl", "params": {"seqlen": 1024, "limit": 8}}])["main"]
    assert {(d.path, d.preset, d.actual) for d in main.fidelity.deviations} >= {
        ("tasks.native/wikitext2_ppl.seqlen", 2048, 1024),
        ("tasks.native/wikitext2_ppl.limit", -1, 8),
    }



def _limit_issues(limit):
    plan, issues = freeze(
        {**GPTQ, "tasks": [{"name": "native/wikitext2_ppl", "params": {"limit": limit}}]},
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    return plan, [(i.code, i.paths) for i in issues if i.severity == Severity.ERROR]


@pytest.mark.parametrize("limit", [0.5, 0.01, 0.999, 3, 1, -1])
def test_a_limit_is_a_count_a_fraction_or_all(limit):
    plan, errors = _limit_issues(limit)
    assert plan is not None and errors == []


@pytest.mark.parametrize("limit", [0, 0.0, 1.0, 1.5, -0.1, -2, 5.0, True, "0.5", float("nan")])
def test_a_limit_that_is_neither_count_fraction_nor_all_is_rejected(limit):
    plan, errors = _limit_issues(limit)
    assert plan is None
    assert errors == [("task.params_invalid", ["tasks[0].params.limit"])]


def test_fractional_integer_and_absent_limits_have_distinct_run_keys():
    keys = [
        {j.run_key for j in _keys([{"name": "native/wikitext2_ppl", "params": params}]).values()}
        for params in ({}, {"limit": 1}, {"limit": 0.1}, {"limit": 0.2})
    ]
    assert all(a.isdisjoint(b) for a, b in itertools.combinations(keys, 2))
    main = _keys([{"name": "native/wikitext2_ppl", "params": {"limit": 0.1}}])["main"]
    assert (main.spec.task.limit, main.spec.task.params) == (0.1, {})
    assert ("tasks.native/wikitext2_ppl.limit", -1, 0.1) in {
        (d.path, d.preset, d.actual) for d in main.fidelity.deviations
    }
    assert run_key(Job.model_validate_json(main.model_dump_json()).spec) == main.run_key

def _generation_jobs(params, generation=None):
    raw = {**GPTQ, "tasks": [{"name": "evalscope/gsm8k", "params": params}]}
    if generation:
        raw["generation"] = generation
    plan, issues = freeze(raw, CATALOG, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return {j.role: j for j in plan.jobs}


def test_a_task_max_new_tokens_overrides_the_run_level_value_for_that_task_only():
    jobs = _generation_jobs({"max_new_tokens": 16}, {"max_new_tokens": 256})
    assert {j.spec.generation.max_new_tokens for j in jobs.values()} == {16}
    assert all(j.spec.task.params == {} for j in jobs.values())
    plan, issues = freeze(
        {
            **GPTQ,
            "generation": {"max_new_tokens": 256},
            "tasks": [{"name": "evalscope/gsm8k", "params": {"max_new_tokens": 16}}, "lm_eval/gsm8k"],
        },
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    assert plan is not None, issues
    lengths = {j.spec.task.name: j.spec.generation.max_new_tokens for j in plan.jobs if j.role == "main"}
    assert lengths == {"evalscope/gsm8k": 16, "lm_eval/gsm8k": 256}

def test_the_effective_generation_length_is_part_of_the_run_key():
    keys = {
        name: {j.run_key for j in _generation_jobs(params, generation).values()}
        for name, (params, generation) in {
            "default": ({}, None),
            "task 16": ({"max_new_tokens": 16}, None),
            "task 32": ({"max_new_tokens": 32}, None),
            "run 16": ({}, {"max_new_tokens": 16}),
        }.items()
    }
    assert keys["task 16"] == keys["run 16"]
    assert keys["default"].isdisjoint(keys["task 16"]) and keys["task 16"].isdisjoint(keys["task 32"])
    same_as_run_level = _generation_jobs({"max_new_tokens": 512})
    assert {j.run_key for j in same_as_run_level.values()} == keys["default"]


@pytest.mark.parametrize(
    ("task", "value", "message"),
    [
        ("lm_eval/arc_challenge", 16, "never generates"),
        ("native/wikitext2_ppl", 16, "never generates"),
        ("evalscope/gsm8k", 0, "greater than or equal to 1"),
        ("evalscope/gsm8k", -5, "greater than or equal to 1"),
        ("evalscope/gsm8k", 1.5, ""),
    ],
)
def test_task_max_new_tokens_is_rejected_where_it_cannot_apply(task, value, message):
    plan, issues = freeze(
        {**GPTQ, "tasks": [{"name": task, "params": {"max_new_tokens": value}}]},
        CATALOG, owner=None, server=None, lookup=None,
    )
    assert plan is None
    errors = [i for i in issues if i.severity == Severity.ERROR]
    assert [(i.code, i.paths) for i in errors] == [("task.params_invalid", ["tasks[0].params.max_new_tokens"])]
    assert message in errors[0].message


def test_only_generating_library_tasks_list_max_new_tokens_in_their_schema():
    assert "max_new_tokens" in CATALOG.tasks["evalscope/gsm8k"].params_schema["properties"]
    assert "max_new_tokens" not in CATALOG.tasks["lm_eval/arc_challenge"].params_schema["properties"]
    assert CATALOG.tasks["evalscope/gsm8k"].params_schema["properties"]["max_new_tokens"]["description"]


GOLDEN_GREEDY_GSM8K = {
    "main": "rk_68e228e5001155faf22a6e9ac031153d046e9ae2729086724374fd633feb94fc",
    "reference": "rk_1a42b6707db6d216aa8edf1fef4fa3e0d0d9098221f51b1e9387fd434b75d38a",
}


def _generation_of(params, generation=None):
    return {role: j.spec.generation for role, j in _generation_jobs(params, generation).items()}


def test_a_task_that_samples_runs_main_and_reference_with_the_default_temperature():
    for role, generation in _generation_of({"do_sample": True}).items():
        assert (generation.do_sample, generation.temperature) == (True, DEFAULT_SAMPLING_TEMPERATURE), role
    assert all(j.spec.task.params == {} for j in _generation_jobs({"do_sample": True}).values())


def test_a_task_temperature_beats_the_run_level_one_which_beats_the_default():
    assert {g.temperature for g in _generation_of({"do_sample": True, "temperature": 0.3}).values()} == {0.3}
    run_level = {"temperature": 0.5}
    assert {g.temperature for g in _generation_of({"do_sample": True}, run_level).values()} == {0.5}
    both = _generation_of({"do_sample": True, "temperature": 0.3}, run_level)
    assert {g.temperature for g in both.values()} == {0.3}


def test_sampling_is_chosen_per_task():
    plan, issues = freeze(
        {**GPTQ, "tasks": [{"name": "evalscope/gsm8k", "params": {"do_sample": True}}, "lm_eval/gsm8k"]},
        CATALOG, owner=None, server=None, lookup=None,
    )
    assert plan is not None, issues
    sampled = {j.spec.task.name: j.spec.generation.do_sample for j in plan.jobs if j.role == "main"}
    assert sampled == {"evalscope/gsm8k": True, "lm_eval/gsm8k": False}


def test_a_task_can_switch_off_run_level_sampling():
    generation = _generation_of({"do_sample": False}, {"do_sample": True, "temperature": 0.9})
    assert {(g.do_sample, g.temperature) for g in generation.values()} == {(False, None)}


def test_greedy_generation_keeps_the_run_keys_recorded_before_sampling_was_a_task_setting():
    for params in ({}, {"do_sample": False}):
        jobs = _generation_jobs(params)
        assert {role: j.run_key for role, j in jobs.items()} == GOLDEN_GREEDY_GSM8K


def test_the_effective_sampling_is_part_of_the_run_key():
    keys = {
        name: {j.run_key for j in _generation_jobs(params, generation).values()}
        for name, (params, generation) in {
            "greedy": ({}, None),
            "sampled": ({"do_sample": True}, None),
            "sampled 0.3": ({"do_sample": True, "temperature": 0.3}, None),
            "sampled at the default": ({"do_sample": True, "temperature": DEFAULT_SAMPLING_TEMPERATURE}, None),
            "run level": ({}, {"do_sample": True}),
        }.items()
    }
    assert keys["sampled"] == keys["sampled at the default"]
    assert keys["greedy"].isdisjoint(keys["sampled"]) and keys["sampled"].isdisjoint(keys["sampled 0.3"])


def test_sampled_repeats_with_varying_seeds_are_distinct_jobs_with_the_same_sampling():
    plan, issues = freeze(
        {
            **GPTQ,
            "tasks": [{"name": "evalscope/gsm8k", "params": {"do_sample": True}}],
            "determinism": {"seed": 4},
            "repeats": {"seeds": [4, 9, 2]},
        },
        CATALOG, owner=None, server=None, lookup=None,
    )
    assert plan is not None, issues
    mains = [j for j in plan.jobs if j.role == "main"]
    assert sorted(j.spec.determinism.seed for j in mains) == [2, 4, 9]
    assert {j.spec.generation.do_sample for j in mains} == {True}


@pytest.mark.parametrize(
    ("task", "params", "path", "message"),
    [
        ("lm_eval/arc_challenge", {"do_sample": True}, "do_sample", "never generates"),
        ("lm_eval/arc_challenge", {"temperature": 0.5}, "temperature", "never generates"),
        ("native/wikitext2_ppl", {"do_sample": True}, "do_sample", "never generates"),
        ("kernel/gemm_w4a16", {"temperature": 0.5}, "temperature", "never generates"),
        ("evalscope/gsm8k", {"temperature": 0.5}, "temperature", "needs do_sample"),
        ("evalscope/gsm8k", {"do_sample": False, "temperature": 0.5}, "temperature", "needs do_sample"),
        ("evalscope/gsm8k", {"do_sample": True, "temperature": 0}, "temperature", "greater than 0"),
        ("evalscope/gsm8k", {"do_sample": True, "temperature": -1}, "temperature", "greater than 0"),
    ],
)
def test_task_sampling_is_rejected_where_it_cannot_apply(task, params, path, message):
    plan, issues = freeze(
        {**GPTQ, "tasks": [{"name": task, "params": params}]}, CATALOG, owner=None, server=None, lookup=None
    )
    assert plan is None
    errors = [i for i in issues if i.severity == Severity.ERROR]
    assert [(i.code, i.paths) for i in errors] == [("task.params_invalid", [f"tasks[0].params.{path}"])]
    assert message in errors[0].message


def test_only_generating_library_tasks_list_sampling_in_their_schema():
    properties = CATALOG.tasks["evalscope/gsm8k"].params_schema["properties"]
    assert properties["do_sample"]["description"] and properties["temperature"]["description"]
    for name in ("lm_eval/arc_challenge", "native/wikitext2_ppl"):
        assert not {"do_sample", "temperature"} & CATALOG.tasks[name].params_schema["properties"].keys()


def test_shard_keeps_model_groups_together():
    plan, _ = freeze(
        {**GPTQ, "model": "qwen3-8b-base", "tasks": ["lm_eval/mmlu", "lm_eval/arc_challenge"]},
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    first, second = shard(plan, 1, 2), shard(plan, 2, 2)
    assert len(first.jobs) + len(second.jobs) == len(plan.jobs) == 4
    for part in (first, second):
        assert len({j.spec.method.name for j in part.jobs}) == 1


def test_errors_return_no_plan():
    plan, issues = freeze(
        {**GPTQ, "tasks": [{"name": "evalscope/mmlu", "mode": "loglikelihood"}]},
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    assert plan is None
    assert "mode.task_unsupported" in [i.code for i in issues]


def _main(task, **extra):
    plan, issues = freeze(
        {**GPTQ, "model": "qwen3-8b-base", "tasks": [task], **extra},
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    assert plan is not None, issues
    return next(j for j in plan.jobs if j.role == "main")


def test_num_fewshot_override_changes_the_run_key_and_the_default_does_not():
    base = _main("lm_eval/gsm8k")
    assert base.spec.task.num_fewshot == 5
    spelled_out = _main({"name": "lm_eval/gsm8k", "params": {"num_fewshot": 5}})
    changed = _main({"name": "lm_eval/gsm8k", "params": {"num_fewshot": 3}})
    assert spelled_out.run_key == base.run_key
    assert changed.run_key != base.run_key
    assert changed.spec.task.num_fewshot == 3
    deviation = next(d for d in changed.fidelity.deviations if d.path == "tasks.lm_eval/gsm8k.num_fewshot")
    assert (deviation.preset, deviation.actual) == (5, 3)


def test_the_same_task_name_in_two_libraries_never_shares_a_run_key():
    assert _main("lm_eval/gsm8k").run_key != _main("evalscope/gsm8k").run_key


def test_passthrough_reaches_only_its_own_evaluator():
    raw = {"evaluator": {"passthrough": {"lm_eval": {"bootstrap_iters": 0}}}}
    plan, issues = freeze(
        {**GPTQ, "model": "qwen3-8b-base", "tasks": ["lm_eval/mmlu", "native/wikitext2_ppl"], **raw},
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    assert plan is not None, issues
    by_task = {j.spec.task.name: j.spec.evaluator for j in plan.jobs if j.role == "main"}
    assert by_task["lm_eval/mmlu"].passthrough == {"bootstrap_iters": 0}
    assert by_task["lm_eval/mmlu"].library == "lm-eval==0.4.13"
    assert by_task["native/wikitext2_ppl"].passthrough == {}
    assert by_task["native/wikitext2_ppl"].library is None


def test_tasks_deviation_is_per_job():
    raw = {
        **GPTQ,
        "tasks": ["native/wikitext2_ppl", "lm_eval/arc_challenge"],
        "method": {"name": "gptq", "config": {"percdamp": 0.1}},
    }
    plan, issues = freeze(raw, CATALOG, owner="bob", server=None, lookup=None)
    assert plan is not None, issues
    mains = {j.spec.task.name: j.fidelity for j in plan.jobs if j.role == "main"}
    paper, extra = mains["native/wikitext2_ppl"], mains["lm_eval/arc_challenge"]
    assert "tasks" in paper.confirmed and all(d.path != "tasks" for d in paper.deviations)
    tasks = next(d for d in extra.deviations if d.path == "tasks")
    assert (tasks.preset, tasks.actual) == (["native/wikitext2_ppl"], "lm_eval/arc_challenge")
    damp = next(d for d in paper.deviations if d.path == "method.config.percdamp")
    assert (damp.preset, damp.actual) == (0.01, 0.1)


def test_a_plan_carries_the_listed_tasks_it_uses_and_round_trips():
    from effeval.core.catalog import builtin_catalog
    from effeval.core.spec import Plan

    plan, issues = freeze(
        {**GPTQ, "model": "qwen3-8b-base", "tasks": ["lm_eval/mmlu"]},
        CATALOG,
        owner=None,
        server=None,
        lookup=None,
    )
    assert plan is not None, issues
    again = Plan.model_validate_json(plan.model_dump_json())
    names = {t.name for t in again.listings["lm_eval"].tasks}
    assert names == {"mmlu", "mmlu_stem", "mmlu_abstract_algebra", "mmlu_anatomy"}
    node = builtin_catalog().with_task_listings(list(again.listings.values()))
    assert "lm_eval/mmlu" in node.tasks and "lm_eval/gsm8k" not in node.tasks


def test_a_plan_of_builtin_tasks_has_no_listings():
    plan, _ = freeze(GPTQ, CATALOG, owner=None, server=None, lookup=None)
    assert plan.listings == {}


GOLDEN_MAIN_KEY = "rk_dc033ef0a9468ea5942dbd611125887f4b6488d85567eff4a5625bf0140c1d3c"
GOLDEN_REFERENCE_KEY = "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b"


def _mains(extra):
    plan, issues = freeze({**GPTQ, **extra}, CATALOG, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return sorted((j for j in plan.jobs if j.role == "main"), key=lambda j: j.spec.repeat), plan


def test_one_repeat_keeps_the_run_keys_recorded_before_repeats_existed():
    for extra in ({}, {"repeats": {"count": 1}}, {"repeats": {"seeds": [0]}}):
        mains, plan = _mains(extra)
        assert [j.run_key for j in mains] == [GOLDEN_MAIN_KEY]
        assert {j.run_key for j in plan.jobs if j.role == "reference"} == {GOLDEN_REFERENCE_KEY}


def test_same_seed_repeats_are_distinct_runs_with_the_same_seed():
    mains, plan = _mains({"determinism": {"seed": 5}, "repeats": {"count": 3}})
    assert [j.spec.repeat for j in mains] == [0, 1, 2]
    assert {j.spec.determinism.seed for j in mains} == {5}
    assert len({j.run_key for j in mains}) == 3
    assert len({j.run_key for j in plan.jobs if j.role == "reference"}) == 3


def test_a_seed_list_plans_one_job_per_listed_seed_in_list_order():
    mains, plan = _mains({"repeats": {"seeds": [7, 3, 11]}})
    assert [j.spec.determinism.seed for j in mains] == [7, 3, 11]
    assert {j.spec.repeat for j in mains} == {0}
    assert len({j.run_key for j in mains}) == 3
    assert len({j.run_key for j in plan.jobs if j.role == "reference"}) == 3


def test_a_seed_list_ignores_determinism_seed_and_a_matching_count_is_accepted():
    mains, _ = _mains({"determinism": {"seed": 99}, "repeats": {"count": 2, "seeds": [3, 5]}})
    assert [j.spec.determinism.seed for j in mains] == [3, 5]


def test_a_job_in_a_seed_list_is_the_same_run_as_a_single_run_with_that_seed():
    single, _ = _mains({"determinism": {"seed": 7}})
    listed, _ = _mains({"repeats": {"seeds": [3, 7]}})
    other, _ = _mains({"repeats": {"seeds": [7, 8, 9]}})
    key = single[0].run_key
    assert key in {j.run_key for j in listed} and key in {j.run_key for j in other}
    assert {j.run_key for j in listed} & {j.run_key for j in other} == {key}


def test_a_stored_single_seed_result_is_reused_by_a_seed_list_containing_it():
    mains, _ = _mains({"determinism": {"seed": 7}})
    lookup = FakeLookup({mains[0].run_key: _record(mains[0].run_key)})
    plan, _ = freeze({**GPTQ, "repeats": {"seeds": [3, 7]}}, CATALOG, owner=None, server=None, lookup=lookup)
    assert [r.run_key for r in plan.reused if r.role == "main"] == [mains[0].run_key]
    assert [j.spec.determinism.seed for j in plan.jobs if j.role == "main"] == [3]


@pytest.mark.parametrize(
    ("repeats", "message"),
    [
        ({"seeds": []}, "at least 1"),
        ({"seeds": [3, 5, 3]}, "duplicate"),
        ({"count": 3, "seeds": [3, 5]}, "lists 2 seeds"),
        ({"seeds": "vary"}, "valid list"),
        ({"count": 0}, "greater than or equal to 1"),
    ],
)
def test_invalid_repeats_are_rejected_with_the_repeats_path(repeats, message):
    plan, issues = freeze({**GPTQ, "repeats": repeats}, CATALOG, owner=None, server=None, lookup=None)
    assert plan is None
    errors = [i for i in issues if i.severity == Severity.ERROR]
    assert errors and message in errors[0].message and errors[0].paths[0].startswith("repeats")


ROTATE = {
    "kind": "rotation",
    "entries": [
        {
            "position": "residual",
            "mode": "offline",
            "source": {"type": "random_hadamard", "seed": 0},
            "options": {"untie_embeddings": True},
        }
    ],
}


@pytest.fixture
def all_positions_ok(monkeypatch):
    table = {
        "residual": {k: {m: "ok" for m in ("offline", "online")} for k in ("rotation", "scaling")},
    }
    monkeypatch.setattr(validate_module, "_position_table", lambda architecture, entry: table)


def _preprocessed(preprocess, extra=None):
    raw = {**GPTQ, "preprocess": preprocess, **(extra or {})}
    plan, issues = freeze(raw, CATALOG, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return plan


def _role(plan, role):
    return next(j for j in plan.jobs if j.role == role)


def test_preprocessing_changes_the_run_key_and_the_model_group_but_not_the_reference(all_positions_ok):
    plain = _role(freeze(GPTQ, CATALOG, owner=None, server=None, lookup=None)[0], "main")
    plan = _preprocessed({"steps": [ROTATE]})
    reseeded = {**ROTATE["entries"][0], "source": {"type": "random_hadamard", "seed": 1}}
    other = _preprocessed({"steps": [{**ROTATE, "entries": [reseeded]}]})
    main, reference = _role(plan, "main"), _role(plan, "reference")
    assert main.spec.preprocess is not None and reference.spec.preprocess is None
    assert main.run_key not in {plain.run_key, _role(other, "main").run_key}
    assert model_group_key(main) not in {model_group_key(plain), model_group_key(_role(other, "main"))}
    assert reference.run_key == plain.reference_key == main.reference_key


def test_source_versions_are_part_of_the_run_key(
    all_positions_ok, monkeypatch
):
    named = _role(_preprocessed({"steps": [ROTATE]}), "main")
    assert named.spec.preprocess.versions == {"rotation.random_hadamard": "1"}
    monkeypatch.setitem(SOURCE_VERSIONS["rotation"], "random_hadamard", "2")
    bumped = _role(_preprocessed({"steps": [ROTATE]}), "main")
    assert bumped.run_key != named.run_key


def test_a_preprocess_preset_fills_steps_and_is_not_part_of_the_run_key(all_positions_ok, monkeypatch):
    preset = PreprocessPresetDef(
        name="demo",
        title="Demo",
        description="d",
        paper=PaperRef(title="t", arxiv="1", official_repo="r"),
        values={"preprocess.steps": PresetValue(value=[ROTATE], source="assumed")},
    )
    monkeypatch.setattr(validate_module, "builtin_preprocess_presets", lambda: {"demo": preset})
    from_preset = _preprocessed({"preset": "demo"})
    explicit = _preprocessed({"steps": [ROTATE]})
    assert _role(from_preset, "main").run_key == _role(explicit, "main").run_key
    assert _role(from_preset, "main").spec.preprocess.preset == "demo"
    assert _role(from_preset, "main").fidelity.deviations == []
    changed = {**ROTATE, "entries": [{**ROTATE["entries"][0], "mode": "online"}]}
    deviating = _preprocessed({"preset": "demo", "steps": [changed]})
    assert [d.path for d in _role(deviating, "main").fidelity.deviations] == ["preprocess.steps"]
