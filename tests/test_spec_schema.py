"""Speculative decoding in the schema: RunSpec.decoding, runtime providers, the speed mode, keys and plans."""

import hashlib

import pytest
from pydantic import Field

from catalogs import CATALOG
from effeval.core.catalog import EvaluatorDef, SampleLimit, TaskDef, TaskMetric
from effeval.core.engines import ENGINES
from effeval.core.spec import RunSpec
from effeval.core.types import Artifact, Capabilities, HandleKind, Mode
from effeval.metrics.quality import COMPARATIVE, Comparative
from effeval.plan import freeze, model_group_key
from effeval.validate import resolve, validate

RTN_RULES = [
    {
        "sites": ["linear.*.weight"],
        "quant": {"type": "int", "bits": 4, "symmetric": False, "granularity": "per_channel"},
    }
]


class SpeedParams(SampleLimit):
    max_new_tokens: int | None = Field(default=None, ge=1)
    do_sample: bool | None = None
    temperature: float | None = Field(default=None, gt=0)


SPEED_TASK = TaskDef(
    name="speed/spec_bench",
    evaluator="speed",
    target="spec_bench",
    version="1",
    category="throughput",
    description="stand-in for the Spec-Bench speed task",
    modes=[Mode.SPEED],
    default_mode=Mode.SPEED,
    num_fewshot=None,
    produces={Mode.SPEED: frozenset({Artifact.PER_SAMPLE})},
    metrics=[TaskMetric(name="throughput_tok_s", higher_is_better=True)],
    sample_scores=["tokens_per_s"],
    params=SpeedParams,
)


def _speed_evaluator(handles: set[HandleKind]) -> EvaluatorDef:
    return EvaluatorDef(
        name="speed",
        version="1",
        description="stand-in",
        capabilities=Capabilities(modes={Mode.SPEED}, handles=handles),
    )


def _catalog(*, handles=frozenset({HandleKind.IN_PROCESS, HandleKind.ENGINE}), field_modes=None, models=None):
    fields = CATALOG.fields
    if field_modes is not None:
        field = CATALOG.fields["speculative_decoding"].model_copy(update={"modes": field_modes})
        fields = {**fields, "speculative_decoding": field}
    return CATALOG.model_copy(
        update={
            "tasks": {**CATALOG.tasks, SPEED_TASK.name: SPEED_TASK},
            "evaluators": {**CATALOG.evaluators, "speed": _speed_evaluator(set(handles))},
            "fields": fields,
            "models": {**CATALOG.models, **(models or {})},
        }
    )


@pytest.fixture(autouse=True)
def _speed_comparatives(monkeypatch):
    """The speed comparatives are implemented elsewhere; the field names them, so register stand-ins."""
    for name in ("speedup", "output_match", "multi_turn_prompt_match"):
        monkeypatch.setitem(
            COMPARATIVE,
            name,
            Comparative(name, frozenset({Artifact.PER_SAMPLE}), frozenset(), "stand-in", lambda *_: {}),
        )


def spec(**overrides):
    base = {
        "model": "qwen3-8b",
        "method": {"name": "base"},
        "field": "speculative_decoding",
        "tasks": ["lm_eval/gsm8k"],
        "decoding": {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": 4},
    }
    base.update(overrides)
    return {k: v for k, v in base.items() if v is not None}


def speed_spec(**overrides):
    return spec(tasks=["speed/spec_bench"], hardware={"gpu": "H100-SXM"}, **overrides)


def codes(issues, severity="error"):
    return [i.code for i in issues if i.severity.value == severity]


def check(raw, catalog=None):
    return codes(validate(raw, catalog or _catalog()))


def frozen(raw, catalog=None):
    plan, issues = freeze(raw, catalog or _catalog(), owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return plan, issues


def role(plan, name):
    return next(j for j in plan.jobs if j.role == name)


def test_a_valid_speculative_spec_has_no_errors_and_batch_size_one_comes_from_the_field():
    res = resolve(spec(), _catalog())
    assert res.ok, res.issues
    assert res.spec.generation.batch_size == 1


def test_default_compare_of_the_field_reaches_the_job():
    main = role(frozen(spec())[0], "main")
    assert {"speedup", "output_match"} <= set(main.compare)


def test_existing_shapes_have_identical_keys_with_explicit_autoregressive():
    plain = {"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}
    explicit = {**plain, "decoding": {"mode": "autoregressive"}}
    a, b = frozen(plain)[0], frozen(explicit)[0]
    assert [(j.run_key, j.reference_key, model_group_key(j)) for j in a.jobs] == [
        (j.run_key, j.reference_key, model_group_key(j)) for j in b.jobs
    ]
    assert all(j.spec.decoding is None for j in b.jobs)


def test_decoding_draft_required():
    raw = spec(decoding={"mode": "speculative", "num_speculative_tokens": 4})
    assert "decoding.draft_required" in check(raw)


def test_decoding_k_required():
    raw = spec(decoding={"mode": "speculative", "draft": {"model": "qwen3-0.6b"}})
    assert "decoding.k_required" in check(raw)


@pytest.mark.parametrize(
    "decoding",
    [
        {"mode": "autoregressive", "num_speculative_tokens": 4},
        {"mode": "autoregressive", "draft": {"model": "qwen3-0.6b"}},
    ],
)
def test_decoding_ineffective_when_autoregressive_sets_draft_or_k(decoding):
    assert "decoding.ineffective" in check(spec(decoding=decoding))


def test_decoding_field_required():
    raw = spec(field="quantization")
    assert "decoding.field_required" in check(raw)


def test_decoding_draft_unknown_model():
    draft = {"model": "no-such-model"}
    raw = spec(decoding={"mode": "speculative", "draft": draft, "num_speculative_tokens": 4})
    assert "decoding.draft_unknown_model" in check(raw)


def test_decoding_vocab_mismatch_and_unknown_shape():
    odd = CATALOG.models["qwen3-0.6b"].model_dump()
    odd.update(checkpoint="Qwen/Qwen3-Odd-Vocab", code="qwen3-odd-vocab")
    noshape = {**odd, "checkpoint": "Qwen/Qwen3-No-Shape", "code": "qwen3-no-shape", "shape": None}
    odd["shape"] = {**odd["shape"], "vocab": 151937}
    models = {
        "qwen3-odd-vocab": CATALOG.models["qwen3-0.6b"].model_validate(odd),
        "qwen3-no-shape": CATALOG.models["qwen3-0.6b"].model_validate(noshape),
    }
    catalog = _catalog(models=models)
    for draft in models:
        raw = spec(decoding={"mode": "speculative", "draft": {"model": draft}, "num_speculative_tokens": 4})
        issues = validate(raw, catalog)
        assert codes(issues) == ["decoding.vocab_mismatch"], draft
    assert "cannot check" in validate(raw, catalog)[0].message


def test_decoding_draft_method_later_for_method_and_for_preprocess():
    rtn = {"name": "rtn", "preset": None, "config": {"rules": RTN_RULES}}
    method = {"model": "qwen3-0.6b", "method": rtn}
    preprocess = {"model": "qwen3-0.6b", "preprocess": {"preset": "smoothattention"}}
    for draft in (method, preprocess):
        raw = spec(decoding={"mode": "speculative", "draft": draft, "num_speculative_tokens": 4})
        assert "decoding.draft_method_later" in check(raw)


def test_decoding_sampling_unavailable_run_level_and_task_level():
    run_level = spec(generation={"do_sample": True})
    task_level = spec(tasks=[{"name": "lm_eval/gsm8k", "params": {"do_sample": True}}])
    for raw in (run_level, task_level):
        assert "decoding.sampling_unavailable" in check(raw)
        issue = next(i for i in validate(raw, _catalog()) if i.code == "decoding.sampling_unavailable")
        assert "speed tasks" in issue.message


def test_speculative_sampling_is_allowed_on_speed_tasks_and_the_reference_samples_alike():
    raw = speed_spec(generation={"do_sample": True})
    assert check(raw) == []
    plan, _ = frozen(raw)
    main, reference = role(plan, "main"), role(plan, "reference")
    assert main.spec.decoding.verification == "sampling"
    assert reference.spec.decoding is None
    assert reference.spec.generation == main.spec.generation and main.spec.generation.do_sample
    assert reference.spec.determinism.seed == main.spec.determinism.seed


def test_decoding_batch_size():
    codes_ = check(spec(generation={"batch_size": 4}))
    assert "decoding.batch_size" in codes_


def test_decoding_no_effect_with_a_loglikelihood_task():
    catalog = _catalog(field_modes=[Mode.LOGLIKELIHOOD, Mode.GENERATION, Mode.SPEED])
    assert "decoding.no_effect" in check(spec(tasks=["lm_eval/mmlu"]), catalog)


def test_speed_hardware_required():
    raw = speed_spec()
    del raw["hardware"]
    assert "speed.hardware_required" in check(raw)
    assert check(speed_spec()) == []


def test_speed_tasks_decode_one_sequence_at_a_time_even_without_speculation():
    raw = speed_spec(decoding=None, generation={"batch_size": 2})
    assert "decoding.batch_size" in check(raw)


def test_runtime_method_unsupported_for_hook_methods_and_preprocess_on_engines():
    target = speed_spec(
        runtime={"name": "vllm"},
        field="quantization",
        method={"name": "rtn", "preset": None, "config": {"rules": RTN_RULES}},
        decoding=None,
    )
    assert "runtime.method_unsupported" in check(target)
    draft = speed_spec(
        runtime={"name": "vllm"},
        decoding={
            "mode": "speculative",
            "draft": {"model": "qwen3-0.6b", "preprocess": {"preset": "smoothattention"}},
            "num_speculative_tokens": 4,
        },
    )
    assert "runtime.method_unsupported" in check(draft)
    assert check(speed_spec(runtime={"name": "vllm"})) == []


def test_runtime_attn_unsupported_only_when_set_explicitly():
    assert "runtime.attn_unsupported" in check(
        speed_spec(runtime={"name": "vllm", "attn_implementation": "eager"})
    )
    assert "runtime.attn_unsupported" not in check(speed_spec(runtime={"name": "sglang"}))


def test_runtime_task_unsupported_for_a_non_speed_task():
    raw = spec(runtime={"name": "vllm"})
    assert "runtime.task_unsupported" in check(raw)


def test_engine_runtime_needs_an_evaluator_with_engine_handles():
    catalog = _catalog(handles={HandleKind.IN_PROCESS})
    assert "handle.unsupported" in check(speed_spec(runtime={"name": "vllm"}), catalog)
    assert check(speed_spec(), catalog) == []


def test_the_method_field_check_applies_to_the_target_only():
    assert "method.field_mismatch" in check(
        spec(method={"name": "rtn", "preset": None, "config": {"rules": RTN_RULES}})
    )
    assert "method.field_mismatch" not in check(spec())


def test_decoding_is_rejected_by_the_schema_when_k_is_out_of_range():
    with pytest.raises(ValueError):
        RunSpec.model_validate(spec(decoding={"mode": "speculative", "num_speculative_tokens": 0}))
    assert "spec.invalid" in check(spec(decoding={"mode": "speculative", "num_speculative_tokens": 17}))


def test_reference_of_speculative_job_is_autoregressive_same_target():
    plan, _ = frozen(spec())
    main, reference = role(plan, "main"), role(plan, "reference")
    assert main.spec.decoding is not None and main.spec.decoding.draft.model.name == "qwen3-0.6b"
    assert main.spec.decoding.verification == "greedy" and main.spec.decoding.num_speculative_tokens == 4
    assert reference.spec.decoding is None
    assert main.reference_key == reference.run_key
    plain = role(frozen(spec(decoding=None))[0], "main")
    assert plain.run_key == reference.run_key and plain.spec == reference.spec


def test_a_plain_autoregressive_spec_of_the_field_has_no_reference():
    plan, _ = frozen(spec(decoding=None))
    assert [j.role for j in plan.jobs] == ["main"] and plan.jobs[0].reference_key is None


def test_distinct_drafts_and_k_give_distinct_keys():
    def main(draft, k):
        decoding = {"mode": "speculative", "draft": {"model": draft}, "num_speculative_tokens": k}
        return role(frozen(spec(decoding=decoding))[0], "main")

    jobs = [main("qwen3-0.6b", 4), main("qwen3-1.7b", 4), main("qwen3-0.6b", 3)]
    assert len({j.run_key for j in jobs}) == 3
    assert len({model_group_key(j) for j in jobs}) == 3
    assert len({j.reference_key for j in jobs}) == 1
    assert main("qwen3-0.6b", 4).run_key == jobs[0].run_key


def test_sweep_over_num_speculative_tokens():
    plan, _ = frozen({**spec(), "sweep": {"decoding.num_speculative_tokens": [2, 3, 5]}})
    mains = [j for j in plan.jobs if j.role == "main"]
    assert sorted(j.spec.decoding.num_speculative_tokens for j in mains) == [2, 3, 5]
    assert len({j.run_key for j in mains}) == 3
    assert len([j for j in plan.jobs if j.role == "reference"]) == 1


def test_engine_runtime_freezes_library():
    plan, _ = frozen(speed_spec(runtime={"name": "vllm"}))
    main, reference = role(plan, "main"), role(plan, "reference")
    assert main.spec.runtime_library == reference.spec.runtime_library == ENGINES["vllm"].library
    assert reference.spec.decoding is None and reference.spec.runtime.name == "vllm"
    hf = role(frozen(speed_spec())[0], "main")
    assert hf.spec.runtime_library is None
    assert hf.run_key != main.run_key and model_group_key(hf) != model_group_key(main)
    sglang = role(frozen(speed_spec(runtime={"name": "sglang"}))[0], "main")
    assert sglang.spec.runtime_library == ENGINES["sglang"].library
    assert sglang.run_key != main.run_key


def test_engine_run_key_depends_on_cuda_graphs_and_capacity():
    def keys(**runtime):
        plan, _ = frozen(speed_spec(runtime={"name": "vllm", **runtime}))
        main, reference = role(plan, "main"), role(plan, "reference")
        return main.run_key, reference.run_key, model_group_key(main), model_group_key(reference)

    base = keys()
    for changed in (keys(cuda_graphs=False), keys(static_cache_tokens=8192)):
        assert all(a != b for a, b in zip(base, changed, strict=True))


def test_sampling_is_accepted_on_an_engine_for_the_run_and_for_a_task():
    assert check(speed_spec(runtime={"name": "vllm"}, decoding=None, generation={"do_sample": True})) == []
    task = {"name": "speed/spec_bench", "params": {"do_sample": True}}
    raw = spec(tasks=[task], hardware={"gpu": "H100-SXM"}, runtime={"name": "sglang"}, decoding=None)
    assert check(raw) == []
    assert check(speed_spec(runtime={"name": "vllm"}, generation={"do_sample": True})) == []


def test_a_draft_dtype_different_from_the_target_is_an_error_on_engines_only():
    draft = {
        "mode": "speculative",
        "draft": {"model": {"name": "qwen3-0.6b", "dtype": "float16"}},
        "num_speculative_tokens": 4,
    }
    assert "runtime.draft_dtype" in check(speed_spec(runtime={"name": "sglang"}, decoding=draft))
    assert "runtime.draft_dtype" not in check(speed_spec(decoding=draft))
    assert "runtime.draft_dtype" not in check(speed_spec(runtime={"name": "sglang"}))


def test_an_engine_runtime_needs_exactly_one_gpu():
    two = {"gpu": "H100-SXM", "count": 2}
    raw = spec(tasks=["speed/spec_bench"], runtime={"name": "vllm"}, hardware=two)
    assert "runtime.single_gpu" in check(raw)
    assert "runtime.single_gpu" not in check(speed_spec(runtime={"name": "vllm"}))


def test_speed_jobs_key_on_the_whole_generation_and_on_the_gpu():
    base = role(frozen(speed_spec())[0], "main")
    longer = role(frozen(speed_spec(generation={"max_new_tokens": 64}))[0], "main")
    other_gpu = role(frozen({**speed_spec(), "hardware": {"gpu": "A100-SXM-80GB"}})[0], "main")
    assert len({base.run_key, longer.run_key, other_gpu.run_key}) == 3


def test_output_match_is_dropped_from_sampled_jobs():
    plan, issues = frozen(spec(decoding=None, generation={"do_sample": True}))
    assert "output_match" not in role(plan, "main").compare
    assert "compare.output_match_dropped" in codes(issues, "info")
    assert "multi_turn_prompt_match" not in role(plan, "main").compare
    assert "compare.multi_turn_prompt_match_dropped" in codes(issues, "info")
    greedy = role(frozen(spec(decoding=None))[0], "main")
    assert "output_match" in greedy.compare


def test_the_draft_model_entry_travels_with_the_plan_when_it_is_not_builtin():
    entry = CATALOG.models["qwen3-0.6b"].model_dump()
    entry.update(checkpoint="Me/Tiny-Draft", code="tiny-draft", revision="abc")
    draft = CATALOG.models["qwen3-0.6b"].model_validate(entry)
    catalog = _catalog(models={"tiny-draft": draft})
    decoding = {"mode": "speculative", "draft": {"model": "tiny-draft"}, "num_speculative_tokens": 4}
    raw = spec(decoding=decoding)
    plan, _ = frozen(raw, catalog)
    assert set(plan.models) == {"tiny-draft"}
    assert frozen(spec())[0].models == {}


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _key_pairs(raw):
    return {j.role: (j.run_key, _sha(model_group_key(j))) for j in frozen(raw)[0].jobs}


def test_greedy_keys_are_unchanged_by_the_sampling_fields():
    assert _key_pairs(speed_spec()) == {
        "reference": (
            "rk_2ab79118a5e781555aa903b62dfbdb4aaebdc95f25126e264c7b21ce4725448a",
            "91fec4b84b4592f93c4a245cc7746266fb45d3410712890a314dfc1c8ab9230d",
        ),
        "main": (
            "rk_1f7c3cdac08104f596a9981475fd64d54eab440d649ac3f697ce3926b3ce222b",
            "7504b5072c5112104152bd6cd249ae31f29b45f1c8bdc296c10dfaede790ba50",
        ),
    }
    assert _key_pairs(speed_spec(runtime={"name": "vllm"})) == {
        "reference": (
            "rk_8dabb56164775ed1254b77d74387a7a57d941d3d903f2cbd035ea3f7cc2ff46e",
            "b98dabced241af353e278787d29ead39e1a28f3bab82b2385c6c5035453ee4f2",
        ),
        "main": (
            "rk_240eae67befadbad181371cae0c4cb69afa1327ccf8a4d5d9057fdb84215571d",
            "fc1a4ce3c77ed36a34874ccff8a5a52829b573768b48798158aa0c9d2fb10dac",
        ),
    }
    assert _key_pairs(speed_spec(generation={"do_sample": True}, decoding=None)) == {
        "main": (
            "rk_6c7792bd031af62c74418fa360e6606193fa067e03e2082a32ba091b5f73579f",
            "91fec4b84b4592f93c4a245cc7746266fb45d3410712890a314dfc1c8ab9230d",
        ),
    }
    plain = role(frozen(speed_spec(generation={"do_sample": True}))[0], "main")
    filtered = role(frozen(speed_spec(generation={"do_sample": True, "top_p": 0.9}))[0], "main")
    assert plain.run_key != filtered.run_key


def test_top_p_top_k_rules():
    assert "generation.filter_needs_sampling" in check(speed_spec(generation={"top_p": 0.9}))
    assert "generation.filter_needs_sampling" in check(speed_spec(generation={"top_k": 5}))
    raw = spec(decoding=None, generation={"do_sample": True, "top_k": 5})
    assert "generation.filter_speed_only" in check(raw)
    for bad in ({"top_p": 1.0}, {"top_p": 0.0}, {"top_k": 0}):
        with pytest.raises(ValueError):
            RunSpec.model_validate(speed_spec(generation={"do_sample": True, **bad}))
        assert "spec.invalid" in check(speed_spec(generation={"do_sample": True, **bad}))
    assert check(speed_spec(generation={"do_sample": True, "top_p": 0.9, "top_k": 5})) == []


def test_a_task_that_does_not_sample_gets_no_top_p_or_top_k():
    other = SPEED_TASK.model_copy(update={"name": "speed/other"})
    catalog = _catalog()
    catalog = catalog.model_copy(update={"tasks": {**catalog.tasks, other.name: other}})
    off = {"name": "speed/spec_bench", "params": {"do_sample": False}}
    raw = spec(
        tasks=[off, "speed/other"],
        hardware={"gpu": "H100-SXM"},
        generation={"do_sample": True, "top_p": 0.9},
    )
    plan, _ = frozen(raw, catalog)
    mains = {j.spec.task.name: j.spec.generation for j in plan.jobs if j.role == "main"}
    assert mains["speed/spec_bench"].top_p is None and not mains["speed/spec_bench"].do_sample
    assert mains["speed/other"].top_p == 0.9


def test_repeats_seeds_give_one_reference_per_seed():
    plan, _ = frozen(speed_spec(generation={"do_sample": True}, repeats={"seeds": [3, 5]}))
    mains = [j for j in plan.jobs if j.role == "main"]
    references = {j.run_key: j for j in plan.jobs if j.role == "reference"}
    assert len(mains) == 2 and len(references) == 2
    assert sorted(j.spec.determinism.seed for j in mains) == [3, 5]
    for main in mains:
        reference = references[main.reference_key]
        assert reference.spec.determinism.seed == main.spec.determinism.seed


def test_sampled_engine_jobs_group_per_seed_and_settings():
    plan, _ = frozen(
        speed_spec(runtime={"name": "vllm"}, generation={"do_sample": True}, repeats={"seeds": [3, 5]})
    )
    assert len(plan.jobs) == 4
    assert len({model_group_key(j) for j in plan.jobs}) == 4
    other = frozen(
        speed_spec(
            runtime={"name": "vllm"}, generation={"do_sample": True, "top_p": 0.9}, repeats={"seeds": [3, 5]}
        )
    )[0]
    assert not {model_group_key(j) for j in plan.jobs} & {model_group_key(j) for j in other.jobs}
    greedy = _key_pairs(speed_spec(runtime={"name": "vllm"}))
    assert greedy["main"][1] == "fc1a4ce3c77ed36a34874ccff8a5a52829b573768b48798158aa0c9d2fb10dac"
    assert greedy["reference"][1] == "b98dabced241af353e278787d29ead39e1a28f3bab82b2385c6c5035453ee4f2"


def test_sglang_rejects_top_k_with_top_p_when_sampling():
    both = {"do_sample": True, "top_p": 0.9, "top_k": 5}
    assert "runtime.sampling_filters" in check(speed_spec(runtime={"name": "sglang"}, generation=both))
    for one in ({"top_p": 0.9}, {"top_k": 5}):
        assert "runtime.sampling_filters" not in check(
            speed_spec(runtime={"name": "sglang"}, generation={"do_sample": True, **one})
        )
    for runtime in ({"name": "vllm"}, {"name": "hf"}):
        assert "runtime.sampling_filters" not in check(speed_spec(runtime=runtime, generation=both))


def test_sampled_count_repeats_get_a_same_seed_note():
    def notes(**overrides):
        return codes(validate(speed_spec(**overrides), _catalog()), "info")

    sampled = {"do_sample": True}
    assert "repeats.sampled_same_seed" in notes(generation=sampled, repeats={"count": 3})
    assert "repeats.sampled_same_seed" not in notes(generation=sampled, repeats={"seeds": [1, 2, 3]})
    assert "repeats.sampled_same_seed" not in notes(generation=sampled, repeats={"count": 1})
    assert "repeats.sampled_same_seed" not in notes(repeats={"count": 3})
