"""Speculative decoding in the schema: RunSpec.decoding, runtime providers, the speed mode, keys and plans."""

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
    for name in ("speedup", "output_match"):
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


def test_speed_jobs_key_on_the_whole_generation_and_on_the_gpu():
    base = role(frozen(speed_spec())[0], "main")
    longer = role(frozen(speed_spec(generation={"max_new_tokens": 64}))[0], "main")
    other_gpu = role(frozen({**speed_spec(), "hardware": {"gpu": "A100-SXM-80GB"}})[0], "main")
    assert len({base.run_key, longer.run_key, other_gpu.run_key}) == 3


def test_output_match_is_dropped_from_sampled_jobs():
    plan, issues = frozen(spec(decoding=None, generation={"do_sample": True}))
    assert "output_match" not in role(plan, "main").compare
    assert "compare.output_match_dropped" in codes(issues, "info")
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
