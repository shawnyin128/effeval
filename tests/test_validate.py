import copy

import pytest

from catalogs import CATALOG
from effeval import validate as validate_module
from effeval.validate import resolve, validate

RTN_RULES = [
    {
        "sites": ["linear.*.weight"],
        "quant": {"type": "int", "bits": 4, "symmetric": False, "granularity": "per_channel"},
    }
]


def spec(**overrides):
    base = {
        "model": "qwen3-8b-base",
        "method": {"name": "rtn", "preset": None, "config": {"rules": copy.deepcopy(RTN_RULES)}},
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
    }
    base.update(overrides)
    return base


def codes(issues, severity="error"):
    return [i.code for i in issues if i.severity.value == severity]


def test_evalscope_task_with_loglikelihood_is_a_conflict():
    issues = validate(spec(tasks=[{"name": "evalscope/mmlu", "mode": "loglikelihood"}]), CATALOG)
    assert codes(issues) == ["mode.task_unsupported"]


def test_task_names_pick_the_evaluator_and_pin_its_library():
    res = resolve(
        spec(tasks=["lm_eval/mmlu", "lm_eval/gsm8k", "evalscope/gsm8k", "native/wikitext2_ppl"]), CATALOG
    )
    assert res.ok, res.issues
    assert [(t.task.name, t.mode.value, t.evaluator.name) for t in res.tasks] == [
        ("lm_eval/mmlu", "loglikelihood", "lm_eval"),
        ("lm_eval/gsm8k", "generation", "lm_eval"),
        ("evalscope/gsm8k", "generation", "evalscope"),
        ("native/wikitext2_ppl", "loglikelihood", "native"),
    ]
    assert (res.tasks[0].evaluator.library, res.tasks[0].evaluator.library_version) == ("lm-eval", "0.4.13")
    assert res.tasks[3].evaluator.library is None


def test_unknown_task_hint_lists_at_most_20_close_names():
    issues = validate(spec(tasks=["lm_eval/nope"]), CATALOG)
    assert codes(issues) == ["task.unknown"]
    assert len(issues[0].hint.split(", ")) <= 20


def test_num_fewshot_is_a_generic_task_setting():
    def effective(params):
        res = resolve(spec(tasks=[{"name": "lm_eval/gsm8k", "params": params}]), CATALOG)
        assert res.ok, res.issues
        return res.tasks[0].num_fewshot

    assert effective({}) == 5
    assert effective({"num_fewshot": 3}) == 3
    assert effective({"num_fewshot": 5}) == 5


def test_a_fixed_fewshot_task_rejects_overrides():
    issues = validate(spec(tasks=[{"name": "lm_eval/aime24", "params": {"num_fewshot": 2}}]), CATALOG)
    assert codes(issues) == ["task.params_invalid"]
    assert issues[0].paths == ["tasks[0].params.num_fewshot"]


def test_passthrough_is_keyed_by_a_known_evaluator():
    issues = validate(spec(evaluator={"passthrough": {"nope": {}}}), CATALOG)
    assert codes(issues) == ["evaluator.unknown"]
    assert issues[0].paths == ["evaluator.passthrough.nope"]
    assert (
        codes(validate(spec(evaluator={"passthrough": {"lm_eval": {"bootstrap_iters": 0}}}), CATALOG)) == []
    )


def test_task_mode_conflict():
    assert codes(validate(spec(tasks=[{"name": "lm_eval/gsm8k", "mode": "loglikelihood"}]), CATALOG)) == [
        "mode.task_unsupported"
    ]


def test_generation_params_are_ineffective_when_every_task_is_loglikelihood():
    issues = validate(spec(generation={"max_new_tokens": 64, "batch_size": 4}), CATALOG)
    assert codes(issues) == ["mode.ineffective_param"]
    assert [i.paths for i in issues if i.code == "mode.ineffective_param"] == [["generation.max_new_tokens"]]
    assert (
        codes(
            validate(
                spec(tasks=["lm_eval/mmlu", "lm_eval/gsm8k"], generation={"max_new_tokens": 64}), CATALOG
            )
        )
        == []
    )


def test_the_quantization_field_accepts_sampling_at_run_and_task_level():
    assert codes(validate(spec(tasks=["lm_eval/gsm8k"], generation={"do_sample": True}), CATALOG)) == []
    sampled = {"name": "lm_eval/gsm8k", "params": {"do_sample": True, "temperature": 0.3}}
    assert codes(validate(spec(tasks=[sampled]), CATALOG)) == []


def test_a_field_generation_pin_still_conflicts_with_a_different_explicit_value():
    catalog = CATALOG.model_copy(deep=True)
    catalog.fields["quantization"].generation["do_sample"] = False
    assert codes(validate(spec(tasks=["lm_eval/gsm8k"], generation={"do_sample": True}), catalog)) == [
        "field.generation_conflict"
    ]


def test_gptq_default_preset_fills_paper_setup_with_provenance():
    res = resolve({"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}, CATALOG)
    assert res.ok, res.issues
    assert (
        res.spec.model.dtype is None and res.model.dtype == "bfloat16"
    )  # native dtype, as torch_dtype='auto'
    assert res.model.checkpoint == "Qwen/Qwen3-0.6B-Base" and res.model.architecture == "Qwen3ForCausalLM"
    assert [t.task.name for t in res.tasks] == ["native/wikitext2_ppl"]
    assert res.tasks[0].evaluator.name == "native"
    assert res.spec.method.config["calibration"] == {
        "dataset": "c4",
        "n_samples": 128,
        "seqlen": 2048,
        "seed": 0,
    }
    assert res.fidelity.preset == "gptq/paper"
    assert res.fidelity.deviations == [] and res.fidelity.assumed == []
    assert "method.config.percdamp" in res.fidelity.confirmed


def test_gptq_explicit_value_differing_from_paper_is_a_deviation():
    res = resolve(
        {
            "model": {"name": "qwen3-0.6b-base", "dtype": "float16"},
            "method": {"name": "gptq", "config": {"act_order": True}},
            "field": "quantization",
        },
        CATALOG,
    )
    assert res.ok, res.issues
    assert {d.path for d in res.fidelity.deviations} == {
        "method.config.act_order"
    }  # dtype is not a paper setting
    assert res.model.dtype == "float16"
    assert "method.config.act_order" not in res.fidelity.confirmed


def test_disabling_the_preset_requires_every_paper_setting():
    issues = validate(
        {
            "model": "qwen3-0.6b-base",
            "method": {"name": "gptq", "preset": None},
            "field": "quantization",
            "tasks": ["native/wikitext2_ppl"],
        },
        CATALOG,
    )
    missing = {i.paths[0] for i in issues if i.code == "preset.required"}
    assert missing == {
        f"method.config.{k}"
        for k in ["rules", "act_order", "static_groups", "percdamp", "blocksize", "calibration"]
    }


def test_unknown_preset_and_unknown_site():
    assert codes(validate(spec(method={"name": "gptq", "preset": "nope"}), CATALOG)) == ["preset.unknown"]
    rules = copy.deepcopy(RTN_RULES)
    rules[0]["sites"] = ["linear.*.wieght"]
    assert codes(
        validate(spec(method={"name": "rtn", "preset": None, "config": {"rules": rules}}), CATALOG)
    ) == ["site.unknown"]


def test_fragment_semantics_are_checked_server_side():
    rules = copy.deepcopy(RTN_RULES)
    rules[0]["quant"]["granularity"] = "per_group"  # missing group_size: JSON Schema cannot express this rule
    issues = validate(spec(method={"name": "rtn", "preset": None, "config": {"rules": rules}}), CATALOG)
    assert codes(issues) == ["method.config_invalid"]
    assert "group_size" in issues[0].message


def test_explicit_compare_needs_artifacts_but_field_defaults_are_skipped_with_a_note():
    explicit = validate(spec(tasks=["lm_eval/gsm8k"], compare=["kl"]), CATALOG)
    assert codes(explicit) == ["metric.missing_artifact"]
    default = validate(spec(tasks=["lm_eval/gsm8k"]), CATALOG)
    assert codes(default) == []
    assert "metric.skipped" in codes(default, "info")


def test_sweep_validates_every_combination():
    raw = spec(sweep={"tasks": [["lm_eval/mmlu"], [{"name": "lm_eval/gsm8k", "mode": "loglikelihood"}]]})
    issues = validate(raw, CATALOG)
    errors = [i for i in issues if i.severity.value == "error"]
    assert len(errors) == 1 and errors[0].message.startswith("[sweep #1]")


def test_schema_errors_are_issues_not_exceptions():
    issues = validate(
        {"model": "qwen3-0.6b-base", "method": {"name": "rtn"}, "field": "quantization", "bogus": 1}, CATALOG
    )
    assert "spec.invalid" in codes(issues)


@pytest.mark.parametrize("name", ["unknown_method"])
def test_unknown_method(name):
    assert codes(validate(spec(method={"name": name}), CATALOG)) == ["method.unknown"]


def test_unknown_model_code_points_to_registration():
    issues = validate(spec(model="Qwen/Qwen3-8B-Base"), CATALOG)
    assert codes(issues) == ["model.unknown"]
    assert "effeval model add" in issues[0].hint


def test_tied_lm_head_weight_cannot_be_edited_but_untied_can():
    rules = [{"sites": ["lm_head.weight"], "quant": RTN_RULES[0]["quant"]}]
    method = {"name": "rtn", "preset": None, "config": {"rules": rules}}
    assert codes(validate(spec(model="qwen3-0.6b-base", method=method), CATALOG)) == ["site.tied_weights"]
    assert codes(validate(spec(model="qwen3-8b-base", method=method), CATALOG)) == []
    acts = [
        {
            "sites": ["lm_head.input"],
            "quant": {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token"},
        }
    ]
    assert (
        codes(validate(spec(model="qwen3-0.6b-base", method={**method, "config": {"rules": acts}}), CATALOG))
        == []
    )


def test_method_architecture_support_and_unpinned_models():
    from effeval.core.catalog import ModelEntry
    from effeval.core.manifest import MethodManifest, schema_hash

    schema = {"type": "object"}
    llama_only = MethodManifest(
        name="llama_only",
        version="1",
        field="quantization",
        runtime="hf",
        entry="x:y",
        architectures=["LlamaForCausalLM"],
        config_schema=schema,
        schema_hash=schema_hash(schema),
    )
    local = ModelEntry(
        code="my-ckpt",
        checkpoint="/data/My-Ckpt",
        revision=None,
        architecture="Qwen3ForCausalLM",
        dtype="bfloat16",
        params=10,
        tie_word_embeddings=False,
    )
    catalog = CATALOG.with_manifests([llama_only]).with_models([local])
    issues = validate(spec(model="my-ckpt", method={"name": "llama_only", "preset": None}), catalog)
    assert codes(issues) == ["method.architecture_unsupported"]
    assert "model.unpinned" in codes(issues, "warning")


POSITIONS = {
    "residual": {
        "rotation": {"offline": "ok", "online": "no: one matrix shared by every layer"},
        "scaling": {"offline": "no: RMSNorm is not invariant to per-channel scaling", "online": "no: same"},
    },
    "norm.input_layernorm.out": {
        "rotation": {
            "offline": "no: producer is a norm with an elementwise gain",
            "online": "no: use linear inputs",
        },
        "scaling": {"offline": "ok", "online": "ok"},
    },
    "linear.mlp.down_proj.input": {
        "rotation": {"offline": "ok", "online": "ok"},
        "scaling": {"offline": "ok", "online": "ok"},
    },
}
UNTIED = {"options": {"untie_embeddings": True}}


def rotation(position="residual", mode="offline", source=None, **entry):
    source = source or {"type": "hadamard"}
    return {"kind": "rotation", "entries": [{"position": position, "mode": mode, "source": source, **entry}]}


def scaling(position="norm.input_layernorm.out", mode="offline", source=None, calibration=None):
    step = {
        "kind": "scaling",
        "entries": [{"position": position, "mode": mode, "source": source or {"type": "smoothquant"}}],
    }
    return {**step, "calibration": calibration} if calibration else step


CALIBRATION = {"dataset": "wikitext2", "n_samples": 8, "seqlen": 64, "seed": 0}


def pre_spec(steps, **overrides):
    return spec(preprocess={"steps": steps}, **overrides)


@pytest.fixture
def positions(monkeypatch):
    monkeypatch.setattr(validate_module, "_position_table", lambda architecture, entry: POSITIONS)


def error_of(issues, code):
    (issue,) = [i for i in issues if i.severity.value == "error" and i.code == code]
    return issue


def test_a_valid_preprocess_has_no_issue(positions):
    steps = [rotation(**UNTIED), scaling(calibration=CALIBRATION)]
    assert codes(validate(pre_spec(steps, model="qwen3-0.6b-base"), CATALOG)) == []


def test_an_unavailable_mode_is_an_error_with_the_graph_reason_on_the_mode_path(positions):
    issue = error_of(
        validate(pre_spec([rotation("norm.input_layernorm.out")]), CATALOG), "preprocess.mode_unavailable"
    )
    assert issue.paths == ["preprocess.steps[0].entries[0].mode"]
    assert "producer is a norm with an elementwise gain" in issue.message


def test_an_unknown_position_lists_the_known_ones(positions):
    issue = error_of(validate(pre_spec([rotation("nowhere")]), CATALOG), "preprocess.position_unknown")
    assert issue.paths == ["preprocess.steps[0].entries[0].position"] and "residual" in issue.hint


def test_scaling_into_a_norm_gain_before_the_residual_rotation_is_a_wrong_order(positions):
    steps = [scaling(calibration=CALIBRATION), rotation(**UNTIED)]
    issue = error_of(validate(pre_spec(steps, model="qwen3-0.6b-base"), CATALOG), "preprocess.order")
    assert issue.paths == ["preprocess.steps"]
    assert "rotation must come first" in issue.message


def test_scaling_into_a_norm_gain_after_the_residual_rotation_is_accepted(positions):
    steps = [rotation(**UNTIED), scaling(calibration=CALIBRATION)]
    assert codes(validate(pre_spec(steps, model="qwen3-0.6b-base"), CATALOG)) == []


@pytest.mark.parametrize("first_is_scaling", [True, False])
@pytest.mark.parametrize(
    ("position", "mode"),
    [("norm.input_layernorm.out", "online"), ("linear.mlp.down_proj.input", "offline")],
)
def test_scaling_not_folded_into_a_norm_gain_has_no_order_constraint(
    positions, first_is_scaling, position, mode
):
    online_scaling = scaling(position, mode, calibration=CALIBRATION)
    steps = [online_scaling, rotation(**UNTIED)] if first_is_scaling else [rotation(**UNTIED), online_scaling]
    assert "preprocess.order" not in codes(validate(pre_spec(steps, model="qwen3-0.6b-base"), CATALOG))


def test_a_trained_rotation_is_rejected_off_the_offline_residual(positions):
    trained = {
        "type": "trained",
        "lr": 1e-4,
        "calibration": {"dataset": "d", "n_samples": 8, "batch_size": 2, "max_length": 8, "epochs": 1},
    }
    steps = [rotation("linear.mlp.down_proj.input", "online", source=trained)]
    res = resolve(pre_spec(steps), CATALOG)
    assert not res.ok
    assert "residual position, offline, only" in " ".join(i.message for i in res.issues)


def test_a_data_driven_source_needs_calibration(positions):
    issue = error_of(validate(pre_spec([scaling()]), CATALOG), "preprocess.calibration_required")
    assert issue.paths == ["preprocess.steps[0].calibration"]


def test_rotating_the_residual_of_a_tied_model_needs_untie_embeddings(positions):
    issues = validate(pre_spec([rotation()], model="qwen3-0.6b-base"), CATALOG)
    assert error_of(issues, "preprocess.tied_embeddings").paths == [
        "preprocess.steps[0].entries[0].options.untie_embeddings"
    ]
    assert codes(validate(pre_spec([rotation(**UNTIED)], model="qwen3-0.6b-base"), CATALOG)) == []


def test_a_global_position_only_takes_all_layers(positions):
    issue = error_of(validate(pre_spec([rotation(layers="0-3")]), CATALOG), "preprocess.layers_global")
    assert issue.paths == ["preprocess.steps[0].entries[0].layers"]


def test_preprocess_without_steps_or_with_an_unknown_preset_is_an_error(positions):
    assert codes(validate(spec(preprocess={"steps": []}), CATALOG)) == ["preprocess.empty"]
    assert codes(validate(spec(preprocess={"preset": "nope"}), CATALOG)) == ["preprocess.preset_unknown"]


def test_an_unknown_source_type_is_a_spec_error_on_its_path(positions):
    issues = validate(pre_spec([rotation(source={"type": "magic"})]), CATALOG)
    assert any(i.code == "spec.invalid" and i.paths[0].startswith("preprocess.steps") for i in issues)


def test_a_model_without_an_inspected_graph_cannot_be_preprocessed(monkeypatch):
    monkeypatch.setattr(validate_module, "_position_table", lambda architecture, entry: None)
    assert codes(validate(pre_spec([rotation()]), CATALOG)) == ["preprocess.architecture_unsupported"]


def test_preprocess_can_be_swept_as_a_whole_and_each_combination_is_checked(positions):
    bad, good = [rotation("norm.input_layernorm.out")], [rotation(**UNTIED)]
    sweep = {"preprocess": [{"steps": bad}, {"steps": good}]}
    issues = validate({**spec(model="qwen3-0.6b-base"), "sweep": sweep}, CATALOG)
    errors = [i for i in issues if i.severity.value == "error"]
    assert [i.code for i in errors] == ["preprocess.mode_unavailable"]
    assert errors[0].message.startswith("[sweep #0]")


SMOOTHATTENTION = {"type": "smoothattention"}
KV_MODEL = {"model": "qwen3-0.6b-base"}


def test_a_linear_input_source_is_rejected_at_a_kv_position():
    steps = [scaling("attn.v_head", source={"type": "smoothquant"}, calibration=CALIBRATION)]
    issue = error_of(validate(pre_spec(steps, **KV_MODEL), CATALOG), "preprocess.source_position")
    assert issue.paths == ["preprocess.steps[0].entries[0].source.type"]
    assert "use smoothattention" in issue.message


def test_smoothattention_is_rejected_at_a_norm_output():
    steps = [scaling("norm.input_layernorm.out", source=SMOOTHATTENTION, calibration=CALIBRATION)]
    issue = error_of(validate(pre_spec(steps, **KV_MODEL), CATALOG), "preprocess.source_position")
    assert "scales the K/V cache" in issue.message


@pytest.mark.parametrize("position", ["attn.qk_pre_rope", "attn.v_head"])
def test_a_valid_smoothattention_spec_has_no_error(position):
    steps = [
        scaling(position, source={**SMOOTHATTENTION, "granularity": "per_head"}, calibration=CALIBRATION)
    ]
    assert codes(validate(pre_spec(steps, **KV_MODEL), CATALOG)) == []


@pytest.mark.parametrize(
    ("position", "mode"), [("attn.v_head", "online"), ("linear.attn.o_proj.input", "online")]
)
def test_v_head_scaling_after_a_value_path_rotation_is_a_wrong_order(position, mode):
    steps = [
        rotation(position, mode), scaling("attn.v_head", source=SMOOTHATTENTION, calibration=CALIBRATION)
    ]
    issue = error_of(validate(pre_spec(steps, **KV_MODEL), CATALOG), "preprocess.order")
    assert issue.paths == ["preprocess.steps"] and issue.hint == "put the scaling step first"


def test_v_head_scaling_before_the_value_path_rotation_is_accepted():
    steps = [
        scaling("attn.v_head", source=SMOOTHATTENTION, calibration=CALIBRATION),
        rotation("attn.v_head", "online"),
    ]
    assert codes(validate(pre_spec(steps, **KV_MODEL), CATALOG)) == []


def test_per_kv_head_needs_a_seeded_rotation_source():
    options = {"options": {"per_kv_head": True}}
    bad = validate(pre_spec([rotation("attn.qk_rope", "online", **options)], **KV_MODEL), CATALOG)
    assert error_of(bad, "preprocess.option").paths == ["preprocess.steps[0].entries[0].options"]
    seeded = rotation("attn.qk_rope", "online", {"type": "random_hadamard"}, **options)
    assert codes(validate(pre_spec([seeded], **KV_MODEL), CATALOG)) == []


def test_the_smoothattention_preset_resolves_with_the_qserve_kv4_quantizer():
    raw = {
        **KV_MODEL,
        "preprocess": {"preset": "smoothattention"},
        "method": {"name": "rtn", "preset": "qserve_kv4"},
        "field": "quantization",
        "tasks": [{"name": "native/wikitext2_ppl"}],
    }
    assert codes(validate(raw, CATALOG)) == []


def quarot_spec(method):
    return spec(method=method, preprocess={"preset": "quarot"})


@pytest.mark.parametrize("name", ["gptq", "rtn"])
def test_a_paired_preprocess_with_the_implicit_method_preset_is_an_error_listing_the_pairs(name):
    issues = validate(quarot_spec({"name": name}), CATALOG)
    (issue,) = [i for i in issues if i.code == "preset.needs_choice"]
    assert issue.severity.value == "error" and issue.paths == ["method.preset"]
    assert "quarot_w4a16" in issue.message and "quarot_w4a4kv4" in issue.message


def test_an_explicit_paired_method_preset_passes():
    for preset in ("quarot_w4a16", "quarot_w4a4kv4"):
        issues = validate(quarot_spec({"name": "gptq", "preset": preset}), CATALOG)
        assert [i.code for i in issues if i.severity.value != "info"] == []


def test_an_unpaired_method_preset_is_only_a_warning():
    issues = validate(quarot_spec({"name": "gptq", "preset": "paper"}), CATALOG)
    assert codes(issues) == [] and codes(issues, "warning") == ["preset.unpaired"]


def test_a_null_method_preset_with_a_full_config_has_no_preset_issue():
    issues = validate(quarot_spec({"name": "rtn", "preset": None, "config": {"rules": RTN_RULES}}), CATALOG)
    assert not [i for i in issues if i.code.startswith("preset.")]


def test_the_implicit_method_preset_without_preprocess_is_unchanged():
    assert "preset.needs_choice" not in codes(validate(spec(method={"name": "gptq"}), CATALOG))


STEPS_PREPROCESS = {
    "steps": [
        {
            "kind": "rotation",
            "entries": [
                {"position": "residual", "mode": "offline", "source": {"type": "hadamard"}},
            ],
        }
    ]
}


def test_a_named_preprocess_without_a_pairing_for_the_method_still_needs_a_method_preset():
    issues = validate(spec(method={"name": "paroquant"}, preprocess={"preset": "quarot"}), CATALOG)
    (issue,) = [i for i in issues if i.code == "preset.needs_choice"]
    assert issue.severity.value == "error" and issue.paths == ["method.preset"]
    assert "paper_w4g128" in issue.hint


@pytest.mark.parametrize("name", ["gptq", "rtn", "paroquant"])
def test_a_steps_preprocess_with_the_implicit_method_preset_is_an_error_listing_the_presets(name):
    issues = validate(spec(method={"name": name}, preprocess=STEPS_PREPROCESS), CATALOG)
    (issue,) = [i for i in issues if i.code == "preset.needs_choice"]
    assert issue.severity.value == "error" and issue.paths == ["method.preset"]
    assert issue.hint and "set method.preset to one of [" in issue.hint


def test_a_steps_preprocess_with_an_explicit_or_null_method_preset_has_no_preset_issue():
    explicit = validate(
        spec(method={"name": "paroquant", "preset": "paper_w4g128"}, preprocess=STEPS_PREPROCESS), CATALOG
    )
    null = validate(spec(preprocess=STEPS_PREPROCESS), CATALOG)
    assert not [i for i in explicit + null if i.code.startswith("preset.")]


@pytest.mark.parametrize(
    "preprocess", [{"preset": "quarot"}, {"preset": "smoothquant_quarot"}, STEPS_PREPROCESS]
)
def test_paroquant_with_an_explicit_preset_and_a_preprocess_validates(preprocess):
    method = {"name": "paroquant", "preset": "paper_w4g128"}
    issues = validate(spec(method=method, preprocess=preprocess), CATALOG)
    assert codes(issues) == []


def test_a_model_too_big_for_the_gpus_asks_for_hardware_count():
    one = validate(spec(model="qwen3-30b-a3b"), CATALOG)
    assert "hardware.too_small" in codes(one)
    issue = next(i for i in one if i.code == "hardware.too_small")
    assert issue.paths == ["hardware.count"] and "hardware.count" in issue.hint
    two = validate(spec(model="qwen3-30b-a3b", hardware={"count": 2}), CATALOG)
    assert "hardware.too_small" not in codes(two)
    assert "hardware.too_small" not in codes(validate(spec(), CATALOG))


def test_hardware_count_is_between_one_and_four():
    assert codes(validate(spec(hardware={"count": 4}), CATALOG)) == []
    for count in (0, 5):
        issues = validate(spec(hardware={"count": count}), CATALOG)
        assert [(i.code, i.paths) for i in issues] == [("spec.invalid", ["hardware.count"])]


def test_hardware_count_does_not_change_a_quality_run_key():
    from effeval.plan import freeze

    def key(count):
        plan, _ = freeze(
            spec(tasks=["native/wikitext2_ppl"], hardware={"count": count}),
            CATALOG,
            owner=None,
            server=None,
            lookup=None,
        )
        return plan.jobs[0].run_key

    assert key(1) == key(2) == key(4)
