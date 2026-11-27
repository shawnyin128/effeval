"""validate's WSVD training checks: the QAT basis and bit width against the job's preprocessing and rules."""

import copy

from catalogs import CATALOG
from effeval.core.preprocess import builtin_preprocess_presets
from effeval.validate import validate


def warnings_of(spec_extra, method_extra=None):
    method = {"name": "wsvd", "preset": "paper_w8a8", **(method_extra or {})}
    spec = {"model": "qwen3-0.6b", "method": method, "field": "quantization", **spec_extra}
    issues = validate(spec, CATALOG)
    assert [i for i in issues if i.severity.value == "error"] == []
    return [(i.code, i.paths) for i in issues if i.severity.value == "warning"]


def qat_issues(spec_extra, method_extra=None):
    method = {"name": "wsvd", "preset": "paper_w8a8", **(method_extra or {})}
    spec = {"model": "qwen3-0.6b", "method": method, "field": "quantization", **spec_extra}
    return [
        (i.code, i.severity.value, i.paths, i.message)
        for i in validate(spec, CATALOG)
        if i.code == "decomposition.qat_basis"
    ]


def preset_steps():
    return copy.deepcopy(builtin_preprocess_presets()["wsvd"].values["preprocess.steps"].value)


def with_steps(steps):
    return {"preprocess": {"steps": steps}}


def custom_preprocess(seed=0, value_rotation=True):
    entries = [
        {
            "position": "residual",
            "layers": "all",
            "mode": "offline",
            "source": {"type": "random_hadamard", "seed": seed},
            "options": {"untie_embeddings": True},
        }
    ]
    if value_rotation:
        entries.append(
            {"position": "attn.v_head", "layers": "all", "mode": "offline", "source": {"type": "hadamard"}}
        )
    return {"steps": [{"kind": "rotation", "entries": entries}]}


def test_the_preset_pair_has_no_warning():
    assert warnings_of({"preprocess": {"preset": "wsvd"}}) == []


def test_a_different_residual_seed_in_the_preprocess_is_an_error():
    got = qat_issues({"preprocess": custom_preprocess(seed=1)})
    assert [(c, sev, paths) for c, sev, paths, _ in got] == [
        ("decomposition.qat_basis", "error", ["method.config.qat.residual_rotation"])
    ]
    assert "seed" in got[0][3] and "1" in got[0][3]


def test_the_matching_custom_preprocess_has_no_basis_issue():
    assert qat_issues({"preprocess": custom_preprocess(seed=0)}) == []


def test_no_preprocess_is_an_error_for_the_basis_and_the_value_hadamard():
    got = qat_issues({})
    assert [(c, sev, paths) for c, sev, paths, _ in got] == [
        ("decomposition.qat_basis", "error", ["method.config.qat.residual_rotation"]),
        ("decomposition.qat_basis", "error", ["method.config.qat.value_hadamard"]),
    ]


def test_a_value_hadamard_without_the_value_rotation_is_an_error():
    got = qat_issues({"preprocess": custom_preprocess(value_rotation=False)})
    assert [(c, sev, paths) for c, sev, paths, _ in got] == [
        ("decomposition.qat_basis", "error", ["method.config.qat.value_hadamard"])
    ]


def test_the_matching_preset_pair_has_no_basis_issue():
    assert qat_issues({"preprocess": {"preset": "wsvd"}}) == []
    assert qat_issues({"preprocess": with_steps(preset_steps())}) == []


def test_a_second_residual_rotation_after_the_matching_one_is_an_error():
    steps = preset_steps()
    steps.append(
        {
            "kind": "rotation",
            "entries": [
                {
                    "position": "residual",
                    "layers": "all",
                    "mode": "offline",
                    "source": {"type": "random_orthogonal", "seed": 3},
                }
            ],
        }
    )
    got = qat_issues(with_steps(steps))
    assert [(c, sev, paths) for c, sev, paths, _ in got] == [
        ("decomposition.qat_basis", "error", ["method.config.qat.residual_rotation"])
    ]
    assert "random_orthogonal" in got[0][3]


def test_a_changed_value_rotation_source_is_an_error():
    steps = preset_steps()
    [entry] = [e for e in steps[0]["entries"] if e["position"] == "attn.v_head"]
    entry["source"] = {"type": "random_orthogonal", "seed": 0}
    got = qat_issues(with_steps(steps))
    assert [(c, sev, paths) for c, sev, paths, _ in got] == [
        ("decomposition.qat_basis", "error", ["method.config.qat.value_hadamard"])
    ]
    assert "random_orthogonal" in got[0][3]


def test_value_rotation_options_that_change_the_matrix_are_an_error():
    steps = preset_steps()
    [entry] = [e for e in steps[0]["entries"] if e["position"] == "attn.v_head"]
    entry["source"] = {"type": "random_hadamard", "seed": 0}
    entry["options"] = {"per_kv_head": True}
    got = qat_issues(with_steps(steps))
    assert [paths for _, _, paths, _ in got] == [["method.config.qat.value_hadamard"]]


def scaling_step(position):
    return {
        "kind": "scaling",
        "entries": [
            {"position": position, "layers": "all", "mode": "offline", "source": {"type": "smoothquant"}}
        ],
    }


def test_a_scaling_on_the_qkv_input_norm_is_an_error():
    steps = [scaling_step("norm.input_layernorm.out"), *preset_steps()]
    got = qat_issues(with_steps(steps))
    assert [(c, sev, paths) for c, sev, paths, _ in got] == [
        ("decomposition.qat_basis", "error", ["preprocess.steps"])
    ]
    assert "norm.input_layernorm.out" in got[0][3]


def test_a_scaling_after_the_rotations_on_a_qkv_position_is_an_error():
    steps = [*preset_steps(), scaling_step("attn.v_head")]
    got = qat_issues(with_steps(steps))
    assert [paths for _, _, paths, _ in got] == [["preprocess.steps"]]


def test_a_scaling_on_a_norm_that_does_not_feed_qkv_is_not_a_basis_issue():
    steps = [scaling_step("norm.post_attention_layernorm.out"), *preset_steps()]
    assert qat_issues(with_steps(steps)) == []


def test_an_unrotated_qat_without_preprocess_has_no_basis_warning():
    got = warnings_of({}, {"config": {"qat": {"residual_rotation": "none", "value_hadamard": False}}})
    assert got == []


def test_weight_rules_with_other_bits_than_the_qat_warn():
    w4 = {
        "sites": ["linear.*.weight"],
        "layers": "all",
        "quant": {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_channel"},
    }
    got = warnings_of({"preprocess": {"preset": "wsvd"}}, {"config": {"rules": [w4]}})
    assert got == [("decomposition.qat_bits", ["method.config.rules[0].quant.bits"])]
