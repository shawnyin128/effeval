import pytest

from effeval.core.fragments import Calibration
from effeval.core.preprocess import (
    NO_PAPER,
    SOURCE_POSITIONS,
    Hadamard,
    Preprocess,
    RandomHadamard,
    RandomOrthogonal,
    SmoothAttentionScaling,
    rotation_option_problem,
    source_position_problem,
)


def test_smoothattention_defaults_to_per_head_scales_with_alpha_one_half():
    source = SmoothAttentionScaling()
    assert (source.alpha, source.granularity, source.act_stat) == (0.5, "per_head", "absmax")


def test_smoothattention_parses_as_a_scaling_source_and_rejects_a_bad_granularity():
    step = {
        "kind": "scaling",
        "entries": [
            {"position": "attn.qk_pre_rope", "mode": "offline", "source": {"type": "smoothattention"}}
        ],
    }
    assert Preprocess.model_validate({"steps": [step]}).steps[0].entries[0].source.type == "smoothattention"
    step["entries"][0]["source"]["granularity"] = "per_layer"
    with pytest.raises(ValueError, match="granularity"):
        Preprocess.model_validate({"steps": [step]})


@pytest.mark.parametrize(
    ("source", "position"),
    [
        ("smoothattention", "attn.qk_pre_rope"),
        ("smoothattention", "attn.qk_rope"),
        ("smoothattention", "attn.v_head"),
        ("smoothquant", "norm.input_layernorm.out"),
        ("awq", "linear.mlp.down_proj.input"),
    ],
)
def test_a_source_at_a_position_it_supports_has_no_problem(source, position):
    assert source_position_problem("scaling", source, position) is None


def test_a_source_not_listed_fits_any_position():
    assert source_position_problem("scaling", "trained", "attn.v_head") is None
    assert source_position_problem("rotation", "hadamard", "residual") is None


def test_smoothattention_off_the_kv_cache_names_the_kv_positions():
    message = source_position_problem("scaling", "smoothattention", "linear.mlp.down_proj.input")
    assert message == (
        "smoothattention scales the K/V cache: use attn.qk_pre_rope, attn.qk_rope or attn.v_head"
    )


def test_linear_input_sources_at_a_kv_position_point_to_smoothattention():
    message = source_position_problem("scaling", "awq", "attn.qk_rope")
    assert message == "awq scales a linear input; at attn.qk_rope use smoothattention"
    assert "residual" in source_position_problem("scaling", "smoothquant", "residual")


def test_no_paper_positions_are_supported_positions():
    for kind, by_source in NO_PAPER.items():
        for source, positions in by_source.items():
            assert set(positions) <= set(SOURCE_POSITIONS[kind][source])


@pytest.mark.parametrize("position", ["attn.qk_rope", "attn.v_head"])
def test_per_kv_head_is_accepted_with_a_seeded_source(position):
    options = {"per_kv_head": True}
    assert rotation_option_problem(position, RandomHadamard(), options) is None
    assert rotation_option_problem(position, RandomOrthogonal(), options) is None
    assert rotation_option_problem(position, Hadamard(), {}) is None
    assert rotation_option_problem(position, Hadamard(), {"per_kv_head": False}) is None


def test_per_kv_head_with_hadamard_is_deterministic_and_rejected():
    message = rotation_option_problem("attn.v_head", Hadamard(), {"per_kv_head": True})
    assert message == "hadamard is deterministic; per_kv_head needs random_hadamard or random_orthogonal"


def test_unknown_and_non_bool_options_are_rejected_at_kv_positions_only():
    assert "unknown options ['axis']" in rotation_option_problem("attn.qk_rope", Hadamard(), {"axis": 1})
    assert rotation_option_problem("attn.v_head", RandomHadamard(), {"per_kv_head": "yes"}) == (
        "per_kv_head must be true or false"
    )
    assert rotation_option_problem("residual", Hadamard(), {"untie_embeddings": True}) is None


def test_random_crop_sampling_needs_pile_val():
    assert Calibration(dataset="pile_val", sampling="random_crop", n_samples=1, seqlen=8, seed=0)
    with pytest.raises(ValueError, match="pile_val uses sampling"):
        Calibration(dataset="c4", sampling="random_crop", n_samples=1, seqlen=8, seed=0)
