from pathlib import Path

import pytest
import yaml

from effeval.core.catalog import ArchitectureDef, builtin_catalog
from effeval.core.graph import (
    LayerGraph,
    ModelShape,
    attention_output_position,
    hadamard_size_ok,
    merge_overrides,
    positions,
)

FIXTURE = Path(__file__).parent / "data" / "qwen3_layer_graph.yaml"
QWEN3_0_6B = ModelShape(
    hidden=1024, intermediate=3072, num_layers=28, heads=16, kv_heads=8, head_dim=128, vocab=151936, tie_word_embeddings=True
)


def qwen3_graph() -> LayerGraph:
    return LayerGraph.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))


# position -> (rotation offline, rotation online, scaling offline, scaling online): "ok" or a reason fragment.
EXPECTED = {
    "residual": ("ok", "shared by the embedding", "not invariant", "not invariant"),
    "norm.input_layernorm.out": ("rotate the residual", "per-linear inputs", "ok", "ok"),
    "norm.post_attention_layernorm.out": ("rotate the residual", "per-linear inputs", "ok", "ok"),
    "linear.attn.q_proj.input": ("same as the norm output", "ok", "shared with sibling linears", "ok"),
    "linear.attn.k_proj.input": ("same as the norm output", "ok", "shared with sibling linears", "ok"),
    "linear.attn.v_proj.input": ("same as the norm output", "ok", "shared with sibling linears", "ok"),
    "linear.attn.o_proj.input": ("across-head rotation", "ok", "on GQA", "ok"),
    "linear.mlp.gate_proj.input": ("same as the norm output", "ok", "shared with sibling linears", "ok"),
    "linear.mlp.up_proj.input": ("same as the norm output", "ok", "shared with sibling linears", "ok"),
    "linear.mlp.down_proj.input": ("silu(gate) * up is a nonlinear product", "ok", "ok", "ok"),
    "attn.v_head": ("ok", "ok", "ok", "fully foldable"),
    "attn.qk_pre_rope": ("RoPE channel pair", "undone before RoPE", "ok", "fully foldable"),
    "attn.qk_rope": ("q_norm/k_norm gains and RoPE sit between", "ok", "attn.qk_pre_rope", "ok"),
}
CELLS = [("rotation", "offline"), ("rotation", "online"), ("scaling", "offline"), ("scaling", "online")]


def check_cell(actual: str, expected: str) -> None:
    if expected == "ok":
        assert actual == "ok"
    else:
        assert actual != "ok" and expected in actual


def test_qwen3_positions_match_the_table_cell_by_cell():
    table = positions(qwen3_graph(), QWEN3_0_6B)
    assert list(table) == list(EXPECTED)
    for position, expected in EXPECTED.items():
        for (kind, mode), want in zip(CELLS, expected, strict=True):
            check_cell(table[position][kind][mode], want)


def test_mha_makes_o_proj_scaling_offline_ok():
    mha = QWEN3_0_6B.model_copy(update={"kv_heads": 16})
    assert positions(qwen3_graph(), mha)["linear.attn.o_proj.input"]["scaling"]["offline"] == "ok"
    assert positions(qwen3_graph(), QWEN3_0_6B)["linear.attn.o_proj.input"]["scaling"]["offline"] != "ok"


def test_layer_norm_blocks_residual_rotation():
    doc = qwen3_graph().model_dump()
    doc["globals"]["norm_kind"] = "layer"
    cell = positions(LayerGraph.model_validate(doc), QWEN3_0_6B)["residual"]["rotation"]["offline"]
    assert "LayerNorm" in cell


def test_a_norm_without_gain_cannot_take_a_folded_scale():
    graph = merge_overrides(qwen3_graph(), {"nodes": {"input_layernorm": {"gain": False}}})
    cell = positions(graph, QWEN3_0_6B)["norm.input_layernorm.out"]["scaling"]["offline"]
    assert "no gain" in cell


def test_a_head_norm_without_gain_cannot_take_a_pre_rope_scale():
    graph = merge_overrides(qwen3_graph(), {"nodes": {"k_norm": {"gain": False}}})
    cell = positions(graph, QWEN3_0_6B)["attn.qk_pre_rope"]["scaling"]["offline"]
    assert cell != "ok" and "k_norm" in cell


def test_a_graph_without_rope_has_no_qk_positions():
    doc = qwen3_graph().model_dump()
    doc["nodes"] = [n for n in doc["nodes"] if n["kind"] != "rope"]
    doc["edges"] = [e for e in doc["edges"] if "rope" not in e[:2]]
    table = positions(LayerGraph.model_validate(doc), QWEN3_0_6B)
    assert "attn.qk_pre_rope" not in table and "attn.qk_rope" not in table


def test_attention_output_position_is_the_o_proj_input():
    assert attention_output_position(qwen3_graph()) == "linear.attn.o_proj.input"
    assert attention_output_position(LayerGraph(unsupported=["moe"])) is None


def test_unsupported_graph_has_no_positions():
    assert positions(LayerGraph(unsupported=["moe"]), QWEN3_0_6B) == {}


def test_a_supported_graph_needs_globals_and_known_edge_nodes():
    with pytest.raises(ValueError, match="needs globals"):
        LayerGraph(nodes=[])
    doc = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="unknown node"):
        LayerGraph.model_validate({**doc, "edges": [["x_in", "nope", None]]})


@pytest.mark.parametrize(
    ("n", "ok"),
    [
        (1024, True),
        (3072, True),
        (2560, True),
        (5120, True),
        (11008, True),
        (12, True),
        (9728, False),
        (17408, False),
        (25600, False),
        (44, False),
        (0, False),
    ],
)
def test_hadamard_sizes(n, ok):
    assert hadamard_size_ok(n) is ok


def test_overrides_patch_nodes_and_edges():
    graph = merge_overrides(
        qwen3_graph(),
        {
            "nodes": {"act": {"fn": "gelu"}},
            "edges_remove": [["x_in", "add1", None]],
            "edges_add": [["x_in", "add2", None]],
        },
    )
    assert graph.node("act").fn == "gelu"
    assert ("x_in", "add1", None) not in graph.edges and ("x_in", "add2", None) in graph.edges
    with pytest.raises(ValueError, match="unknown node"):
        merge_overrides(qwen3_graph(), {"nodes": {"nope": {}}})
    with pytest.raises(ValueError, match="missing edge"):
        merge_overrides(qwen3_graph(), {"edges_remove": [["x_in", "down", None]]})


def test_architecture_applies_overrides_but_does_not_dump_them():
    stored = builtin_catalog().architectures["Qwen3ForCausalLM"]
    patched = ArchitectureDef.model_validate(
        {**stored.model_dump(), "graph_overrides": {"nodes": {"act": {"fn": "gelu"}}}}
    )
    assert patched.effective_graph().node("act").fn == "gelu"
    assert stored.effective_graph().node("act").fn == "silu"
    assert "graph_overrides" not in patched.model_dump()


def test_shipped_qwen3_graph_equals_the_fixture_and_models_carry_shapes():
    catalog = builtin_catalog()
    assert catalog.architectures["Qwen3ForCausalLM"].graph == qwen3_graph()
    assert catalog.models["qwen3-0.6b"].shape == QWEN3_0_6B
    assert all(m.shape is not None and m.shape.head_dim == 128 for m in catalog.models.values())


def test_shipped_models_carry_their_decoder_depth():
    depth = {
        "qwen3-0.6b": 28, "qwen3-1.7b": 28, "qwen3-4b": 36, "qwen3-8b": 36, "qwen3-14b": 40, "qwen3-32b": 64,
    }
    models = builtin_catalog().models
    assert {code: models[code].shape.num_layers for code in depth} == depth
    assert all(m.shape.num_layers == depth[m.code.removesuffix("-base")] for m in models.values())
