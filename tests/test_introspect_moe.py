import json
from importlib import resources

import pytest
import yaml

from effeval.core.catalog import ArchitectureDef, MoELayout, builtin_catalog
from effeval.core.graph import GraphNode, LayerGraph, ModelShape, hadamard_size_ok, input_width, positions

ARCHITECTURES = resources.files("effeval.core.data.architectures")
QWEN3_30B_A3B = ModelShape(
    hidden=2048,
    intermediate=6144,
    num_layers=48,
    heads=32,
    kv_heads=4,
    head_dim=128,
    vocab=151936,
    tie_word_embeddings=False,
    num_experts=128,
    experts_per_token=8,
    moe_intermediate=768,
)


def canonical(doc) -> str:
    return json.dumps(doc, sort_keys=True)


def stored_doc(name: str) -> dict:
    return yaml.safe_load(ARCHITECTURES.joinpath(f"{name}.yaml").read_text(encoding="utf-8"))


def qwen3_30b_config():
    transformers = pytest.importorskip("transformers")
    return transformers.Qwen3MoeConfig(
        vocab_size=151936,
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_hidden_layers=48,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def tiny_moe_config(**overrides):
    transformers = pytest.importorskip("transformers")
    return transformers.Qwen3MoeConfig(
        **{
            "vocab_size": 97,
            "hidden_size": 64,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            **overrides,
        }
    )


def test_dense_layout_and_graph_serialize_exactly_as_the_stored_yaml():
    stored = stored_doc("Qwen3ForCausalLM")
    layout = builtin_catalog().architectures["Qwen3ForCausalLM"]
    dumped = layout.model_dump(mode="json")
    dumped["graph"] = layout.graph.model_dump(mode="json", exclude_none=True)
    assert canonical(dumped) == canonical(stored)
    assert "moe" not in dumped
    for node in layout.graph.nodes:
        assert "layers" not in node.model_dump() and "experts" not in node.model_dump()
    assert layout.site_names() == sorted(layout.site_names())
    assert not any(site.startswith(("moe.", "linear.mlp.experts")) for site in layout.site_names())


def test_dense_model_shapes_serialize_without_moe_keys():
    for entry in builtin_catalog().models.values():
        assert entry.shape is None or not (
            {"num_experts", "experts_per_token", "moe_intermediate", "shared_intermediate"}
            & entry.shape.model_dump().keys()
        )


def test_stored_moe_layout_round_trips():
    layout = builtin_catalog().architectures["Qwen3MoeForCausalLM"]
    assert layout.moe is not None and layout.moe.num_experts == 128
    assert layout.moe.layers == list(range(48))
    assert layout.model_dump(mode="json")["moe"]["router"] == "mlp.gate"
    assert ArchitectureDef.model_validate(layout.model_dump(mode="json")) == layout


def test_moe_site_names_add_expert_linears_and_router_sites():
    layout = builtin_catalog().architectures["Qwen3MoeForCausalLM"]
    names = set(layout.site_names())
    assert {f"moe.router.{t}" for t in ("weight", "input", "output")} <= names
    assert {
        f"linear.mlp.experts.{p}.{t}"
        for p in ("gate_proj", "up_proj", "down_proj")
        for t in ("weight", "input")
    } <= names
    assert not any(n.startswith("linear.mlp.router") or n.startswith("linear.mlp.gate") for n in names)


@pytest.mark.compute
def test_extraction_of_qwen3_30b_a3b_matches_the_stored_layout():
    from effeval.ops.introspect import extract_architecture

    live = extract_architecture(qwen3_30b_config(), source="test")
    dumped = live.model_dump(mode="json", exclude={"extracted_with"})
    dumped["graph"] = live.graph.model_dump(mode="json", exclude_none=True)
    stored = stored_doc("Qwen3MoeForCausalLM")
    del stored["extracted_with"]
    assert canonical(dumped) == canonical(stored)
    assert live.uncovered_weights == []


@pytest.mark.compute
def test_moe_graph_has_router_experts_and_route_nodes():
    from effeval.ops.introspect import extract_architecture

    graph = extract_architecture(tiny_moe_config(), source="test").graph
    kinds = {n.id: n.kind for n in graph.nodes}
    assert {k: kinds[k] for k in ("router", "expert_gate", "expert_up", "expert_down")} == dict.fromkeys(
        ("router", "expert_gate", "expert_up", "expert_down"), "linear"
    )
    assert kinds["route"] == "route" and kinds["expert_act"] == "act" and kinds["expert_mul"] == "mul"
    assert graph.node("router").module == "mlp.gate" and graph.node("router").canon == "mlp.router"
    assert graph.node("expert_down").module == "mlp.experts.*.down_proj"
    assert all(
        graph.node(n).experts for n in ("expert_gate", "expert_up", "expert_down", "expert_act", "expert_mul")
    )
    assert all(n.layers is None for n in graph.nodes)
    edges = set(graph.edges)
    assert ("post_attention_layernorm", "router", "linear.mlp.router.input") in edges
    assert ("expert_down", "route", None) in edges and ("router", "route", None) in edges
    assert ("route", "add2", None) in edges
    assert ("post_attention_layernorm", "add2", None) not in edges
    assert [src for src, dst, _ in graph.edges if dst == "route"] == ["router", "expert_down"]


@pytest.mark.compute
def test_attention_half_of_the_moe_graph_equals_the_dense_graph():
    from effeval.ops.introspect import extract_architecture

    moe = extract_architecture(tiny_moe_config(), source="test").graph
    dense = LayerGraph.model_validate(
        yaml.safe_load((ARCHITECTURES.joinpath("Qwen3ForCausalLM.yaml")).read_text(encoding="utf-8"))["graph"]
    )
    attn = {
        "x_in",
        "input_layernorm",
        "q",
        "q_norm",
        "k",
        "k_norm",
        "v",
        "rope",
        "qk",
        "softmax",
        "pv",
        "o",
        "add1",
    }
    assert [n for n in moe.nodes if n.id in attn] == [n for n in dense.nodes if n.id in attn]
    assert [e for e in moe.edges if e[0] in attn and e[1] in attn] == [
        e for e in dense.edges if e[0] in attn and e[1] in attn
    ]


@pytest.mark.compute
def test_dense_blocks_in_a_moe_model_are_refused():
    from effeval.ops.introspect import extract_architecture

    with pytest.raises(ValueError, match="dense layers"):
        extract_architecture(tiny_moe_config(mlp_only_layers=[0]), source="test")


@pytest.mark.compute
def test_moe_layout_lists_only_the_layers_with_a_block():
    from effeval.ops.introspect import _layers_path, _moe_layout, meta_model

    config = tiny_moe_config(num_hidden_layers=3, mlp_only_layers=[0])
    model = meta_model(config)
    layout = _moe_layout(model.get_submodule(_layers_path(model, config)))
    assert layout.layers == [1, 2]


@pytest.mark.compute
def test_model_shape_reads_the_moe_dimensions():
    from effeval.ops.introspect import model_shape

    shape = model_shape(qwen3_30b_config())
    assert (shape.num_experts, shape.experts_per_token, shape.moe_intermediate) == (128, 8, 768)
    assert shape.shared_intermediate is None and shape.intermediate == 6144


@pytest.mark.compute
def test_shared_expert_chain_feeds_the_route_node():
    from effeval.ops.introspect import _append_moe

    nodes = {
        "x_in": GraphNode(id="x_in", kind="residual"),
        "norm": GraphNode(
            id="norm", kind="norm", module="post_attention_layernorm", gain=True, axis="hidden"
        ),
        "add2": GraphNode(id="add2", kind="add"),
    }
    edges = [("x_in", "norm", None), ("norm", "add2", None), ("x_in", "add2", None)]
    moe = MoELayout(
        block="mlp",
        router="mlp.gate",
        experts="mlp.experts",
        expert_linears=["gate_proj", "up_proj", "down_proj"],
        shared="mlp.shared_experts",
        num_experts=4,
        layers=[0],
    )
    linears = {
        f"mlp.shared_experts.{n}": f"mlp.shared_experts.{n}" for n in ("gate_proj", "up_proj", "down_proj")
    }
    _append_moe(nodes, edges, moe, linears, "silu")
    assert nodes["shared_gate"].module == "mlp.shared_experts.gate_proj" and not nodes["shared_gate"].experts
    assert ("shared_down", "route", None) in edges and ("expert_down", "route", None) in edges
    assert ("norm", "shared_up", "linear.mlp.shared_experts.up_proj.input") in edges
    assert ("norm", "add2", None) not in edges


@pytest.mark.compute
def test_moe_positions_and_expert_widths():
    from effeval.ops.introspect import extract_architecture

    graph = extract_architecture(tiny_moe_config(), source="test").graph
    table = positions(graph, QWEN3_30B_A3B)
    assert list(table) == [
        "residual",
        "norm.input_layernorm.out",
        "norm.post_attention_layernorm.out",
        "linear.attn.q_proj.input",
        "linear.attn.k_proj.input",
        "linear.attn.v_proj.input",
        "linear.attn.o_proj.input",
        "linear.mlp.router.input",
        "linear.mlp.experts.gate_proj.input",
        "linear.mlp.experts.up_proj.input",
        "linear.mlp.experts.down_proj.input",
        "attn.v_head",
        "attn.qk_pre_rope",
        "attn.qk_rope",
    ]
    assert table["residual"]["rotation"]["offline"] == "ok"
    down = table["linear.mlp.experts.down_proj.input"]
    assert down["rotation"]["online"] == "ok" and down["scaling"]["offline"] == "ok"
    assert "nonlinear product" in down["rotation"]["offline"]
    assert "sibling" in table["linear.mlp.router.input"]["scaling"]["offline"]
    assert input_width(graph, QWEN3_30B_A3B, graph.node("expert_down")) == 768
    assert input_width(graph, QWEN3_30B_A3B, graph.node("expert_gate")) == 2048
    assert input_width(graph, QWEN3_30B_A3B, graph.node("router")) == 2048
    assert input_width(graph, QWEN3_30B_A3B, graph.node("o")) == 32 * 128
    assert hadamard_size_ok(768)


@pytest.mark.compute
def test_a_moe_width_missing_from_the_shape_is_an_error():
    from effeval.ops.introspect import extract_architecture

    graph = extract_architecture(tiny_moe_config(), source="test").graph
    dense_shape = QWEN3_30B_A3B.model_copy(update={"moe_intermediate": None})
    with pytest.raises(ValueError, match="MoE width"):
        input_width(graph, dense_shape, graph.node("expert_down"))


def test_position_dim_uses_the_expert_width_for_the_expert_down_input():
    from effeval.core.preprocess import position_dim

    graph = builtin_catalog().architectures["Qwen3MoeForCausalLM"].graph
    assert position_dim(graph, QWEN3_30B_A3B, "linear.mlp.experts.down_proj.input") == 768
    assert position_dim(graph, QWEN3_30B_A3B, "linear.mlp.experts.gate_proj.input") == 2048
    assert position_dim(graph, QWEN3_30B_A3B, "linear.mlp.router.input") == 2048
    assert position_dim(graph, QWEN3_30B_A3B, "linear.attn.o_proj.input") == 32 * 128


@pytest.mark.compute
def test_unfused_expert_children_are_named_by_the_adapter_table():
    from effeval.core.moe import EXPERT_ADAPTERS
    from effeval.runtime.moe import FUSED_EXPERTS, unfuse_experts

    assert set(EXPERT_ADAPTERS) == FUSED_EXPERTS
    transformers = pytest.importorskip("transformers")
    model = transformers.Qwen3MoeForCausalLM(tiny_moe_config())
    block = model.model.layers[0].mlp
    adapter = EXPERT_ADAPTERS[type(block.experts).__name__]
    assert hasattr(block, adapter.router)
    assert all(hasattr(block.experts, name) for name in adapter.fused_params)
    unfuse_experts(model)
    assert [name for name, _ in block.experts.experts[0].named_children() if name != "act_fn"] == list(
        adapter.linears
    )
