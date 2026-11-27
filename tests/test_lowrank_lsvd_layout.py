"""LSVD layouts (per-head q/k/v, hybrid FFN), layer-restricted site resolution, budgets and validation."""

import pytest

torch = pytest.importorskip("torch")

from catalogs import CATALOG  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import QuantRule  # noqa: E402
from effeval.core.graph import positions  # noqa: E402
from effeval.core.lowrank import (  # noqa: E402
    ARCHITECTURE_ATTR,
    ARCHITECTURE_TRANSFORMS,
    LSVD,
    QKV_SHARED,
    factor_qkv_architecture,
    factor_qkv_per_head_architecture,
    ffn_branch_canons,
    hybrid_ffn_architecture,
    hybrid_ffn_sizes,
    lsvd_architecture,
    per_head_budget,
    transform_architecture,
)
from effeval.ops import sites  # noqa: E402
from effeval.validate import validate  # noqa: E402

pytestmark = pytest.mark.compute

DENSE = builtin_catalog().architectures["Qwen3ForCausalLM"]
INT8 = {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_channel"}


def test_per_head_linears_put_each_down_factor_before_its_projection():
    assert factor_qkv_per_head_architecture(DENSE).linears == {
        "self_attn.q_down": "attn.q_down",
        "self_attn.q_proj.up": "attn.q_proj",
        "self_attn.k_down": "attn.k_down",
        "self_attn.k_proj.up": "attn.k_proj",
        "self_attn.v_down": "attn.v_down",
        "self_attn.v_proj.up": "attn.v_proj",
        "self_attn.o_proj": "attn.o_proj",
        "mlp.gate_proj": "mlp.gate_proj",
        "mlp.up_proj": "mlp.up_proj",
        "mlp.down_proj": "mlp.down_proj",
    }


def test_per_head_graph_gives_each_projection_its_own_down_node():
    layout = factor_qkv_per_head_architecture(DENSE)
    graph = layout.graph
    for name in ("q", "k", "v"):
        node = graph.node(f"{name}_down")
        assert (node.canon, node.module) == (f"attn.{name}_down", f"self_attn.{name}_down")
        assert graph.node(name).module == f"self_attn.{name}_proj.up"
    assert [e for e in graph.edges if e not in DENSE.graph.edges] == [
        ("input_layernorm", "q_down", "linear.attn.q_down.input"),
        ("q_down", "q", "linear.attn.q_proj.input"),
        ("input_layernorm", "k_down", "linear.attn.k_down.input"),
        ("k_down", "k", "linear.attn.k_proj.input"),
        ("input_layernorm", "v_down", "linear.attn.v_down.input"),
        ("v_down", "v", "linear.attn.v_proj.input"),
    ]
    assert [e for e in DENSE.graph.edges if e not in graph.edges] == [
        ("input_layernorm", n, f"linear.attn.{n}_proj.input") for n in ("q", "k", "v")
    ]
    assert layout.graph_overrides == {}
    ids = [n.id for n in graph.nodes]
    assert all(ids.index(f"{n}_down") + 1 == ids.index(n) for n in ("q", "k", "v"))


def test_the_shared_down_layout_is_unchanged():
    shared = factor_qkv_architecture(DENSE)
    assert "self_attn.qkv_down" in shared.linears and "self_attn.q_down" not in shared.linears
    assert shared.graph.node("qkv_down").canon == "attn.qkv_down"


def test_per_head_layout_rejects_a_graph_without_one_shared_norm():
    llama_like = DENSE.model_copy(
        update={
            "graph_overrides": {
                "edges_remove": [["input_layernorm", "v", "linear.attn.v_proj.input"]],
                "edges_add": [["post_attention_layernorm", "v", "linear.attn.v_proj.input"]],
            }
        }
    )
    with pytest.raises(ValueError, match="one shared norm"):
        factor_qkv_per_head_architecture(llama_like)
    with pytest.raises(ValueError, match="no supported layer graph"):
        factor_qkv_per_head_architecture(DENSE.model_copy(update={"graph": None}))


def test_ffn_branch_canons_name_the_gate_and_up_branches():
    assert ffn_branch_canons("mlp.gate_proj") == ("mlp.gate_svd_down", "mlp.gate_svd_up")
    assert ffn_branch_canons("mlp.up_proj") == ("mlp.up_svd_down", "mlp.up_svd_up")
    with pytest.raises(ValueError, match="not one of"):
        ffn_branch_canons("mlp.down_proj")


@pytest.mark.parametrize("hybrid_layers", [[0, 1], None])
def test_hybrid_ffn_nodes_edges_and_layers(hybrid_layers):
    layout = hybrid_ffn_architecture(DENSE, hybrid_layers)
    graph = layout.graph
    assert layout.linears == {
        "self_attn.q_proj": "attn.q_proj",
        "self_attn.k_proj": "attn.k_proj",
        "self_attn.v_proj": "attn.v_proj",
        "self_attn.o_proj": "attn.o_proj",
        "mlp.gate_proj.dense": "mlp.gate_proj",
        "mlp.gate_proj.svd_down": "mlp.gate_svd_down",
        "mlp.gate_proj.svd_up": "mlp.gate_svd_up",
        "mlp.up_proj.dense": "mlp.up_proj",
        "mlp.up_proj.svd_down": "mlp.up_svd_down",
        "mlp.up_proj.svd_up": "mlp.up_svd_up",
        "mlp.down_proj": "mlp.down_proj",
    }
    assert [graph.node(n).module for n in ("gate", "up")] == ["mlp.gate_proj.dense", "mlp.up_proj.dense"]
    assert graph.node("gate").layers is None and graph.node("down").layers is None
    for c in ("gate", "up"):
        down, up = graph.node(f"{c}_svd_down"), graph.node(f"{c}_svd_up")
        assert (down.canon, down.module) == (f"mlp.{c}_svd_down", f"mlp.{c}_proj.svd_down")
        assert (up.canon, up.module) == (f"mlp.{c}_svd_up", f"mlp.{c}_proj.svd_up")
        assert down.layers == hybrid_layers and up.layers == hybrid_layers
    assert [e for e in graph.edges if e not in DENSE.graph.edges] == [
        ("post_attention_layernorm", "gate_svd_down", "linear.mlp.gate_svd_down.input"),
        ("gate_svd_down", "gate_svd_up", "linear.mlp.gate_svd_up.input"),
        ("gate_svd_up", "act", None),
        ("post_attention_layernorm", "up_svd_down", "linear.mlp.up_svd_down.input"),
        ("up_svd_down", "up_svd_up", "linear.mlp.up_svd_up.input"),
        ("up_svd_up", "mul", None),
    ]
    assert all(e in graph.edges for e in DENSE.graph.edges)


def test_hybrid_ffn_rejects_layouts_it_cannot_split():
    no_act = DENSE.model_copy(
        update={
            "graph_overrides": {"edges_remove": [["gate", "act", None]], "edges_add": [["gate", "mul", None]]}
        }
    )
    with pytest.raises(ValueError, match="feed exactly one 'act'"):
        hybrid_ffn_architecture(no_act, None)
    other_norm = DENSE.model_copy(
        update={
            "graph_overrides": {
                "edges_remove": [["post_attention_layernorm", "up", "linear.mlp.up_proj.input"]],
                "edges_add": [["input_layernorm", "up", "linear.mlp.up_proj.input"]],
            }
        }
    )
    with pytest.raises(ValueError, match="same norm"):
        hybrid_ffn_architecture(other_norm, None)
    with pytest.raises(ValueError, match="no supported layer graph"):
        hybrid_ffn_architecture(DENSE.model_copy(update={"graph": None}), None)


def test_lsvd_architecture_adds_the_ffn_branches_only_when_enabled():
    off = lsvd_architecture(DENSE, {"ffn": {"enabled": False}})
    assert off == factor_qkv_per_head_architecture(DENSE)
    assert not any("svd" in n.id for n in off.graph.nodes)
    on = lsvd_architecture(DENSE, {"ffn": {"enabled": True}})
    assert on == hybrid_ffn_architecture(factor_qkv_per_head_architecture(DENSE), None)
    assert {"gate_svd_down", "up_svd_up", "q_down", "v_down"} <= {n.id for n in on.graph.nodes}


def test_transform_registry_dispatches_lsvd_with_the_config():
    assert set(ARCHITECTURE_TRANSFORMS) == {QKV_SHARED, LSVD}
    assert transform_architecture(DENSE, LSVD, {"ffn": {"enabled": True}}) == lsvd_architecture(
        DENSE, {"ffn": {"enabled": True}}
    )
    assert transform_architecture(DENSE, LSVD) == lsvd_architecture(DENSE, {})
    assert transform_architecture(DENSE, QKV_SHARED, {"ffn": {"enabled": True}}) == factor_qkv_architecture(
        DENSE
    )


@pytest.mark.parametrize(
    ("layers", "hidden", "head_dim", "heads", "expected"),
    [
        (28, 1024, 128, 16, 25472),
        (36, 2560, 128, 32, 70208),
        (28, 1024, 128, 8, 12736),
        (36, 2560, 128, 8, 17552),
    ],
)
def test_per_head_budget_matches_the_released_count(layers, hidden, head_dim, heads, expected):
    assert per_head_budget(layers, hidden, head_dim, heads, 0.5) == expected


@pytest.mark.parametrize(
    ("d_ffn", "hidden", "svd_ratio", "expected"),
    [(3072, 1024, 0.5, (1536, 552)), (9728, 2560, 0.5, (4864, 1509)), (3072, 1024, 0.2, (614, 345))],
)
def test_hybrid_ffn_sizes_match_the_released_rule(d_ffn, hidden, svd_ratio, expected):
    assert hybrid_ffn_sizes(d_ffn, hidden, svd_ratio, 0.9) == expected


def test_hybrid_ffn_sizes_clamp_to_one_dense_channel_and_the_rank_bounds():
    assert hybrid_ffn_sizes(8, 4, 1.0, 0.9)[0] == 7
    assert hybrid_ffn_sizes(8, 4, 0.0, 0.9)[0] == 1
    assert hybrid_ffn_sizes(8, 4, 0.5, 100.0) == (4, 4)
    assert hybrid_ffn_sizes(8, 4, 0.5, 0.0)[1] == 1


def _tiny_hybrid_model():
    """Tiny Qwen3 whose layer 0 has SVD branches on gate/up and layer 1 has the dense branch only."""
    from torch import nn

    model = build_tiny_qwen3()
    for index, layer in enumerate(model.model.layers):
        for name in ("gate_proj", "up_proj"):
            dense = getattr(layer.mlp, name)
            holder = nn.Module()
            holder.dense = dense
            if index == 0:
                holder.svd_down = nn.Linear(64, 4, bias=False)
                holder.svd_up = nn.Linear(4, 8, bias=False)
            setattr(layer.mlp, name, holder)
    setattr(model, ARCHITECTURE_ATTR, hybrid_ffn_architecture(DENSE, [0]))
    return model


def test_resolve_skips_layers_outside_a_nodes_layers():
    model = _tiny_hybrid_model()
    rules = [QuantRule(sites=["linear.mlp.gate_svd_down.weight"], layers="all", quant=INT8)]
    assignment, hits = sites.resolve(model, rules)
    assert [key for key in assignment] == [("linear.mlp.gate_svd_down.weight", 0)]
    assert hits[0].locations == 1 and hits[0].layers == [0]
    dense_hits = sites.resolve(
        model, [QuantRule(sites=["linear.mlp.gate_proj.weight"], layers="all", quant=INT8)]
    )[1]
    assert dense_hits[0].layers == [0, 1]
    with pytest.raises(sites.SiteError, match="matches nothing"):
        sites.resolve(model, [QuantRule(sites=["linear.mlp.gate_svd_down.weight"], layers="1", quant=INT8)])


def test_resolve_still_rejects_a_missing_module_in_an_unrestricted_node():
    model = _tiny_hybrid_model()
    del model.model.layers[1].mlp.gate_proj.dense
    with pytest.raises(AttributeError):
        sites.resolve(model, [QuantRule(sites=["linear.mlp.down_proj.weight"], layers="all", quant=INT8)])


def test_positions_treat_the_svd_down_like_a_norm_consumer_and_the_latents_as_unavailable():
    shape = CATALOG.models["qwen3-0.6b"].shape
    table = positions(lsvd_architecture(DENSE, {"ffn": {"enabled": True}}).graph, shape)
    dense = table["linear.mlp.gate_proj.input"]
    assert table["linear.mlp.gate_svd_down.input"] == dense
    assert dense["rotation"]["offline"] == "same as the norm output: rotate the residual instead"
    for site in ("linear.mlp.gate_svd_up.input", "linear.mlp.up_svd_up.input"):
        assert all("latent" in text for kinds in table[site].values() for text in kinds.values())
    for name in ("q", "k", "v"):
        assert table[f"linear.attn.{name}_down.input"]["rotation"]["online"] == "ok"
        assert all(
            "latent" in text
            for kinds in table[f"linear.attn.{name}_proj.input"].values()
            for text in kinds.values()
        )


def lsvd_manifest():
    """The RTN manifest renamed `lsvd_probe` with an `ffn` section and the `lsvd` transform."""
    rtn = CATALOG.manifests["rtn"]
    schema = {
        **rtn.config_schema,
        "properties": {**rtn.config_schema["properties"], "ffn": {"type": "object"}},
    }
    return rtn.model_copy(
        update={"name": "lsvd_probe", "config_schema": schema, "architecture_transform": LSVD}
    )


def lsvd_issues(rules, ffn=None, preprocess=None):
    config = {"rules": rules, **({} if ffn is None else {"ffn": ffn})}
    spec = {
        "model": "qwen3-0.6b",
        "method": {"name": "lsvd_probe", "preset": None, "config": config},
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
    }
    if preprocess is not None:
        spec["preprocess"] = preprocess
    return validate(spec, CATALOG.with_manifests([lsvd_manifest()]))


def errors_of(issues):
    return [i.code for i in issues if i.severity.value == "error"]


WEIGHT = {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_channel"}
ACT = {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token"}
RESIDUAL_ROTATION = lambda seed: {  # noqa: E731
    "steps": [
        {
            "kind": "rotation",
            "entries": [
                {
                    "position": "residual",
                    "mode": "offline",
                    "source": {"type": "random_hadamard", "seed": seed},
                    "options": {"untie_embeddings": True},
                }
            ],
        }
    ]
}


def test_validate_accepts_per_head_down_sites_and_ffn_branch_sites_only_when_enabled():
    ffn_on = {"enabled": True}
    rules = [
        {"sites": ["linear.attn.q_down.weight"], "quant": WEIGHT},
        {"sites": ["linear.mlp.gate_svd_up.input"], "quant": ACT},
    ]
    assert errors_of(lsvd_issues(rules, ffn_on)) == []
    off = lsvd_issues(rules, {"enabled": False})
    assert errors_of(off) == ["site.unknown"]
    (unknown,) = [i for i in off if i.code == "site.unknown"]
    assert unknown.paths == ["method.config.rules[1].sites[0]"]


def test_validate_reports_the_latent_reason_for_an_online_rotation_on_a_projection_input():
    steps = [
        {
            "kind": "rotation",
            "entries": [
                {"position": "linear.attn.k_proj.input", "mode": "online", "source": {"type": "hadamard"}}
            ],
        }
    ]
    issues = lsvd_issues(
        [{"sites": ["linear.*.weight"], "quant": WEIGHT}], {"enabled": False}, {"steps": steps}
    )
    errors = [i for i in issues if i.severity.value == "error"]
    assert [i.code for i in errors] == ["preprocess.mode_unavailable"]
    assert "the input is the latent of attn.k_down" in errors[0].message


def selection_basis(issues):
    return [i for i in issues if i.code == "decomposition.selection_basis"]


RULES = [{"sites": ["linear.attn.q_down.weight"], "quant": WEIGHT}]
HADAMARD_FFN = {"enabled": True, "selection_rotation": "random_hadamard", "selection_rotation_seed": 0}


def test_selection_basis_warning_fires_without_a_matching_residual_rotation():
    (issue,) = selection_basis(lsvd_issues(RULES, HADAMARD_FFN))
    assert issue.severity.value == "warning" and issue.paths == ["method.config.ffn.selection_rotation"]
    assert "random Hadamard with seed 0" in issue.message and "applies no residual rotation" in issue.message
    assert issue.hint == "set selection_rotation to none without a residual rotation, or match the seed"
    (other,) = selection_basis(lsvd_issues(RULES, HADAMARD_FFN, RESIDUAL_ROTATION(5)))
    assert "applies random_hadamard rotation" in other.message


def test_selection_basis_warning_is_silent_when_the_basis_matches_or_is_not_used():
    assert not selection_basis(lsvd_issues(RULES, HADAMARD_FFN, RESIDUAL_ROTATION(0)))
    assert not selection_basis(lsvd_issues(RULES, HADAMARD_FFN, {"preset": "qsvd"}))
    assert not selection_basis(lsvd_issues(RULES, {**HADAMARD_FFN, "selection_rotation": "none"}))
    assert not selection_basis(lsvd_issues(RULES, {**HADAMARD_FFN, "enabled": False}))
    seeded = {**HADAMARD_FFN, "selection_rotation_seed": 5}
    assert not selection_basis(lsvd_issues(RULES, seeded, RESIDUAL_ROTATION(5)))
    assert selection_basis(lsvd_issues(RULES, seeded, RESIDUAL_ROTATION(0)))
