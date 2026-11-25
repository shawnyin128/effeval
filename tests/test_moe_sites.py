"""Site resolution over routed experts: (site, layer, expert) keys, the router family, routed-token hooks."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from effeval.core.fragments import IntFormat, QuantRule
from effeval.ops import online, sites
from effeval.runtime import moe
from test_moe_experts import EXPERTS, HIDDEN, INTER, LAYERS, TOP_K, build

INT4 = IntFormat(type="int", bits=4, symmetric=True, granularity="per_token")
INT4_W = IntFormat(type="int", bits=4, symmetric=True, granularity="per_channel")


def unfused_model(case: str = "qwen3_moe"):
    model = build(case, torch.float32)
    moe.unfuse_experts(model)
    return model


def rule(*patterns: str, quant=INT4, layers="all") -> QuantRule:
    return QuantRule(sites=list(patterns), layers=layers, quant=quant)


def test_weight_rule_hits_every_expert_linear_and_attention_but_not_the_router():
    model = unfused_model()
    assignment, hits = sites.resolve(model, [rule("linear.*.weight", quant=INT4_W)])
    keys = list(assignment)
    expert_keys = [k for k in keys if len(k) == 3]
    assert len(expert_keys) == LAYERS * EXPERTS * 3
    projs = ("gate_proj", "up_proj", "down_proj")
    assert {k[0] for k in expert_keys} == {f"linear.mlp.experts.{p}.weight" for p in projs}
    assert {k[2] for k in expert_keys} == set(range(EXPERTS))
    attention = [k for k in keys if len(k) == 2]
    assert len(attention) == LAYERS * 4
    assert all(k[0].startswith("linear.attn.") for k in attention)
    assert not any(k[0].startswith("moe.router") for k in keys)
    assert hits[0].locations == LAYERS * (EXPERTS * 3 + 4)
    for key in expert_keys:
        loc = assignment[key][1]
        assert loc.expert == key[2]
        assert loc.module is model.model.layers[key[1]].mlp.experts.experts[key[2]].get_submodule(
            key[0].split(".")[-2]
        )


def test_router_family_is_its_own_site_and_resolves_once_per_layer():
    model = unfused_model()
    assignment, hits = sites.resolve(model, [rule("moe.router.weight", quant=INT4_W)])
    assert sorted(assignment) == [("moe.router.weight", layer) for layer in range(LAYERS)]
    assert hits[0].locations == LAYERS
    for (_site, layer), (_i, loc, _fmt) in assignment.items():
        assert loc.module is model.model.layers[layer].mlp.gate


def test_router_weight_rule_quantizes_only_the_router():
    model = unfused_model()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    attachment = sites.attach(model, [rule("moe.router.weight", quant=INT4_W)])
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {f"model.layers.{layer}.mlp.gate.weight" for layer in range(LAYERS)}
    attachment.remove()


def test_router_output_site_is_rejected_because_it_is_a_tuple():
    model = unfused_model()
    with pytest.raises(sites.SiteError, match="tuple"):
        sites.attach(model, [rule("moe.router.output")])


def test_layer_selector_limits_expert_locations():
    model = unfused_model()
    down = rule("linear.mlp.experts.down_proj.weight", layers="1", quant=INT4_W)
    assignment, _ = sites.resolve(model, [down])
    assert sorted(assignment) == [("linear.mlp.experts.down_proj.weight", 1, e) for e in range(EXPERTS)]


def test_fused_experts_are_rejected_with_a_pointer_to_unfusing():
    model = build("qwen3_moe", torch.float32)
    with pytest.raises(sites.SiteError, match="unfuse"):
        sites.resolve(model, [rule("linear.*.weight", quant=INT4_W)])


def test_expert_input_hook_fires_only_for_routed_tokens():
    model = unfused_model()
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(5))
    seen: dict[tuple[int, int], int] = {}
    routed: dict[tuple[int, int], int] = {}
    handles = []
    for li, layer in enumerate(model.model.layers):
        for e, expert in enumerate(layer.mlp.experts.experts):
            handles.append(
                expert.gate_proj.register_forward_pre_hook(
                    lambda _m, args, key=(li, e): seen.__setitem__(key, seen.get(key, 0) + args[0].shape[0])
                )
            )

        def count_routes(_m, _args, out, li=li):
            for e in out[-1].flatten().tolist():
                routed[(li, e)] = routed.get((li, e), 0) + 1

        handles.append(layer.mlp.gate.register_forward_hook(count_routes))
    with torch.no_grad():
        model(input_ids=ids)
    for h in handles:
        h.remove()
    assert sum(seen.values()) == LAYERS * ids.numel() * TOP_K
    assert seen == routed

    # with a quantizer attached at the expert inputs the hook still sees only routed tokens
    fired: dict[tuple[int, int], int] = {}
    attachment = sites.attach(model, [rule("linear.mlp.experts.gate_proj.input")])
    assert len(attachment.handles) == LAYERS * EXPERTS
    probes = [
        expert.gate_proj.register_forward_pre_hook(
            lambda _m, args, key=(li, e): fired.__setitem__(key, fired.get(key, 0) + args[0].shape[0])
        )
        for li, layer in enumerate(model.model.layers)
        for e, expert in enumerate(layer.mlp.experts.experts)
    ]
    with torch.no_grad():
        model(input_ids=ids)
    for h in probes:
        h.remove()
    attachment.remove()
    assert fired == routed


def test_online_op_attaches_to_each_expert_down_proj():
    model = unfused_model()
    ops = [online.OnlineOp("linear.mlp.experts.down_proj.input", lambda x: x * 2.0, "all")]
    attachment = online.register(model, ops)
    assert len(attachment.handles) == LAYERS * EXPERTS
    attachment.remove()


def test_dense_model_keys_stay_two_tuples():
    from conftest import build_tiny_qwen3

    model = build_tiny_qwen3()
    assignment, _ = sites.resolve(model, [rule("linear.*.weight", quant=INT4_W)])
    assert all(len(k) == 2 for k in assignment)


def test_path_absent_in_a_layer_is_skipped_in_a_moe_model():
    config = Qwen3MoeConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=96, moe_intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K, decoder_sparse_step=1, mlp_only_layers=[0],
        max_position_embeddings=64, tie_word_embeddings=False,
    )  # fmt: skip
    config._experts_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).eval()
    moe.unfuse_experts(model)
    assignment, _ = sites.resolve(model, [rule("linear.*.weight", quant=INT4_W), rule("moe.router.weight")])
    assert {k[1] for k in assignment if len(k) == 3} == {1}
    assert [k for k in assignment if k[0] == "moe.router.weight"] == [("moe.router.weight", 1)]
