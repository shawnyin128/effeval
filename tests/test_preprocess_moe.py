"""Rotation preprocessing over MoE layers: the residual rotation folds through router and experts."""

from __future__ import annotations

import pytest
import torch

from effeval.methods.preprocess import engine
from effeval.runtime import moe
from test_moe_experts import EXPERTS, INTER, LAYERS, build
from test_preprocess_quarot import RESIDUAL, apply_to, logits_of, make_preprocess

ATOL = 1e-5
DOWN_PROJ = {
    "position": "linear.mlp.experts.down_proj.input",
    "mode": "online",
    "source": {"type": "hadamard"},
}


def prepared_model():
    model = build("qwen3_moe", torch.float32)
    moe.unfuse_experts(model)
    generator = torch.Generator().manual_seed(7)
    for name, param in model.named_parameters():
        if name.endswith("layernorm.weight") or name == "model.norm.weight":
            param.data.copy_(1 + 0.5 * torch.rand(param.shape, generator=generator))
    return model


def net_of(model) -> engine.Net:
    return engine.make_context(model, None, {"checkpoint": "tiny"}).net


def test_net_modules_expand_experts_and_stream_producers_reach_through_route():
    model = prepared_model()
    net = net_of(model)
    down = net.linear_node("mlp.experts.down_proj")
    modules = net.modules(1, down)
    assert modules == [e.down_proj for e in model.model.layers[1].mlp.experts.experts]
    assert net.modules(0, net.linear_node("mlp.router")) == [model.model.layers[0].mlp.gate]
    assert {n.canon for n in net.stream_producers()} == {"attn.o_proj", "mlp.experts.down_proj"}
    consumers = {n.canon for _norm, linears in net.stream_consumers() for n in linears}
    assert {"mlp.router", "mlp.experts.gate_proj", "mlp.experts.up_proj"} <= consumers
    with pytest.raises(engine.EngineError, match=r"Net.modules"):
        net.module(0, down)


def test_residual_rotation_with_expert_down_proj_online_keeps_logits():
    model = prepared_model()
    reference = logits_of(model)
    router_before = [layer.mlp.gate.weight.detach().clone() for layer in model.model.layers]
    expert_before = model.model.layers[0].mlp.experts.experts[0].down_proj.weight.detach().clone()

    applied = apply_to(model, make_preprocess(RESIDUAL, DOWN_PROJ))

    for layer, before in zip(model.model.layers, router_before, strict=True):
        assert not torch.allclose(layer.mlp.gate.weight, before)
    assert not torch.allclose(model.model.layers[0].mlp.experts.experts[0].down_proj.weight, expert_before)
    for layer in model.model.layers:
        assert (layer.post_attention_layernorm.weight == 1).all()
    assert len(applied.online_ops.handles) == LAYERS * EXPERTS
    try:
        got = logits_of(model)
    finally:
        applied.online_ops.remove()
    assert (got - reference).abs().max().item() < ATOL


def test_online_down_proj_hook_runs_on_every_routed_expert_input():
    model = prepared_model()
    reference = logits_of(model)
    applied = apply_to(model, make_preprocess(DOWN_PROJ))
    widths: set[int] = set()
    probes = [
        expert.down_proj.register_forward_pre_hook(lambda _m, args: widths.add(args[0].shape[-1]))
        for layer in model.model.layers
        for expert in layer.mlp.experts.experts
    ]
    got = logits_of(model)
    for probe in probes:
        probe.remove()
    applied.online_ops.remove()
    assert widths == {INTER}
    assert (got - reference).abs().max().item() < ATOL
    assert len(applied.edits) == 1
    assert f"{LAYERS} layers" in applied.edits[0]
