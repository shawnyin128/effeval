"""GPTQ over routed experts: groups from the graph, per-expert Hessians, RTN fallback for starved experts."""

from __future__ import annotations

import logging

import pytest
import torch

from effeval.core.catalog import builtin_catalog
from effeval.core.fragments import IntFormat
from effeval.methods.quantization.gptq import (
    GPTQConfig,
    HessianAccumulator,
    quantize_model,
    quantize_weight,
    sequential_groups,
)
from effeval.ops.quant.quant import fake_quantize
from effeval.runtime import moe
from test_moe_experts import EXPERTS, LAYERS, build

FMT = IntFormat(type="int", bits=4, symmetric=False, granularity="per_channel")
SEQLEN = 16


def config(*patterns: str, **overrides) -> GPTQConfig:
    base = {
        "rules": [{"sites": list(patterns or ["linear.*.weight"]), "quant": FMT.model_dump()}],
        "act_order": False,
        "static_groups": False,
        "percdamp": 0.01,
        "blocksize": 128,
        "calibration": {"dataset": "c4", "n_samples": 4, "seqlen": SEQLEN, "seed": 0},
    }
    base.update(overrides)
    return GPTQConfig.model_validate(base)


def windows(count: int = 4) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(3)
    return [torch.randint(0, 128, (1, SEQLEN), generator=gen) for _ in range(count)]


def moe_model():
    model = build("qwen3_moe", torch.float32)
    moe.unfuse_experts(model)
    return model


def never_route_to(model, expert: int, replacement: int) -> None:
    """Redirect the router's choice of `expert` to `replacement` so `expert` sees no token."""
    for layer in model.model.layers:
        layer.mlp.gate.register_forward_hook(
            lambda _m, _i, out: (out[0], out[1], torch.where(out[2] == expert, replacement, out[2]))
        )


def routed_counts(model, wins, layer: int) -> dict[int, int]:
    """Tokens each expert of `layer` receives over `wins`, from the router's indices."""
    counts: dict[int, int] = {}

    def count(_module, _inputs, out) -> None:
        for e in out[2].flatten().tolist():
            counts[e] = counts.get(e, 0) + 1

    handle = model.model.layers[layer].mlp.gate.register_forward_hook(count)
    with torch.no_grad():
        for w in wins:
            model(input_ids=w)
    handle.remove()
    return counts


def levels(weight: torch.Tensor) -> int:
    return max(row.unique().numel() for row in weight)


def test_dense_groups_follow_the_official_true_sequential_order():
    graph = builtin_catalog().architectures["Qwen3ForCausalLM"].effective_graph()
    assert [set(g) for g in sequential_groups(graph)] == [
        {"attn.k_proj", "attn.v_proj", "attn.q_proj"},
        {"attn.o_proj"},
        {"mlp.up_proj", "mlp.gate_proj"},
        {"mlp.down_proj"},
    ]


def test_moe_groups_put_router_with_gate_up_and_down_last():
    graph = builtin_catalog().architectures["Qwen3MoeForCausalLM"].effective_graph()
    assert [set(g) for g in sequential_groups(graph)] == [
        {"attn.k_proj", "attn.v_proj", "attn.q_proj"},
        {"attn.o_proj"},
        {"mlp.router", "mlp.experts.gate_proj", "mlp.experts.up_proj"},
        {"mlp.experts.down_proj"},
    ]


@pytest.mark.parametrize("true_sequential", [False, True])
def test_every_expert_lands_on_the_grid_and_the_router_is_left_alone(true_sequential):
    model = moe_model()
    router_before = [layer.mlp.gate.weight.clone() for layer in model.model.layers]
    expert_before = model.model.layers[0].mlp.experts.experts[1].gate_proj.weight.clone()
    quantize_model(model, windows(), config(true_sequential=true_sequential))
    for layer, before in zip(model.model.layers, router_before, strict=True):
        assert torch.equal(layer.mlp.gate.weight, before)
        assert levels(layer.self_attn.q_proj.weight) <= 16
        for expert in layer.mlp.experts.experts:
            for linear in (expert.gate_proj, expert.up_proj, expert.down_proj):
                assert levels(linear.weight) <= 16
    assert not torch.equal(model.model.layers[0].mlp.experts.experts[1].gate_proj.weight, expert_before)


def test_router_site_rule_quantizes_the_router_with_gptq():
    model = moe_model()
    router_before = [layer.mlp.gate.weight.clone() for layer in model.model.layers]
    quantize_model(model, windows(), config("moe.router.weight", true_sequential=True))
    for layer, before in zip(model.model.layers, router_before, strict=True):
        assert not torch.equal(layer.mlp.gate.weight, before)
        assert levels(layer.mlp.gate.weight) <= 16


def test_expert_hessian_uses_only_its_routed_tokens():
    model = moe_model()
    wins = windows()
    original = moe_model()
    seen: dict[int, HessianAccumulator] = {}
    handles = []
    for e, expert in enumerate(original.model.layers[0].mlp.experts.experts):
        seen[e] = HessianAccumulator(expert.gate_proj.weight.shape[1], torch.device("cpu"))
        handles.append(
            expert.gate_proj.register_forward_hook(lambda _m, inp, _o, e=e: seen[e].add_batch(inp[0].data))
        )
    with torch.no_grad():
        for w in wins:
            original(input_ids=w)
    for h in handles:
        h.remove()
    assert all(a.tokens > 0 for a in seen.values())

    quantize_model(model, wins, config("linear.mlp.experts.gate_proj.weight"))
    for e, expert in enumerate(model.model.layers[0].mlp.experts.experts):
        reference = quantize_weight(
            original.model.layers[0].mlp.experts.experts[e].gate_proj.weight.data,
            seen[e].H,
            FMT,
            blocksize=128,
            percdamp=0.01,
            act_order=False,
            static_groups=False,
        )
        torch.testing.assert_close(expert.gate_proj.weight.data, reference, rtol=0, atol=1e-5)


def test_expert_that_is_never_routed_gets_rtn_weights_and_a_log_line(caplog):
    model = moe_model()
    never_route_to(model, expert=3, replacement=2)
    before = [layer.mlp.experts.experts[3].down_proj.weight.clone() for layer in model.model.layers]
    with caplog.at_level(logging.WARNING):
        quantize_model(model, windows(), config())
    for layer, original in zip(model.model.layers, before, strict=True):
        starved = layer.mlp.experts.experts[3]
        assert torch.equal(starved.down_proj.weight, fake_quantize(original, FMT, kind="weight"))
        assert starved.down_proj.weight.abs().sum() > 0
    messages = [r.getMessage() for r in caplog.records]
    assert [f"expert {layer}.3: 0 tokens, RTN" in messages for layer in range(LAYERS)] == [True] * LAYERS
    assert sum("RTN" in m for m in messages) == LAYERS


def test_moe_min_tokens_boundary_follows_the_routed_token_count(caplog):
    wins = windows()
    counts = routed_counts(moe_model(), wins, layer=0)
    assert len(counts) == EXPERTS
    threshold = sorted(counts.values())[1] + 1
    expected = {e for e, n in counts.items() if n < threshold}
    assert 0 < len(expected) < EXPERTS
    model = moe_model()
    with caplog.at_level(logging.WARNING):
        quantize_model(model, wins, config("linear.mlp.experts.gate_proj.weight", moe_min_tokens=threshold))
    logged = {
        int(m.split()[1].rstrip(":").split(".")[1])
        for m in (r.getMessage() for r in caplog.records)
        if m.startswith("expert 0.")
    }
    assert logged == expected
    assert all(f"expert 0.{e}: {counts[e]} tokens, RTN" in caplog.text for e in expected)


def test_true_sequential_makes_expert_down_proj_see_the_quantized_gate_and_up():
    wins = windows()
    plain, sequential = moe_model(), moe_model()
    quantize_model(plain, wins, config())
    quantize_model(sequential, wins, config(true_sequential=True))
    plain_layer, sequential_layer = plain.model.layers[0], sequential.model.layers[0]
    assert torch.equal(plain_layer.self_attn.q_proj.weight, sequential_layer.self_attn.q_proj.weight)
    plain_down = plain_layer.mlp.experts.experts[0].down_proj.weight
    assert not torch.equal(plain_down, sequential_layer.mlp.experts.experts[0].down_proj.weight)
