"""Expert unfusing: the per-expert `nn.Linear` model is bit-identical to HF's eager fused experts."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from effeval.core.spec import ResolvedModel, RuntimeSpec
from effeval.runtime import hf, moe

EXPERTS, TOP_K, LAYERS, HIDDEN, INTER = 4, 2, 2, 64, 32


def qwen3_moe_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=96, moe_intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=EXPERTS, num_experts_per_tok=TOP_K, decoder_sparse_step=1, mlp_only_layers=[],
        max_position_embeddings=64, tie_word_embeddings=False,
    )  # fmt: skip


def deepseek_v2_config() -> DeepseekV2Config:
    return DeepseekV2Config(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=96, moe_intermediate_size=INTER,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=8, qk_nope_head_dim=16, v_head_dim=16,
        n_routed_experts=EXPERTS, n_shared_experts=1, num_experts_per_tok=TOP_K, first_k_dense_replace=0,
        n_group=1, topk_group=1, topk_method="greedy", max_position_embeddings=64,
    )  # fmt: skip


def dense_config() -> Qwen3Config:
    return Qwen3Config(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=96, num_hidden_layers=LAYERS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=64,
    )  # fmt: skip


CASES = {
    "qwen3_moe": (qwen3_moe_config, Qwen3MoeForCausalLM),
    "deepseek_v2": (deepseek_v2_config, DeepseekV2ForCausalLM),
}


def spread_routers(model: nn.Module) -> None:
    """Give every router a random weight so that all experts are selected."""
    for name, param in model.named_parameters():
        if name.endswith("mlp.gate.weight"):
            nn.init.normal_(param, std=1.0)


def build(case: str, dtype: torch.dtype) -> nn.Module:
    make_config, cls = CASES[case]
    config = make_config()
    config._experts_implementation = "eager"
    torch.manual_seed(0)
    model = cls(config)
    spread_routers(model)
    return model.to(dtype).eval()


def logits(model: nn.Module) -> torch.Tensor:
    ids = torch.randint(0, 128, (2, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        return model(input_ids=ids).logits


def routed_experts(model: nn.Module) -> set[int]:
    seen: set[int] = set()
    hooks = [
        m.register_forward_hook(lambda _m, _i, out: seen.update(out[-1].flatten().tolist()))
        for n, m in model.named_modules()
        if n.endswith("mlp.gate")
    ]
    logits(model)
    for h in hooks:
        h.remove()
    return seen


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_unfused_logits_bit_identical_to_fused_eager(case, dtype):
    model = build(case, dtype)
    assert routed_experts(model) == set(range(EXPERTS))
    reference = logits(model)
    assert moe.unfuse_experts(model) == LAYERS
    assert torch.equal(logits(model), reference)


@pytest.mark.parametrize("case", sorted(CASES))
def test_unfused_structure_and_weights(case):
    model = build(case, torch.float32)
    fused = {
        n: (m.gate_up_proj.detach().clone(), m.down_proj.detach().clone())
        for n, m in model.named_modules()
        if n.endswith("mlp.experts")
    }
    assert len(fused) == LAYERS
    moe.unfuse_experts(model)
    assert not any(type(m).__name__ in moe.FUSED_EXPERTS for m in model.modules())
    assert not any(n.endswith(("gate_up_proj", "down_proj")) for n, _ in model.named_parameters())
    for name, (gate_up, down) in fused.items():
        block = model.get_submodule(name)
        assert isinstance(block, moe.ExpertLinears) and len(block.experts) == EXPERTS
        for e, expert in enumerate(block.experts):
            assert isinstance(expert.gate_proj, nn.Linear) and expert.gate_proj.bias is None
            assert torch.equal(expert.gate_proj.weight, gate_up[e, :INTER])
            assert torch.equal(expert.up_proj.weight, gate_up[e, INTER:])
            assert torch.equal(expert.down_proj.weight, down[e])


def test_unfusing_is_idempotent():
    model = build("qwen3_moe", torch.float32)
    assert moe.unfuse_experts(model) == LAYERS
    assert moe.unfuse_experts(model) == 0


def test_unfusing_frees_fused_modules_held_by_accelerate_hooks():
    from accelerate.hooks import ModelHook, add_hook_to_module

    model = build("qwen3_moe", torch.float32)
    fused = [layer.mlp.experts for layer in model.model.layers]
    for module in fused:
        add_hook_to_module(module, ModelHook())
    refs = [weakref.ref(module) for module in fused]
    del fused, module
    gc.disable()
    try:
        moe.unfuse_experts(model)
        assert [ref() for ref in refs] == [None] * LAYERS
    finally:
        gc.enable()


def resolved(path, architecture, dtype):
    return ResolvedModel(
        name="tiny", checkpoint=str(path), revision=None, architecture=architecture, dtype=dtype
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_load_model_unfuses_moe_checkpoint(tmp_path, dtype):
    torch.manual_seed(0)
    source = Qwen3MoeForCausalLM(qwen3_moe_config())
    spread_routers(source)
    source.save_pretrained(tmp_path)
    loaded = hf.load_model(resolved(tmp_path, "Qwen3MoeForCausalLM", dtype), RuntimeSpec()).model
    assert all(isinstance(layer.mlp.experts, moe.ExpertLinears) for layer in loaded.model.layers)

    reference = AutoModelForCausalLM.from_pretrained(
        tmp_path, dtype=getattr(torch, dtype), experts_implementation="eager"
    ).eval()
    assert type(reference.model.layers[0].mlp.experts).__name__ == "Qwen3MoeExperts"
    assert torch.equal(logits(loaded), logits(reference))


def test_load_model_leaves_dense_checkpoint_unchanged(tmp_path, monkeypatch):
    torch.manual_seed(0)
    Qwen3ForCausalLM(dense_config()).save_pretrained(tmp_path)
    monkeypatch.setattr(
        moe, "unfuse_experts", lambda _m: pytest.fail("unfuse_experts called for a dense model")
    )
    loaded = hf.load_model(resolved(tmp_path, "Qwen3ForCausalLM", "float32"), RuntimeSpec()).model
    plain = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    assert [(n, type(m)) for n, m in loaded.named_modules()] == [
        (n, type(m)) for n, m in plain.named_modules()
    ]
    state = plain.state_dict()
    assert loaded.state_dict().keys() == state.keys()
    assert all(torch.equal(v, state[k]) for k, v in loaded.state_dict().items())
    assert torch.equal(logits(loaded), logits(plain))


def test_unfuse_experts_is_a_noop_on_dense_model():
    model = Qwen3ForCausalLM(dense_config())
    before = [(n, type(m)) for n, m in model.named_modules()]
    assert moe.unfuse_experts(model) == 0
    assert [(n, type(m)) for n, m in model.named_modules()] == before
