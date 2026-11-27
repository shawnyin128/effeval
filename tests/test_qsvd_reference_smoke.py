from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import reference.qsvd_official as qsvd  # noqa: E402


def _tiny_mha_qwen3():
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).to(torch.float32).eval()


def _args(**overrides):
    base = dict(
        act_aware=True, act_alpha=0.5, grad_alpha=1.0, rank_ratio=1.0, svd_mode="0.2",
        had_rank=True, seed=0, qkv_fuse=True, calib_method="abs_mean",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _windows(count=4, length=16):
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


def test_official_pipeline_factors_every_qkv_and_keeps_logits_finite():
    model = _tiny_mha_qwen3()
    layers = model.model.layers
    qsvd.qsvd_lm_decompose(model, layers, _windows(), _args())
    model.eval()
    kept = 0
    for layer in layers:
        for name in ("q_proj", "k_proj", "v_proj"):
            assert isinstance(getattr(layer.self_attn, name), qsvd.SVDLinear)
        kept += layer.self_attn.q_proj.truncation_rank
    assert kept == int(1.0 / 2 * 3 * 64)
    with torch.no_grad():
        logits = model(_windows(1)[0]).logits
    assert torch.isfinite(logits).all()
