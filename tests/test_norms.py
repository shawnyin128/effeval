import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.ops.norms import HeadwiseRMSNorm, install_headwise_norm  # noqa: E402
from test_sites import padded_batch  # noqa: E402

pytestmark = pytest.mark.compute


def _logits(model):
    input_ids, attention_mask = padded_batch()
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).logits


def _original_norm(dim=16, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    norm = transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm(dim, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.rand(dim, generator=generator) + 0.5)
    return norm.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_equal_head_gains_are_bit_identical_to_qwen3_rmsnorm(dtype):
    norm = _original_norm(dtype=dtype)
    headwise = HeadwiseRMSNorm.from_shared(norm, heads=4)
    x = torch.randn(2, 5, 4, 16, generator=torch.Generator().manual_seed(1)).to(dtype)
    expected = norm(x)
    got = headwise(x)
    assert headwise.weight.shape == (4, 16)
    assert headwise.variance_epsilon == norm.variance_epsilon == 1e-5
    assert got.dtype == expected.dtype == dtype
    assert torch.equal(got, expected)


def test_a_different_gain_per_head_changes_only_that_head():
    norm = _original_norm()
    headwise = HeadwiseRMSNorm.from_shared(norm, heads=4)
    with torch.no_grad():
        headwise.weight[2] *= 3.0
    x = torch.randn(2, 5, 4, 16, generator=torch.Generator().manual_seed(1))
    expected = norm(x)
    got = headwise(x)
    for head in (0, 1, 3):
        assert torch.equal(got[:, :, head], expected[:, :, head])
    assert torch.allclose(got[:, :, 2], 3.0 * expected[:, :, 2], rtol=1e-6)


def test_install_headwise_norm_keeps_the_logits_of_tiny_qwen3_and_rejects_a_second_install():
    model = build_tiny_qwen3(head_dim=32)
    before = _logits(model)
    config = model.config
    for layer in model.model.layers:
        attention = layer.self_attn
        q = install_headwise_norm(attention, "q_norm", config.num_attention_heads)
        k = install_headwise_norm(attention, "k_norm", config.num_key_value_heads)
        assert attention.q_norm is q and attention.k_norm is k
        assert q.weight.shape == (4, 32) and k.weight.shape == (2, 32)
    assert torch.equal(_logits(model), before)
    with pytest.raises(ValueError, match="already"):
        install_headwise_norm(model.model.layers[0].self_attn, "q_norm", 4)


def test_per_head_gain_in_the_model_changes_the_logits():
    model = build_tiny_qwen3(head_dim=32)
    before = _logits(model)
    k = install_headwise_norm(model.model.layers[0].self_attn, "k_norm", model.config.num_key_value_heads)
    with torch.no_grad():
        k.weight[1] *= 2.0
    assert (_logits(model) - before).abs().max() > 1e-3
