import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_llama, build_tiny_qwen3  # noqa: E402
from effeval.ops import online, rotation, sites  # noqa: E402
from effeval.ops.scaling import ChannelDivide, HeadChannelScale  # noqa: E402
from test_sites import padded_batch  # noqa: E402

pytestmark = pytest.mark.compute

MODELS = {"qwen3": lambda: build_tiny_qwen3(head_dim=32), "llama": build_tiny_llama}


def _capture(model, ops, site="attn.k", layer=0):
    """Operand seen at `site` of `layer` after `ops`, on one padded batch."""
    seen = []

    def observe(x):
        seen.append(x.detach().clone())
        return x

    attachment = online.register(model, [*ops, online.OnlineOp(site, observe, [layer])])
    try:
        input_ids, attention_mask = padded_batch()
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        attachment.remove()
    return seen[0]


@pytest.mark.parametrize("name", MODELS)
def test_chain_applies_ops_in_registration_order(name, register_architecture):
    model = MODELS[name]()
    register_architecture(model)
    head_dim = model.model.layers[0].self_attn.head_dim
    raw = _capture(model, [])
    generator = torch.Generator().manual_seed(1)
    rotation = torch.linalg.qr(torch.randn(head_dim, head_dim, generator=generator))[0]
    scale = torch.rand(head_dim, generator=generator) + 0.5
    ops = [
        online.OnlineOp("attn.k", online.RightMultiply(rotation)),
        online.OnlineOp("attn.k", ChannelDivide(scale)),
    ]
    got = _capture(model, ops)
    assert torch.allclose(got, (raw @ rotation) / scale, atol=1e-5)
    assert not torch.allclose(got, (raw / scale) @ rotation, atol=1e-3)
    assert not hasattr(model.model.layers[0].self_attn, sites.ATTN_OPS_ATTR)
    assert model.config._attn_implementation == "sdpa"


def test_chain_module_runs_functions_first_to_last():
    chain = online.Chain([lambda x: x + 1, lambda x: x * 2])
    assert torch.equal(chain(torch.tensor([1.0, 2.0])), torch.tensor([4.0, 6.0]))


def test_headwise_right_multiply_equals_a_per_head_loop():
    generator = torch.Generator().manual_seed(0)
    matrices = torch.randn(3, 8, 8, generator=generator)
    x = torch.randn(2, 3, 5, 8, generator=generator)
    got = online.HeadwiseRightMultiply(matrices)(x)
    expected = torch.stack([x[:, h] @ matrices[h] for h in range(3)], dim=1)
    assert got.shape == x.shape
    assert torch.allclose(got, expected, atol=1e-5)
    assert not torch.allclose(got[:, 0], x[:, 0] @ matrices[1], atol=1e-3)
    same = online.HeadwiseRightMultiply(matrices[:1].expand(3, 8, 8))(x)
    assert torch.allclose(same, online.RightMultiply(matrices[0])(x), atol=1e-6)


def test_headwise_right_multiply_keeps_dtype_and_rejects_wrong_head_count():
    matrices = torch.eye(4)[None].repeat(3, 1, 1)
    x = torch.randn(1, 3, 2, 4).to(torch.bfloat16)
    assert online.HeadwiseRightMultiply(matrices)(x).dtype == torch.bfloat16
    with pytest.raises(ValueError, match="expected x"):
        online.HeadwiseRightMultiply(matrices)(torch.randn(1, 2, 2, 4))
    with pytest.raises(ValueError, match="heads, D, D"):
        online.HeadwiseRightMultiply(torch.eye(4))


def test_head_channel_scale_per_head_and_shared():
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(2, 3, 5, 8, generator=generator)
    per_head = torch.rand(3, 8, generator=generator) + 0.5
    got = HeadChannelScale(per_head)(x)
    assert torch.allclose(got, torch.stack([x[:, h] * per_head[h] for h in range(3)], dim=1))
    shared = per_head[0]
    assert torch.allclose(HeadChannelScale(shared)(x), x * shared)
    assert HeadChannelScale(per_head)(x.to(torch.bfloat16)).dtype == torch.bfloat16


def test_head_channel_scale_rejects_mismatched_shapes():
    x = torch.randn(1, 3, 2, 8)
    with pytest.raises(ValueError, match="does not match"):
        HeadChannelScale(torch.ones(2, 8))(x)
    with pytest.raises(ValueError, match="does not match"):
        HeadChannelScale(torch.ones(4))(x)
    with pytest.raises(ValueError, match="must be"):
        HeadChannelScale(torch.ones(1, 3, 8))


def test_kron_right_multiply_equals_the_dense_hadamard_product_and_the_signed_one():
    n = 9728
    x = torch.randn(3, 5, n)
    base, m = rotation.hadamard_parts(n)
    dense = x.double() @ rotation.hadamard(n)
    got = online.KronRightMultiply(base, m)(x)
    assert got.dtype == x.dtype
    assert (got - dense.float()).abs().max() < 1e-4 * x.abs().max()
    signed = online.KronRightMultiply(base, m, rotation.random_signs(n, 3))(x)
    dense_signed = x.double() @ rotation.random_hadamard(n, 3)
    assert (signed - dense_signed.float()).abs().max() < 1e-4 * x.abs().max()
    assert (signed - got).abs().max() > 0.1


def test_kron_right_multiply_returns_the_input_dtype_for_half_inputs():
    base, m = rotation.hadamard_parts(9728)
    x = torch.randn(2, 9728).to(torch.bfloat16)
    assert online.KronRightMultiply(base, m)(x).dtype == torch.bfloat16
