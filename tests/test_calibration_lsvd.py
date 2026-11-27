import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import IntFormat  # noqa: E402
from effeval.ops.calibration import act_stats, collect_input_rows, output_grad_grams  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402

SITE = "linear.attn.k_proj.input"


def _windows(count: int = 3, length: int = 12, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


def _k_inputs(model, windows, layer):
    seen: list[torch.Tensor] = []
    handle = model.model.layers[layer].self_attn.k_proj.register_forward_pre_hook(
        lambda _m, args: seen.append(args[0].detach())
    )
    with torch.no_grad():
        for w in windows:
            model(w)
    handle.remove()
    return seen


def _hand_gram(seen, quant=None):
    total = None
    for x in seen:
        if quant is not None:
            x = fake_quantize(x, quant, "activation")
        x = x.float()
        add = (x.transpose(1, 2) @ x).sum(dim=0)
        total = add if total is None else total + add
    return total


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gram_equals_hand_accumulation(dtype):
    model = build_tiny_qwen3().to(dtype)
    windows = _windows()
    stats = act_stats(model, windows, [SITE], "gram")
    for layer in range(2):
        value = stats[SITE, layer]
        assert value.dtype == torch.float32 and value.shape == (64, 64)
        assert torch.equal(value, _hand_gram(_k_inputs(model, windows, layer)))


def test_gram_prequant_quantizes_before_accumulating():
    model = build_tiny_qwen3()
    windows = _windows()
    fmt = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token", clip_ratio=1.0)
    stats = act_stats(model, windows, [SITE], "gram", prequant=fmt)
    plain = act_stats(model, windows, [SITE], "gram")
    for layer in range(2):
        seen = _k_inputs(model, windows, layer)
        assert torch.equal(stats[SITE, layer], _hand_gram(seen, fmt))
        assert not torch.equal(stats[SITE, layer], plain[SITE, layer])


def test_prequant_is_rejected_for_other_statistics():
    model = build_tiny_qwen3()
    fmt = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token")
    with pytest.raises(ValueError, match="prequant"):
        act_stats(model, _windows(1), [SITE], "absmax", prequant=fmt)


@pytest.mark.parametrize("batch", [1, 2])
def test_output_grad_grams_equal_autograd_of_the_loss_and_remove_hooks(batch):
    model = build_tiny_qwen3()
    gen = torch.Generator().manual_seed(3)
    windows = [torch.randint(0, 97, (batch, 10), generator=gen) for _ in range(2)]
    attn = model.model.layers[0].self_attn
    heads, head_dim = 4, 32
    params = [attn.q_proj.weight]

    q_out: list[torch.Tensor] = []
    handle = attn.q_proj.register_forward_hook(lambda _m, _a, out: q_out.append(out))
    expected = torch.zeros(heads, head_dim, head_dim)
    all_params = list(model.parameters())
    flags = [p.requires_grad for p in all_params]
    for p in all_params:
        p.requires_grad_(p is params[0])
    for w in windows:
        loss = model(w, labels=w).loss
        (grad,) = torch.autograd.grad(loss, q_out[-1])
        g = grad.view(batch, -1, heads, head_dim).permute(0, 2, 1, 3).float()
        expected += (g.transpose(-1, -2) @ g).sum(dim=0)
    handle.remove()
    for p, flag in zip(all_params, flags, strict=True):
        p.requires_grad_(flag)

    hooks_before = len(attn.q_proj._backward_hooks)
    grams = output_grad_grams(model, windows, {"q0": (attn.q_proj, heads, head_dim)}, params)
    assert grams["q0"].shape == (heads, head_dim, head_dim) and grams["q0"].dtype == torch.float32
    assert torch.allclose(grams["q0"], expected, rtol=1e-6, atol=1e-12)
    assert expected.abs().max() > 0
    assert len(attn.q_proj._backward_hooks) == hooks_before == 0


def test_collect_input_rows_equals_indexing_the_concatenated_stream():
    model = build_tiny_qwen3()
    windows = _windows(count=3, length=12)
    rows = torch.tensor([30, 0, 13, 11, 35, 12])
    got = collect_input_rows(model, windows, SITE, rows)
    for layer in range(2):
        stream = torch.cat([x.reshape(-1, 64) for x in _k_inputs(model, windows, layer)])
        assert got[layer].shape == (6, 64)
        assert torch.equal(got[layer], stream[rows])


def test_collect_input_rows_keeps_activation_dtype_and_validates_range():
    model = build_tiny_qwen3().to(torch.bfloat16)
    windows = _windows(count=2, length=8)
    got = collect_input_rows(model, windows, SITE, torch.tensor([15]))
    assert got[0].dtype == torch.bfloat16 and got[0].device.type == "cpu"
    with pytest.raises(ValueError, match="stream"):
        collect_input_rows(model, windows, SITE, torch.tensor([16]))
    with pytest.raises(ValueError, match="stream"):
        collect_input_rows(model, windows, SITE, torch.tensor([-1]))
