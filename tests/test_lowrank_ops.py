"""Low-rank ops: SVD factors, global rank selection, and the shared-down q/k/v installation."""

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import QuantRule  # noqa: E402
from effeval.core.lowrank import ARCHITECTURE_ATTR  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.lowrank import (  # noqa: E402
    LowRankProjection,
    factor_weights,
    global_top_k,
    install_shared_qkv,
    joint_whitened_svd,
    projected_gradient,
    rotate_latent,
)

pytestmark = pytest.mark.compute

INT4 = {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_token"}
OUT_SIZES = (128, 64, 64)


def _dense_qkv(layer):
    attn = layer.self_attn
    return [attn.q_proj.weight.data, attn.k_proj.weight.data, attn.v_proj.weight.data]


def _full_rank_factors(model, scale=None, beta=0.5):
    factors = []
    for layer in model.model.layers:
        U, S, V = joint_whitened_svd(_dense_qkv(layer), scale)
        factors.append(factor_weights(U, S, V, list(range(S.numel())), scale, beta, OUT_SIZES, torch.float32))
    return factors


def _logits(model):
    ids = torch.arange(24).view(1, -1) % 97
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_full_rank_install_reproduces_dense_logits_and_shares_one_down_per_layer():
    model = build_tiny_qwen3(layers=2)
    scale = 0.5 + torch.rand(64, generator=torch.Generator().manual_seed(3))
    dense = _logits(model)
    install_shared_qkv(model, _full_rank_factors(model, scale))
    assert torch.allclose(_logits(model), dense, atol=1e-5)
    names = [n for n, _ in model.named_parameters()]
    assert len(names) == len(set(names))
    for layer in range(2):
        assert names.count(f"model.layers.{layer}.self_attn.qkv_down.weight") == 1
        assert not any(f"layers.{layer}.self_attn.{p}_proj.down" in n for n in names for p in "qkv")


def test_installed_layout_has_eight_input_sites_per_layer_and_runs_down_three_times():
    model = build_tiny_qwen3(layers=2)
    install_shared_qkv(model, _full_rank_factors(model))
    assignment, _ = sites.resolve(model, [QuantRule(sites=["linear.*.input"], layers="all", quant=INT4)])
    per_layer = [sum(1 for key in assignment if key[1] == layer) for layer in range(2)]
    assert per_layer == [8, 8]

    attn = model.model.layers[0].self_attn
    calls, widths = [], []
    attn.qkv_down.register_forward_hook(lambda m, i, o: calls.append(1))
    attn.q_proj.up.register_forward_pre_hook(lambda m, args: widths.append(args[0].shape[-1]))
    _logits(model)
    assert len(calls) == 3
    assert widths == [attn.qkv_down.out_features]


def test_install_rejects_a_second_installation_and_moves_bias_to_the_up_factor():
    model = build_tiny_qwen3(layers=1)
    attn = model.model.layers[0].self_attn
    attn.q_proj.bias = nn.Parameter(torch.randn(128))
    bias = attn.q_proj.bias.data.clone()
    factors = _full_rank_factors(model)
    install_shared_qkv(model, factors)
    assert getattr(model, ARCHITECTURE_ATTR) is not None
    assert torch.equal(attn.q_proj.up.bias.data, bias)
    assert attn.k_proj.up.bias is None and attn.qkv_down.bias is None
    with pytest.raises(ValueError, match="already"):
        install_shared_qkv(model, factors)


def test_projected_gradient_is_the_diagonal_of_the_projected_matrix():
    g = torch.Generator().manual_seed(0)
    w, grad = torch.randn(20, 12, generator=g), torch.randn(20, 12, generator=g)
    scale = 0.5 + torch.rand(12, generator=g)
    U, _, V = joint_whitened_svd([w], scale)
    got = projected_gradient(U, V, grad, 1 / scale)
    assert torch.allclose(got, torch.diag(U.T @ (grad / scale) @ V), atol=1e-5)
    assert torch.allclose(projected_gradient(U, V, grad, None), torch.diag(U.T @ grad @ V), atol=1e-5)


def _official_top_k(scores, k):
    entries = [(layer, i, s.item()) for layer, sc in enumerate(scores) for i, s in enumerate(sc)]
    entries.sort(key=lambda x: x[2], reverse=True)
    out = [[] for _ in scores]
    for layer, i, _ in entries[:k]:
        out[layer].append(i)
    return out


@pytest.mark.parametrize("k", [1, 7, 20, 41, 60])
def test_global_top_k_matches_the_official_stable_sort_with_many_ties(k):
    g = torch.Generator().manual_seed(1)
    levels = torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float16)
    scores = [levels[torch.randint(0, 8, (n,), generator=g)] for n in (20, 0, 25, 15)]
    assert global_top_k(scores, k) == _official_top_k(scores, k)


def test_global_top_k_rejects_nan_and_oversized_k():
    with pytest.raises(ValueError):
        global_top_k([torch.tensor([1.0, float("nan")])], 1)
    with pytest.raises(ValueError):
        global_top_k([torch.ones(3)], 4)


@pytest.mark.parametrize("beta", [0.0, 0.2, 0.5, 1.0])
def test_factors_reconstruct_the_dense_weights_for_every_beta(beta):
    g = torch.Generator().manual_seed(2)
    ws = [torch.randn(n, 16, generator=g) for n in OUT_SIZES]
    scale = 0.5 + torch.rand(16, generator=g)
    U, S, V = joint_whitened_svd(ws, scale)
    down, ups = factor_weights(U, S, V, list(range(16)), scale, beta, OUT_SIZES, torch.float32)
    assert down.shape == (16, 16) and [u.shape[0] for u in ups] == list(OUT_SIZES)
    assert torch.allclose(torch.cat(ups) @ down, torch.cat(ws), atol=1e-5)


def test_beta_sets_the_split_of_singular_values_between_down_and_up():
    g = torch.Generator().manual_seed(4)
    ws = [torch.randn(n, 16, generator=g) for n in OUT_SIZES]
    U, S, V = joint_whitened_svd(ws, None)
    down, ups = factor_weights(U, S, V, list(range(16)), None, 1.0, OUT_SIZES, torch.float32)
    assert torch.allclose(down.norm(dim=1), S, atol=1e-4)
    assert torch.allclose(torch.cat(ups).norm(dim=0), torch.ones(16), atol=1e-5)


def test_factor_weights_keeps_selection_order_and_rejects_empty():
    g = torch.Generator().manual_seed(5)
    ws = [torch.randn(n, 16, generator=g) for n in OUT_SIZES]
    U, S, V = joint_whitened_svd(ws, None)
    down, ups = factor_weights(U, S, V, [3, 0], None, 0.5, OUT_SIZES, torch.float32)
    assert torch.allclose(down[0], V[:, 3] * S[3].sqrt(), atol=1e-6)
    assert torch.allclose(torch.cat(ups)[:, 1], U[:, 0] * S[0].sqrt(), atol=1e-6)
    with pytest.raises(ValueError):
        factor_weights(U, S, V, [], None, 0.5, OUT_SIZES, torch.float32)


def test_rotate_latent_with_an_orthogonal_matrix_keeps_the_product():
    g = torch.Generator().manual_seed(6)
    ws = [torch.randn(n, 16, generator=g) for n in OUT_SIZES]
    U, S, V = joint_whitened_svd(ws, None)
    down_w, up_ws = factor_weights(U, S, V, list(range(10)), None, 0.5, OUT_SIZES, torch.float32)
    down = nn.Linear(16, 10, bias=False)
    down.weight.data.copy_(down_w)
    ups = [nn.Linear(10, n, bias=False) for n in OUT_SIZES]
    for up, w in zip(ups, up_ws, strict=True):
        up.weight.data.copy_(w)
    before = torch.cat([u.weight.data for u in ups]) @ down.weight.data
    h, _ = torch.linalg.qr(torch.randn(10, 10, generator=g, dtype=torch.float64))
    rotate_latent(down, ups, h)
    assert not torch.allclose(down.weight.data, down_w, atol=1e-3)
    assert torch.allclose(torch.cat([u.weight.data for u in ups]) @ down.weight.data, before, atol=1e-5)


def test_low_rank_projection_does_not_register_the_down_factor_as_a_child():
    down, up = nn.Linear(4, 3, bias=False), nn.Linear(3, 5, bias=False)
    proj = LowRankProjection(down, up)
    assert [n for n, _ in proj.named_parameters()] == ["up.weight"]
    assert (proj.in_features, proj.out_features) == (4, 5)
