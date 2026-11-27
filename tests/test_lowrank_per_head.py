"""Per-head LSVD attention ops: Cholesky, whitened SVD, gradient projection, allocation, installation."""

import pytest

torch = pytest.importorskip("torch")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import QuantRule  # noqa: E402
from effeval.core.lowrank import ARCHITECTURE_ATTR  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.lowrank import (  # noqa: E402
    cholesky_lower,
    cholesky_upper_heads,
    install_per_head_qkv,
    per_head_allocation,
    per_head_factors,
    per_head_projected_gradient,
    per_head_whitened_svd,
)

pytestmark = pytest.mark.compute

INT4 = {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_token"}


def _pd(n, seed, batch=None):
    generator = torch.Generator().manual_seed(seed)
    shape = (n, n) if batch is None else (batch, n, n)
    a = torch.randn(*shape, generator=generator, dtype=torch.float64)
    return a @ a.transpose(-1, -2) + n * torch.eye(n, dtype=torch.float64)


def test_cholesky_lower_reconstructs_and_rejects_a_singular_gram():
    gram = _pd(6, 0)
    lower = cholesky_lower(gram)
    assert lower.dtype == torch.float64
    assert torch.allclose(lower @ lower.T, gram)
    assert torch.equal(lower, torch.tril(lower))
    singular = torch.ones(4, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="not positive definite"):
        cholesky_lower(singular)


def test_cholesky_upper_heads_returns_upper_factor_and_its_inverse_and_rejects_a_bad_head():
    gram = _pd(5, 1, batch=3).float()
    upper, upper_inv = cholesky_upper_heads(gram)
    assert upper.dtype == upper_inv.dtype == torch.float32
    assert torch.equal(upper, torch.triu(upper))
    assert torch.allclose(upper.transpose(1, 2) @ upper, gram, atol=1e-4)
    assert torch.allclose(upper @ upper_inv, torch.eye(5).expand(3, 5, 5), atol=1e-5)
    gram[1] = torch.ones(5, 5)
    with pytest.raises(ValueError, match="not positive definite"):
        cholesky_upper_heads(gram)


def _whitening(heads, hd, d, seed):
    lower = cholesky_lower(_pd(d, seed)).float()
    upper, upper_inv = cholesky_upper_heads(_pd(hd, seed + 1, batch=heads).float())
    return lower, upper, upper_inv


def test_whitened_svd_reconstructs_the_whitened_blocks_and_projected_gradient_matches_the_formula():
    heads, hd, d = 3, 4, 10
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(heads * hd, d, generator=generator)
    grad = torch.randn(heads * hd, d, generator=generator)
    lower, upper, upper_inv = _whitening(heads, hd, d, 5)
    U, S, V = per_head_whitened_svd(weight, heads, lower, upper)
    assert U.shape == (heads, hd, hd) and S.shape == (heads, hd) and V.shape == (heads, d, hd)
    expected = upper @ (weight.view(heads, hd, d) @ lower)
    assert torch.allclose(U @ torch.diag_embed(S) @ V.transpose(1, 2), expected, atol=1e-4)

    input_inv_t = torch.linalg.inv(lower.double()).float().T
    score = per_head_projected_gradient(U, V, grad, input_inv_t, upper_inv, torch.float32)
    assert score.shape == (heads, hd)
    g = grad.view(heads, hd, d)
    for h in range(heads):
        full = U[h].T @ upper_inv[h].T @ g[h] @ input_inv_t @ V[h]
        assert torch.allclose(score[h], torch.diagonal(full), atol=1e-4)
    low = per_head_projected_gradient(U, V, grad, input_inv_t, upper_inv, torch.bfloat16)
    assert low.dtype == torch.bfloat16


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_all_indices_factors_reconstruct_the_dense_weight(beta):
    heads, hd, d = 3, 4, 10
    weight = torch.randn(heads * hd, d, generator=torch.Generator().manual_seed(3))
    lower, upper, upper_inv = _whitening(heads, hd, d, 7)
    U, S, V = per_head_whitened_svd(weight, heads, lower, upper)
    down, blocks = per_head_factors(U, S, V, [list(range(hd))] * heads, lower, upper_inv, beta, torch.float32)
    assert down.shape == (heads * hd, d) and [b.shape for b in blocks] == [(hd, hd)] * heads
    assert torch.allclose(torch.block_diag(*blocks) @ down, weight, atol=1e-4)


def test_factors_with_a_subset_use_each_heads_own_ranks_and_reject_an_empty_head():
    heads, hd, d = 2, 4, 8
    weight = torch.randn(heads * hd, d, generator=torch.Generator().manual_seed(4))
    lower, upper, upper_inv = _whitening(heads, hd, d, 9)
    U, S, V = per_head_whitened_svd(weight, heads, lower, upper)
    down, blocks = per_head_factors(U, S, V, [[0, 2], [1]], lower, upper_inv, 0.0, torch.float32)
    assert down.shape == (3, d) and [b.shape for b in blocks] == [(hd, 2), (hd, 1)]
    with pytest.raises(ValueError, match="head 1"):
        per_head_factors(U, S, V, [[0], []], lower, upper_inv, 0.0, torch.float32)


def _official_top_k(scores, ks, joint):
    """Pure-Python replica of `get_top_k_scores` (per-head q/k/v) and the `[0]` floor of the builder."""
    heads = [pool[0].shape[0] for pool in scores]
    layers = len(scores[0])
    chosen = [[{} for _ in range(layers)] for _ in scores]

    def entries(p):
        return [
            (p, layer, i, h, scores[p][layer][h][i].item())
            for layer in range(layers)
            for h in range(heads[p])
            for i in range(scores[p][layer].shape[1])
        ]

    if joint:
        shared = [e for p in range(len(scores)) for e in entries(p)]
        shared.sort(key=lambda e: e[-1], reverse=True)
        k = min(sum(k_ * h for k_, h in zip(ks, heads, strict=True)), len(shared))
        groups = [shared[:k]]
    else:
        groups = []
        for p in range(len(scores)):
            pool = entries(p)
            pool.sort(key=lambda e: e[-1], reverse=True)
            groups.append(pool[: min(ks[p] * heads[p], len(pool))])
    for group in groups:
        for p, layer, i, h, _ in group:
            chosen[p][layer].setdefault(h, []).append(i)
    if any(not chosen[p][layer] for p in range(len(scores)) for layer in range(layers)):
        return None, 0
    floored = 0
    result = []
    for p in range(len(scores)):
        per_layer = []
        for layer in range(layers):
            per_head = []
            for h in range(heads[p]):
                indices = chosen[p][layer].get(h, [])
                if not indices:
                    indices = [0]
                    floored += 1
                per_head.append(indices)
            per_layer.append(per_head)
        result.append(per_layer)
    return result, floored


def _bf16_pools(seed):
    generator = torch.Generator().manual_seed(seed)
    heads = (4, 2, 2)
    return [
        [torch.randint(0, 8, (h, 8), generator=generator).to(torch.bfloat16) for _ in range(3)] for h in heads
    ]


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("ks", [(3, 2, 2), (5, 4, 6), (8, 8, 8), (2, 1, 1), (1, 1, 1)])
def test_allocation_equals_the_official_selection_on_tied_bf16_scores(joint, ks):
    for seed in range(4):
        pools = _bf16_pools(seed)
        expected, expected_floor = _official_top_k(pools, ks, joint)
        if expected is None:
            with pytest.raises(ValueError, match="no selected singular value"):
                per_head_allocation(pools, ks, joint)
            continue
        indices, floored = per_head_allocation(pools, ks, joint)
        assert indices == expected
        assert floored == expected_floor


def test_allocation_floors_empty_heads_and_raises_for_an_empty_layer():
    hd = 4
    layer0 = torch.zeros(2, hd)
    layer0[0] = torch.tensor([9.0, 8.0, 0.0, 0.0])
    layer1 = torch.zeros(2, hd)
    layer1[1] = torch.tensor([7.0, 6.0, 0.0, 0.0])
    indices, floored = per_head_allocation([[layer0, layer1]], [2], joint=False)
    assert indices[0][0][0] == [0, 1] and indices[0][1][1] == [0, 1]
    assert floored == 2
    assert indices[0][0][1] == [0] and indices[0][1][0] == [0]
    with pytest.raises(ValueError, match="layer 1"):
        per_head_allocation([[layer0, layer1 * 0 - 1]], [1], joint=False)


def _factor_model(layers=2):
    model = build_tiny_qwen3(layers=layers)
    heads = {"q_proj": 4, "k_proj": 2, "v_proj": 2}
    factors = []
    for layer in model.model.layers:
        per_layer = {}
        for name, h in heads.items():
            weight = getattr(layer.self_attn, name).weight.data
            hd = weight.shape[0] // h
            lower, upper, upper_inv = _whitening(h, hd, 64, 11)
            U, S, V = per_head_whitened_svd(weight, h, lower, upper)
            per_layer[name] = per_head_factors(
                U, S, V, [list(range(hd))] * h, lower, upper_inv, 0.5, torch.float32
            )
        factors.append(per_layer)
    return model, factors


def _logits(model):
    ids = torch.arange(24).view(1, -1) % 97
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_full_rank_install_reproduces_dense_logits_with_own_downs_and_ten_input_sites():
    model, factors = _factor_model()
    dense = _logits(model)
    install_per_head_qkv(model, factors)
    assert torch.allclose(_logits(model), dense, atol=1e-4)
    names = [n for n, _ in model.named_parameters()]
    assert len(names) == len(set(names))
    for layer in range(2):
        for p in "qkv":
            assert names.count(f"model.layers.{layer}.self_attn.{p}_down.weight") == 1
    assignment, _ = sites.resolve(model, [QuantRule(sites=["linear.*.input"], layers="all", quant=INT4)])
    assert [sum(1 for key in assignment if key[1] == layer) for layer in range(2)] == [10, 10]
    up = model.model.layers[0].self_attn.q_proj.up.weight
    assert up.shape == (128, 128)
    assert torch.count_nonzero(up[:32, 32:]) == 0


def test_install_rejects_a_second_installation_and_moves_bias_to_the_up_factor():
    model, factors = _factor_model(layers=1)
    attn = model.model.layers[0].self_attn
    attn.q_proj.bias = torch.nn.Parameter(torch.randn(128))
    bias = attn.q_proj.bias.data.clone()
    install_per_head_qkv(model, factors)
    assert torch.equal(attn.q_proj.up.bias.data, bias)
    assert getattr(model, ARCHITECTURE_ATTR) is not None
    with pytest.raises(ValueError, match="already"):
        install_per_head_qkv(model, factors)
