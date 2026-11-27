"""Hybrid FFN ops: block-Hadamard, whitening, Phase-1 channel order, SVD branch and installation."""

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import QuantRule  # noqa: E402
from effeval.core.lowrank import ARCHITECTURE_ATTR  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.hybrid_ffn import (  # noqa: E402
    HybridLayer,
    HybridProjection,
    block_hadamard_lastdim,
    install_hybrid_ffn,
    partition_channels,
    rotated_selection_inputs,
    strided_rows,
    svd_aware_order,
    svd_branch,
    whitening_factor,
)

pytestmark = pytest.mark.compute

INT8 = {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token"}


def _sylvester(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / n**0.5


def test_block_hadamard_is_the_orthogonal_blockdiag_of_power_of_two_blocks():
    out = block_hadamard_lastdim(torch.eye(24, dtype=torch.float64))
    expected = torch.block_diag(_sylvester(16), _sylvester(8))
    assert torch.allclose(out, expected, atol=1e-12)
    assert torch.allclose(out @ out.T, torch.eye(24, dtype=torch.float64), atol=1e-12)
    x = torch.randn(3, 5, 24)
    assert torch.allclose(block_hadamard_lastdim(block_hadamard_lastdim(x)), x, atol=1e-5)
    assert block_hadamard_lastdim(torch.randn(4, 1)).shape == (4, 1)


def test_strided_rows_takes_every_row_up_to_the_cap_and_a_strided_prefix_beyond_it():
    assert strided_rows(5, 8).tolist() == [0, 1, 2, 3, 4]
    assert strided_rows(8, 8).tolist() == list(range(8))
    assert strided_rows(10, 4).tolist() == [0, 2, 4, 6]
    assert strided_rows(11, 4).tolist() == [0, 2, 4, 6]
    assert strided_rows(3, 2).tolist() == [0, 1]


def test_whitening_factor_without_gain_is_the_released_formula_bit_for_bit():
    x = torch.randn(200, 12, generator=torch.Generator().manual_seed(0))
    xd = x.double()
    gram = xd.T @ xd / xd.shape[0]
    gram.diagonal().add_(0.01 * gram.diagonal().mean().clamp(min=1e-8))
    assert torch.equal(whitening_factor(x, 0.01, None), torch.linalg.cholesky(gram))


def test_whitening_factor_with_gain_equals_the_gain_scaled_factor_of_the_normalised_input():
    generator = torch.Generator().manual_seed(1)
    x_n = torch.randn(300, 12, generator=generator, dtype=torch.float64)
    gain = 0.3 + 2 * torch.rand(12, generator=generator, dtype=torch.float64)
    damping = 0.05
    gram_n = x_n.T @ x_n / x_n.shape[0]
    ridge = damping * gram_n.diagonal().mean() * torch.eye(12, dtype=torch.float64)
    expected = torch.diag(gain) @ (gram_n + ridge) @ torch.diag(gain)
    chol = whitening_factor(x_n * gain, damping, gain)
    assert chol.dtype == torch.float64
    assert torch.allclose(chol @ chol.T, expected, rtol=1e-10, atol=1e-10)
    assert torch.equal(chol, torch.tril(chol))
    with pytest.raises(ValueError, match="non-zero"):
        whitening_factor(x_n, damping, torch.zeros(12))


def test_rotated_selection_inputs_reproduce_the_ffn_output_in_the_rotated_basis():
    generator = torch.Generator().manual_seed(2)
    d, f = 16, 24
    w_gate, w_up = (torch.randn(f, d, generator=generator, dtype=torch.float64) for _ in range(2))
    w_down = torch.randn(d, f, generator=generator, dtype=torch.float64)
    x = torch.randn(10, d, generator=generator, dtype=torch.float64)
    gain = 0.5 + torch.rand(d, generator=generator, dtype=torch.float64)
    rotation = torch.linalg.qr(torch.randn(d, d, generator=generator, dtype=torch.float64))[0]
    gate, up, down, inputs = rotated_selection_inputs(w_gate, w_up, w_down, x, gain, rotation, torch.float64)
    silu = torch.nn.functional.silu
    original = (silu(x @ w_gate.T) * (x @ w_up.T)) @ w_down.T
    rotated = (silu(inputs @ gate.T) * (inputs @ up.T)) @ down.T
    assert torch.allclose(rotated.double(), original @ rotation, rtol=1e-4, atol=1e-4)
    assert gate.dtype == torch.float32
    low = rotated_selection_inputs(w_gate, w_up, w_down, x, gain, rotation, torch.bfloat16)[0]
    assert torch.equal(low, ((w_gate * gain) @ rotation).to(torch.bfloat16).float())


def _structured_ffn(d=24, f=40, rank=6, hard=7, seed=3):
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(rank, d, generator=generator)
    w_gate = torch.randn(f, rank, generator=generator) @ basis * 0.3
    w_up = torch.randn(f, rank, generator=generator) @ basis * 0.3
    w_gate[hard] = torch.randn(d, generator=generator) * 3
    w_up[hard] = torch.randn(d, generator=generator) * 3
    w_down = torch.randn(d, f, generator=generator) * 0.3
    x = torch.randn(300, d, generator=generator)
    return w_gate, w_up, w_down, x, rank


@pytest.mark.parametrize("use_chol", [False, True])
def test_svd_aware_order_moves_the_channel_the_low_rank_branch_cannot_represent_first(use_chol):
    w_gate, w_up, w_down, x, rank = _structured_ffn(hard=7)
    chol = whitening_factor(x, 0.01, None) if use_chol else None
    order = svd_aware_order(w_gate, w_up, w_down, x, rank, 8, 16, 1.0, 0.0, chol, False)
    assert order[0] == 7
    assert len(set(order)) == len(order)


def test_svd_aware_order_does_not_accept_a_channel_whose_switch_changes_nothing():
    w_gate, w_up, w_down, x, rank = _structured_ffn(hard=7)
    w_down[:, 11] = 0.0
    order = svd_aware_order(w_gate, w_up, w_down, x, rank, 8, 16, 1.0, 0.0, None, False)
    assert 7 in order
    assert 11 not in order


def test_svd_aware_order_uses_only_the_first_8192_rows():
    w_gate, w_up, w_down, x, rank = _structured_ffn(hard=7)
    big = torch.cat([x.repeat(28, 1), torch.randn(500, x.shape[1]) * 50])
    assert big.shape[0] > 8192
    kwargs = {"rank": rank, "w_bits": 8, "a_bits": 16, "a_clip_ratio": 1.0}
    kwargs |= {"beta": 0.0, "chol": None, "latent_rotation": False}
    assert svd_aware_order(w_gate, w_up, w_down, big, **kwargs) == svd_aware_order(
        w_gate, w_up, w_down, big[:8192], **kwargs
    )


def test_partition_channels_reslices_the_greedy_order_and_pads_a_short_one_in_index_order():
    assert partition_channels([5, 2, 7, 0], 8, 5) == [0, 1, 3, 4, 6, 5, 2, 7]
    assert partition_channels([5, 2], 6, 3) == [1, 3, 4, 5, 2, 0]
    assert partition_channels([5, 2, 7, 0], 8, 4) == [1, 3, 4, 6, 5, 2, 7, 0]
    assert partition_channels([5, 2, 7, 0], 8, 4, 4) == [1, 3, 4, 6, 0, 2, 5, 7]
    assert partition_channels([5, 2, 7, 0], 8, 4, 5) == [1, 3, 4, 6, 5, 2, 7, 0]
    assert partition_channels([], 4, 2) == [2, 3, 0, 1]
    with pytest.raises(ValueError):
        partition_channels([1, 1], 4, 2)
    with pytest.raises(ValueError):
        partition_channels([], 4, 4)
    with pytest.raises(ValueError):
        partition_channels([9], 4, 2)


@pytest.mark.parametrize("latent_rotation", [False, True])
@pytest.mark.parametrize("beta", [0.0, 0.5])
def test_svd_branch_at_full_rank_reconstructs_the_weight(latent_rotation, beta):
    generator = torch.Generator().manual_seed(4)
    w = torch.randn(24, 40, generator=generator)
    x = torch.randn(200, 40, generator=generator)
    chol = whitening_factor(x, 0.01, None)
    up, down = svd_branch(w, 24, chol, beta, latent_rotation, torch.float32)
    assert up.shape == (24, 24) and down.shape == (24, 40)
    assert torch.allclose(up @ down, w, atol=1e-4)


def test_svd_branch_truncation_error_in_the_whitened_norm_is_the_discarded_singular_values():
    generator = torch.Generator().manual_seed(5)
    w = torch.randn(24, 40, generator=generator)
    chol = whitening_factor(torch.randn(200, 40, generator=generator), 0.01, None)
    for latent_rotation in (False, True):
        up, down = svd_branch(w, 8, chol, 0.5, latent_rotation, torch.float32)
        residual = (w - up @ down).double() @ chol
        tail = torch.linalg.svdvals(w.double() @ chol)[8:]
        assert residual.square().sum().item() == pytest.approx(tail.square().sum().item(), rel=1e-3)
    with pytest.raises(ValueError, match="beta"):
        svd_branch(w, 8, chol, 0.3, False, torch.float32)
    with pytest.raises(ValueError, match="rank"):
        svd_branch(w, 25, chol, 0.0, False, torch.float32)


def _logits(model):
    ids = torch.arange(24).view(1, -1) % 97
    with torch.no_grad():
        return model(input_ids=ids).logits


def _full_rank_layer(layer, generator, n_svd=48):
    mlp = layer.mlp
    d_ffn = mlp.gate_proj.out_features
    perm = torch.randperm(d_ffn, generator=generator).tolist()
    factors = []
    for proj in (mlp.gate_proj, mlp.up_proj):
        rows = proj.weight.data[perm][:n_svd]
        chol = whitening_factor(torch.randn(256, rows.shape[1], generator=generator), 0.01, None)
        factors.append(svd_branch(rows, min(n_svd, rows.shape[1]), chol, 0.0, True, torch.float32))
    return HybridLayer(perm=perm, n_svd=n_svd, gate=factors[0], up=factors[1])


def test_full_rank_installation_reproduces_dense_logits_and_extends_the_layout():
    model = build_tiny_qwen3(layers=2)
    dense = _logits(model)
    generator = torch.Generator().manual_seed(6)
    install_hybrid_ffn(model, [_full_rank_layer(model.model.layers[0], generator), None])
    assert torch.allclose(_logits(model), dense, atol=1e-4)

    hybrid, skipped = (layer.mlp.gate_proj for layer in model.model.layers)
    assert isinstance(hybrid, HybridProjection) and isinstance(skipped, HybridProjection)
    assert hybrid.out_features == skipped.out_features == 96
    assert hybrid.svd_up.out_features == 48 and hybrid.dense.out_features == 48
    assert not hasattr(skipped, "svd_down")
    assert getattr(model, ARCHITECTURE_ATTR).linears["mlp.gate_proj.dense"] == "mlp.gate_proj"

    rule = QuantRule(sites=["linear.mlp.*.input"], layers="all", quant=INT8)
    assignment, _ = sites.resolve(model, [rule])
    assert [sum(1 for key in assignment if key[1] == layer) for layer in range(2)] == [7, 3]


def test_installation_permutes_the_dense_rows_and_the_down_columns_with_the_same_permutation():
    model = build_tiny_qwen3(layers=1)
    original = {n: p.detach().clone() for n, p in model.named_parameters() if "mlp" in n}
    generator = torch.Generator().manual_seed(7)
    entry = _full_rank_layer(model.model.layers[0], generator, n_svd=30)
    install_hybrid_ffn(model, [entry])
    mlp = model.model.layers[0].mlp
    perm = torch.tensor(entry.perm)
    assert torch.equal(mlp.gate_proj.dense.weight, original["model.layers.0.mlp.gate_proj.weight"][perm][30:])
    assert torch.equal(mlp.up_proj.dense.weight, original["model.layers.0.mlp.up_proj.weight"][perm][30:])
    assert torch.equal(mlp.down_proj.weight, original["model.layers.0.mlp.down_proj.weight"][:, perm])


def test_install_rejects_a_second_installation_a_bias_and_a_misfit():
    model = build_tiny_qwen3(layers=1)
    generator = torch.Generator().manual_seed(8)
    entry = _full_rank_layer(model.model.layers[0], generator)
    with pytest.raises(ValueError, match="no hybrid layer"):
        install_hybrid_ffn(model, [None])
    with pytest.raises(ValueError, match="entries"):
        install_hybrid_ffn(model, [entry, entry])
    bad = HybridLayer(perm=entry.perm, n_svd=40, gate=entry.gate, up=entry.up)
    with pytest.raises(ValueError, match="do not fit"):
        install_hybrid_ffn(model, [bad])
    install_hybrid_ffn(model, [entry])
    with pytest.raises(ValueError):
        install_hybrid_ffn(model, [entry])

    biased = build_tiny_qwen3(layers=1)
    biased.model.layers[0].mlp.up_proj.bias = nn.Parameter(torch.zeros(96))
    with pytest.raises(ValueError, match="bias"):
        install_hybrid_ffn(biased, [_full_rank_layer(biased.model.layers[0], generator)])
