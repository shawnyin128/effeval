"""WSVD training ops: whitening, budget, factor init, STE quantizer, Stiefel SGD, per-projection FT/QAT."""

import random

import pytest

torch = pytest.importorskip("torch")

from effeval.ops import rotation as rot  # noqa: E402
from effeval.ops.wsvd import (  # noqa: E402
    TRAIN_DTYPE,
    FTSettings,
    QATSettings,
    StiefelSGD,
    cholesky_lower_f32,
    fisher_weight,
    init_factors,
    no_tf32,
    prequantize_dense,
    released_head_budget,
    row_scale,
    ste_quantize,
    train_projection,
)

pytestmark = pytest.mark.compute


def _spd(n, seed):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(n, n, generator=generator, dtype=torch.float64)
    return a @ a.T + n * torch.eye(n, dtype=torch.float64)


def test_cholesky_lower_f32_factors_and_inverts():
    gram = _spd(16, 0)
    lower, inverse = cholesky_lower_f32(gram)
    assert lower.dtype == torch.float32 and inverse.dtype == torch.float32
    assert torch.allclose((lower.double() @ lower.double().T), gram, rtol=1e-5, atol=1e-4)
    assert torch.allclose(inverse @ lower, torch.eye(16), atol=1e-4)
    assert torch.equal(torch.triu(lower, 1), torch.zeros(16, 16))


def test_cholesky_lower_f32_rejects_indefinite_gram():
    with pytest.raises(ValueError, match="positive definite"):
        cholesky_lower_f32(-torch.eye(4))


def test_released_head_budget_hand_computed():
    # Qwen3-0.6B q_proj: 16 heads, hidden 1024, 28 layers: 2097152 / 1152 / 16 * 28 * 0.5 = 1592.9
    assert released_head_budget(2048 * 1024, 1024, 2048, 16, 28, 0.5) == 1592
    # k_proj of the same model: 8 heads, 1024 outputs
    assert released_head_budget(1024 * 1024, 1024, 1024, 8, 28, 0.9) == int(
        0.9 * (28 * (1024 * 1024 / (1024 + 128) / 8))
    )


def test_init_factors_reproduce_heads_at_full_rank():
    heads, hd, d = 2, 8, 16
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(heads, hd, d, generator=generator)
    lower, l_inv = cholesky_lower_f32(_spd(d, 2))
    U, S, Vt = torch.linalg.svd(weight @ lower, full_matrices=False)
    V = Vt.transpose(1, 2)
    indices = [list(range(hd)), list(range(hd))]
    a_list, b_list = init_factors(U, S, V, indices, l_inv)
    for h in range(heads):
        assert a_list[h].dtype == TRAIN_DTYPE and b_list[h].dtype == TRAIN_DTYPE
        assert a_list[h].shape == (hd, hd) and b_list[h].shape == (hd, d)
        error = (a_list[h].float() @ b_list[h].float() - weight[h]).abs().max()
        assert error < 1e-2 * weight[h].abs().max() * 10


def test_init_factors_selects_components_in_given_order():
    heads, hd, d = 2, 8, 16
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(heads, hd, d, generator=generator)
    lower, l_inv = cholesky_lower_f32(_spd(d, 4))
    U, S, Vt = torch.linalg.svd(weight @ lower, full_matrices=False)
    a_list, b_list = init_factors(U, S, Vt.transpose(1, 2), [[3, 0], [1]], l_inv)
    assert a_list[0].shape == (hd, 2) and b_list[0].shape == (2, d) and a_list[1].shape == (hd, 1)
    expected = (U[0][:, 3] * S[0][3].sqrt()).to(TRAIN_DTYPE)
    assert torch.equal(a_list[0][:, 0], expected)


def test_fisher_weight_keeps_dtype():
    accumulated = torch.tensor([[-8.0, 2.0]], dtype=torch.bfloat16)
    out = fisher_weight(accumulated, 4)
    assert out.dtype == torch.bfloat16
    assert torch.allclose(out.float(), torch.tensor([[2.0**0.5, 0.5**0.5]]), atol=1e-2)


def test_row_scale_is_absmax_over_maxq_with_zero_in_range():
    x = torch.tensor([[1.0, 2.0, 3.0], [-4.0, -2.0, -1.0], [0.0, 0.0, 0.0]])
    scale = row_scale(x, 8)
    assert scale.shape == (3, 1) and scale.dtype == torch.float32
    assert torch.allclose(scale.flatten(), torch.tensor([3 / 127, 4 / 127, 1e-5 / 127]))


def test_ste_quantize_levels_and_identity_gradient_including_clipped():
    scale = torch.tensor([[0.5], [0.5]])
    x = torch.tensor([[0.2, 3.0, 1000.0, -1000.0], [-64.0, 63.5, 0.3, -0.3]], requires_grad=True)
    y = ste_quantize(x, scale, 8)
    levels = y / scale
    assert torch.equal(levels, levels.round())
    assert levels.min() == -128 and levels.max() == 127
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


def test_ste_quantize_lowest_level_is_minus_maxq_plus_one():
    scale = torch.tensor([[1.0]])
    assert ste_quantize(torch.tensor([[-128.4, -200.0]]), scale, 8).tolist() == [[-128.0, -128.0]]
    assert ste_quantize(torch.tensor([[127.4, 200.0]]), scale, 8).tolist() == [[127.0, 127.0]]
    assert ste_quantize(torch.tensor([[-8.4]]), scale, 4).tolist() == [[-8.0]]


def test_ste_quantize_output_dtype_follows_input():
    x = torch.randn(3, 4).to(torch.bfloat16)
    assert ste_quantize(x, row_scale(x, 8), 8).dtype == torch.bfloat16


def test_prequantize_dense_quantizes_bf16_rounded_values_per_row():
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(6, 32, generator=generator) * torch.tensor(
        [[0.1], [1.0], [3.0], [0.01], [1.0], [2.0]]
    )
    out = prequantize_dense(weight, 8)
    assert out.dtype == weight.dtype
    rounded = weight.to(torch.bfloat16).float()
    scale = row_scale(rounded, 8)
    levels = out / scale
    assert torch.allclose(levels, levels.round(), atol=1e-3)
    assert levels.abs().max() <= 128
    assert (out - rounded).abs().max() <= scale.max() / 2 + 1e-6
    assert prequantize_dense(weight.to(torch.bfloat16), 8).dtype == torch.bfloat16


def _orthogonality_error(matrices):
    return max((m.float().T @ m.float() - torch.eye(m.shape[0])).abs().max().item() for m in matrices)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_stiefel_sgd_keeps_rotations_near_orthogonal(dtype):
    generator = torch.Generator().manual_seed(6)
    params = [torch.nn.Parameter(rot.random_orthogonal(8, seed).to(dtype)) for seed in range(3)]
    optimizer = StiefelSGD(params, lr=0.5, rng=random.Random(0))
    for _ in range(20):
        for p in params:
            p.grad = torch.randn(8, 8, generator=generator).to(dtype)
        optimizer.step()
    assert _orthogonality_error(params) < 0.1
    assert all(not torch.equal(p, rot.random_orthogonal(8, s).to(dtype)) for s, p in enumerate(params))


def test_stiefel_sgd_qr_retraction_branch_keeps_orthogonality():
    class AlwaysOne:
        def randint(self, low, high):
            return 1

    generator = torch.Generator().manual_seed(7)
    p = torch.nn.Parameter(rot.random_orthogonal(8, 1).to(torch.float32))
    optimizer = StiefelSGD([p], lr=0.3, rng=AlwaysOne())
    for _ in range(5):
        p.grad = torch.randn(8, 8, generator=generator)
        optimizer.step()
    assert _orthogonality_error([p]) < 1e-3


def test_stiefel_sgd_descends_a_procrustes_loss():
    target = rot.random_orthogonal(8, 11).float()
    p = torch.nn.Parameter(rot.random_orthogonal(8, 12).float())
    optimizer = StiefelSGD([p], lr=0.1, rng=random.Random(0))
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        loss = (p - target).pow(2).sum()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    assert _orthogonality_error([p]) < 1e-3


def test_stiefel_sgd_rejects_bad_shapes():
    with pytest.raises(ValueError, match="rows <= cols"):
        StiefelSGD([torch.nn.Parameter(torch.randn(8, 4))], lr=0.1, rng=random.Random(0))
    with pytest.raises(ValueError, match="2-D"):
        StiefelSGD([torch.nn.Parameter(torch.randn(8))], lr=0.1, rng=random.Random(0))


def _problem(heads=2, hd=8, d=16, rank=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(heads, hd, d, generator=generator).to(TRAIN_DTYPE)
    a = [torch.randn(hd, rank, generator=generator).mul(0.3).to(TRAIN_DTYPE) for _ in range(heads)]
    b = [torch.randn(rank, d, generator=generator).mul(0.3).to(TRAIN_DTYPE) for _ in range(heads)]
    weight = torch.rand(heads, hd, d, generator=generator).add(0.5).to(TRAIN_DTYPE)
    return w, a, b, weight


def _qat(**overrides):
    values = dict(
        start_fraction=0.5,
        lr_factors=1e-3,
        lr_rotation=0.5,
        scale_update_every=1,
        bits=8,
        learned_rotation=True,
    )
    values.update(overrides)
    return QATSettings(**values)


def _settings(iters, qat=None, steps=1, lr=1e-2, seed=0):
    return FTSettings(iters=iters, lr=lr, steps_per_iteration=steps, seed=seed, qat=qat)


def test_train_projection_ft_loss_decreases_and_inputs_are_untouched():
    w, a, b, weight = _problem()
    a_before, b_before = [x.clone() for x in a], [x.clone() for x in b]
    result = train_projection(a, b, w, weight, _settings(30), None, None)
    assert len(result.ft_losses) == 30 and result.qat_losses == []
    assert result.ft_losses[-1] < result.ft_losses[0]
    assert all(torch.equal(x, y) for x, y in zip(a + b, a_before + b_before, strict=True))
    assert all(x.dtype == TRAIN_DTYPE for x in result.A + result.B)
    assert result.A[0].shape == a[0].shape and result.B[0].shape == b[0].shape


def test_train_projection_without_weight_uses_plain_normalised_loss():
    w, a, b, _ = _problem()
    result = train_projection(a, b, w, None, _settings(1, lr=0.0), None, None)
    y = torch.cat([(x @ z).unsqueeze(0) for x, z in zip(a, b, strict=True)], dim=0)
    expected = ((y - w).pow(2).mean() / w.pow(2).mean()).float().item()
    assert result.ft_losses == [expected]


def test_train_projection_phase_lengths_and_steps_per_iteration():
    w, a, b, weight = _problem()
    result = train_projection(a, b, w, weight, _settings(10, _qat(), steps=2), None, None)
    assert len(result.ft_losses) == 10 and len(result.qat_losses) == 10


def test_train_projection_qat_differs_from_run_without_rotation():
    w, a, b, weight = _problem()
    rotated = train_projection(a, b, w, weight, _settings(10, _qat()), None, None)
    plain = train_projection(a, b, w, weight, _settings(10, _qat(learned_rotation=False)), None, None)
    assert any(not torch.equal(x, y) for x, y in zip(rotated.A, plain.A, strict=True))
    assert any(not torch.equal(x, y) for x, y in zip(rotated.B, plain.B, strict=True))


def test_train_projection_qat_phase_uses_qat_learning_rate_on_the_same_optimizer():
    w, a, b, weight = _problem()
    frozen_qat = _qat(lr_factors=0.0, learned_rotation=False)
    full = train_projection(a, b, w, weight, _settings(10, frozen_qat), None, None)
    ft_only = train_projection(a, b, w, weight, _settings(5), None, None)
    assert all(torch.equal(x, y) for x, y in zip(full.A + full.B, ft_only.A + ft_only.B, strict=True))
    moving = train_projection(a, b, w, weight, _settings(10, _qat(learned_rotation=False)), None, None)
    assert any(not torch.equal(x, y) for x, y in zip(moving.A, ft_only.A, strict=True))


def test_train_projection_zero_iterations_fold_the_initial_rotation():
    w, a, b, weight = _problem()
    result = train_projection(a, b, w, weight, _settings(0, _qat()), None, None)
    r0 = rot.random_orthogonal(a[0].shape[1], 0).to(TRAIN_DTYPE)
    for h in range(len(a)):
        assert torch.equal(result.A[h], (a[h].float() @ r0.float()).to(TRAIN_DTYPE))
        assert torch.equal(result.B[h], (r0.T.float() @ b[h].float()).to(TRAIN_DTYPE))
        assert not torch.equal(result.A[h], a[h])
    assert result.ft_losses == [] and result.qat_losses == []


def test_train_projection_zero_iterations_without_rotation_returns_inputs():
    w, a, b, weight = _problem()
    result = train_projection(a, b, w, weight, _settings(0, _qat(learned_rotation=False)), None, None)
    assert all(torch.equal(x, y) for x, y in zip(result.A + result.B, a + b, strict=True))


def test_train_projection_qat_target_is_rotated_by_value_hadamard_and_residual():
    heads, hd, d = 2, 8, 16
    generator = torch.Generator().manual_seed(9)
    w = torch.randn(heads, hd, d, generator=generator).to(TRAIN_DTYPE)
    a = [torch.eye(hd).to(TRAIN_DTYPE) for _ in range(heads)]
    b = [w[h].clone() for h in range(heads)]
    hadamard = rot.hadamard(hd).float()
    residual = rot.random_orthogonal(d, 0)
    qat = _qat(start_fraction=0.0, lr_factors=0.0, learned_rotation=False, bits=16)
    settings = _settings(1, qat)
    matched = train_projection(a, b, w, None, settings, residual, hadamard).qat_losses[0]
    assert matched < 1e-3
    coarse = _qat(start_fraction=0.0, lr_factors=0.0, learned_rotation=False, bits=3)
    assert train_projection(a, b, w, None, _settings(1, coarse), residual, hadamard).qat_losses[0] > 0.01


def test_train_projection_value_hadamard_is_applied_as_the_deployed_fold_output():
    heads, hd, d = 1, 12, 16
    hadamard = rot.hadamard(hd)
    assert not torch.equal(hadamard, hadamard.T)
    generator = torch.Generator().manual_seed(4)
    w = torch.randn(heads, hd, d, generator=generator).to(TRAIN_DTYPE)
    a = [torch.randn(hd, hd, generator=generator).to(TRAIN_DTYPE)]
    b = [torch.randn(hd, d, generator=generator).to(TRAIN_DTYPE)]
    qat = _qat(start_fraction=0.0, lr_factors=0.0, learned_rotation=False, bits=16)
    weight = torch.zeros(heads, hd, d).to(TRAIN_DTYPE)
    weight[:, 0] = 1
    got = train_projection(a, b, w, weight, _settings(1, qat), None, hadamard.float()).qat_losses[0]
    deployed_a = rot.fold_output(a[0], hadamard)
    deployed_w = rot.fold_output(w[0], hadamard)
    y = (weight[0] * (deployed_a @ b[0])).double()
    out = (weight[0] * deployed_w).double()
    expected = ((y - out).pow(2).mean() / out.pow(2).mean()).item()
    assert abs(got - expected) < 0.05 * expected


def test_no_tf32_sets_flags_false_inside_and_restores_after():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with no_tf32():
            assert torch.backends.cuda.matmul.allow_tf32 is False
            assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cuda.matmul.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
