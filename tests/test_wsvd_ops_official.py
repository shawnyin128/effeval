"""Bit-for-bit equivalence of `effeval.ops.wsvd` with the vendored official WSVD code.

Every comparison is `torch.equal` (or exact equality of Python values) on CPU: the Stiefel SGD step, the
quantizer and STE, the dense prequantization, the budget and top-k allocation, the factor initialisation and
one local FT / QAT run (value-path Hadamard, learned rotation and its fold).
"""

import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import reference.wsvd_official as ref  # noqa: E402
from effeval.ops import lowrank as lowrank_ops  # noqa: E402
from effeval.ops import rotation as rot  # noqa: E402
from effeval.ops import wsvd as ops  # noqa: E402

pytestmark = pytest.mark.compute


def _generator(seed):
    return torch.Generator().manual_seed(seed)


def _seed_with_retraction(draws):
    """Smallest seed whose first `draws` `randint(1, 101)` draws contain a 1 (a QR retraction)."""
    for seed in range(100000):
        rng = random.Random(seed)
        if any(rng.randint(1, 101) == 1 for _ in range(draws)):
            return seed
    raise AssertionError("no seed found")


def _orthogonal_f64(n, seed):
    q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=_generator(seed)))
    return q


# ---------------------------------------------------------------------------------------------- V1.1


@pytest.mark.parametrize("size", [8, 1])
def test_stiefel_sgd_matches_official_sgdg(size):
    seed = _seed_with_retraction(1)
    assert random.Random(seed).randint(1, 101) == 1
    start = rot.random_orthogonal(size, 3).to(torch.bfloat16)
    grads = [torch.randn(size, size, generator=_generator(20 + i)).to(torch.bfloat16) for i in range(5)]
    p_ref = start.clone().requires_grad_(True)
    p_new = start.clone().requires_grad_(True)
    optimizer_ref = ref.SGDG([p_ref], lr=1.0, stiefel=True)
    rng = random.Random(seed)
    optimizer_new = ops.StiefelSGD([p_new], lr=1.0, rng=rng)
    random.seed(seed)
    for grad in grads:
        p_ref.grad = grad.clone()
        p_new.grad = grad.clone()
        optimizer_ref.step()
        optimizer_new.step()
        assert torch.equal(p_ref, p_new)
    assert size == 1 or not torch.equal(p_new, start)
    assert random.getstate() == rng.getstate()


# ---------------------------------------------------------------------------------------------- V1.2


def _official_quantizer(x):
    quantizer = ref.WeightQuantizer()
    quantizer.configure(8, perchannel=True, sym=True, mse=False)
    quantizer.find_params(x)
    return quantizer


def _quantizer_input(dtype):
    x = torch.randn(6, 16, generator=_generator(5)) * 3
    x[2] = 0.0
    x[3, 5] = -40.0
    x[4] = x[4].abs()
    x[5, 0] = 25.0
    return x.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_scale_and_ste_quantize_match_official(dtype):
    x = _quantizer_input(dtype)
    quantizer = _official_quantizer(x)
    scale = ops.row_scale(x, 8)
    assert quantizer.scale.dtype == scale.dtype
    assert torch.equal(quantizer.scale, scale)

    x_ref = x.clone().requires_grad_(True)
    x_new = x.clone().requires_grad_(True)
    y_ref = quantizer.quantize(x_ref)
    y_new = ops.ste_quantize(x_new, scale, 8)
    assert y_ref.dtype == y_new.dtype == dtype
    assert torch.equal(y_ref, y_new)
    grad = torch.randn(x.shape, generator=_generator(6)).to(dtype)
    y_ref.backward(grad)
    y_new.backward(grad)
    assert torch.equal(x_ref.grad, grad)
    assert torch.equal(x_new.grad, x_ref.grad)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ste_quantize_clips_like_official_with_a_stale_scale(dtype):
    x = _quantizer_input(dtype)
    quantizer = _official_quantizer(x)
    stale = quantizer.scale * 0.5
    quantizer.scale = stale.clone()
    ratio = x.float() / stale
    assert ratio.min() < -128 and ratio.max() > 127
    y_ref = quantizer.quantize(x.clone().requires_grad_(True))
    y_new = ops.ste_quantize(x.clone().requires_grad_(True), stale, 8)
    assert torch.equal(y_ref, y_new)
    assert torch.equal((y_new.float() / stale).round().min(), torch.tensor(-128.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_prequantize_dense_matches_dense_prequant_text(dtype):
    shapes = {"o": (12, 20), "up": (24, 10), "gate": (24, 10), "down": (10, 24)}
    linears = {}
    for i, (name, (rows, cols)) in enumerate(shapes.items()):
        linear = torch.nn.Linear(cols, rows, bias=False, dtype=dtype)
        weight = torch.randn(rows, cols, generator=_generator(30 + i)) * (1 + i)
        weight[1] = 0.0
        weight[2, 3] = -50.0
        linear.weight.data.copy_(weight)
        linears[name] = linear
    originals = {name: linear.weight.data.clone() for name, linear in linears.items()}
    layer = SimpleNamespace(
        self_attn=SimpleNamespace(o_proj=linears["o"]),
        mlp=SimpleNamespace(up_proj=linears["up"], gate_proj=linears["gate"], down_proj=linears["down"]),
    )
    ref.dense_prequant_text([layer], ref.wsvd_args(), {})
    for name, linear in linears.items():
        expected = ops.prequantize_dense(originals[name], 8)
        assert linear.weight.dtype == expected.dtype == dtype
        assert torch.equal(linear.weight.data, expected)
        assert not torch.equal(expected, originals[name])


# ---------------------------------------------------------------------------------------------- V1.3

QWEN3_SHAPES = {
    "tiny": dict(layers=3, q_heads=4, kv_heads=2, head_dim=8, hidden=16),
    "qwen3_0.6b": dict(layers=28, q_heads=16, kv_heads=8, head_dim=128, hidden=1024),
    "qwen3_4b": dict(layers=36, q_heads=32, kv_heads=8, head_dim=128, hidden=2560),
}


def _fake_attention_layers(shape, seed):
    """Layers whose q/k/v hold meta weights of the real shape plus tied bf16 `S` and `S_grad_info`."""
    config = SimpleNamespace(num_attention_heads=shape["q_heads"], num_key_value_heads=shape["kv_heads"])
    heads = {"q_proj": shape["q_heads"], "k_proj": shape["kv_heads"], "v_proj": shape["kv_heads"]}
    layers, scores = [], {name: [] for name in heads}
    generator = _generator(seed)
    for _ in range(shape["layers"]):
        projections = {}
        for name, count in heads.items():
            sigma = torch.randint(1, 6, (count, shape["head_dim"]), generator=generator).to(torch.bfloat16)
            gradient = torch.randint(0, 4, (count, shape["head_dim"]), generator=generator).to(torch.bfloat16)
            gradient[0] = 0
            projections[name] = SimpleNamespace(
                weight=torch.empty(count * shape["head_dim"], shape["hidden"], device="meta"),
                in_features=shape["hidden"],
                out_features=count * shape["head_dim"],
                svd_info={"S": sigma},
                S_grad_info=gradient,
            )
            scores[name].append(sigma.abs() * gradient.abs().pow(1.0))
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(config=config, **projections)))
    return layers, scores, heads


@pytest.mark.parametrize("ratio", [0.3, 0.5, 0.9])
@pytest.mark.parametrize("shape_name", list(QWEN3_SHAPES))
def test_budget_and_allocation_match_official(shape_name, ratio):
    shape = QWEN3_SHAPES[shape_name]
    layers, scores, heads = _fake_attention_layers(shape, seed=7)
    probe = layers[0].self_attn
    names = ["q_proj", "k_proj", "v_proj"]
    budgets = [
        ops.released_head_budget(
            getattr(probe, name).weight.numel(),
            shape["hidden"],
            getattr(probe, name).out_features,
            heads[name],
            shape["layers"],
            ratio,
        )
        for name in names
    ]
    top_indices, _, official = ref.svd_qkv_with_grad_info_text(layers, ref.wsvd_args(rank_ratio=ratio))
    for name, budget in zip(names, budgets, strict=True):
        assert len(top_indices[name]) == budget * heads[name]

    indices, floored = lowrank_ops.per_head_allocation(
        [scores[name] for name in names], budgets, False, floor_index=1
    )
    assert floored > 0
    for p, name in enumerate(names):
        for layer in range(shape["layers"]):
            for head in range(heads[name]):
                expected = official[layer][name].get(head, [1])
                assert indices[p][layer][head] == expected, (name, layer, head)


# ---------------------------------------------------------------------------------------------- V1.4


def _spd_gram(d, seed):
    x = torch.randn(4 * d, d, dtype=torch.float64, generator=_generator(seed))
    return x.T @ x


@pytest.mark.parametrize("heads,head_dim,d", [(3, 8, 24), (2, 16, 48)])
def test_init_factors_match_official(heads, head_dim, d):
    gram = _spd_gram(d, 40)
    lower, lower_inv = ops.cholesky_lower_f32(gram)
    weight = torch.randn(heads * head_dim, d, generator=_generator(41))
    selected = {0: [5, 1, 2], 2: [0, 7, 3, 4]}
    if heads == 2:
        selected = {1: [3, 0, 9, 2]}
    indices = [selected.get(h, [1]) for h in range(heads)]

    args = ref.wsvd_args()
    k_linear = SimpleNamespace(scaling_diag_matrixS=lower.clone())
    q_linear = SimpleNamespace(weight=weight, in_features=d, out_features=heads * head_dim)
    scaled, scaling, scaling_inv = ref.create_and_fuse_scalingmatrix_to_W(
        weight.view(heads, head_dim, d).float(), torch.ones(d), args, k_linear, keys="q"
    )
    dictionary = {0: {"q_proj": {h: idx for h, idx in enumerate(indices) if h in selected}}}
    u_list, s_list, v_list = ref.create_svd_decomposition(
        scaled, args, dictionary, q_linear, 0, num_heads=heads, keys="q_proj"
    )
    v_list = ref.fuse_invariance_S_to_V(
        v_list, scaling_inv, scaling, args, keys="q", layer_indices_dict=dictionary, num_heads=heads
    )
    a_ref, b_ref, r_ref = ref.create_ABlinear_and_quantizer(
        u_list, s_list, v_list, args, {0: {}}, 0, keys="q", num_heads=heads
    )

    u, s, vt = torch.linalg.svd(scaled.to(torch.float32), full_matrices=False)
    a_new, b_new = ops.init_factors(u, s, vt.transpose(1, 2), indices, lower_inv)
    for h in range(heads):
        assert a_new[h].dtype == b_new[h].dtype == ops.TRAIN_DTYPE
        assert torch.equal(a_ref[h].data, a_new[h])
        assert torch.equal(b_ref[h].data, b_new[h])
        rank = len(indices[h])
        assert torch.equal(r_ref[h].data, rot.random_orthogonal(rank, args.seed).to(torch.bfloat16))
    assert torch.equal(lower_inv, scaling_inv)


# ---------------------------------------------------------------------------------------------- V1.5


def _problem(key, head_dim, d, ranks, seed):
    """Factors and targets of one projection: official lists (float32 SVD pieces) and effeval bf16 tensors."""
    generator = _generator(seed)
    heads = len(ranks)
    w = torch.randn(heads, head_dim, d, generator=generator).to(torch.bfloat16)
    fisher = (torch.rand(heads, head_dim, d, generator=generator) + 0.5).to(torch.bfloat16)
    u_list, s_list, v_list = [], [], []
    for rank in ranks:
        u_list.append(torch.randn(head_dim, rank, generator=generator))
        s_list.append(torch.rand(rank, generator=generator) + 0.5)
        v_list.append(torch.randn(d, rank, generator=generator))
    a0 = [u.mul(s.sqrt()).contiguous().to(torch.bfloat16) for u, s in zip(u_list, s_list, strict=True)]
    b0 = [
        v.t().mul(s.sqrt().view(-1, 1)).contiguous().to(torch.bfloat16)
        for v, s in zip(v_list, s_list, strict=True)
    ]
    return w, fisher, (u_list, s_list, v_list), a0, b0


def _run_official(key, w, fisher, svd, q_matrix, args):
    heads, _, d = w.shape
    linear = SimpleNamespace(
        weight=w.reshape(-1, d).float(),
        in_features=d,
        W_grad_info=fisher.clone(),
        svd_info_before_rot={"U": svd[0], "S": svd[1], "V": svd[2]},
    )
    layer = SimpleNamespace(self_attn=SimpleNamespace(**{key + "_proj": linear}))
    losses, _ = ref.train_projection_official(layer, 0, key, heads, args, {0: {}}, {}, {}, q_matrix)
    info = linear.svd_info
    return [a.data for a in info["U"]], [b.data for b in info["V"]], losses


def _run_effeval(key, w, fisher, a0, b0, q_matrix, settings, head_dim):
    hadamard = rot.hadamard(head_dim).to(torch.float32) if key == "v" else None
    return ops.train_projection(a0, b0, w, fisher, settings, q_matrix, hadamard)


def _assert_train_equal(key, head_dim, d, ranks, iters, steps, start, every, seed, data_seed):
    w, fisher, svd, a0, b0 = _problem(key, head_dim, d, ranks, data_seed)
    q_matrix = _orthogonal_f64(d, data_seed + 1)
    lr, lr_qat, lr_rot = 1e-2, 2e-2, 1.0
    args = ref.wsvd_args(
        localft_iters=iters,
        localft_lr=lr,
        qat_lr_UV=lr_qat,
        qat_lr_R=lr_rot,
        qat_start_iter=start,
        qat_param_update_freq=every,
        nsamples=steps,
        bs=1,
        seed=seed,
    )
    a_ref, b_ref, losses_ref = _run_official(key, w, fisher, svd, q_matrix, args)

    qat = ops.QATSettings(
        start_fraction=start,
        lr_factors=lr_qat,
        lr_rotation=lr_rot,
        scale_update_every=every,
        bits=8,
        learned_rotation=True,
    )
    settings = ops.FTSettings(iters=iters, lr=lr, steps_per_iteration=steps, seed=seed, qat=qat)
    a0_before = [a.clone() for a in a0]
    result = _run_effeval(key, w, fisher, a0, b0, q_matrix, settings, head_dim)

    assert all(torch.equal(x, y) for x, y in zip(a0, a0_before, strict=True))
    assert len(result.ft_losses) + len(result.qat_losses) == iters * steps
    assert result.qat_losses, "no QAT step ran"
    assert result.ft_losses + result.qat_losses == losses_ref
    assert all(torch.isfinite(torch.tensor(losses_ref)))
    for h in range(len(ranks)):
        assert torch.equal(a_ref[h], result.A[h]), (key, "A", h, seed)
        assert torch.equal(b_ref[h], result.B[h]), (key, "B", h, seed)
        assert not torch.equal(result.A[h], a0[h])


@pytest.mark.parametrize("key", ["q", "v"])
@pytest.mark.parametrize(
    "iters,steps,start,every",
    [
        (2, 1, 0.5, 1),
        (3, 2, 0.5, 1),
        (5, 1, 0.4, 2),
    ],
)
def test_train_projection_matches_official(key, iters, steps, start, every):
    _assert_train_equal(key, 8, 16, (4, 3), iters, steps, start, every, seed=0, data_seed=50)


@pytest.mark.parametrize("key", ["q", "v"])
def test_train_projection_matches_official_with_a_retraction(key):
    iters, steps, ranks = 3, 2, (4, 3)
    seed = _seed_with_retraction(len(ranks) * steps * 2)
    assert seed > 0
    _assert_train_equal(key, 8, 16, ranks, iters, steps, 0.5, 1, seed=seed, data_seed=60)


@pytest.mark.parametrize("seed", range(6))
def test_train_projection_matches_official_at_value_head_size(seed):
    _assert_train_equal("v", 128, 64, (64, 48), 3, 1, 0.5, 1, seed=seed, data_seed=70 + seed)
