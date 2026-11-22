import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from effeval.core.fragments import QuantRule  # noqa: E402
from effeval.ops import online, rotation, scaling, sites  # noqa: E402
from reference import quarot_official as quarot  # noqa: E402
from test_sites import INT4, padded_batch, tiny_qwen3  # noqa: E402

pytestmark = pytest.mark.compute

HADAMARD_SIZES = [12, 16, 128, 1024, 2048, 3072, 2560]
BASE_SIZES = [*rotation.HADAMARD_BASES, *(2 * k for k in rotation.HADAMARD_BASES)]
LOGITS_ATOL = 1e-5


def _logits(model):
    """Logits at the non-padding positions; fully masked padding rows differ between attention backends."""
    input_ids, attention_mask = padded_batch()
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[attention_mask.bool()]


def _identity(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=torch.float64)


@pytest.mark.parametrize("n", sorted({*HADAMARD_SIZES, *BASE_SIZES}))
def test_hadamard_equals_quarot_matrix(n):
    ours = rotation.hadamard(n)
    official = quarot.matmul_hadU(_identity(n))
    assert torch.equal(ours.sign(), official.sign())
    # QuaRot divides by a float32 sqrt(n), so its entries differ from the exact 1/sqrt(n) by about 1e-9.
    assert (ours - official).abs().max() < 1e-8
    assert (ours @ ours.T - _identity(n)).abs().max() < 1e-10


def test_hadamard_factor_follows_quarot_search_order_and_rejects_unsupported_sizes():
    assert rotation.hadamard_factor(2560) == (40, 64)
    assert rotation.hadamard_factor(3072) == (12, 256)
    assert rotation.hadamard_factor(256) == (1, 256)
    for n in (9728, 17408, 44, 3):
        with pytest.raises(ValueError, match="no Hadamard matrix"):
            rotation.hadamard(n)
        with pytest.raises(AssertionError):
            quarot.get_hadK(n)


@pytest.mark.parametrize(("n", "seed"), [(64, 0), (96, 1), (3072, 0), (1024, 7)])
def test_random_hadamard_equals_quarot_after_global_seed(n, seed):
    torch.manual_seed(seed)
    official = quarot.random_hadamard_matrix(n, "cpu")
    ours = rotation.random_hadamard(n, seed)
    assert torch.equal(ours.sign(), official.sign())
    assert (ours - official).abs().max() < 1e-8
    assert (ours @ ours.T - _identity(n)).abs().max() < 1e-10
    assert not torch.equal(rotation.random_hadamard(n, seed + 1).sign(), ours.sign())


def test_random_orthogonal_is_orthogonal_seeded_and_sign_fixed():
    q = rotation.random_orthogonal(48, seed=3)
    assert (q @ q.T - _identity(48)).abs().max() < 1e-10
    assert torch.equal(q, rotation.random_orthogonal(48, seed=3))
    assert not torch.equal(q, rotation.random_orthogonal(48, seed=4))
    torch.manual_seed(3)
    reference = torch.linalg.qr(torch.randn(48, 48, dtype=torch.float64))
    expected = reference.Q * torch.sign(torch.diag(reference.R)).unsqueeze(0)
    assert torch.equal(q, expected)


def test_block_hadamard_is_block_diagonal_and_rejects_small_blocks():
    n = 9728
    assert n & -n == 512
    block = rotation.block_hadamard(n)
    assert block.shape == (n, n)
    assert torch.equal(block[:512, :512], rotation.hadamard(512))
    assert block[:512, 512:].abs().max() == 0
    assert (block[:1024] @ block[:1024].T - _identity(1024)).abs().max() < 1e-10
    with pytest.raises(ValueError, match="below the minimum block"):
        rotation.block_hadamard(12)


@pytest.mark.parametrize(("heads", "head_dim"), [(16, 128), (12, 64), (4, 16)])
def test_kron_hadamard_equals_full_hadamard_for_power_of_two_head_dim(heads, head_dim):
    assert torch.equal(rotation.kron_hadamard(heads, head_dim), rotation.hadamard(heads * head_dim))


def test_fold_helpers_equal_quarot_apply_exact_had_to_linear():
    torch.manual_seed(0)
    linear = torch.nn.Linear(96, 64, bias=False)
    weight = linear.weight.detach().clone()

    quarot.apply_exact_had_to_linear(linear, had_dim=-1, output=False)
    assert torch.allclose(linear.weight, rotation.fold_input(weight, rotation.hadamard(96)), atol=1e-6)

    linear.weight.data = weight.clone()
    quarot.apply_exact_had_to_linear(linear, had_dim=-1, output=True)
    assert torch.allclose(linear.weight, rotation.fold_output(weight, rotation.hadamard(64)), atol=1e-6)

    linear.weight.data = weight.clone()
    quarot.apply_exact_had_to_linear(linear, had_dim=16, output=True)
    expected = rotation.fold_output_per_head(weight, rotation.hadamard(16), head_dim=16)
    assert torch.allclose(linear.weight, expected, atol=1e-6)


def test_fold_conventions_preserve_a_linear_product():
    torch.manual_seed(1)
    x = torch.randn(5, 96, dtype=torch.float64)
    weight = torch.randn(40, 96, dtype=torch.float64)
    rot = rotation.random_hadamard(96, seed=2)
    assert not torch.allclose(rot, rot.T)
    assert torch.allclose((x @ rot) @ rotation.fold_input(weight, rot).T, x @ weight.T)
    producer_96 = torch.randn(96, 40, dtype=torch.float64)
    x_40 = torch.randn(5, 40, dtype=torch.float64)
    assert torch.allclose(x_40 @ rotation.fold_output(producer_96, rot).T, (x_40 @ producer_96.T) @ rot)

    head_rot = rotation.random_hadamard(16, seed=3)
    rotated_x = (x.reshape(5, 6, 16) @ head_rot).reshape(5, 96)
    folded_in = rotation.fold_input_per_head(weight, head_rot, head_dim=16)
    assert torch.allclose(rotated_x @ folded_in.T, x @ weight.T)
    producer = torch.randn(48, 96, dtype=torch.float64)
    folded_out = rotation.fold_output_per_head(producer, head_rot, head_dim=16)
    expected = (x @ producer.T).reshape(5, 3, 16) @ head_rot
    assert torch.allclose(x @ folded_out.T, expected.reshape(5, 48))


@torch.no_grad()
def test_online_hadamard_on_down_proj_with_offline_fold_keeps_logits():
    model = tiny_qwen3()
    base = _logits(model)
    rot = rotation.random_hadamard(96, seed=1)
    for layer in model.model.layers:
        down = layer.mlp.down_proj
        down.weight.copy_(rotation.fold_input(down.weight, rot))
    op = online.OnlineOp("linear.mlp.down_proj.input", online.RightMultiply(rot))
    handle = online.register(model, [op])
    got = _logits(model)
    print("down_proj online max abs logit diff", (got - base).abs().max().item())
    assert torch.allclose(got, base, atol=LOGITS_ATOL)
    handle.remove()
    assert not torch.allclose(_logits(model), base, atol=1e-3)


@torch.no_grad()
def test_quantizer_sees_the_online_transformed_tensor(monkeypatch):
    seen: dict[str, list[torch.Tensor]] = {"base": [], "rotated": []}
    current = {"key": "base"}

    def recording_quantize(x, fmt, kind):
        seen[current["key"]].append(x.clone())
        return x

    monkeypatch.setattr(sites, "fake_quantize", recording_quantize)
    rule = QuantRule(sites=["linear.mlp.down_proj.input"], layers="all", quant=INT4)
    rot = rotation.random_hadamard(96, seed=1)

    model = tiny_qwen3()
    base_attachment = sites.attach(model, [rule])
    _logits(model)
    base_attachment.remove()

    current["key"] = "rotated"
    for layer in model.model.layers:
        layer.mlp.down_proj.weight.copy_(rotation.fold_input(layer.mlp.down_proj.weight, rot))
    ops = online.register(model, [online.OnlineOp("linear.mlp.down_proj.input", online.RightMultiply(rot))])
    sites.attach(model, [rule])
    _logits(model)
    ops.remove()

    assert len(seen["base"]) == len(seen["rotated"]) == 3
    for plain, rotated in zip(seen["base"], seen["rotated"], strict=True):
        assert (plain @ rot.float() - rotated).abs().max() < 1e-4
        assert (plain - rotated).abs().max() > 1e-2


@torch.no_grad()
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_online_qk_rotation_keeps_logits_and_restores_backend(attn):
    model = tiny_qwen3(attn)
    base = _logits(model)
    rot = rotation.hadamard(16)
    ops = [online.OnlineOp(site, online.RightMultiply(rot)) for site in ("attn.q", "attn.k")]
    handle = online.register(model, ops)
    assert model.config._attn_implementation == sites.SDPA_IMPL
    got = _logits(model)
    print("qk online max abs logit diff", (got - base).abs().max().item())
    assert torch.allclose(got, base, atol=LOGITS_ATOL)
    handle.remove()
    assert model.config._attn_implementation == attn
    assert not hasattr(model.model.layers[0].self_attn, sites.ATTN_OPS_ATTR)


@torch.no_grad()
def test_online_op_on_only_q_changes_logits():
    model = tiny_qwen3()
    base = _logits(model)
    handle = online.register(model, [online.OnlineOp("attn.q", online.RightMultiply(rotation.hadamard(16)))])
    assert not torch.allclose(_logits(model), base, atol=1e-3)
    handle.remove()


@torch.no_grad()
def test_quarot_value_output_path_keeps_logits():
    """v_proj per-head Hadamard offline, o_proj full fold offline, online H_heads (x) I before o_proj."""
    model = tiny_qwen3()
    base = _logits(model)
    heads, head_dim = 4, 16
    head_rot = rotation.hadamard(head_dim)
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.v_proj.weight.copy_(rotation.fold_output_per_head(attn.v_proj.weight, head_rot, head_dim))
        full = rotation.kron_hadamard(heads, head_dim)
        attn.o_proj.weight.copy_(rotation.fold_input(attn.o_proj.weight, full))
    mix = online.HeadMix(rotation.hadamard(heads), head_dim)
    handle = online.register(model, [online.OnlineOp("linear.attn.o_proj.input", mix)])
    got = _logits(model)
    print("v/o path max abs logit diff", (got - base).abs().max().item())
    assert torch.allclose(got, base, atol=LOGITS_ATOL)
    handle.remove()


@torch.no_grad()
def test_head_mix_with_non_symmetric_matrix_equals_kron_with_identity():
    model = tiny_qwen3()
    base = _logits(model)
    mix_matrix = rotation.random_orthogonal(4, seed=9)
    assert not torch.allclose(mix_matrix, mix_matrix.T)
    full = torch.kron(mix_matrix, _identity(16))
    for layer in model.model.layers:
        o_proj = layer.self_attn.o_proj
        o_proj.weight.copy_(rotation.fold_input(o_proj.weight, full))
    handle = online.register(
        model, [online.OnlineOp("linear.attn.o_proj.input", online.HeadMix(mix_matrix, head_dim=16))]
    )
    assert torch.allclose(_logits(model), base, atol=LOGITS_ATOL)
    handle.remove()


@torch.no_grad()
def test_online_v_operand_with_per_head_fold_keeps_logits():
    model = tiny_qwen3()
    base = _logits(model)
    rot = rotation.random_hadamard(16, seed=5)
    for layer in model.model.layers:
        o_proj = layer.self_attn.o_proj
        o_proj.weight.copy_(rotation.fold_input_per_head(o_proj.weight, rot, 16))
    handle = online.register(model, [online.OnlineOp("attn.v", online.RightMultiply(rot))])
    assert torch.allclose(_logits(model), base, atol=LOGITS_ATOL)
    handle.remove()


@torch.no_grad()
def test_online_op_layer_selection_and_unmatched_site():
    model = tiny_qwen3()
    calls: list[int] = []

    def counting(x):
        calls.append(1)
        return x

    handle = online.register(model, [online.OnlineOp("linear.mlp.down_proj.input", counting, layers=[1])])
    _logits(model)
    assert len(calls) == 1
    handle.remove()
    with pytest.raises(sites.SiteError, match="matches nothing"):
        online.register(model, [online.OnlineOp("lm_head.input", lambda x: x)])


@torch.no_grad()
def test_scaling_into_norm_gain_and_consumer_columns_keeps_logits():
    model = tiny_qwen3()
    base = _logits(model)
    generator = torch.Generator().manual_seed(0)
    for layer in model.model.layers:
        s = torch.rand(64, generator=generator) + 0.5
        norm = layer.post_attention_layernorm
        norm.weight.copy_(scaling.fold_into_norm_gain(norm.weight, s))
        for proj in (layer.mlp.gate_proj, layer.mlp.up_proj):
            proj.weight.copy_(scaling.scale_input_columns(proj.weight, s))
    got = _logits(model)
    print("norm-gain scaling max abs logit diff", (got - base).abs().max().item())
    assert torch.allclose(got, base, atol=LOGITS_ATOL)


@torch.no_grad()
def test_scaling_producer_rows_and_online_division_keep_logits():
    model = tiny_qwen3()
    base = _logits(model)
    generator = torch.Generator().manual_seed(1)
    for layer in model.model.layers:
        s = torch.rand(96, generator=generator) + 0.5
        layer.mlp.up_proj.weight.copy_(scaling.scale_output_rows(layer.mlp.up_proj.weight, s))
        layer.mlp.down_proj.weight.copy_(scaling.scale_input_columns(layer.mlp.down_proj.weight, s))
    assert torch.allclose(_logits(model), base, atol=LOGITS_ATOL)

    model = tiny_qwen3()
    scale = torch.rand(64, generator=generator) + 0.5
    for layer in model.model.layers:
        layer.self_attn.o_proj.weight.copy_(scaling.scale_input_columns(layer.self_attn.o_proj.weight, scale))
    op = online.OnlineOp("linear.attn.o_proj.input", scaling.ChannelDivide(scale))
    handle = online.register(model, [op])
    assert torch.allclose(_logits(model), base, atol=LOGITS_ATOL)
    handle.remove()


def test_hadamard_scale_is_one_over_sqrt_n():
    assert rotation.hadamard(1024).abs().max() == pytest.approx(1 / math.sqrt(1024))


@torch.no_grad()
def test_attention_quantizer_sees_the_key_after_its_online_op(monkeypatch):
    seen: list[torch.Tensor] = []

    def recording_quantize(x, fmt, kind):
        seen.append(x.clone())
        return x

    monkeypatch.setattr(sites, "fake_quantize", recording_quantize)
    rule = QuantRule(sites=["attn.k"], layers="all", quant=INT4)
    rot = rotation.random_hadamard(16, seed=2)
    model = tiny_qwen3()

    handle = sites.attach(model, [rule])
    _logits(model)
    handle.remove()
    plain = seen.copy()
    seen.clear()

    ops = online.register(model, [online.OnlineOp("attn.k", online.RightMultiply(rot))])
    handle = sites.attach(model, [rule])
    _logits(model)
    handle.remove()
    ops.remove()

    assert len(plain) == len(seen) == len(model.model.layers)
    # Only layer 0 sees the same keys in both runs: the online op changes every later layer's input.
    assert (plain[0] @ rot.float() - seen[0]).abs().max() < 1e-4
    assert (plain[0] - seen[0]).abs().max() > 1e-2


def test_per_head_block_folds_with_equal_matrices_equal_the_shared_folds_bit_for_bit():
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(6, 8, dtype=torch.float64, generator=generator)
    matrix = rotation.random_orthogonal(2, seed=1)
    stacked = matrix[None].expand(4, 2, 2)
    assert torch.equal(
        rotation.fold_input_per_head_blocks(weight, stacked), rotation.fold_input_per_head(weight, matrix, 2)
    )
    producer = torch.randn(8, 6, dtype=torch.float64, generator=generator)
    assert torch.equal(
        rotation.fold_output_per_head_blocks(producer, stacked),
        rotation.fold_output_per_head(producer, matrix, 2),
    )


def test_per_head_block_folds_with_distinct_matrices_match_a_block_loop_and_check_sizes():
    generator = torch.Generator().manual_seed(0)
    matrices = torch.stack([rotation.random_orthogonal(4, seed=s) for s in range(3)])
    weight = torch.randn(5, 12, dtype=torch.float64, generator=generator)
    expected = torch.cat([weight[:, i * 4 : (i + 1) * 4] @ matrices[i] for i in range(3)], dim=1)
    assert torch.allclose(rotation.fold_input_per_head_blocks(weight, matrices), expected, atol=1e-14)
    producer = torch.randn(12, 5, dtype=torch.float64, generator=generator)
    expected = torch.cat([matrices[i].T @ producer[i * 4 : (i + 1) * 4] for i in range(3)], dim=0)
    assert torch.allclose(rotation.fold_output_per_head_blocks(producer, matrices), expected, atol=1e-14)
    assert not torch.allclose(
        rotation.fold_output_per_head_blocks(producer, matrices),
        rotation.fold_output_per_head(producer, matrices[0], 4),
        atol=1e-3,
    )
    with pytest.raises(ValueError, match="input columns"):
        rotation.fold_input_per_head_blocks(weight, matrices[:2])
    with pytest.raises(ValueError, match="output rows"):
        rotation.fold_output_per_head_blocks(producer, matrices[:2])
