"""Smoke tests of the vendored official WSVD code (`reference/wsvd_official.py`) on a tiny GQA Qwen3.

They check that the text drivers run end to end on CPU, that every q/k/v projection ends with per-head
factors of the allocated ranks, that the dense projections are quantized, and that the vendored Stiefel
optimizer keeps its matrix orthogonal. Numerical comparisons against effeval live in the equivalence tests.
"""

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import reference.wsvd_official as ref  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402

pytestmark = pytest.mark.compute

HEAD_DIM = 16
LAYERS = 2
RANK_RATIO = 0.5
PROJECTIONS = ["q_proj", "k_proj", "v_proj"]


def tiny_model():
    return build_tiny_qwen3(head_dim=HEAD_DIM, tie=False, layers=LAYERS).to(torch.float32)


def windows(count=4, length=16, seed=1):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


def expected_total_rank(linear, heads):
    """released_head_budget(...) * heads of one projection, from the formula of grad_info_utils (E4)."""
    numel, d_in, d_out = linear.weight.numel(), linear.in_features, linear.out_features
    return int(RANK_RATIO * (LAYERS * (numel / (d_in + d_out / heads) / heads))) * heads


@pytest.fixture(scope="module")
def run():
    model = tiny_model()
    layers = model.model.layers
    original = copy.deepcopy(layers)
    args = ref.wsvd_args(
        rank_ratio=RANK_RATIO,
        localft_iters=2,
        qat_start_iter=0.5,
        nsamples=4,
        bs=4,
        w_bits=8,
        is_q_headnum=True,
    )
    result = ref.wsvd_attention_text(model, layers, windows(), args)
    return model, original, result


def test_qkv_factors_have_the_allocated_ranks(run):
    model, _, result = run
    config = model.config
    allocation = result["layer_indices_dict"]
    heads = {
        "q_proj": config.num_attention_heads,
        "k_proj": config.num_key_value_heads,
        "v_proj": config.num_key_value_heads,
    }
    for name in PROJECTIONS:
        total = 0
        for idx, layer in enumerate(model.model.layers):
            info = getattr(layer.self_attn, name).svd_info
            assert info["S"] is None
            assert len(info["U"]) == len(info["V"]) == heads[name]
            for head, (a, b) in enumerate(zip(info["U"], info["V"], strict=True)):
                rank = len(allocation[idx][name].get(head, [1]))
                assert tuple(a.shape) == (HEAD_DIM, rank)
                assert tuple(b.shape) == (rank, config.hidden_size)
                assert a.dtype == b.dtype == torch.bfloat16
                assert torch.isfinite(a).all() and torch.isfinite(b).all()
                total += rank
        assert total == expected_total_rank(getattr(model.model.layers[0].self_attn, name), heads[name])


def test_svd_linear_modules_stack_the_heads(run):
    model, _, result = run
    for (idx, name), module in result["svdlinears"].items():
        info = getattr(model.model.layers[idx].self_attn, name).svd_info
        assert len(module.ALinear) == len(info["U"])
        assert module.BLinear.weight.shape[0] == sum(b.shape[0] for b in info["V"])
        assert module.BLinear.weight.shape[1] == model.config.hidden_size


def test_dense_projections_are_quantized_to_the_8_bit_grid(run):
    model, original, _ = run
    for layer, before in zip(model.model.layers, original, strict=True):
        for after_w, before_w in [
            (layer.self_attn.o_proj.weight, before.self_attn.o_proj.weight),
            (layer.mlp.up_proj.weight, before.mlp.up_proj.weight),
            (layer.mlp.gate_proj.weight, before.mlp.gate_proj.weight),
            (layer.mlp.down_proj.weight, before.mlp.down_proj.weight),
        ]:
            assert not torch.equal(after_w, before_w)
            scale = after_w.abs().amax(dim=1, keepdim=True) / 127
            steps = after_w / scale
            assert torch.allclose(steps, steps.round(), atol=1e-2)


def test_stiefel_optimizer_keeps_the_rotation_orthogonal():
    torch.manual_seed(0)
    start = ref.get_orthogonal_matrix(8, "random", 0).to(torch.bfloat16)
    rotation = torch.nn.Parameter(start.clone())
    optimizer = ref.SGDG([rotation], lr=1.0, stiefel=True)
    x = torch.randn(8, 16).to(torch.bfloat16)
    y = torch.randn(8, 16).to(torch.bfloat16)
    eye = torch.eye(8)
    for _ in range(3):
        optimizer.zero_grad()
        (rotation @ x - y).float().pow(2).sum().backward()
        optimizer.step()
        err = (rotation.detach().float().T @ rotation.detach().float() - eye).abs().max()
        assert err < 0.1
    assert (rotation.detach() - start).abs().max() > 0.01
