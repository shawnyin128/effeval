import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import reference.lsvd_official as lsvd  # noqa: E402

D_FFN = 96


def _tiny_gqa_qwen3(intermediate_size=D_FFN):
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=intermediate_size,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).to(torch.float32).eval()


def _args(**overrides):
    base = dict(
        w_bits=8,
        a_bits=8,
        k_bits=16,
        v_bits=16,
        w_asym=False,
        w_clip=True,
        w_groupsize=-1,
        a_clip_ratio=1.0,
        int8_down_proj=False,
        seed=0,
        svd_modules="qkv",
        svd_mode="U",
        is_per_head_svd=True,
        is_q_headnum=False,
        qkv_fuse=False,
        kv_fuse=False,
        had_rank=False,
        act_aware=True,
        act_alpha=0.5,
        calib_method="cholesky",
        act_calib_fake_quant=True,
        a_bits_calib=None,
        loss_aware=True,
        taylor_order=2,
        is_taylor=False,
        add_taylor_first=False,
        grad_alpha=1.0,
        label_shift=False,
        is_rank_allocate_ft=False,
        use_S_gradinfo_init=False,
        rank_ratio=0.5,
        use_true_param_ratio=True,
        use_hybrid_ffn=True,
        svd_ffn_ratio=0.2,
        svd_ffn_calib_method="svd_aware",
        hybrid_ffn_rank_ratio=0.9,
        hybrid_ffn_act_aware=True,
        hybrid_ffn_act_aware_mode="chol",
        hybrid_ffn_svd_latent_rotate=False,
        hybrid_ffn_had=False,
        hybrid_ffn_skip_layers=[2],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _windows(count=4, length=16):
    gen = torch.Generator().manual_seed(1)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


def _logits(model):
    with torch.no_grad():
        return model(_windows(1)[0]).logits


def test_official_attention_decomposition_replaces_every_qkv_with_per_head_svd_linear():
    model = _tiny_gqa_qwen3()
    layers = model.model.layers
    for table in (lsvd.total_rank_sum, lsvd.total_linear_count):
        table.update({key: 0 for key in table})
    lsvd.lsvd_attention_decompose(model, layers, _windows(), _args())
    model.eval()
    for layer in layers:
        for name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(layer.self_attn, name)
            assert isinstance(proj, lsvd.SVDLinear)
            assert proj.is_per_head_svd and len(proj.rank_split) == proj.head_num
    # slots per head: int(rank_ratio * L * hd * d / (d + hd)) = 19; an empty head falls back to rank 1
    slots = int(0.5 * (3 * 16 * 64 / (64 + 16)))
    for name, heads in (("q_proj", 4), ("k_proj", 2), ("v_proj", 2)):
        assert slots * heads <= lsvd.total_rank_sum[name] <= slots * heads + 3 * heads
        assert lsvd.total_linear_count[name] == 3
    assert torch.isfinite(_logits(model)).all()


def test_official_hybrid_ffn_phase1_groups_and_setup_with_skipped_layer():
    model = _tiny_gqa_qwen3()
    layers = model.model.layers
    windows = _windows()
    lsvd.lsvd_attention_decompose(model, layers, windows, _args())
    model.eval()
    layer_inps = lsvd._collect_layer_inputs(model, layers, windows)

    phase1_args = _args()
    groups = {"layers": {}}
    for idx, layer in enumerate(layers):
        lin = {n: getattr(layer.mlp, n).weight.data.float() for n in ("gate_proj", "up_proj", "down_proj")}
        entry = lsvd.phase1_layer_groups(
            lin["gate_proj"], lin["up_proj"], lin["down_proj"], layer_inps[idx], phase1_args, idx
        )
        assert sorted(entry["perm"]) == list(range(D_FFN))
        assert sorted(entry["svd_idx"] + entry["quant_idx"]) == list(range(D_FFN))
        assert entry["perm"] == entry["svd_idx"] + entry["quant_idx"]
        assert sorted(entry["greedy_accept_order"]) == sorted(set(entry["greedy_accept_order"]))
        groups["layers"][str(idx)] = entry

    for layer in layers:
        lsvd.add_actquant(layer.mlp)
    lsvd.hybrid_ffn_setup(
        model, _args(svd_ffn_ratio=0.5, hybrid_ffn_svd_latent_rotate=True), layer_inps, groups
    )
    model.eval()
    hybrid = [layer.mlp for layer in layers if isinstance(layer.mlp, lsvd.HybridSVDQuantFFN)]
    assert len(hybrid) == 2
    assert not isinstance(layers[2].mlp, lsvd.HybridSVDQuantFFN)
    for mlp in hybrid:
        assert mlp.n_svd == D_FFN // 2
        assert sorted(mlp.perm.tolist()) == list(range(D_FFN))
    assert torch.isfinite(_logits(model)).all()


def test_official_skipped_layer_online_hadamard_without_offline_fold_is_not_function_preserving():
    model = _tiny_gqa_qwen3(intermediate_size=128)
    layers = model.model.layers
    reference_logits = _logits(model)
    for layer in layers:
        lsvd.add_actquant(layer.mlp)
    lsvd.hybrid_ffn_setup(model, _args(hybrid_ffn_skip_layers=[0, 1, 2]), None, {"layers": {}})
    assert all(layer.mlp.down_proj.online_full_had for layer in layers)
    released = _logits(model)
    assert (released - reference_logits).abs().max() > 1e-2 * reference_logits.abs().max()

    folded_model = copy.deepcopy(model)
    for layer in folded_model.model.layers:
        weight = layer.mlp.down_proj.module.weight
        weight.data = lsvd._hadamard_transform_power2_lastdim(weight.data)
    folded = _logits(folded_model)
    assert torch.allclose(folded, reference_logits, atol=1e-4, rtol=1e-4)
