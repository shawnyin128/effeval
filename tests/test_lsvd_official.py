"""effeval's LSVD pieces against the vendored official code (`reference/lsvd_official.py`), tensor by tensor.

Both paths run on deepcopies of one tiny GQA Qwen3 (head_dim = hidden / heads, as the official code assumes)
and the same calibration windows; every comparison is `torch.equal` unless stated. effeval runs with
`selection_rotation: none` and one skipped layer; the post-attention norm gains are 1 so the gain-aware
damping of the hybrid FFN reduces to the released one. The official hybrid FFN collects its own MLP inputs
from the official attention-factored model.
"""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import reference.lsvd_official as ref  # noqa: E402
from conftest import _randomize_norm_gains  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import Calibration, IntFormat, QuantRule  # noqa: E402
from effeval.methods.quantization import lsvd as lsvd_method  # noqa: E402
from effeval.methods.quantization.lsvd import LSVDFFN, LSVDConfig, LSVDDecomposition  # noqa: E402
from effeval.ops import artifacts  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402

pytestmark = pytest.mark.compute

LAYERS, HIDDEN, HEAD_DIM, D_FFN = 3, 64, 16, 96
SELECTION_CAP = 8192
PROJECTIONS = ["q_proj", "k_proj", "v_proj"]
BETA_MODES = {0.0: "U", 0.5: "UV", 1.0: "V"}


def tiny_gqa(max_positions=64):
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=HIDDEN,
        intermediate_size=D_FFN,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        max_position_embeddings=max_positions,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
    _randomize_norm_gains(model, 1)
    for layer in model.model.layers:
        layer.post_attention_layernorm.weight.data.fill_(1.0)
    return model.to(torch.float32)


def windows(seed=1, count=8, length=32):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


WINDOWS = windows()
LONG_WINDOWS = windows(seed=3, count=48, length=256)
HELD_OUT = windows(seed=2, count=1)[0]


@pytest.fixture(autouse=True)
def _calibration_windows(monkeypatch):
    monkeypatch.setattr(lsvd_method, "calibration_windows", lambda *_: WINDOWS)


def official_args(rank_ratio=0.5, beta=0.0, qaw=False, joint=False, **overrides):
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
        svd_mode=BETA_MODES[beta],
        is_per_head_svd=True,
        is_q_headnum=False,
        qkv_fuse=False,
        kv_fuse=False,
        had_rank=False,
        act_aware=True,
        act_alpha=0.5,
        calib_method="cholesky",
        act_calib_fake_quant=qaw,
        a_bits_calib=None,
        loss_aware=True,
        taylor_order=2,
        is_taylor=False,
        add_taylor_first=False,
        grad_alpha=1.0,
        label_shift=False,
        is_rank_allocate_ft=False,
        use_S_gradinfo_init=False,
        rank_ratio=rank_ratio,
        use_true_param_ratio=True,
        truncate_mixed_qkv_budget=joint,
        use_hybrid_ffn=True,
        svd_ffn_ratio=0.2,
        svd_ffn_calib_method="svd_aware",
        hybrid_ffn_rank_ratio=0.9,
        hybrid_ffn_act_aware=True,
        hybrid_ffn_act_aware_mode="chol",
        hybrid_ffn_svd_latent_rotate=False,
        hybrid_ffn_had=False,
        hybrid_ffn_skip_layers=[LAYERS - 1],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def effeval_config(
    rank_ratio=0.5,
    beta=0.0,
    qaw=False,
    joint=False,
    ffn=False,
    rules=(),
    weight_scope="all",
    qaw_clip_ratio=1.0,
    ffn_options=None,
):
    return LSVDConfig(
        rules=list(rules),
        weight_scope=weight_scope,
        calibration=Calibration(dataset="wikitext2", n_samples=8, seqlen=32, seed=0),
        decomposition=LSVDDecomposition(
            rank_ratio=rank_ratio,
            beta=beta,
            qaw=qaw,
            qaw_clip_ratio=qaw_clip_ratio,
            rank_pools="joint" if joint else "per_projection",
            k_grad_normalisation="twice",
        ),
        ffn=LSVDFFN(enabled=ffn, skip_last_layers=1, selection_rotation="none", **(ffn_options or {})),
    )


class Recorder:
    """Records the arguments and results of the module-level ops `decompose` calls."""

    def __init__(self, patch, *names):
        self.calls = {name: [] for name in names}
        for name in names:
            real = getattr(lsvd_method, name)

            def wrapper(*args, _name=name, _real=real, **kwargs):
                result = _real(*args, **kwargs)
                self.calls[_name].append((args, kwargs, result))
                return result

            patch.setattr(lsvd_method, name, wrapper)

    def results(self, name):
        return [result for _args, _kwargs, result in self.calls[name]]


RECORDED = (
    "cholesky_lower",
    "cholesky_upper_heads",
    "per_head_whitened_svd",
    "per_head_allocation",
    "collect_input_rows",
    "svd_aware_order",
    "install_hybrid_ffn",
)


def official_stages(model, args):
    layers = model.model.layers
    ref.calib_input_cholesky_text(model, layers, WINDOWS, args)
    ref.calib_grad_distribution_text(model, layers, WINDOWS, args)
    ref.calib_grad_info_text(model, layers, WINDOWS, args)
    _top, _scores, layer_indices = ref.svd_qkv_with_grad_info(layers, args)
    return layer_indices


def effeval_s_grad():
    """The cached once-normalised S_grad statistic of the last `decompose`: {"<layer>.<proj>": [H, hd]}."""
    for entry in artifacts.default_cache()._memory.values():
        first = next(iter(entry.values()))
        if "0.q_proj" in entry and first.dim() == 2:
            return entry
    raise AssertionError("no S_grad entry in the artifact cache")


CASES = [
    pytest.param(0.5, 0.0, False, False, 1.0, id="r0.5-U"),
    pytest.param(0.5, 0.0, True, False, 1.0, id="r0.5-U-qaw"),
    pytest.param(0.5, 0.0, True, False, 0.7, id="r0.5-U-qaw-clip"),
    pytest.param(0.3, 0.5, False, False, 1.0, id="r0.3-UV"),
    pytest.param(1.2, 1.0, False, False, 1.0, id="r1.2-V-clamp"),
    pytest.param(0.27, 0.0, False, False, 1.0, id="r0.27-U-floor"),
    pytest.param(0.4, 0.0, False, True, 1.0, id="r0.4-U-joint-floor"),
    pytest.param(0.5, 0.5, True, True, 1.0, id="r0.5-UV-qaw-joint"),
]


@pytest.mark.parametrize(("rank_ratio", "beta", "qaw", "joint", "clip"), CASES)
def test_attention_decomposition_matches_the_official_code_stage_by_stage(
    request, monkeypatch, rank_ratio, beta, qaw, joint, clip
):
    base = tiny_gqa()
    args = official_args(rank_ratio, beta, qaw, joint, a_clip_ratio=clip)

    staged = copy.deepcopy(base)
    layer_indices = official_stages(staged, args)
    proj = [[getattr(layer.self_attn, name) for name in PROJECTIONS] for layer in staged.model.layers]

    recorder = Recorder(monkeypatch, *RECORDED)
    model = copy.deepcopy(base)
    lsvd_method.decompose(model, None, effeval_config(rank_ratio, beta, qaw, joint, qaw_clip_ratio=clip))

    # O1: input Cholesky factor of every layer (k_proj's, shared by q/k/v)
    for layer, factor in enumerate(recorder.results("cholesky_lower")):
        assert torch.equal(factor, proj[layer][1].scaling_diag_matrixS)
    # O2 and O3: output-gradient Cholesky factors, their inverses, and the per-head whitened SVDs
    uppers = recorder.results("cholesky_upper_heads")
    svds = recorder.results("per_head_whitened_svd")
    for layer in range(LAYERS):
        for p in range(3):
            official = proj[layer][p]
            upper, upper_inv = uppers[layer * 3 + p]
            assert torch.equal(upper, official.output_grad_scaling_matrix)
            assert torch.equal(upper_inv, official.svd_info["output_grad_scaling_matrix_inv"])
            for ours, theirs in zip(svds[layer * 3 + p], "USV", strict=True):
                assert torch.equal(ours, official.svd_info[theirs])
    # O4: S_grad of q and v, and the scores (which carry k's double normalisation)
    s_grad = effeval_s_grad()
    ((allocation_args, _kw, (indices, floored)),) = recorder.calls["per_head_allocation"]
    for layer in range(LAYERS):
        for p, name in enumerate(PROJECTIONS):
            official = proj[layer][p]
            if name != "k_proj":
                assert torch.equal(s_grad[f"{layer}.{name}"], official.S_grad_info)
            score = (
                official.svd_info["S"].to(torch.bfloat16).abs()
                * official.S_grad_info.to(torch.bfloat16).abs()
            )
            assert torch.equal(allocation_args[0][p][layer], score)
    # O5: the per-head index lists, in order
    for layer in range(LAYERS):
        for p, name in enumerate(PROJECTIONS):
            for head, ours in enumerate(indices[p][layer]):
                assert ours == (layer_indices[layer][name].get(head) or [0])
    if "floor" in request.node.callspec.id:
        assert floored > 0

    # O6: factors against the replaced official modules
    full = copy.deepcopy(base)
    ref.lsvd_attention_decompose(full, full.model.layers, WINDOWS, args)
    for layer, official_layer in enumerate(full.model.layers):
        attention = model.model.layers[layer].self_attn
        for name, down_name in zip(PROJECTIONS, ("q_down", "k_down", "v_down"), strict=True):
            official = getattr(official_layer.self_attn, name)
            assert torch.equal(getattr(attention, down_name).weight, official.BLinear.weight)
            up = getattr(attention, name).up.weight
            expected = torch.block_diag(*official.ALinear)
            assert torch.equal(up, expected)
    assert torch.isfinite(torch.stack([t.float().sum() for t in model.parameters()])).all()


def official_ffn_model(base, wins, args_attention, phase1, setup):
    """Official attention decomposition, MLP input collection on it, then Phase 1 and the hybrid FFN setup."""
    model = copy.deepcopy(base)
    layers = model.model.layers
    ref.lsvd_attention_decompose(model, layers, wins, args_attention)
    model.eval()
    inputs = ref._collect_layer_inputs(model, layers, wins)
    groups = {"layers": {}}
    for index in range(LAYERS - 1):
        mlp = base.model.layers[index].mlp
        groups["layers"][str(index)] = ref.phase1_layer_groups(
            mlp.gate_proj.weight.data.float(),
            mlp.up_proj.weight.data.float(),
            mlp.down_proj.weight.data.float(),
            inputs[index],
            phase1,
            index,
        )
    for layer in layers:
        ref.add_actquant(layer.self_attn)
        ref.add_actquant(layer.mlp)
    ref.hybrid_ffn_setup(model, setup, inputs, groups)
    model.eval()
    return model, groups


def weight_of(module):
    return module.module.weight if hasattr(module, "module") else module.weight


def official_weights(layer, hybrid):
    """The official tensors that effeval holds as `(attention, mlp)` weights, keyed by role."""
    out = {}
    for name in PROJECTIONS:
        module = getattr(layer.self_attn, name)
        out[f"{name}.down"] = weight_of(module.BLinear)
        out[f"{name}.up"] = torch.block_diag(*module.ALinear)
    out["o_proj"] = weight_of(layer.self_attn.o_proj)
    mlp = layer.mlp
    if hybrid:
        for name in ("gate", "up"):
            out[f"{name}.svd_up"] = weight_of(getattr(mlp, f"{name}_svd").ALinear)
            out[f"{name}.svd_down"] = weight_of(getattr(mlp, f"{name}_svd").BLinear)
            out[f"{name}.dense"] = weight_of(getattr(mlp, f"{name}_quant"))
    else:
        for name in ("gate", "up"):
            out[f"{name}.dense"] = weight_of(getattr(mlp, f"{name}_proj"))
    out["down"] = weight_of(mlp.down_proj)
    return out


def effeval_weights(layer, hybrid):
    attention, mlp = layer.self_attn, layer.mlp
    out = {}
    for name, down_name in zip(PROJECTIONS, ("q_down", "k_down", "v_down"), strict=True):
        out[f"{name}.down"] = getattr(attention, down_name).weight
        out[f"{name}.up"] = getattr(attention, name).up.weight
    out["o_proj"] = attention.o_proj.weight
    for name in ("gate", "up"):
        projection = getattr(mlp, f"{name}_proj")
        if hybrid:
            out[f"{name}.svd_up"] = projection.svd_up.weight
            out[f"{name}.svd_down"] = projection.svd_down.weight
        out[f"{name}.dense"] = projection.dense.weight
    out["down"] = mlp.down_proj.weight
    return out


def logits_of(model):
    with torch.no_grad():
        return model(HELD_OUT).logits


FFN_SCENARIOS = [
    pytest.param(WINDOWS, 0.0, True, (8, 8, 0.2), 0.9, False, id="8bit-selection-accepts-all"),
    pytest.param(WINDOWS, 0.0, True, (2, 4, 0.2), 0.9, True, id="2bit-selection-rejects-U-qaw"),
    pytest.param(WINDOWS, 0.5, False, (2, 8, 0.8), 2.0, True, id="2bit-selection-rejects-UV-full-rank"),
    pytest.param(LONG_WINDOWS, 0.0, False, (2, 8, 0.2), 0.9, True, id="long-calibration-token-subsets"),
    pytest.param(
        LONG_WINDOWS, 0.0, False, (2, 4, 0.2), 0.9, True, id="stored-selection-count-equals-requested"
    ),
]


@pytest.mark.parametrize(("wins", "beta", "qaw", "selection", "ffn_rank_ratio", "rejects"), FFN_SCENARIOS)
def test_hybrid_ffn_matches_the_official_code_from_channel_order_to_rtn_weights(
    monkeypatch, wins, beta, qaw, selection, ffn_rank_ratio, rejects
):
    monkeypatch.setattr(lsvd_method, "calibration_windows", lambda *_: wins)
    select_w_bits, select_a_bits, select_ratio = selection
    base = tiny_gqa(max_positions=256)
    recorder = Recorder(monkeypatch, *RECORDED)
    model = copy.deepcopy(base)
    options = {
        "selection_w_bits": select_w_bits,
        "selection_a_bits": select_a_bits,
        "selection_ratio": select_ratio,
        "rank_ratio": ffn_rank_ratio,
    }
    config = effeval_config(beta=beta, qaw=qaw, ffn=True, ffn_options=options)
    lsvd_method.decompose(model, None, config)
    phase1 = official_args(
        beta=beta,
        w_bits=select_w_bits,
        a_bits=select_a_bits,
        svd_ffn_ratio=select_ratio,
        hybrid_ffn_rank_ratio=ffn_rank_ratio,
    )
    setup = official_args(
        beta=beta,
        svd_ffn_ratio=0.5,
        hybrid_ffn_rank_ratio=ffn_rank_ratio,
        hybrid_ffn_svd_latent_rotate=True,
    )
    official, groups = official_ffn_model(base, wins, official_args(beta=beta, qaw=qaw), phase1, setup)

    # O7: the greedy acceptance order of every hybrid layer, with channels rejected where expected
    orders = recorder.results("svd_aware_order")
    assert len(orders) == LAYERS - 1
    for index, order in enumerate(orders):
        assert order == groups["layers"][str(index)]["greedy_accept_order"]
        assert (len(order) < D_FFN) == rejects
    if wins is LONG_WINDOWS:
        total = sum(window.numel() for window in wins)
        assert total > SELECTION_CAP and total // 4096 == 3
    # O8: permutation, dense rows, SVD factors and down_proj columns
    (installed,) = recorder.calls["install_hybrid_ffn"]
    entries = installed[0][1]
    assert entries[LAYERS - 1] is None
    for index in range(LAYERS - 1):
        mlp = official.model.layers[index].mlp
        assert entries[index].perm == mlp.perm.tolist()
        assert entries[index].n_svd == mlp.n_svd
        ours = effeval_weights(model.model.layers[index], True)
        theirs = official_weights(official.model.layers[index], True)
        for role in (
            "gate.svd_up",
            "gate.svd_down",
            "gate.dense",
            "up.svd_up",
            "up.svd_down",
            "up.dense",
            "down",
        ):
            assert torch.equal(ours[role], theirs[role]), role
    # O9: full-precision logits, then every weight after RTN in the released scope
    assert torch.allclose(logits_of(model), logits_of(official), atol=1e-5)

    ref.rtn_fwrd(official.model.layers, ref.DEVICE, official_args())
    weight_rule = [
        QuantRule.model_validate(rule)
        for rule in builtin_catalog().presets["lsvd"]["paper_w8a8"].values["method.config.rules"].value
        if rule["sites"] == ["linear.*.weight"]
    ]
    scoped = effeval_config(
        beta=beta, qaw=qaw, ffn=True, rules=weight_rule, weight_scope="released", ffn_options=options
    )
    lsvd_method.lsvd(model, None, scoped)
    for index in range(LAYERS):
        hybrid = index < LAYERS - 1
        ours = effeval_weights(model.model.layers[index], hybrid)
        theirs = official_weights(official.model.layers[index], hybrid)
        assert ours.keys() == theirs.keys()
        for role in ours:
            assert torch.equal(ours[role], theirs[role]), (index, role)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fake_quantize_equals_the_official_quantizers(dtype):
    generator = torch.Generator().manual_seed(3)
    activation = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token", clip_ratio=1.0)
    for shape in ((2, 10, 24), (4, 37)):
        x = (torch.randn(shape, generator=generator) * 3).to(dtype)
        quantizer = ref.ActQuantizer()
        quantizer.configure(bits=8, groupsize=-1, sym=True, clip_ratio=1.0)
        quantizer.find_params(x)
        assert torch.equal(fake_quantize(x, activation, "activation"), quantizer(x))
    weight = IntFormat.model_validate(
        {
            "type": "int",
            "bits": 8,
            "symmetric": True,
            "granularity": "per_channel",
            "clip_search": {"type": "mse", "grid": 100, "max_shrink": 0.8, "norm": 2.4},
        }
    )
    for shape in ((16, 11), (11, 64), (48, 64)):
        w = (torch.randn(shape, generator=generator) * 0.1).to(dtype)
        quantizer = ref.WeightQuantizer()
        quantizer.configure(8, perchannel=True, sym=True, mse=True)
        quantizer.find_params(w)
        assert torch.equal(fake_quantize(w, weight, "weight"), quantizer.quantize(w).to(dtype))
