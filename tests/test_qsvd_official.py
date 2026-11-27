"""effeval's QSVD pieces against the vendored official code (`reference/qsvd_official.py`), tensor by tensor.

Both paths run on deepcopies of one tiny MHA Qwen3 and the same calibration windows; every comparison is
`torch.equal` unless stated.
"""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import reference.qsvd_official as ref  # noqa: E402
from conftest import _randomize_norm_gains  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import Calibration, QuantRule  # noqa: E402
from effeval.methods.quantization import qsvd as qsvd_method  # noqa: E402
from effeval.methods.quantization.qsvd import QSVDConfig, QSVDDecomposition  # noqa: E402
from effeval.ops import artifacts  # noqa: E402
from effeval.ops.calibration import act_stats  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402

pytestmark = pytest.mark.compute

DTYPES = ["float32", "float16"]
RANK_RATIOS = [0.5, 1.0, 1.5, 2.0]
FACTOR_CASES = [(0.2, "0.2"), (0.0, "0.0"), (1.0, "1.0"), (0.5, "UV"), (0.5, "0.5")]
STAT_SITE = "linear.attn.k_proj.input"


def tiny_mha(dtype_name):
    """Qwen3 with hidden 64, 4 heads, 4 kv heads (MHA), head_dim 16, 3 untied layers and random norm gains."""
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
    _randomize_norm_gains(model, 1)
    return model.to(getattr(torch, dtype_name))


def windows(seed=1, count=8, length=32):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


WINDOWS = windows()
HELD_OUT = windows(seed=2, count=1)[0]


def official_args(rank_ratio, svd_mode, rotate, grad_alpha=1.0, seed=0):
    return SimpleNamespace(
        act_aware=True,
        act_alpha=0.5,
        grad_alpha=grad_alpha,
        rank_ratio=rank_ratio,
        svd_mode=svd_mode,
        had_rank=rotate,
        seed=seed,
        qkv_fuse=True,
        calib_method="abs_mean",
    )


def effeval_config(rank_ratio, beta, rotate, rules=(), grad_alpha=1.0, seed=0):
    return QSVDConfig(
        rules=list(rules),
        calibration=Calibration(dataset="wikitext2", n_samples=8, seqlen=32, seed=0),
        decomposition=QSVDDecomposition(
            rank_ratio=rank_ratio,
            beta=beta,
            latent_rotation="random_orthogonal" if rotate else "none",
            grad_alpha=grad_alpha,
            latent_seed=seed,
        ),
    )


class RecordingCache(artifacts.ArtifactCache):
    """In-memory cache keeping every `get_or_compute` result in call order (statistic, then S_grad)."""

    def __init__(self):
        super().__init__(None)
        self.results = []

    def get_or_compute(self, key, compute):
        result = super().get_or_compute(key, compute)
        self.results.append(result)
        return result


def run_official(model, args):
    """The official decomposition; returns its per-layer SVD info, S_grad and per-layer index lists."""
    captured = {}
    real = ref.svd_qkv_with_grad_info

    def spy(layers, a):
        out = real(layers, a)
        captured["indices"] = out[2]
        return out

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ref, "svd_qkv_with_grad_info", spy)
        ref.qsvd_lm_decompose(model, model.model.layers, WINDOWS, args)
    attentions = [layer.self_attn for layer in model.model.layers]
    return SimpleNamespace(
        svd=[a.qkv_svd_info for a in attentions],
        s_grad=[a.S_grad_info for a in attentions],
        indices=[captured["indices"].get(i, []) for i in range(len(attentions))],
    )


def run_effeval(model, cfg):
    """effeval's `decompose` with spies on the statistic cache, the SVD and the allocation."""
    cache, svds, picked = RecordingCache(), [], []
    real_svd, real_top = qsvd_method.joint_whitened_svd, qsvd_method.global_top_k

    def svd_spy(*args):
        svds.append(real_svd(*args))
        return svds[-1]

    def top_spy(*args):
        picked.append(real_top(*args))
        return picked[-1]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qsvd_method, "calibration_windows", lambda *_: WINDOWS)
        patch.setattr(qsvd_method, "default_cache", lambda: cache)
        patch.setattr(qsvd_method, "joint_whitened_svd", svd_spy)
        patch.setattr(qsvd_method, "global_top_k", top_spy)
        qsvd_method.decompose(model, None, cfg)
    return SimpleNamespace(stat=cache.results[0], s_grad=cache.results[1], svd=svds, indices=picked[0])


_PAIRS = {}


def pair(dtype_name, rank_ratio, beta, svd_mode, rotate, grad_alpha=1.0, seed=0):
    key = (dtype_name, rank_ratio, beta, svd_mode, rotate, grad_alpha, seed)
    if key not in _PAIRS:
        base = tiny_mha(dtype_name)
        official, ours = copy.deepcopy(base), copy.deepcopy(base)
        official_run = run_official(official, official_args(rank_ratio, svd_mode, rotate, grad_alpha, seed))
        ours_run = run_effeval(
            ours, effeval_config(rank_ratio, beta, rotate, grad_alpha=grad_alpha, seed=seed)
        )
        _PAIRS[key] = SimpleNamespace(
            official=official, official_run=official_run, ours=ours, ours_run=ours_run
        )
    return _PAIRS[key]


def default_pair(dtype_name):
    return pair(dtype_name, 1.0, 0.2, "0.2", True)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_k_proj_input_statistic_equals_official_scaling_diag_matrix(dtype_name):
    base = tiny_mha(dtype_name)
    official, ours = copy.deepcopy(base), copy.deepcopy(base)
    ref.calib_input_distribution_text(official, WINDOWS)
    stats = act_stats(ours, WINDOWS, [STAT_SITE], "window_absmean_sum")
    for index, layer in enumerate(official.model.layers):
        assert torch.equal(stats[(STAT_SITE, index)], layer.self_attn.k_proj.scaling_diag_matrix)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_whitened_svd_equals_official_qkv_svd_info(dtype_name):
    run = default_pair(dtype_name)
    for (U, S, V), info in zip(run.ours_run.svd, run.official_run.svd, strict=True):
        assert torch.equal(U, info["U"])
        assert torch.equal(S, info["S"])
        assert torch.equal(V, info["V"])


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_singular_value_gradients_equal_official_s_grad_info(dtype_name):
    run = default_pair(dtype_name)
    for index, official in enumerate(run.official_run.s_grad):
        assert torch.equal(run.ours_run.s_grad[str(index)], official)


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("rank_ratio", RANK_RATIOS)
def test_rank_allocation_equals_official_layer_indices_in_order(dtype_name, rank_ratio):
    run = pair(dtype_name, rank_ratio, 0.2, "0.2", True)
    assert run.ours_run.indices == run.official_run.indices
    assert sum(map(len, run.ours_run.indices)) == int(rank_ratio / 2 * 3 * 64)


def test_grad_alpha_changes_the_allocation_and_equals_official():
    default = default_pair("float32")
    run = pair("float32", 1.0, 0.2, "0.2", True, grad_alpha=0.0)
    assert run.ours_run.indices == run.official_run.indices
    assert run.ours_run.indices != default.ours_run.indices


def test_latent_seed_changes_the_rotation_and_equals_official():
    default = default_pair("float32")
    run = pair("float32", 1.0, 0.2, "0.2", True, seed=7)
    for ours, official in zip(run.ours.model.layers, run.official.model.layers, strict=True):
        assert torch.equal(ours.self_attn.qkv_down.weight, official.self_attn.q_proj.BLinear.weight)
    ours_default = default.ours.model.layers[0].self_attn.qkv_down.weight
    assert not torch.equal(run.ours.model.layers[0].self_attn.qkv_down.weight, ours_default)


@pytest.mark.parametrize("dtype_name", DTYPES)
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize(("beta", "svd_mode"), FACTOR_CASES)
def test_factors_equal_official_linear_weights(dtype_name, beta, svd_mode, rotate):
    run = pair(dtype_name, 1.0, beta, svd_mode, rotate)
    for ours, official in zip(run.ours.model.layers, run.official.model.layers, strict=True):
        down = ours.self_attn.qkv_down.weight
        for name in ("q_proj", "k_proj", "v_proj"):
            official_projection = getattr(official.self_attn, name)
            assert torch.equal(down, official_projection.BLinear.weight)
            assert torch.equal(getattr(ours.self_attn, name).up.weight, official_projection.ALinear.weight)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_decomposed_logits_equal_official(dtype_name):
    run = default_pair(dtype_name)
    with torch.no_grad():
        assert torch.equal(run.ours.eval()(HELD_OUT).logits, run.official.eval()(HELD_OUT).logits)


def preset_rules():
    values = builtin_catalog().presets["qsvd"]["paper_w4a4"].values
    return [QuantRule.model_validate(rule) for rule in values["method.config.rules"].value]


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
def test_activation_format_equals_official_act_quantizer(dtype_name):
    dtype = getattr(torch, dtype_name)
    activation_rule = preset_rules()[1]
    assert activation_rule.sites == ["linear.*.input"]
    gen = torch.Generator().manual_seed(3)
    x = (torch.randn(1, 32, 64, generator=gen) * 3).to(dtype)
    x[0, 5] = 0.0
    quantizer = ref.ActQuantizer()
    quantizer.configure(4, sym=True, clip_ratio=0.9)
    quantizer.find_params(x)
    expected = quantizer(x).to(dtype)
    assert torch.equal(fake_quantize(x, activation_rule.quant, kind="activation"), expected)


@pytest.mark.parametrize("dtype_name", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("shape", [(64, 40), (40, 64)])
def test_weight_format_equals_official_weight_quantizer(dtype_name, shape):
    dtype = getattr(torch, dtype_name)
    weight_rule = preset_rules()[0]
    assert weight_rule.sites == ["linear.*.weight"]
    gen = torch.Generator().manual_seed(4)
    weight = torch.randn(*shape, generator=gen).to(dtype)
    weight[:, 0] *= 8
    quantizer = ref.WeightQuantizer()
    quantizer.configure(4, perchannel=True, sym=True, mse=True)
    quantizer.find_params(weight)
    expected = quantizer.quantize(weight)
    assert torch.equal(fake_quantize(weight, weight_rule.quant, kind="weight"), expected)


@pytest.mark.parametrize("dtype_name", DTYPES)
def test_w4a4_without_rotation_gives_official_logits(dtype_name):
    rules = preset_rules()
    base = tiny_mha(dtype_name)
    official, ours = copy.deepcopy(base), copy.deepcopy(base)
    run_official(official, official_args(1.0, "0.2", True))
    layers = official.model.layers
    ref.add_actquant(layers)
    for wrapper in ref.find_qlayers(official, layers=[ref.ActQuantWrapper]).values():
        wrapper.quantizer.configure(bits=4, groupsize=-1, sym=True, clip_ratio=0.9)
    weight_args = SimpleNamespace(w_bits=4, w_groupsize=-1, w_asym=False, w_clip=True, int8_down_proj=False)
    ref.rtn_fwrd(layers, ref.DEVICE, weight_args)

    cfg = effeval_config(1.0, 0.2, True, rules)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(qsvd_method, "calibration_windows", lambda *_: WINDOWS)
        patch.setattr(qsvd_method, "default_cache", lambda: RecordingCache())
        qsvd_method.decompose(ours, None, cfg)
    qsvd_method.qsvd(ours, None, cfg)

    with torch.no_grad():
        assert torch.equal(ours.eval()(HELD_OUT).logits, official.eval()(HELD_OUT).logits)
