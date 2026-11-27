"""`wsvd.decompose` against the vendored official WSVD code (`reference/wsvd_official.py`), stage by stage.

Both paths run on deepcopies of one tiny GQA Qwen3 (hidden 64, 4 heads, 2 kv heads, head_dim 16, 2 layers,
float32, untied) and the same 8 calibration windows of 32 tokens; the official path runs with
`is_q_headnum=True`. Every comparison is `torch.equal` (or exact equality of Python values). The effeval
intermediates are read through recording wrappers around the functions `decompose` calls; the official ones
from the attributes and return values of the vendored stage functions.
"""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import reference.wsvd_official as ref  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import Calibration  # noqa: E402
from effeval.core.preprocess import RandomHadamard  # noqa: E402
from effeval.methods.preprocess.rotation import rotation_matrix  # noqa: E402
from effeval.methods.quantization import wsvd as wsvd_method  # noqa: E402
from effeval.methods.quantization.wsvd import WSVDQAT, WSVDConfig, WSVDDecomposition, WSVDLocalFT  # noqa: E402

pytestmark = pytest.mark.compute

LAYERS, HIDDEN, HEAD_DIM = 2, 64, 16
PROJECTIONS = ["q_proj", "k_proj", "v_proj"]
RECORDED = [
    "act_stats",
    "cholesky_lower_f32",
    "per_head_whitened_svd",
    "fisher_weight",
    "per_head_allocation",
    "init_factors",
    "train_projection",
]


def tiny_model():
    return build_tiny_qwen3(head_dim=HEAD_DIM, tie=False, layers=LAYERS).to(torch.float32)


def make_windows(seed=1, count=8, length=32):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


WINDOWS = make_windows()


def heads_of(model):
    config = model.config
    return {
        "q_proj": config.num_attention_heads,
        "k_proj": config.num_key_value_heads,
        "v_proj": config.num_key_value_heads,
    }


def make_config(windows, rank_ratio=0.5, iters=0, qat=None, decomposition=None):
    return WSVDConfig(
        calibration=Calibration(
            dataset="wikitext2", n_samples=len(windows), seqlen=windows[0].shape[1], seed=0
        ),
        decomposition=WSVDDecomposition(rank_ratio=rank_ratio, **(decomposition or {})),
        local_ft=WSVDLocalFT(iters=iters, batch_size=64, seed=0),
        qat=WSVDQAT(**(qat or {})),
    )


def run_effeval(model, windows, config):
    """Runs `decompose` on `model`; returns the recorded calls as `{name: [(args, kwargs, result)]}`."""
    calls = {name: [] for name in RECORDED}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(wsvd_method, "calibration_windows", lambda *_: windows)
        for name in RECORDED:
            real = getattr(wsvd_method, name)

            def wrapper(*args, _real=real, _log=calls[name], **kwargs):
                result = _real(*args, **kwargs)
                _log.append((args, kwargs, result))
                return result

            patch.setattr(wsvd_method, name, wrapper)
        wsvd_method.decompose(model, None, config)
    return calls


def official_args(rank_ratio=0.5, **overrides):
    return ref.wsvd_args(rank_ratio=rank_ratio, is_q_headnum=True, nsamples=len(WINDOWS), bs=64, **overrides)


def official_front(model, windows, args):
    """The stages of `wsvd_attention_text` up to the allocation, with the intermediates they leave behind.

    Returns a namespace with `grams` (inputs of every `torch.linalg.cholesky` call, three per layer: q, k, v),
    `chol` (float32 L per layer), `svd` and `s_grad` (keyed `(layer, projection)`) and `indices`.
    """
    layers = list(model.model.layers)
    args.svd_modules = "qkv"
    grams = []
    with pytest.MonkeyPatch.context() as patch:
        real = torch.linalg.cholesky

        def capture(matrix, *a, **k):
            grams.append(matrix.clone())
            return real(matrix, *a, **k)

        patch.setattr(torch.linalg, "cholesky", capture)
        ref.calib_input_cholesky_text(model, layers, windows, args)
    chol = [layer.self_attn.k_proj.scaling_diag_matrixS.clone() for layer in layers]
    ref.calib_grad_info_text(model, layers, windows, args)
    svd, s_grad = {}, {}
    for idx, layer in enumerate(layers):
        for name in PROJECTIONS:
            proj = getattr(layer.self_attn, name)
            svd[(idx, name)] = {key: value.clone() for key, value in proj.svd_info.items()}
            s_grad[(idx, name)] = proj.S_grad_info.clone()
    _, _, indices = ref.svd_qkv_with_grad_info_text(layers, args)
    return SimpleNamespace(grams=grams, chol=chol, svd=svd, s_grad=s_grad, indices=indices, layers=layers)


def whiten_stages(model, rank_ratio=0.5, decomposition=None):
    """Both fronts on deepcopies of `model`: `(official namespace, effeval recorded calls)`."""
    official = official_front(copy.deepcopy(model), WINDOWS, official_args(rank_ratio))
    effeval = run_effeval(
        copy.deepcopy(model), WINDOWS, make_config(WINDOWS, rank_ratio, decomposition=decomposition)
    )
    return official, effeval


# ---------------------------------------------------------------------------------------------- V2.1


def test_input_gram_and_float32_cholesky_factor_match_official():
    official, effeval = whiten_stages(tiny_model())
    grams = effeval["act_stats"][0][2]
    chol_calls = effeval["cholesky_lower_f32"]
    assert len(chol_calls) == LAYERS
    for layer in range(LAYERS):
        ours_gram = grams[("linear.attn.k_proj.input", layer)]
        for proj in range(3):
            assert torch.equal(official.grams[3 * layer + proj], ours_gram.double()), (layer, proj)
        lower = chol_calls[layer][2][0]
        assert lower.dtype == torch.float32
        assert torch.equal(official.chol[layer], lower), layer
    # the factors `decompose` feeds the SVD and the factor initialisation are the official ones
    for i, call in enumerate(effeval["per_head_whitened_svd"]):
        whitening = call[0][2]
        assert whitening.dtype == torch.float32
        assert torch.equal(whitening, official.chol[i // 3]), i
    for i, call in enumerate(effeval["init_factors"]):
        assert torch.equal(call[0][4], torch.linalg.inv(official.chol[i // 3])), i


# ---------------------------------------------------------------------------------------------- V2.2


def test_per_head_svd_matches_official_with_q_split_by_attention_heads():
    model = tiny_model()
    official, effeval = whiten_stages(model)
    heads = heads_of(model)
    svds = effeval["per_head_whitened_svd"]
    assert len(svds) == LAYERS * 3
    for i, ((layer, name), expected) in enumerate(official.svd.items()):
        U, S, V = svds[i][2]
        assert S.shape[0] == heads[name], (layer, name)
        assert torch.equal(expected["U"], U), (layer, name, "U")
        assert torch.equal(expected["S"], S), (layer, name, "S")
        assert torch.equal(expected["V"], V), (layer, name, "V")


# ---------------------------------------------------------------------------------------------- V2.3


def test_gradient_statistics_match_official_after_normalisation():
    official, effeval = whiten_stages(tiny_model())
    fishers = [call[2] for call in effeval["fisher_weight"]]
    assert len(fishers) == 2 * LAYERS * 3
    pools = effeval["per_head_allocation"][0][0][0]
    for i, ((layer, name), s_grad) in enumerate(official.s_grad.items()):
        p = PROJECTIONS.index(name)
        if name == "k_proj":
            # effeval normalises k once more after `fisher_weight`; its effect is visible in the score pool
            score = (
                official.svd[(layer, name)]["S"].to(torch.bfloat16).abs() * s_grad.to(torch.bfloat16).abs()
            )
            assert torch.equal(score, pools[p][layer]), (layer, name)
        else:
            assert torch.equal(s_grad, fishers[i]), (layer, name)


# ---------------------------------------------------------------------------------------------- V2.4


@pytest.mark.parametrize("rank_ratio", [0.2, 0.5])
def test_scores_and_allocation_match_official(rank_ratio):
    model = tiny_model()
    official, effeval = whiten_stages(model, rank_ratio)
    heads = heads_of(model)
    (pools, _ks, _joint), kwargs, (indices, floored) = effeval["per_head_allocation"][0]
    assert kwargs == {"floor_index": 1}
    official_floored = 0
    for p, name in enumerate(PROJECTIONS):
        for layer in range(LAYERS):
            score = (
                official.svd[(layer, name)]["S"].to(torch.bfloat16).abs()
                * official.s_grad[(layer, name)].to(torch.bfloat16).abs()
            )
            assert torch.equal(score, pools[p][layer]), (name, layer)
            kept = official.indices[layer][name]
            for head in range(heads[name]):
                official_floored += head not in kept
                assert indices[p][layer][head] == kept.get(head, [1]), (name, layer, head)
    assert floored == official_floored
    if rank_ratio == 0.2:
        assert floored >= 1


# ---------------------------------------------------------------------------------------------- V2.5


def official_factors(layers):
    """`{(layer, projection): (down [R, d], [A_h])}` from the official `svd_info`, as float32."""
    out = {}
    for idx, layer in enumerate(layers):
        for name in PROJECTIONS:
            info = getattr(layer.self_attn, name).svd_info
            out[(idx, name)] = (
                torch.cat([b.detach().float() for b in info["V"]], 0),
                [a.detach().float() for a in info["U"]],
            )
    return out


def assert_installed_factors_equal_official(model, factors):
    for idx, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        for name in PROJECTIONS:
            down, blocks = factors[(idx, name)]
            down_name = name.replace("_proj", "_down")
            assert torch.equal(getattr(attention, down_name).weight, down), (idx, name, "down")
            up = getattr(attention, name).up.weight
            row = col = 0
            for h, block in enumerate(blocks):
                rows, cols = block.shape
                assert torch.equal(up[row : row + rows, col : col + cols], block), (idx, name, h)
                row, col = row + rows, col + cols
            assert col == up.shape[1] and row == up.shape[0]


def with_effeval_rotation(monkeypatch, matrix):
    """Makes the official driver use `matrix` as its residual rotation, keeping its seeding side effects."""
    real = ref.get_orthogonal_matrix

    def patched(size, mode, seed=42, device=ref.DEVICE):
        out = real(size, mode, seed, device)
        return matrix.to(out.dtype) if mode == "hadamard" else out

    monkeypatch.setattr(ref, "get_orthogonal_matrix", patched)


def test_qat_factors_and_dense_prequant_match_official(monkeypatch):
    base = tiny_model()
    qat = {"enabled": True, "start_fraction": 0.5, "w_bits": 8, "residual_rotation": "random_hadamard"}
    ours = copy.deepcopy(base)
    calls = run_effeval(ours, WINDOWS, make_config(WINDOWS, iters=2, qat=qat))
    residual = rotation_matrix(RandomHadamard(seed=0), HIDDEN, "residual")
    assert torch.equal(calls["train_projection"][0][0][5], residual)

    theirs = copy.deepcopy(base)
    with_effeval_rotation(monkeypatch, residual)
    args = official_args(localft_iters=2, qat_start_iter=0.5, w_bits=8, seed=0)
    ref.wsvd_attention_text(theirs, list(theirs.model.layers), WINDOWS, args)

    expected = official_factors(theirs.model.layers)
    assert_installed_factors_equal_official(ours, expected)
    untrained = copy.deepcopy(base)
    run_effeval(untrained, WINDOWS, make_config(WINDOWS, iters=0, qat={"enabled": False}))
    assert not torch.equal(
        ours.model.layers[0].self_attn.q_down.weight, untrained.model.layers[0].self_attn.q_down.weight
    )
    for ours_layer, their_layer in zip(ours.model.layers, theirs.model.layers, strict=True):
        for ours_w, their_w in [
            (ours_layer.self_attn.o_proj.weight, their_layer.self_attn.o_proj.weight),
            (ours_layer.mlp.gate_proj.weight, their_layer.mlp.gate_proj.weight),
            (ours_layer.mlp.up_proj.weight, their_layer.mlp.up_proj.weight),
            (ours_layer.mlp.down_proj.weight, their_layer.mlp.down_proj.weight),
        ]:
            assert torch.equal(ours_w, their_w)
    assert not torch.equal(ours.model.layers[0].mlp.up_proj.weight, base.model.layers[0].mlp.up_proj.weight)


def test_local_ft_without_qat_matches_official():
    base = tiny_model()
    ours = copy.deepcopy(base)
    run_effeval(ours, WINDOWS, make_config(WINDOWS, iters=2, qat={"enabled": False}))

    theirs = copy.deepcopy(base)
    args = official_args(localft_iters=2, is_quant_aware_ft=False)
    front = official_front(theirs, WINDOWS, args)
    ref.calib_localft_grad_info_text(theirs, front.layers, WINDOWS, args, front.indices)
    ref.disable_grad_for_svd_modules(theirs, args)
    ref.local_ft_qkv_text(front.layers, args, front.indices, None)

    expected = official_factors(front.layers)
    assert_installed_factors_equal_official(ours, expected)
    for ours_layer, base_layer in zip(ours.model.layers, base.model.layers, strict=True):
        assert torch.equal(ours_layer.mlp.up_proj.weight, base_layer.mlp.up_proj.weight)


# ---------------------------------------------------------------------------------------------- fallback


def test_non_positive_definite_gram_follows_the_official_eigenvalue_shift():
    base = tiny_model()
    windows = make_windows(count=4, length=8)
    shifted = []
    with pytest.MonkeyPatch.context() as patch:
        real = torch.linalg.eigvalsh

        def counting(*a, **k):
            shifted.append(1)
            return real(*a, **k)

        patch.setattr(torch.linalg, "eigvalsh", counting)
        theirs = copy.deepcopy(base)
        args = official_args()
        official = official_front(theirs, windows, args)
    assert shifted, "the official eigenvalue-shift fallback did not trigger; the windows are too many"

    ours = copy.deepcopy(base)
    calls = run_effeval(ours, windows, make_config(windows))
    for layer in range(LAYERS):
        assert torch.equal(official.chol[layer], calls["cholesky_lower_f32"][layer][2][0]), layer
    (_pools, _ks, _joint), _, (indices, _floored) = calls["per_head_allocation"][0]
    heads = heads_of(base)
    for p, name in enumerate(PROJECTIONS):
        for layer in range(LAYERS):
            for head in range(heads[name]):
                assert indices[p][layer][head] == official.indices[layer][name].get(head, [1])
