"""QSVD on the tiny GQA Qwen3: factorisation, shapes, rotation equivalence, `_apply`, GPTQ, cache, errors."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from catalogs import CATALOG  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import Calibration, QuantRule  # noqa: E402
from effeval.core.preprocess import Preprocess, builtin_preprocess_presets, resolve_preprocess  # noqa: E402
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.methods.quantization import gptq  # noqa: E402
from effeval.methods.quantization import qsvd as qsvd_method  # noqa: E402
from effeval.methods.quantization.qsvd import QSVDConfig, QSVDDecomposition  # noqa: E402
from effeval.ops.artifacts import ArtifactCache, artifact_key  # noqa: E402
from effeval.plan import freeze  # noqa: E402
from effeval.runtime import hf  # noqa: E402

pytestmark = pytest.mark.compute

LAYERS = 2
HIDDEN = 64


def make_windows(seed=1, count=4, length=16):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


WINDOWS = make_windows()
HELD_OUT = make_windows(seed=2, count=1)[0]


@pytest.fixture(autouse=True)
def _calibration_windows(monkeypatch):
    monkeypatch.setattr(qsvd_method, "calibration_windows", lambda *_: WINDOWS)


def preset_values():
    return builtin_catalog().presets["qsvd"]["paper_w4a4"].values


def preset_rules():
    return [QuantRule.model_validate(rule) for rule in preset_values()["method.config.rules"].value]


def make_config(rank_ratio=1.0, beta=0.2, rotate=True, rules=(), **decomposition):
    return QSVDConfig(
        rules=list(rules),
        calibration=Calibration(dataset="wikitext2", n_samples=4, seqlen=16, seed=0),
        decomposition=QSVDDecomposition(
            rank_ratio=rank_ratio,
            beta=beta,
            latent_rotation="random_orthogonal" if rotate else "none",
            **decomposition,
        ),
    )


def logits_of(model):
    with torch.no_grad():
        return model(HELD_OUT).logits


def decomposed(model=None, **kwargs):
    model = build_tiny_qwen3(layers=LAYERS) if model is None else model
    qsvd_method.decompose(model, None, make_config(**kwargs))
    return model


@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("beta", [0.0, 0.2, 0.5, 1.0])
def test_keeping_every_singular_value_reproduces_the_dense_logits(beta, rotate):
    model = build_tiny_qwen3(layers=LAYERS)
    dense = logits_of(model)
    decomposed(model, rank_ratio=2.0, beta=beta, rotate=rotate)
    assert torch.allclose(logits_of(model), dense, atol=1e-4)
    assert not torch.equal(logits_of(model), dense)


@pytest.mark.parametrize("rank_ratio", [0.5, 1.0, 1.5, 2.0])
def test_factor_shapes_follow_the_grouped_query_split_and_the_budget(rank_ratio):
    model = decomposed(rank_ratio=rank_ratio)
    total = 0
    for layer in model.model.layers:
        attention = layer.self_attn
        rank = attention.qkv_down.weight.shape[0]
        total += rank
        assert attention.qkv_down.weight.shape == (rank, HIDDEN)
        assert attention.q_proj.up.weight.shape == (128, rank)
        assert attention.k_proj.up.weight.shape == (64, rank)
        assert attention.v_proj.up.weight.shape == (64, rank)
    assert total == int(rank_ratio / 2 * LAYERS * HIDDEN)


def qsvd_preprocess():
    steps = copy.deepcopy(builtin_preprocess_presets()["qsvd"].values["preprocess.steps"].value)
    return resolve_preprocess(Preprocess.model_validate({"preset": "qsvd", "steps": steps}))


def test_qsvd_preprocess_keeps_logits_and_rotates_only_what_it_should():
    model = decomposed(rank_ratio=1.0)
    before = logits_of(model)
    attentions = [layer.self_attn for layer in model.model.layers]
    down = [a.qkv_down.weight.detach().clone() for a in attentions]
    ups = {
        n: [getattr(a, n).up.weight.detach().clone() for a in attentions]
        for n in ("q_proj", "k_proj", "v_proj")
    }
    engine.apply(model, qsvd_preprocess(), engine.make_context(model, None, {"checkpoint": "tiny"}))
    assert torch.allclose(logits_of(model), before, atol=1e-4)
    assert not torch.equal(logits_of(model), before)
    for index, attention in enumerate(attentions):
        assert not torch.equal(attention.qkv_down.weight, down[index])
        assert torch.equal(attention.q_proj.up.weight, ups["q_proj"][index])
        assert torch.equal(attention.k_proj.up.weight, ups["k_proj"][index])
        assert not torch.equal(attention.v_proj.up.weight, ups["v_proj"][index])


def apply_through_hf(preprocess):
    manifest = builtin_catalog().manifests["qsvd"]
    config = make_config(rank_ratio=1.0, rules=preset_rules()).model_dump(mode="json")
    handle = hf.InProcess(model=build_tiny_qwen3(layers=LAYERS), tokenizer=None)
    model = SimpleNamespace(checkpoint="ckpt", revision=None)
    result = hf._apply(handle, model, SimpleNamespace(config=config), preprocess, manifest)
    return result.model


@pytest.mark.parametrize("with_preprocess", [False, True])
def test_apply_decomposes_rotates_and_quantizes_through_the_runtime(with_preprocess):
    model = apply_through_hf(qsvd_preprocess() if with_preprocess else None)
    assert "qkv_down" in dict(model.model.layers[0].self_attn.named_children())
    assert torch.isfinite(logits_of(model)).all()
    for layer in model.model.layers:
        quantizing = [
            module
            for module in layer.modules()
            for hook in module._forward_pre_hooks.values()
            if hasattr(hook, "_effeval_format")
        ]
        assert len(quantizing) == 8


def test_gptq_quantizes_the_down_factor_and_the_up_factors():
    model = decomposed(rank_ratio=2.0)
    rule = QuantRule.model_validate(preset_rules()[0].model_dump(mode="json"))
    cfg = gptq.GPTQConfig(
        rules=[rule],
        act_order=False,
        static_groups=False,
        percdamp=0.01,
        blocksize=32,
        true_sequential=True,
        calibration=Calibration(dataset="wikitext2", n_samples=4, seqlen=16, seed=0),
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    gptq.quantize_model(model, WINDOWS, cfg)
    attention = model.model.layers[0].self_attn
    for module in (attention.qkv_down, attention.q_proj.up, attention.k_proj.up, attention.v_proj.up):
        assert all(row.unique().numel() <= 16 for row in module.weight)
    assert not torch.equal(attention.qkv_down.weight, before["model.layers.0.self_attn.qkv_down.weight"])
    assert torch.equal(model.model.embed_tokens.weight, before["model.embed_tokens.weight"])


class Spy:
    def __init__(self, patch, name):
        self.calls = 0
        real = getattr(qsvd_method, name)

        def wrapper(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        patch.setattr(qsvd_method, name, wrapper)


def test_statistics_come_from_the_cache_unless_their_inputs_change(monkeypatch):
    stats, grads = Spy(monkeypatch, "act_stats"), Spy(monkeypatch, "causal_lm_grads")
    base = build_tiny_qwen3(layers=LAYERS)
    decomposed(copy.deepcopy(base), rank_ratio=1.0)
    assert (stats.calls, grads.calls) == (1, 1)
    decomposed(copy.deepcopy(base), rank_ratio=1.5, beta=0.5)
    assert (stats.calls, grads.calls) == (1, 1)
    decomposed(copy.deepcopy(base), rank_ratio=1.0, act_alpha=0.7)
    assert (stats.calls, grads.calls) == (1, 2)
    decomposed(build_tiny_qwen3(layers=LAYERS, seed=5), rank_ratio=1.0)
    assert (stats.calls, grads.calls) == (2, 3)


def test_a_layer_without_singular_values_is_an_error_naming_the_layer(monkeypatch):
    real = qsvd_method.joint_whitened_svd
    calls = []

    def zero_second_layer(weights, scale):
        U, S, V = real(weights, scale)
        calls.append(1)
        return (U, S * 0, V) if len(calls) == 2 else (U, S, V)

    monkeypatch.setattr(qsvd_method, "joint_whitened_svd", zero_second_layer)
    with pytest.raises(ValueError, match="layer 1 keeps no singular value"):
        decomposed(rank_ratio=0.5)


def test_clip_search_is_rejected_on_activation_sites():
    rule = QuantRule.model_validate(
        {
            "sites": ["linear.*.input"],
            "quant": {
                "type": "int",
                "bits": 4,
                "symmetric": True,
                "granularity": "per_token",
                "clip_search": {"type": "mse"},
            },
        }
    )
    with pytest.raises(ValueError, match=r"clip_search applies to '.weight' sites only"):
        make_config(rules=[rule])


GOLDEN_RUN_KEYS = [
    ("reference", "rk_109244c472f67abd73e52402292211b295795dc483fceb2b5332917eafe94f43"),
    ("reference", "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b"),
    ("main", "rk_0f9f74897876adbbd925d0a0179d688e4ac089c9d91dfe80db651e925b83484d"),
    ("main", "rk_ac41a961e0f1b8c8c37ed5f3992da46a961893cd9b803031e76dfd0369bdc065"),
]


def test_run_key_of_the_paper_preset_spec_is_frozen():
    spec = {
        "model": "qwen3-0.6b-base",
        "method": {"name": "qsvd", "preset": "paper_w4a4"},
        "preprocess": {"preset": "qsvd"},
        "field": "quantization",
    }
    plan, issues = freeze(spec, CATALOG, owner="bob", server=None, lookup=None)
    assert plan is not None, issues
    assert [(job.role, job.run_key) for job in plan.jobs] == GOLDEN_RUN_KEYS


def preprocess_identity(monkeypatch, method_name, config):
    """The `model_id` that `hf._apply` hands to the preprocessing context for a tiny Qwen3."""
    seen = []
    real = engine.make_context

    def spy(model, tokenizer, model_id, **kwargs):
        seen.append(model_id)
        return real(model, tokenizer, model_id, **kwargs)

    monkeypatch.setattr(engine, "make_context", spy)
    monkeypatch.setattr(engine, "apply", lambda *_: SimpleNamespace(artifacts={}, edits=[]))
    handle = hf.InProcess(model=build_tiny_qwen3(layers=LAYERS), tokenizer=None)
    model = SimpleNamespace(checkpoint="ckpt", revision=None)
    manifest = builtin_catalog().manifests[method_name]
    hf._apply(handle, model, SimpleNamespace(config=config), qsvd_preprocess(), manifest)
    (model_id,) = seen
    return model_id


def test_preprocess_caches_are_keyed_on_the_decomposition(monkeypatch):
    rules = [rule.model_dump(mode="json") for rule in preset_rules()]
    dense = preprocess_identity(monkeypatch, "rtn", {"rules": rules})
    factored = {
        ratio: preprocess_identity(
            monkeypatch,
            "qsvd",
            make_config(rank_ratio=ratio, rules=preset_rules()).model_dump(mode="json"),
        )
        for ratio in (1.0, 1.5)
    }
    assert dense == {"checkpoint": "ckpt", "revision": None, "dtype": "torch.float32", "device": "cpu"}

    def key(model_id):
        return artifact_key(model=model_id, calibration=None, statistic="s", upstream=[])

    assert len({key(dense), key(factored[1.0]), key(factored[1.5])}) == 3
    assert factored[1.0]["decomposition"]["method"] == "qsvd"


@pytest.mark.parametrize("rank_ratio", [0.5, 1.0, 1.5])
def test_budget_uses_the_hidden_size_when_the_key_width_differs(rank_ratio):
    model = build_tiny_qwen3(head_dim=16, layers=LAYERS)
    assert model.model.layers[0].self_attn.k_proj.out_features == 32 != HIDDEN
    decomposed(model, rank_ratio=rank_ratio)
    kept = sum(layer.self_attn.qkv_down.weight.shape[0] for layer in model.model.layers)
    assert kept == int(rank_ratio / 2 * LAYERS * HIDDEN)


def test_logged_compression_ratios_follow_the_layer_shapes(caplog):
    model = build_tiny_qwen3(head_dim=16, layers=LAYERS)
    with caplog.at_level("INFO", logger=qsvd_method.logger.name):
        decomposed(model, rank_ratio=1.0)
    ranks = [layer.self_attn.qkv_down.weight.shape[0] for layer in model.model.layers]
    q_out, k_out, v_out = 64, 32, 32
    total = q_out + k_out + v_out
    r1 = sum(r * (HIDDEN + total) for r in ranks) / (LAYERS * HIDDEN * total)
    r2 = sum(ranks) / (LAYERS * (k_out + v_out))
    (message,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("qsvd rank")]
    assert f"R1 {r1:.4f}, R2 {r2:.4f}" in message


def test_every_layer_builds_its_statistics_on_its_own_device(monkeypatch):
    model = build_tiny_qwen3(layers=LAYERS)
    model.model.layers[1].self_attn.q_proj.weight = torch.nn.Parameter(
        torch.empty(128, HIDDEN, device="meta"), requires_grad=False
    )
    monkeypatch.setattr(qsvd_method, "default_cache", lambda: ArtifactCache(None))
    monkeypatch.setattr(qsvd_method, "weights_fingerprint", lambda _model: "fingerprint")
    monkeypatch.setattr(
        qsvd_method,
        "act_stats",
        lambda *_: {(qsvd_method.STATISTIC_SITE, layer): torch.ones(HIDDEN) for layer in range(LAYERS)},
    )
    seen = []
    real_svd = qsvd_method.joint_whitened_svd

    def record(weights, scale):
        seen.append((weights[0].device.type, scale.device.type))
        if len(seen) == LAYERS:
            raise StopIteration
        return real_svd(weights, scale)

    monkeypatch.setattr(qsvd_method, "joint_whitened_svd", record)
    with pytest.raises(StopIteration):
        qsvd_method.decompose(model, None, make_config())
    assert seen == [("cpu", "cpu"), ("meta", "meta")]
