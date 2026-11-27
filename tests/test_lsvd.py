"""LSVD on the tiny GQA Qwen3: full-rank reconstruction, layout, rotation, `_apply`, cache, errors."""

import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from catalogs import CATALOG  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import Calibration, QuantRule  # noqa: E402
from effeval.core.lowrank import transform_architecture  # noqa: E402
from effeval.core.preprocess import Preprocess, RandomHadamard, builtin_preprocess_presets, resolve_preprocess  # noqa: E402
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.methods.preprocess.rotation import rotation_matrix  # noqa: E402
from effeval.methods.quantization import lsvd as lsvd_method  # noqa: E402
from effeval.methods.quantization.lsvd import LSVDFFN, LSVDConfig, LSVDDecomposition  # noqa: E402
from effeval.ops import rotation as rot  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from effeval.ops.calibration import collect_input_rows  # noqa: E402
from effeval.ops.hybrid_ffn import rotated_selection_inputs  # noqa: E402
from effeval.plan import freeze  # noqa: E402
from effeval.runtime import hf  # noqa: E402
from effeval.validate import validate  # noqa: E402

pytestmark = pytest.mark.compute

LAYERS = 3
SKIP = 1


def make_windows(seed=1, count=8, length=32):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


WINDOWS = make_windows()
HELD_OUT = make_windows(seed=2, count=1)[0]
ACT = {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token", "clip_ratio": 1.0}


@pytest.fixture(autouse=True)
def _calibration_windows(monkeypatch):
    monkeypatch.setattr(lsvd_method, "calibration_windows", lambda *_: WINDOWS)


def make_config(rank_ratio=0.5, ffn=True, rules=(), weight_scope="all", ffn_options=None, **decomposition):
    return LSVDConfig(
        rules=list(rules),
        weight_scope=weight_scope,
        calibration=Calibration(dataset="wikitext2", n_samples=8, seqlen=32, seed=0),
        decomposition=LSVDDecomposition(rank_ratio=rank_ratio, **decomposition),
        ffn=LSVDFFN(enabled=ffn, **{"skip_last_layers": SKIP, **(ffn_options or {})}),
    )


def logits_of(model):
    with torch.no_grad():
        return model(HELD_OUT).logits


def decomposed(model=None, **kwargs):
    model = build_tiny_qwen3(layers=LAYERS) if model is None else model
    lsvd_method.decompose(model, None, make_config(**kwargs))
    return model


def lsvd_preprocess():
    steps = copy.deepcopy(builtin_preprocess_presets()["lsvd"].values["preprocess.steps"].value)
    return resolve_preprocess(Preprocess.model_validate({"preset": "lsvd", "steps": steps}))


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_keeping_every_value_reproduces_the_dense_logits(beta):
    model = build_tiny_qwen3(layers=LAYERS)
    dense = logits_of(model)
    with_ffn = beta != 1.0
    decomposed(model, rank_ratio=1.6, beta=beta, ffn=with_ffn, ffn_options={"rank_ratio": 2.0})
    assert torch.allclose(logits_of(model), dense, atol=1e-4)
    assert not torch.equal(logits_of(model), dense)
    if with_ffn:
        assert len(model.model.layers[0].mlp.gate_proj.svd_down.weight) == 48


def strip_layers(node):
    if isinstance(node, dict):
        return {k: strip_layers(v) for k, v in node.items() if k != "layers"}
    if isinstance(node, list):
        return [strip_layers(v) for v in node]
    return node


def test_runtime_layout_matches_the_transform_and_restricts_the_ffn_branches_to_hybrid_layers():
    model = decomposed()
    stored = builtin_catalog().architectures["Qwen3ForCausalLM"]
    config = make_config().model_dump(mode="json")
    expected = transform_architecture(stored, "lsvd", config)
    runtime = sites.architecture(model)
    assert runtime.linears == expected.linears
    assert strip_layers(runtime.graph.model_dump()) == strip_layers(expected.graph.model_dump())
    restricted = {n.id: n.layers for n in runtime.graph.nodes if n.layers is not None}
    assert restricted and all(layers == list(range(LAYERS - SKIP)) for layers in restricted.values())
    assert {"gate_svd_down", "gate_svd_up", "up_svd_down", "up_svd_up"} <= set(restricted)
    placeholder = QuantRule.model_validate({"sites": ["linear.*.input"], "quant": ACT})
    assignment, _ = sites.resolve(model, [placeholder])
    per_layer = [sum(1 for key in assignment if key[1] == layer) for layer in range(LAYERS)]
    assert per_layer == [14, 14, 10]


def test_preprocess_keeps_logits_and_leaves_the_value_up_factor_block_diagonal():
    model = decomposed()
    before = logits_of(model)
    ups = [layer.self_attn.v_proj.up.weight.detach().clone() for layer in model.model.layers]
    engine.apply(model, lsvd_preprocess(), engine.make_context(model, None, {"checkpoint": "tiny"}))
    assert torch.allclose(logits_of(model), before, atol=1e-4)
    assert not torch.equal(logits_of(model), before)
    head_dim = 32
    for layer, old in zip(model.model.layers, ups, strict=True):
        new = layer.self_attn.v_proj.up.weight
        assert not torch.equal(new, old)
        for head in range(old.shape[0] // head_dim):
            rows = slice(head * head_dim, (head + 1) * head_dim)
            outside = ~(old[rows] != 0).any(dim=0)
            assert outside.any()
            assert (new[rows][:, outside] == 0).all()


def apply_through_hf(config, preprocess):
    manifest = builtin_catalog().manifests["lsvd"]
    handle = hf.InProcess(model=build_tiny_qwen3(layers=LAYERS), tokenizer=None)
    model = SimpleNamespace(checkpoint="ckpt", revision=None)
    return hf._apply(
        handle, model, SimpleNamespace(config=config.model_dump(mode="json")), preprocess, manifest
    ).model


def preset_config(name="paper_w8a8"):
    values = builtin_catalog().presets["lsvd"][name].values
    rules = [QuantRule.model_validate(rule) for rule in values["method.config.rules"].value]
    return make_config(
        rules=rules,
        weight_scope=values["method.config.weight_scope"].value,
        qaw=values["method.config.decomposition.qaw"].value,
        k_grad_normalisation=values["method.config.decomposition.k_grad_normalisation"].value,
    )


@pytest.mark.parametrize("with_preprocess", [False, True])
def test_apply_decomposes_rotates_and_quantizes_through_the_runtime(with_preprocess):
    model = apply_through_hf(preset_config(), lsvd_preprocess() if with_preprocess else None)
    assert "q_down" in dict(model.model.layers[0].self_attn.named_children())
    assert torch.isfinite(logits_of(model)).all()


def test_released_weight_scope_skips_attention_weights_of_hybrid_layers_only():
    model = decomposed()
    rules = preset_config().rules
    hybrid = lsvd_method.hybrid_layer_ids(make_config(), LAYERS)
    assert hybrid == [0, 1]
    scoped = lsvd_method.scoped_rules(model, rules, hybrid)
    assignment, _ = sites.resolve(model, scoped)
    weights = {(key[0], key[1]) for key in assignment if key[0].endswith(".weight")}
    assert {layer for site, layer in weights if site.startswith("linear.attn.")} == {2}
    assert {layer for site, layer in weights if site.startswith("linear.mlp.")} == {0, 1, 2}
    inputs = {(key[0], key[1]) for key in assignment if key[0].endswith(".input")}
    assert {layer for site, layer in inputs if site.startswith("linear.attn.")} == {0, 1, 2}
    unscoped, _ = sites.resolve(model, rules)
    assert len(unscoped) - len(assignment) == len(hybrid) * 7


def test_the_released_skipped_layer_hadamard_drops_the_offline_fold_of_the_skipped_layers_only():
    standard = decomposed()
    released = decomposed(ffn_options={"skipped_layer_hadamard": "online_only"})
    for model in (standard, released):
        engine.apply(model, lsvd_preprocess(), engine.make_context(model, None, {"checkpoint": "tiny"}))
    lsvd_method.lsvd(released, None, make_config(ffn_options={"skipped_layer_hadamard": "online_only"}))
    for index in range(LAYERS - SKIP):
        assert torch.equal(
            standard.model.layers[index].mlp.down_proj.weight,
            released.model.layers[index].mlp.down_proj.weight,
        )
    skipped = LAYERS - 1
    unfolded = rot.fold_input(standard.model.layers[skipped].mlp.down_proj.weight, rot.hadamard(96).T)
    assert torch.equal(released.model.layers[skipped].mlp.down_proj.weight, unfolded)
    assert not torch.allclose(logits_of(standard), logits_of(released), atol=1e-3)
    unprepared = decomposed()
    with pytest.raises(ValueError, match="online Hadamard"):
        lsvd_method.lsvd(unprepared, None, make_config(ffn_options={"skipped_layer_hadamard": "online_only"}))


class Spy:
    def __init__(self, patch, name):
        self.calls = 0
        real = getattr(lsvd_method, name)

        def wrapper(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        patch.setattr(lsvd_method, name, wrapper)


def test_statistics_come_from_the_cache_unless_their_inputs_change(monkeypatch):
    gram, output, grads = (Spy(monkeypatch, n) for n in ("act_stats", "output_grad_grams", "causal_lm_grads"))
    base = build_tiny_qwen3(layers=LAYERS)

    def counts():
        return (gram.calls, output.calls, grads.calls)

    decomposed(copy.deepcopy(base), rank_ratio=0.5)
    assert counts() == (1, 1, 1)
    decomposed(copy.deepcopy(base), rank_ratio=0.8, beta=0.5)
    assert counts() == (1, 1, 1)
    decomposed(copy.deepcopy(base), rank_ratio=0.5, qaw=True)
    assert counts() == (2, 1, 2)
    decomposed(copy.deepcopy(base), rank_ratio=0.5, score_dtype="float32")
    assert counts() == (2, 1, 3)
    decomposed(build_tiny_qwen3(layers=LAYERS, seed=5), rank_ratio=0.5)
    assert counts() == (3, 2, 4)


def test_selection_inputs_equal_the_weights_and_inputs_of_the_actually_rotated_model():
    model = build_tiny_qwen3(layers=LAYERS)
    rotated = copy.deepcopy(model)
    steps = copy.deepcopy(builtin_preprocess_presets()["lsvd"].values["preprocess.steps"].value)
    steps[0]["entries"] = steps[0]["entries"][:1]
    residual_only = resolve_preprocess(Preprocess.model_validate({"steps": steps}))
    engine.apply(rotated, residual_only, engine.make_context(rotated, None, {"checkpoint": "tiny"}))

    rows = torch.arange(sum(w.numel() for w in WINDOWS))
    site = lsvd_method.MLP_SITE
    inputs = collect_input_rows(model, WINDOWS, site, rows)
    rotated_inputs = collect_input_rows(rotated, WINDOWS, site, rows)
    rotation = rotation_matrix(RandomHadamard(seed=0), 64, "residual")
    for index, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        gain = layer.post_attention_layernorm.weight
        gate, up, down, x = rotated_selection_inputs(
            mlp.gate_proj.weight,
            mlp.up_proj.weight,
            mlp.down_proj.weight,
            inputs[index],
            gain,
            rotation,
            torch.float32,
        )
        target = rotated.model.layers[index].mlp
        for ours, theirs in (
            (gate, target.gate_proj.weight),
            (up, target.up_proj.weight),
            (down, target.down_proj.weight),
            (x, rotated_inputs[index]),
        ):
            assert torch.allclose(ours, theirs, atol=1e-5, rtol=1e-4)


def test_a_projection_without_selected_values_is_an_error_naming_projection_and_layer(monkeypatch):
    real = lsvd_method.per_head_whitened_svd
    calls = []

    def zero_second_layer_keys(weight, heads, L, R):
        U, S, V = real(weight, heads, L, R)
        calls.append(1)
        return (U, S * 0, V) if len(calls) == 5 else (U, S, V)

    monkeypatch.setattr(lsvd_method, "per_head_whitened_svd", zero_second_layer_keys)
    with pytest.raises(ValueError, match="projection 1 has no selected singular value in layer 1"):
        decomposed(rank_ratio=0.3)


def test_ffn_errors_are_raised_before_any_statistic_is_computed():
    with pytest.raises(ValueError, match="leaves no hybrid layer"):
        decomposed(ffn_options={"skip_last_layers": LAYERS})
    model = build_tiny_qwen3(layers=LAYERS)
    model.model.layers[0].mlp.up_proj.bias = torch.nn.Parameter(torch.zeros(96))
    with pytest.raises(ValueError, match="has a bias"):
        decomposed(model)


GOLDEN_RUN_KEYS = [
    ("reference", "rk_109244c472f67abd73e52402292211b295795dc483fceb2b5332917eafe94f43"),
    ("reference", "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b"),
    ("main", "rk_6f157aa1b822dfaa1d0e1e7d852b1fd9573c50552ad9e8f9062b0b2a0bd7e422"),
    ("main", "rk_ff846a2b077f3088adea60653b60941c16bea6dfe5e41a61ed32a074d3b68f8b"),
]


def test_run_key_of_the_paper_preset_spec_is_frozen():
    spec = {
        "model": "qwen3-0.6b-base",
        "method": {"name": "lsvd", "preset": "paper_w8a8"},
        "preprocess": {"preset": "lsvd"},
        "field": "quantization",
    }
    plan, issues = freeze(spec, CATALOG, owner="bob", server=None, lookup=None)
    assert plan is not None, issues
    assert [(job.role, job.run_key) for job in plan.jobs] == GOLDEN_RUN_KEYS


def issues_of(spec_extra):
    spec = {
        "model": "qwen3-0.6b",
        "method": {"name": "lsvd", "preset": "paper_w8a8"},
        "field": "quantization",
        **spec_extra,
    }
    return validate(spec, CATALOG)


def test_the_preset_pair_validates_cleanly_and_without_preprocess_warns_about_the_selection_basis():
    paired = issues_of({"preprocess": {"preset": "lsvd"}})
    assert [i for i in paired if i.severity.value in ("error", "warning")] == []
    alone = issues_of({})
    assert [i.code for i in alone if i.severity.value == "warning"] == ["decomposition.selection_basis"]
    assert [i.code for i in alone if i.severity.value == "error"] == []
    for name in ("paper_noq", "all_weights_w8a8"):
        spec = {"model": "qwen3-0.6b", "method": {"name": "lsvd", "preset": name}, "field": "quantization"}
        assert [i for i in validate(spec, CATALOG) if i.severity.value == "error"] == []


def test_the_hybrid_ffn_supports_only_the_singular_value_splits_the_branch_implements():
    for beta in (0.0, 0.5):
        make_config(beta=beta)
    make_config(ffn=False, beta=1.0)
    with pytest.raises(ValidationError, match="beta"):
        make_config(beta=1.0)
    with pytest.raises(ValidationError, match="beta"):
        make_config(beta=0.25)


def test_the_selection_token_cap_is_the_released_8192():
    LSVDFFN(selection_tokens=8192)
    with pytest.raises(ValidationError):
        LSVDFFN(selection_tokens=8193)


def test_every_layer_builds_its_factorization_statistics_on_its_own_device(monkeypatch):
    model = build_tiny_qwen3(layers=LAYERS)
    hidden = model.config.hidden_size
    for name in lsvd_method.PROJECTIONS:
        projection = getattr(model.model.layers[1].self_attn, name)
        projection.weight = torch.nn.Parameter(
            torch.empty(projection.weight.shape, device="meta"), requires_grad=False
        )
    monkeypatch.setattr(lsvd_method, "default_cache", lambda: ArtifactCache(None))
    monkeypatch.setattr(lsvd_method, "weights_fingerprint", lambda _model: "fingerprint")
    monkeypatch.setattr(
        lsvd_method,
        "act_stats",
        lambda *_a, **_k: {(lsvd_method.INPUT_SITE, layer): torch.eye(hidden) for layer in range(LAYERS)},
    )
    monkeypatch.setattr(
        lsvd_method,
        "output_grad_grams",
        lambda *_a: {
            f"{layer}.{name}": torch.ones(1) for layer in range(LAYERS) for name in lsvd_method.PROJECTIONS
        },
    )
    monkeypatch.setattr(lsvd_method, "cholesky_lower", lambda gram: gram)
    monkeypatch.setattr(lsvd_method, "cholesky_upper_heads", lambda gram: (gram, gram))
    seen = []

    def record(weight, heads, chol, upper):
        seen.append((weight.device.type, chol.device.type, upper.device.type))
        if len(seen) == 6:
            raise StopIteration
        return None

    monkeypatch.setattr(lsvd_method, "per_head_whitened_svd", record)
    with pytest.raises(StopIteration):
        lsvd_method.decompose(model, None, make_config(ffn=False))
    assert seen == [("cpu",) * 3] * 3 + [("meta",) * 3] * 3


def test_the_ffn_svd_branch_does_not_depend_on_where_the_post_attention_norm_gain_is_stored(monkeypatch):
    stored = build_tiny_qwen3(layers=LAYERS)
    folded = copy.deepcopy(stored)
    for layer in folded.model.layers:
        gain = layer.post_attention_layernorm.weight.data
        assert gain.std() > 0.05
        layer.mlp.gate_proj.weight.data *= gain
        layer.mlp.up_proj.weight.data *= gain
        gain.fill_(1.0)
    monkeypatch.setattr(lsvd_method, "svd_aware_order", lambda *_a: list(range(0, 96, 2)))
    options = {"damping": 0.5}
    for model in (stored, folded):
        lsvd_method.decompose(model, None, make_config(ffn_options=options))

    inputs = torch.randn(6, 64, generator=torch.Generator().manual_seed(4))
    for index in range(LAYERS - SKIP):
        gain = stored.model.layers[index].post_attention_layernorm.weight
        for name in ("gate_proj", "up_proj"):
            ours = getattr(stored.model.layers[index].mlp, name)(inputs * gain)
            theirs = getattr(folded.model.layers[index].mlp, name)(inputs)
            assert torch.allclose(ours, theirs, atol=1e-4), (index, name)
