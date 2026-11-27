"""WSVD on the tiny GQA Qwen3: full-rank reconstruction, GQA split, train/deploy basis, cache, dense prequant,
failures and the runtime path."""

import copy
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import Calibration, QuantRule  # noqa: E402
from effeval.core.preprocess import (  # noqa: E402
    Preprocess,
    RandomHadamard,
    builtin_preprocess_presets,
    resolve_preprocess,
)
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.methods.preprocess.rotation import rotation_matrix  # noqa: E402
from effeval.methods.quantization import wsvd as wsvd_method  # noqa: E402
from effeval.methods.quantization.wsvd import WSVDQAT, WSVDConfig, WSVDDecomposition, WSVDLocalFT  # noqa: E402
from effeval.ops import artifacts  # noqa: E402
from effeval.runtime import hf  # noqa: E402

pytestmark = pytest.mark.compute

LAYERS = 2
HEAD_DIM = 16
HIDDEN = 64
NAMES = ["q_down", "k_down", "v_down"]


def make_windows(seed=1, count=8, length=32):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


WINDOWS = make_windows()
HELD_OUT = make_windows(seed=2, count=1)[0]


@pytest.fixture(autouse=True)
def _calibration_windows(monkeypatch):
    monkeypatch.setattr(wsvd_method, "calibration_windows", lambda *_: WINDOWS)


def make_config(rules=(), decomposition=None, local_ft=None, qat=None):
    return WSVDConfig(
        rules=list(rules),
        calibration=Calibration(dataset="wikitext2", n_samples=len(WINDOWS), seqlen=32, seed=0),
        decomposition=WSVDDecomposition(**(decomposition or {})),
        local_ft=WSVDLocalFT(**{"iters": 2, **(local_ft or {})}),
        qat=WSVDQAT(**(qat or {})),
    )


def tiny_model():
    return build_tiny_qwen3(head_dim=HEAD_DIM, tie=False, layers=LAYERS)


def logits_of(model):
    with torch.no_grad():
        return model(HELD_OUT).logits


def spy(monkeypatch, module, name):
    calls = []
    real = getattr(module, name)

    def wrapper(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


def wsvd_preprocess():
    steps = copy.deepcopy(builtin_preprocess_presets()["wsvd"].values["preprocess.steps"].value)
    return resolve_preprocess(Preprocess.model_validate({"preset": "wsvd", "steps": steps}))


def test_config_validates_its_bounds_and_builds_a_schema():
    assert WSVDConfig.model_json_schema()["title"] == "WSVDConfig"
    with pytest.raises(ValidationError):
        WSVDQAT(w_bits=1)
    with pytest.raises(ValidationError):
        WSVDDecomposition(rank_ratio=0)


def test_keeping_every_value_reproduces_the_dense_logits():
    model = tiny_model()
    dense = logits_of(model)
    wsvd_method.decompose(model, None, make_config(decomposition={"rank_ratio": 5.0}, local_ft={"iters": 0}))
    assert torch.allclose(logits_of(model), dense, atol=5e-2)
    assert not torch.equal(logits_of(model), dense)


def test_heads_split_by_attention_heads_for_q_and_by_kv_heads_for_k_and_v(monkeypatch):
    installs = spy(monkeypatch, wsvd_method, "install_per_head_qkv")
    model = tiny_model()
    wsvd_method.decompose(model, None, make_config())
    factors = installs[0][0][1]
    config = model.config
    expected = {
        "q_proj": config.num_attention_heads,
        "k_proj": config.num_key_value_heads,
        "v_proj": config.num_key_value_heads,
    }
    for layer_factors in factors:
        for name, heads in expected.items():
            down, blocks = layer_factors[name]
            assert len(blocks) == heads
            assert down.shape[0] == sum(block.shape[1] for block in blocks)
    attention = model.model.layers[0].self_attn
    assert attention.q_proj.up.weight.shape[0] == config.num_attention_heads * HEAD_DIM


def test_the_deployed_down_factors_are_the_trained_ones_in_the_basis_the_qat_used(monkeypatch):
    trainings = spy(monkeypatch, wsvd_method, "train_projection")
    model = tiny_model()
    wsvd_method.decompose(model, None, make_config(qat={"enabled": True}))
    residual = trainings[0][0][5]
    layers = model.model.layers
    gains = [layer.input_layernorm.weight.detach().double().clone() for layer in layers]
    before = [
        {name: layer.self_attn.get_submodule(name).weight.detach().double().clone() for name in NAMES}
        for layer in layers
    ]
    logits = logits_of(model)
    engine.apply(model, wsvd_preprocess(), engine.make_context(model, None, {"checkpoint": "tiny"}))
    assert torch.allclose(logits_of(model), logits, atol=1e-3)
    assert torch.allclose(residual, rotation_matrix(RandomHadamard(seed=0), HIDDEN, "residual"))
    for layer, gain, old in zip(layers, gains, before, strict=True):
        for name in NAMES:
            deployed = layer.self_attn.get_submodule(name).weight.double()
            assert torch.allclose(deployed, (old[name] * gain) @ residual, atol=1e-3)


def test_the_cache_serves_a_repeat_and_a_changed_learning_rate_recomputes_only_the_factors(monkeypatch):
    grads = spy(monkeypatch, wsvd_method, "causal_lm_grads")
    base = tiny_model()
    first, second, third = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    cache = artifacts.default_cache()
    wsvd_method.decompose(first, None, make_config())
    assert (cache.computed, len(grads)) == (3, 1)
    wsvd_method.decompose(second, None, make_config())
    assert (cache.computed, len(grads)) == (3, 1)
    for (name, a), (_, b) in zip(first.state_dict().items(), second.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    wsvd_method.decompose(third, None, make_config(local_ft={"lr": 5e-4}))
    assert (cache.computed, len(grads)) == (4, 1)
    assert not torch.equal(
        first.model.layers[0].self_attn.q_down.weight, third.model.layers[0].self_attn.q_down.weight
    )


def test_the_qat_trains_in_the_basis_of_the_residual_rotation_seed_not_the_finetuning_seed(monkeypatch):
    trainings = spy(monkeypatch, wsvd_method, "train_projection")
    config = make_config(local_ft={"seed": 5}, qat={"enabled": True, "residual_rotation_seed": 3})
    wsvd_method.decompose(tiny_model(), None, config)
    residual = trainings[0][0][5]
    assert torch.allclose(residual, rotation_matrix(RandomHadamard(seed=3), HIDDEN, "residual"))
    assert not torch.allclose(residual, rotation_matrix(RandomHadamard(seed=5), HIDDEN, "residual"))


def test_the_factors_are_trained_with_tf32_disabled(monkeypatch):
    seen = []
    real = wsvd_method.train_projection

    def recording(*args, **kwargs):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return real(*args, **kwargs)

    monkeypatch.setattr(wsvd_method, "train_projection", recording)
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        wsvd_method.decompose(tiny_model(), None, make_config())
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (True, True)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = before
    assert seen and set(seen) == {(False, False)}


def recorded_keys(monkeypatch, config, device_name="cpu", torch_version=None):
    """statistic -> (key, downstream) of every artifact_key a decompose of the tiny model computes."""
    seen = {}
    real = wsvd_method.artifact_key

    def recording(**kwargs):
        key = real(**kwargs)
        seen[kwargs["statistic"]] = (key, kwargs.get("downstream"))
        return key

    monkeypatch.setattr(wsvd_method, "artifact_key", recording)
    monkeypatch.setattr(wsvd_method, "_device_name", lambda _device: device_name)
    if torch_version is not None:
        monkeypatch.setattr(torch, "__version__", torch_version)
    wsvd_method.decompose(tiny_model(), None, config)
    names = {
        "gram": wsvd_method.INPUT_GRAM_NAME,
        "grads": wsvd_method.GRAD_NAME,
        "factors": wsvd_method.FACTORS_NAME,
    }
    return {name: seen.get(statistic, (None, None)) for name, statistic in names.items()}


def key_changes(base, other):
    return {name for name in base if base[name][0] != other[name][0]}


def test_the_qat_settings_change_only_the_factor_key(monkeypatch):
    base = recorded_keys(monkeypatch, make_config())
    other = recorded_keys(monkeypatch, make_config(qat={"enabled": True}))
    assert key_changes(base, other) == {"factors"}


def test_the_qat_settings_are_in_the_factor_key_itself(monkeypatch):
    keys = recorded_keys(monkeypatch, make_config(qat={"enabled": True}))
    assert keys["factors"][1]["qat"]["enabled"] is True
    assert "dense_prequant" not in keys["factors"][1]["qat"]


def test_the_step_count_is_in_the_factor_key(monkeypatch):
    base = recorded_keys(monkeypatch, make_config())
    other = recorded_keys(monkeypatch, make_config(local_ft={"batch_size": 2}))
    assert base["factors"][1]["steps"] == 1
    assert other["factors"][1]["steps"] == len(WINDOWS) // 2


def test_the_score_dtype_changes_the_gradient_statistics_key_and_not_the_gram_key(monkeypatch):
    base = recorded_keys(monkeypatch, make_config())
    other = recorded_keys(monkeypatch, make_config(decomposition={"score_dtype": "float32"}))
    assert key_changes(base, other) == {"grads", "factors"}


def test_the_device_changes_every_key(monkeypatch):
    base = recorded_keys(monkeypatch, make_config())
    other = recorded_keys(monkeypatch, make_config(), device_name="another gpu")
    assert key_changes(base, other) == {"gram", "grads", "factors"}


def test_the_torch_version_changes_the_factor_key_only(monkeypatch):
    base = recorded_keys(monkeypatch, make_config())
    other = recorded_keys(monkeypatch, make_config(), torch_version="0.0.0")
    assert key_changes(base, other) == {"factors"}


def on_eight_bit_grid(weight):
    scale = weight.abs().amax(dim=1, keepdim=True) / 127
    ratio = weight / scale
    return bool(((ratio - ratio.round()).abs() < 1e-3).all())


DENSE = ["self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def test_dense_weights_are_prequantized_only_with_the_switch():
    base = tiny_model()
    reference = {
        (i, path): layer.get_submodule(path).weight.detach().clone()
        for i, layer in enumerate(base.model.layers)
        for path in DENSE
    }
    for switch in (True, False):
        model = copy.deepcopy(base)
        wsvd_method.decompose(
            model, None, make_config(local_ft={"iters": 0}, qat={"enabled": True, "dense_prequant": switch})
        )
        for i, layer in enumerate(model.model.layers):
            for path in DENSE:
                weight = layer.get_submodule(path).weight.detach()
                if switch:
                    assert on_eight_bit_grid(weight), (i, path)
                    assert not torch.equal(weight, reference[(i, path)])
                else:
                    assert torch.equal(weight, reference[(i, path)]), (i, path)
                    assert not on_eight_bit_grid(weight), (i, path)


def test_a_singular_gram_is_shifted_and_a_bad_floor_index_raises(monkeypatch, caplog):
    zero = {("linear.attn.k_proj.input", layer): torch.zeros(HIDDEN, HIDDEN) for layer in range(LAYERS)}
    monkeypatch.setattr(wsvd_method, "act_stats", lambda *_args, **_kwargs: zero)
    wsvd_method.decompose(tiny_model(), None, make_config())
    shifted = [r.getMessage() for r in caplog.records if "not positive definite" in r.getMessage()]
    assert len(shifted) == LAYERS
    with pytest.raises(ValueError, match="empty_head_index"):
        wsvd_method.decompose(tiny_model(), None, make_config(decomposition={"empty_head_index": HEAD_DIM}))


def preset_config(name="paper_w8a8"):
    values = builtin_catalog().presets["wsvd"][name].values
    fields: dict = {}
    for key, entry in values.items():
        if key.startswith("method.config."):
            node = fields
            *parents, leaf = key.removeprefix("method.config.").split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = entry.value
    fields["rules"] = [QuantRule.model_validate(rule) for rule in fields["rules"]]
    fields["calibration"] = Calibration(dataset="wikitext2", n_samples=len(WINDOWS), seqlen=32, seed=0)
    fields["local_ft"]["iters"] = 2
    return WSVDConfig.model_validate(fields)


@pytest.mark.parametrize("name", ["paper_w8a8", "paper_noq"])
def test_presets_load_as_configs(name):
    config = preset_config(name)
    assert config.qat.enabled == (name == "paper_w8a8")
    assert bool(config.rules) == (name == "paper_w8a8")


def test_the_runtime_applies_decomposition_preprocess_and_rules_to_finite_logits():
    manifest = builtin_catalog().manifests["wsvd"]
    handle = hf.InProcess(model=tiny_model(), tokenizer=None)
    config = preset_config()
    model = hf._apply(
        handle,
        SimpleNamespace(checkpoint="ckpt", revision=None),
        SimpleNamespace(config=config.model_dump(mode="json")),
        wsvd_preprocess(),
        manifest,
    ).model
    assert "q_down" in dict(model.model.layers[0].self_attn.named_children())
    assert torch.isfinite(logits_of(model)).all()
