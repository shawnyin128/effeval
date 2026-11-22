import copy
import functools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval import validate as validate_module  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import Calibration, IntFormat  # noqa: E402
from effeval.core.preprocess import Preprocess, builtin_preprocess_presets, resolve_preprocess  # noqa: E402
from effeval.methods.preprocess import duquant, engine  # noqa: E402
from effeval.ops import online  # noqa: E402
from effeval.ops import rotation as rot  # noqa: E402
from effeval.ops import scaling as scaling_ops  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from effeval.ops.online import OnlineOp  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402
from reference import duquant_official as official  # noqa: E402
from test_sites import padded_batch  # noqa: E402

compute = pytest.mark.compute
LOGITS_ATOL = 1e-4
BLOCK = 16
SOURCE = {"type": "duquant", "block_size": BLOCK, "max_steps": 8, "permutation_times": 1, "seed": 2}
LINEARS = [
    "linear.attn.q_proj.input",
    "linear.attn.k_proj.input",
    "linear.attn.v_proj.input",
    "linear.attn.o_proj.input",
    "linear.mlp.gate_proj.input",
    "linear.mlp.up_proj.input",
    "linear.mlp.down_proj.input",
]
CALIBRATION = {"dataset": "wikitext2", "n_samples": 3, "seqlen": 16, "seed": 2}


PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def activation(seed: int, tokens: int = 48, channels: int = 64) -> torch.Tensor:
    """Gaussian activations with a few outlier channels and one outlier entry, so the search takes steps."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, channels, generator=generator)
    x[:, 5] *= 30
    x[:, 40] *= 12
    x[7, 19] = 90
    return x


def official_quantizer(steps: int, times: int, **kwargs) -> official.UniformAffineQuantizer:
    return official.UniformAffineQuantizer(
        n_bits=4, block_size=BLOCK, max_rotation_step=steps, permutation_times=times, **kwargs
    )


# --- search and apply path against the vendored official code ---


@pytest.mark.parametrize("seed", [0, 2, 7])
@pytest.mark.parametrize("times", [0, 1, 2])
def test_search_gives_the_official_rotations_and_permutations_exactly(seed, times):
    x = activation(seed)
    torch.manual_seed(seed)
    reference = official_quantizer(8, times)
    reference.online_duquant_cali(x)

    generator = torch.Generator().manual_seed(seed)
    rotations, permutations = duquant.search_transform(x, BLOCK, 8, times, generator)

    assert rotations.shape == (times + 1, BLOCK, BLOCK)
    assert torch.equal(rotations, reference.R)
    if times == 0:
        assert reference.permutation_list is None and permutations.shape == (0, 64)
    else:
        assert torch.equal(permutations, reference.permutation_list)
        assert not torch.equal(permutations[0], torch.arange(64)), "the zigzag permutation is the identity"
    assert not torch.equal(rotations[0], torch.eye(BLOCK)), "the search kept no step"


def test_consecutive_searches_consume_the_random_stream_like_the_official_code():
    xs = [activation(seed) for seed in (1, 2, 3)]
    torch.manual_seed(2)
    expected = []
    for x in xs:
        reference = official_quantizer(8, 1)
        reference.online_duquant_cali(x)
        expected.append((reference.R, reference.permutation_list))

    generator = torch.Generator().manual_seed(2)
    for x, (rotations_ref, permutations_ref) in zip(xs, expected, strict=True):
        rotations, permutations = duquant.search_transform(x, BLOCK, 8, 1, generator)
        assert torch.equal(rotations, rotations_ref)
        assert torch.equal(permutations, permutations_ref)


@pytest.mark.parametrize("times", [0, 1, 2])
def test_apply_path_equals_the_official_activation_transform(times):
    x = activation(0)
    torch.manual_seed(0)
    reference = official_quantizer(8, times)
    reference.online_duquant_cali(x)
    reference.init_duquant_params = torch.tensor(1)
    other = activation(11, tokens=20)
    expected = reference.init_duquant(other.clone())

    rotations, permutations = reference.R, reference.permutation_list
    if permutations is None:
        permutations = torch.zeros(0, 64, dtype=torch.long)
    got = duquant.apply_transform(other, rotations, permutations, BLOCK)
    assert torch.equal(got, expected)

    transform = duquant.BlockRotatePermute(rotations, permutations, BLOCK)
    assert torch.allclose(transform(other), expected, atol=1e-5)
    matrix = transform.matrix(64)
    assert torch.allclose(matrix @ matrix.T, torch.eye(64, dtype=torch.float64), atol=1e-5)
    assert torch.allclose((other.double() @ matrix).float(), expected, atol=1e-4)


def test_search_exchange_uses_the_official_row_index_modulo_block():
    # The maximum sits at row 2, column 5 of the [-1, 16] view; the official code swaps channel 2 (row % 16).
    x = torch.zeros(4, 32)
    x[0, 3] = 1.0
    x[1, 5] = 50.0
    torch.manual_seed(2)
    reference = official_quantizer(1, 0)
    reference.online_duquant_cali(x)
    rotations, _ = duquant.search_transform(x, BLOCK, 1, 0, torch.Generator().manual_seed(2))
    assert torch.equal(rotations, reference.R)
    base = duquant.random_rotation(BLOCK, torch.Generator().manual_seed(2))
    assert torch.equal(rotations[0], duquant._swap_channel_zero(base, 2))
    assert not torch.equal(rotations[0], duquant._swap_channel_zero(base, 5))


def test_block_hadamard_on_qk_is_the_official_sylvester_matrix():
    assert torch.allclose(rot.hadamard(128).float(), official.get_hadamard(128), atol=1e-7)


# --- asymmetric clip quantizer against the official UniformAffineQuantizer ---


@pytest.mark.parametrize("bits", [4, 6])
def test_asymmetric_clipped_activation_and_weight_quantizers_equal_the_official_ones(bits):
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(40, 128, generator=generator) * 3 + 0.5
    w = torch.randn(30, 128, generator=generator)
    act_ref = official.UniformAffineQuantizer(n_bits=bits, symmetric=False, lac=0.9)
    weight_ref = official.UniformAffineQuantizer(n_bits=bits, symmetric=False, swc=0.8)
    act_fmt = IntFormat(type="int", bits=bits, symmetric=False, granularity="per_token", clip_ratio=0.9)
    weight_fmt = IntFormat(type="int", bits=bits, symmetric=False, granularity="per_channel", clip_ratio=0.8)
    assert torch.allclose(fake_quantize(x, act_fmt, "activation"), act_ref.quantize(x.clone()), atol=1e-6)
    assert torch.allclose(fake_quantize(w, weight_fmt, "weight"), weight_ref.quantize(w.clone()), atol=1e-6)


def test_quantizers_differ_from_the_official_one_only_for_rows_that_exclude_zero():
    # Official: range [0.9 min, 0.9 max]. effeval: the min-max range always contains zero (GPTQ convention).
    x = torch.rand(8, 64) + 4.0
    reference = official.UniformAffineQuantizer(n_bits=4, symmetric=False, lac=0.9).quantize(x.clone())
    fmt = IntFormat(type="int", bits=4, symmetric=False, granularity="per_token", clip_ratio=0.9)
    ours = fake_quantize(x, fmt, "activation")
    assert not torch.allclose(ours, reference, atol=1e-3)
    assert (ours - x).abs().mean() > (reference - x).abs().mean()


# --- the source on tiny Qwen3 models ---

MODELS = {
    "qwen3_wide": lambda: build_tiny_qwen3(head_dim=32, tie=True),
    "qwen3_square": lambda: build_tiny_qwen3(head_dim=16, tie=False),
}


@pytest.fixture(params=sorted(MODELS))
def model(request):
    return MODELS[request.param]()


def windows_for(model, n: int = 3, length: int = 16) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    return [torch.randint(0, model.config.vocab_size, (1, length), generator=generator) for _ in range(n)]


@torch.no_grad()
def logits_of(model) -> torch.Tensor:
    input_ids, attention_mask = padded_batch()
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[attention_mask.bool()]


def step_for(positions: list[str], source: dict | None = None, calibration: dict | None = CALIBRATION):
    entries = [
        {"position": p, "mode": "online", "source": copy.deepcopy(source or SOURCE)} for p in positions
    ]
    step: dict = {"kind": "rotation", "entries": entries}
    if calibration is not None:
        step["calibration"] = calibration
    return resolve_preprocess(Preprocess.model_validate({"steps": [step]})).steps[0]


def run_step(model, step, windows=None, pending=(), cache=None):
    """Run the duquant source on `model` as the engine would; returns (result, ctx, registered online ops)."""
    ctx = engine.make_context(model, None, {"checkpoint": "tiny"}, artifacts=cache or ArtifactCache())
    ctx.step, ctx.upstream = step, []
    ctx.pending_online = list(pending)
    if step.calibration is not None:
        ctx._windows[step.calibration.model_dump_json()] = windows or windows_for(model)
    result = duquant.apply_duquant(model, ctx.net.graph, step.entries, ctx)
    attachment = online.register(model, [*pending, *result.online])
    return result, ctx, attachment


@compute
def test_all_linear_inputs_keep_logits_and_the_online_ops_are_needed(model):
    base = logits_of(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result, _ctx, attachment = run_step(model, step_for(LINEARS))
    got = logits_of(model)
    print("max abs logit diff", (got - base).abs().max().item(), "of scale", base.abs().max().item())
    assert torch.allclose(got, base, atol=LOGITS_ATOL)
    names = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert names == {
        f"model.layers.{i}.{m}.weight" for i in range(len(model.model.layers)) for m in PROJECTIONS
    }
    attachment.remove()
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    assert len(result.online) == len(LINEARS) * len(model.model.layers)


@compute
def test_inputs_read_by_several_linears_share_one_transform_per_layer(model):
    result, _ctx, attachment = run_step(model, step_for(LINEARS))
    attachment.remove()
    by_site: dict[tuple[str, int], OnlineOp] = {(op.site, op.layers[0]): op for op in result.online}
    for layer in range(len(model.model.layers)):
        assert (
            by_site[("linear.attn.q_proj.input", layer)].fn is by_site[("linear.attn.k_proj.input", layer)].fn
        )
        assert (
            by_site[("linear.attn.q_proj.input", layer)].fn is by_site[("linear.attn.v_proj.input", layer)].fn
        )
        assert (
            by_site[("linear.mlp.gate_proj.input", layer)].fn
            is by_site[("linear.mlp.up_proj.input", layer)].fn
        )
        assert len({id(by_site[(p, layer)].fn) for p in LINEARS}) == 4


GROUP_LEAD = {
    "linear.attn.q_proj.input": "linear.attn.q_proj.input",
    "linear.attn.k_proj.input": "linear.attn.q_proj.input",
    "linear.attn.v_proj.input": "linear.attn.q_proj.input",
    "linear.attn.o_proj.input": "linear.attn.o_proj.input",
    "linear.mlp.gate_proj.input": "linear.mlp.gate_proj.input",
    "linear.mlp.up_proj.input": "linear.mlp.gate_proj.input",
    "linear.mlp.down_proj.input": "linear.mlp.down_proj.input",
}


def weight_name(layer: int, position: str) -> str:
    group, proj = position[len("linear.") : -len(".input")].split(".")
    return f"model.layers.{layer}.{'self_attn' if group == 'attn' else 'mlp'}.{proj}.weight"


@compute
def test_folded_weights_equal_the_official_weight_transform():
    model = build_tiny_qwen3(head_dim=32)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result, ctx, attachment = run_step(model, step_for(LINEARS))
    attachment.remove()
    found = ctx.artifacts.get_or_compute(result.artifacts["duquant"].key, lambda: pytest.fail("not cached"))
    for position, lead in GROUP_LEAD.items():
        reference = official_quantizer(8, 1)
        reference.R = found[f"1/{lead}/R"]
        reference.permutation_list = found[f"1/{lead}/P"]
        reference.init_duquant_params = torch.tensor(1)
        expected = reference.init_duquant(before[weight_name(1, position)].clone())
        got = model.get_parameter(weight_name(1, position))
        assert torch.allclose(got, expected, atol=1e-5), position


# --- smoothing before rotation: search input and equivalence ---


def smooth_like_duquant(model, seed: int = 0):
    """Function-preserving per-channel smoothing at the four DuQuant sites; returns (online ops, scales)."""
    generator = torch.Generator().manual_seed(seed)
    ops: list[OnlineOp] = []
    scales: list[dict[str, torch.Tensor]] = []

    def draw(n: int) -> torch.Tensor:
        return 0.5 + 2 * torch.rand(n, generator=generator)

    for index, layer in enumerate(model.model.layers):
        attn, mlp = layer.self_attn, layer.mlp
        s = {
            "q": draw(attn.q_proj.in_features),
            "o": draw(attn.o_proj.in_features),
            "gate": draw(mlp.gate_proj.in_features),
            "down": draw(mlp.down_proj.in_features),
        }
        layer.input_layernorm.weight.data.copy_(
            scaling_ops.fold_into_norm_gain(layer.input_layernorm.weight, s["q"])
        )
        for linear in (attn.q_proj, attn.k_proj, attn.v_proj):
            linear.weight.data.copy_(scaling_ops.scale_input_columns(linear.weight, s["q"]))
        layer.post_attention_layernorm.weight.data.copy_(
            scaling_ops.fold_into_norm_gain(layer.post_attention_layernorm.weight, s["gate"])
        )
        for linear in (mlp.gate_proj, mlp.up_proj):
            linear.weight.data.copy_(scaling_ops.scale_input_columns(linear.weight, s["gate"]))
        mlp.up_proj.weight.data.copy_(scaling_ops.scale_output_rows(mlp.up_proj.weight, s["down"]))
        mlp.down_proj.weight.data.copy_(scaling_ops.scale_input_columns(mlp.down_proj.weight, s["down"]))
        attn.o_proj.weight.data.copy_(scaling_ops.scale_input_columns(attn.o_proj.weight, s["o"]))
        ops.append(OnlineOp("linear.attn.o_proj.input", scaling_ops.ChannelDivide(s["o"]), [index]))
        scales.append(s)
    return ops, scales


def recorded_search_inputs(monkeypatch) -> list[torch.Tensor]:
    seen: list[torch.Tensor] = []
    real = duquant.search_transform

    def wrapper(x, *args, **kwargs):
        seen.append(x.detach().clone())
        return real(x, *args, **kwargs)

    monkeypatch.setattr(duquant, "search_transform", wrapper)
    return seen


@compute
def test_smoothing_then_rotation_keeps_logits_and_the_searched_input_is_the_smoothed_activation(
    model, monkeypatch
):
    base_model = copy.deepcopy(model)
    base = logits_of(model)
    pending, scales = smooth_like_duquant(model)
    smoothing = online.register(model, pending)
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL), "smoothing alone must be exact"
    smoothing.remove()

    window = windows_for(model, n=1)
    oracle: dict[tuple[int, str], torch.Tensor] = {}
    names = {
        "self_attn.q_proj": "q",
        "self_attn.o_proj": "o",
        "mlp.gate_proj": "gate",
        "mlp.down_proj": "down",
    }

    def observe(_module, args, key: tuple[int, str]) -> None:
        oracle.setdefault(key, args[0].detach().float().clone())

    handles = []
    for index, layer in enumerate(base_model.model.layers):
        for path, key in names.items():
            module = layer.get_submodule(path)
            handles.append(module.register_forward_pre_hook(functools.partial(observe, key=(index, key))))
    with torch.no_grad():
        base_model(window[0])
    for handle in handles:
        handle.remove()

    seen = recorded_search_inputs(monkeypatch)
    _result, _ctx, attachment = run_step(model, step_for(LINEARS), windows=window, pending=pending)
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    attachment.remove()

    order = [(i, key) for i in range(len(model.model.layers)) for key in ("q", "o", "gate", "down")]
    assert len(seen) == len(order)
    for x, (index, key) in zip(seen, order, strict=True):
        expected = oracle[(index, key)] / scales[index][key]
        assert torch.allclose(x.reshape(expected.shape), expected, atol=1e-4, rtol=1e-4), (index, key)


@compute
def test_the_search_sequence_is_the_mean_of_the_calibration_inputs(monkeypatch):
    model = build_tiny_qwen3(head_dim=32)
    windows = windows_for(model, n=3)
    seen = recorded_search_inputs(monkeypatch)
    _result, _ctx, attachment = run_step(model, step_for(LINEARS[:1]), windows=windows)
    attachment.remove()
    with torch.no_grad():
        mean = torch.stack([model.model.embed_tokens(w) for w in windows]).mean(dim=0)
        expected = model.model.layers[0].input_layernorm(mean)
    assert torch.allclose(seen[0].reshape(expected.shape), expected, atol=1e-5)


@compute
def test_the_search_follows_the_official_random_order_over_layers_and_groups(model, monkeypatch):
    seen = recorded_search_inputs(monkeypatch)
    result, ctx, attachment = run_step(model, step_for(LINEARS))
    attachment.remove()
    found = ctx.artifacts.get_or_compute(result.artifacts["duquant"].key, lambda: pytest.fail("not cached"))
    leads = ["linear.attn.q_proj.input", "linear.attn.o_proj.input", "linear.mlp.gate_proj.input"]
    leads.append("linear.mlp.down_proj.input")
    torch.manual_seed(SOURCE["seed"])
    calls = iter(seen)
    for layer in range(len(model.model.layers)):
        for lead in leads:
            reference = official_quantizer(SOURCE["max_steps"], SOURCE["permutation_times"])
            reference.online_duquant_cali(next(calls))
            assert torch.equal(found[f"{layer}/{lead}/R"], reference.R), (layer, lead)
            assert torch.equal(found[f"{layer}/{lead}/P"], reference.permutation_list), (layer, lead)


@compute
def test_layer_selection_and_the_artifact_cache():
    model = build_tiny_qwen3(head_dim=32)
    cache = ArtifactCache()
    step = step_for(LINEARS[3:4])
    step.entries[0].layers = [1]
    result, _ctx, attachment = run_step(model, step, cache=cache)
    attachment.remove()
    assert [(op.site, op.layers) for op in result.online] == [("linear.attn.o_proj.input", [1])]
    assert cache.computed == 1
    again = build_tiny_qwen3(head_dim=32)
    _result, _ctx, attachment = run_step(again, step, cache=cache)
    attachment.remove()
    assert cache.computed == 1 and cache.memory_hits == 1
    assert torch.equal(
        again.model.layers[1].self_attn.o_proj.weight, model.model.layers[1].self_attn.o_proj.weight
    )


@compute
def test_the_source_is_registered_with_the_engine():
    engine._discover()
    assert engine._SOURCES[("rotation", "duquant")] is duquant.apply_duquant


@compute
@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda step: setattr(step.entries[0], "mode", "offline"), "not available"),
        (lambda step: setattr(step.entries[0], "position", "attn.qk_rope"), "qk_rope"),
        (
            lambda step: setattr(
                step.entries[1], "source", step.entries[1].source.model_copy(update={"seed": 9})
            ),
            "same source",
        ),
        (
            lambda step: setattr(
                step.entries[0], "source", step.entries[0].source.model_copy(update={"block_size": 48})
            ),
            "power of two",
        ),
        (
            lambda step: setattr(
                step.entries[1], "source", step.entries[1].source.model_copy(update={"block_size": 32})
            ),
            "same source",
        ),
    ],
)
def test_invalid_entries_are_refused(mutate, message):
    model = build_tiny_qwen3(head_dim=32)
    step = step_for(LINEARS[:2])
    mutate(step)
    with pytest.raises(engine.EngineError, match=message):
        run_step(model, step)


@compute
def test_width_not_divisible_by_the_block_size_and_missing_calibration_are_refused():
    model = build_tiny_qwen3(head_dim=32)
    wide = step_for(LINEARS[:1])
    wide.entries[0].source = wide.entries[0].source.model_copy(update={"block_size": 128})
    with pytest.raises(engine.EngineError, match="not a multiple"):
        run_step(model, wide)
    with pytest.raises(engine.EngineError, match="calibration"):
        run_step(model, step_for(LINEARS[:1], calibration=None))


# --- presets ---


def duquant_preset_steps() -> list[dict]:
    return copy.deepcopy(builtin_preprocess_presets()["duquant"].values["preprocess.steps"].value)


@compute
def test_engine_apply_runs_a_duquant_step_and_exposes_earlier_online_ops_to_later_functions(
    model, monkeypatch
):
    base = logits_of(model)
    step = step_for(LINEARS)
    preprocess = resolve_preprocess(Preprocess.model_validate({"steps": [step.model_dump(mode="json")]}))
    ctx = engine.make_context(model, None, {"checkpoint": "tiny"}, artifacts=ArtifactCache())
    ctx._windows[step.calibration.model_dump_json()] = windows_for(model)
    seen_pending: list[int] = []
    real = duquant.search_transform

    def spy(x, *args, **kwargs):
        seen_pending.append(len(ctx.pending_online))
        return real(x, *args, **kwargs)

    monkeypatch.setattr(duquant, "search_transform", spy)
    applied = engine.apply(model, preprocess, ctx)
    assert applied.online_ops is not None and set(seen_pending) == {0}
    assert len(ctx.pending_online) == len(LINEARS) * len(model.model.layers)
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    applied.online_ops.remove()
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


def test_a_duquant_step_without_calibration_is_refused_by_the_validator():
    steps = duquant_preset_steps()
    del steps[1]["calibration"]
    raw = {
        "field": "quantization",
        "model": {"name": "qwen3-0.6b"},
        "method": {"name": "rtn", "preset": "duquant_w4a4"},
        "preprocess": {"preset": "duquant", "steps": steps},
    }
    res = validate_module.resolve(raw, builtin_catalog())
    assert [i.code for i in res.issues if i.severity == "error"] == ["preprocess.calibration_required"]
    assert res.issues[0].paths == ["preprocess.steps[1].calibration"]


def resolve_duquant(method_preset: str = "duquant_w4a4", model: str = "qwen3-0.6b"):
    raw = {
        "field": "quantization",
        "model": {"name": model},
        "method": {"name": "rtn", "preset": method_preset},
        "preprocess": {"preset": "duquant"},
    }
    return validate_module.resolve(raw, builtin_catalog())


def test_the_duquant_preset_encodes_the_official_run_sh_settings():
    scaling, rotation = Preprocess.model_validate({"steps": duquant_preset_steps()}).steps
    expected = Calibration(dataset="wikitext2", n_samples=128, seqlen=2048, seed=2)
    assert scaling.calibration == expected and rotation.calibration == expected
    assert {(e.position, e.mode) for e in scaling.entries} == {
        ("norm.input_layernorm.out", "offline"),
        ("norm.post_attention_layernorm.out", "offline"),
        ("linear.mlp.down_proj.input", "offline"),
        ("linear.attn.o_proj.input", "online"),
    }
    assert all(
        e.source.model_dump()
        == {"type": "smoothquant", "alpha": 0.6, "weight_reduce": "first_consumer", "act_stat": "absmax"}
        for e in scaling.entries
    )
    duquant_entries = [e for e in rotation.entries if e.source.type == "duquant"]
    assert [e.position for e in duquant_entries] == LINEARS
    assert all(
        e.source.model_dump()
        == {"type": "duquant", "block_size": 128, "max_steps": 256, "permutation_times": 1, "seed": 2}
        for e in duquant_entries
    )
    assert [(e.position, e.source.type) for e in rotation.entries[-1:]] == [("attn.qk_rope", "hadamard")]


@pytest.mark.parametrize(("preset", "bits"), [("duquant_w4a4", 4), ("duquant_w6a6", 6)])
def test_duquant_quantizer_presets_validate_with_asymmetric_clipped_formats(preset, bits):
    res = resolve_duquant(preset)
    assert res.ok, [i.message for i in res.issues]
    rules = res.spec.method.config["rules"]
    formats = {tuple(r["sites"]): r["quant"] for r in rules}
    base = {"type": "int", "bits": bits, "symmetric": False}
    assert formats[("linear.*.weight",)] == {**base, "granularity": "per_channel", "clip_ratio": 0.8}
    assert formats[("linear.*.input",)] == {**base, "granularity": "per_token", "clip_ratio": 0.9}
    assert formats[("attn.q", "attn.k")] == {**base, "granularity": "per_token", "clip_ratio": 0.9}
    assert formats[("attn.v",)] == {**base, "granularity": "per_token"}
    assert len(rules) == 4


@compute
def test_the_preset_rotation_step_scaled_to_the_tiny_models_keeps_logits(model):
    step = copy.deepcopy(duquant_preset_steps()[1])
    entries = [e for e in step["entries"] if e["source"]["type"] == "duquant"]
    for entry in entries:
        entry["source"].update(block_size=BLOCK, max_steps=8)
    step["entries"] = entries
    step["calibration"] = CALIBRATION
    resolved = resolve_preprocess(Preprocess.model_validate({"steps": [step]})).steps[0]
    base = logits_of(model)
    _result, _ctx, attachment = run_step(model, resolved)
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    attachment.remove()
