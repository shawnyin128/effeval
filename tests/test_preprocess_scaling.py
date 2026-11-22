import contextlib
import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_llama, build_tiny_qwen3  # noqa: E402
from effeval import validate as validate_module  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import QuantRule  # noqa: E402
from effeval.core.preprocess import (  # noqa: E402
    Preprocess,
    ResolvedPreprocess,
    builtin_preprocess_presets,
    resolve_preprocess,
)
from effeval.core.types import InProcess  # noqa: E402
from effeval.methods.preprocess import engine, scaling  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402
from effeval.runtime import hf  # noqa: E402
from reference import awq_official as awq_ref  # noqa: E402
from reference import duquant_smooth_official as dq_ref  # noqa: E402
from reference import smoothquant_official as sq_ref  # noqa: E402
from test_sites import padded_batch  # noqa: E402

compute = pytest.mark.compute
LOGITS_ATOL = 1e-4
WEIGHT_ATOL = 1e-6
GROUP = 16
N_TOKENS = 16

MODELS = {
    "qwen3_wide": lambda: build_tiny_qwen3(head_dim=32, tie=True),
    "qwen3_square": lambda: build_tiny_qwen3(head_dim=16, tie=False),
}
CALIBRATION = {"dataset": "wikitext2", "n_samples": 8, "seqlen": 32, "seed": 0}
SQ_POSITIONS = ("norm.input_layernorm.out", "norm.post_attention_layernorm.out")
AWQ_POSITIONS = (*SQ_POSITIONS, "linear.mlp.down_proj.input")
W4_RULES = [
    QuantRule.model_validate(
        {
            "sites": ["linear.*.weight"],
            "quant": {
                "type": "int",
                "bits": 4,
                "symmetric": False,
                "granularity": "per_group",
                "group_size": GROUP,
            },
        }
    )
]
W4_FORMAT = W4_RULES[0].quant
CLIP = {"n_grid": 20, "max_shrink": 0.5, "n_tokens": N_TOKENS, "skip": ["q_proj", "k_proj"]}


def fake_windows(n: int = 8, length: int = 32, seed: int = 3) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=generator) for _ in range(n)]


@pytest.fixture
def windows(monkeypatch):
    data = fake_windows()
    monkeypatch.setattr(engine, "calibration_windows", lambda _calibration, _tokenizer_id: data)
    return data


def smoothquant_preprocess(alpha: float = 0.5, weight_reduce: str = "group_max") -> ResolvedPreprocess:
    entries = [
        {
            "position": position,
            "mode": "offline",
            "source": {"type": "smoothquant", "alpha": alpha, "weight_reduce": weight_reduce},
        }
        for position in SQ_POSITIONS
    ]
    step = {"kind": "scaling", "calibration": CALIBRATION, "entries": entries}
    return resolve_preprocess(Preprocess.model_validate({"steps": [step]}))


def awq_preprocess(clip: dict | None, positions=AWQ_POSITIONS, n_grid: int = 20) -> ResolvedPreprocess:
    source = {"type": "awq", "n_grid": n_grid, "clip_search": clip}
    entries = [{"position": p, "mode": "offline", "source": source} for p in positions]
    step = {"kind": "scaling", "calibration": CALIBRATION, "entries": entries}
    return resolve_preprocess(Preprocess.model_validate({"steps": [step]}))


def apply_to(model, preprocess, downstream=None, cache=None) -> tuple[engine.Applied, engine.Context]:
    ctx = engine.make_context(
        model,
        SimpleNamespace(name_or_path="tiny"),
        {"checkpoint": "tiny"},
        downstream=downstream,
        artifacts=cache if cache is not None else ArtifactCache(),
    )
    return engine.apply(model, preprocess, ctx), ctx


@torch.no_grad()
def logits_of(model) -> torch.Tensor:
    input_ids, attention_mask = padded_batch()
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[attention_mask.bool()]


def max_abs_diff(a, b) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a.parameters(), b.parameters(), strict=True))


# ------------------------------------------------------------ SmoothQuant


@pytest.fixture(params=sorted(MODELS))
def qwen3(request):
    return MODELS[request.param]()


@compute
def test_smoothquant_keeps_logits_and_changes_the_norm_gains(qwen3, windows):
    base = logits_of(qwen3)
    before = {n: p.detach().clone() for n, p in qwen3.named_parameters()}
    applied, _ = apply_to(qwen3, smoothquant_preprocess())
    assert applied.online_ops is None and len(applied.edits) == 2
    assert torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL)
    for name in ("model.layers.0.input_layernorm.weight", "model.layers.1.post_attention_layernorm.weight"):
        assert (dict(qwen3.named_parameters())[name] - before[name]).abs().max() > 1e-2
    untouched = ("self_attn.o_proj", "mlp.down_proj", "q_norm", "k_norm")
    for name, parameter in qwen3.named_parameters():
        if any(part in name for part in untouched):
            assert torch.equal(parameter, before[name]), name


@compute
@pytest.mark.parametrize("alpha", [0.5, 0.85])
def test_smoothquant_matches_the_official_code_on_tiny_llama(register_architecture, windows, alpha):
    model = build_tiny_llama()
    register_architecture(model)
    official = copy.deepcopy(model)
    base = copy.deepcopy(model)

    act_scales = sq_ref.get_act_scales(official, windows)
    for index, layer in enumerate(official.model.layers):
        prefix = f"model.layers.{index}."
        attn, mlp = layer.self_attn, layer.mlp
        sq_ref.smooth_ln_fcs_llama_like(
            layer.input_layernorm,
            [attn.q_proj, attn.k_proj, attn.v_proj],
            act_scales[prefix + "self_attn.q_proj"],
            alpha,
        )
        sq_ref.smooth_ln_fcs_llama_like(
            layer.post_attention_layernorm,
            [mlp.gate_proj, mlp.up_proj],
            act_scales[prefix + "mlp.gate_proj"],
            alpha,
        )

    apply_to(model, smoothquant_preprocess(alpha))
    assert max_abs_diff(model, base) > 1e-2, "the comparison needs a model that smoothing really changes"
    assert max_abs_diff(model, official) <= WEIGHT_ATOL


@compute
def test_smoothquant_scale_formula_and_weight_reduce():
    generator = torch.Generator().manual_seed(1)
    act = torch.rand(8, generator=generator) * 4 + 0.1
    weights = [torch.randn(5, 8, generator=generator) for _ in range(3)]
    expected = (act.pow(0.7) / torch.stack([w.abs().amax(0) for w in weights]).amax(0).pow(0.3)).clamp(
        min=1e-5
    )
    assert torch.equal(scaling.smoothquant_scales(act, weights, 0.7, "group_max"), expected)
    first = (act.pow(0.7) / weights[0].abs().amax(0).pow(0.3)).clamp(min=1e-5)
    assert torch.equal(scaling.smoothquant_scales(act, weights, 0.7, "first_consumer"), first)
    assert not torch.allclose(first, expected)


# ------------------------------------------------------------ SmoothQuant at down_proj / o_proj

DOWN = ("linear.mlp.down_proj.input", "offline")
O_PROJ = ("linear.attn.o_proj.input", "online")
LN1 = ("norm.input_layernorm.out", "offline")
LN2 = ("norm.post_attention_layernorm.out", "offline")


def smoothquant_at(*positions: tuple[str, str], alpha: float = 0.6, reduce: str = "first_consumer"):
    entries = [
        {
            "position": position,
            "mode": mode,
            "source": {"type": "smoothquant", "alpha": alpha, "weight_reduce": reduce},
        }
        for position, mode in positions
    ]
    step = {"kind": "scaling", "calibration": CALIBRATION, "entries": entries}
    return resolve_preprocess(Preprocess.model_validate({"steps": [step]}))


@compute
@pytest.mark.parametrize(
    "positions", [[DOWN], [O_PROJ], [LN1, LN2, DOWN, O_PROJ]], ids=["down", "o_proj", "all"]
)
def test_smoothquant_at_down_and_o_proj_keeps_logits(qwen3, windows, positions):
    base = logits_of(qwen3)
    before = {n: p.detach().clone() for n, p in qwen3.named_parameters()}
    applied, _ = apply_to(qwen3, smoothquant_at(*positions))
    assert torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL)
    params = dict(qwen3.named_parameters())

    def changed(name: str) -> bool:
        return not torch.equal(params[name], before[name])

    wanted = {pos for pos, _ in positions}
    assert changed("model.layers.0.mlp.down_proj.weight") == (DOWN[0] in wanted)
    assert changed("model.layers.0.mlp.up_proj.weight") == (DOWN[0] in wanted or LN2[0] in wanted)
    assert changed("model.layers.0.self_attn.o_proj.weight") == (O_PROJ[0] in wanted)
    assert (applied.online_ops is not None) == (O_PROJ[0] in wanted)
    if applied.online_ops is not None:
        applied.online_ops.remove()
        assert not torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL), (
            "the online division does nothing"
        )


@compute
def test_first_consumer_is_q_proj_for_the_attention_input_and_up_proj_for_the_mlp_input():
    model = MODELS["qwen3_wide"]()
    ctx = engine.make_context(model, None, {"checkpoint": "tiny"})
    entry = smoothquant_at(LN1, LN2).steps[0].entries
    picked = [scaling.first_consumer(scaling._group(ctx.net, e)).canon for e in entry]
    assert picked == ["attn.q_proj", "mlp.up_proj"]


@compute
def test_duquant_style_smoothing_matches_the_vendored_code_on_tiny_llama(register_architecture, windows):
    model = build_tiny_llama()
    register_architecture(model)
    official = copy.deepcopy(model)
    base = copy.deepcopy(model)
    alpha = 0.6
    act = sq_ref.get_act_scales(official, windows)
    expected_online = []
    for index, layer in enumerate(official.model.layers):
        prefix = f"model.layers.{index}."
        attn, mlp = layer.self_attn, layer.mlp
        qkv = dq_ref.smooth_scale(act[prefix + "self_attn.q_proj"], attn.q_proj.weight, alpha)
        fc1 = dq_ref.smooth_scale(act[prefix + "mlp.up_proj"], mlp.up_proj.weight, alpha)
        down = dq_ref.smooth_scale(act[prefix + "mlp.down_proj"], mlp.down_proj.weight, alpha)
        out = dq_ref.smooth_scale(act[prefix + "self_attn.o_proj"], attn.o_proj.weight, alpha)
        dq_ref.smooth_ln_fcs_inplace(layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj], qkv)
        dq_ref.smooth_ln_fcs_inplace(layer.post_attention_layernorm, [mlp.up_proj, mlp.gate_proj], fc1)
        dq_ref.smooth_fc_fc_inplace(mlp.up_proj, mlp.down_proj, down)
        attn.o_proj.act_quantizer = SimpleNamespace()
        dq_ref.smooth_fc_inplace(attn.o_proj, out)  # the GQA fallback: online division plus weight columns
        expected_online.append(attn.o_proj.act_quantizer.smooth_scales.reshape(-1))

    applied, _ = apply_to(model, smoothquant_at(LN1, LN2, DOWN, O_PROJ, alpha=alpha))
    assert max_abs_diff(model, base) > 1e-2
    assert max_abs_diff(model, official) <= WEIGHT_ATOL
    ops = [op for result in applied.results for op in result.online]
    assert len(ops) == len(expected_online)
    for op, expected in zip(ops, expected_online, strict=True):
        assert torch.allclose(op.fn.scale, expected, atol=WEIGHT_ATOL, rtol=0)


# ------------------------------------------------------------ AWQ: equivalence of scaling alone


@compute
def test_awq_scaling_alone_keeps_logits(qwen3, windows):
    base = logits_of(qwen3)
    before = {n: p.detach().clone() for n, p in qwen3.named_parameters()}
    applied, _ = apply_to(qwen3, awq_preprocess(clip=None), downstream=W4_RULES)
    assert applied.online_ops is None and len(applied.edits) == 3
    assert torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL)
    params = dict(qwen3.named_parameters())
    for name in ("model.layers.0.input_layernorm.weight", "model.layers.0.mlp.up_proj.weight"):
        assert (params[name] - before[name]).abs().max() > 1e-3, name
    assert torch.equal(
        params["model.layers.0.self_attn.o_proj.weight"], before["model.layers.0.self_attn.o_proj.weight"]
    )


@compute
def test_awq_clip_changes_weights_only_inside_the_clip_range(qwen3, windows):
    scaled = copy.deepcopy(qwen3)
    apply_to(scaled, awq_preprocess(clip=None), downstream=W4_RULES)
    clipped = copy.deepcopy(qwen3)
    applied, _ = apply_to(clipped, awq_preprocess(clip=CLIP), downstream=W4_RULES)
    assert any("clipped" in line for line in applied.edits)
    skipped = ("q_proj", "k_proj")
    for (name, a), (_, b) in zip(scaled.named_parameters(), clipped.named_parameters(), strict=True):
        if "layers" in name and "proj" in name and not any(s in name for s in skipped):
            assert b.abs().max() <= a.abs().max() + 1e-7
        else:
            assert torch.equal(a, b), name
    assert any(
        not torch.equal(a, b)
        for (_, a), (_, b) in zip(scaled.named_parameters(), clipped.named_parameters(), strict=True)
    )


# ------------------------------------------------------------ AWQ: against the vendored official code


def official_quantizers(kind: str):
    """(scale-search quantizer, clip-search quantizer or None) for the vendored code"""
    if kind == "official":
        return (
            lambda w: awq_ref.pseudo_quantize_tensor(w, n_bit=4, zero_point=True, q_group_size=GROUP),
            None,
        )
    unit = W4_FORMAT.model_copy(update={"granularity": "per_group", "group_size": GROUP})
    return (
        lambda w: fake_quantize(w, W4_FORMAT, "weight"),
        lambda cur: fake_quantize(cur.reshape(-1, GROUP), unit, "weight").reshape(cur.shape),
    )


@torch.no_grad()
def run_official_awq(model, data, kind: str) -> dict[str, torch.Tensor]:
    """The official layer loop (pre_quant.py run_awq) on tiny Llama; edits `model`, returns its scales etc."""
    search_quantize, clip_quantize = official_quantizers(kind)
    layers = model.model.layers
    inps, layer_kwargs = [], {}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError

    model.config.use_cache = (
        False  # else transformers 5 hands every layer a DynamicCache that each call extends
    )
    layers[0] = Catcher(layers[0])
    with contextlib.suppress(ValueError):
        model(torch.cat(data, dim=0))
    layers[0] = layers[0].module
    hidden = inps[0]
    out: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(layers):
        named = {n: m for n, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
        captured: dict[str, list] = {n: [] for n in named}
        handles = [
            m.register_forward_hook(lambda _m, x, _y, n=n, sink=captured: sink[n].append(x[0].detach()))
            for n, m in named.items()
        ]
        result = layer(hidden, **layer_kwargs)
        hidden = result[0] if isinstance(result, tuple) else result
        for handle in handles:
            handle.remove()
        feat = {n: torch.cat(v, dim=0) for n, v in captured.items()}
        kwargs = {k: v for k, v in layer_kwargs.items() if k != "use_cache"}
        attn, mlp = layer.self_attn, layer.mlp
        searches = {
            "norm.input_layernorm.out": awq_ref.search_module_scale(
                attn,
                [attn.q_proj, attn.k_proj, attn.v_proj],
                feat["self_attn.q_proj"],
                search_quantize,
                kwargs,
            ),
            "norm.post_attention_layernorm.out": awq_ref.search_module_scale(
                mlp, [mlp.gate_proj, mlp.up_proj], feat["mlp.gate_proj"], search_quantize
            ),
            "linear.mlp.down_proj.input": awq_ref.search_module_scale(
                mlp.down_proj, [mlp.down_proj], feat["mlp.down_proj"], search_quantize
            ),
        }
        for position, (scales, ratio) in searches.items():
            out[f"scale.{index}.{position}"] = scales.clone()
            out[f"ratio.{index}.{position}"] = torch.tensor(ratio)
        s_attn, s_mlp, s_down = (searches[p][0] for p in AWQ_POSITIONS)
        awq_ref.scale_ln_fcs(layer.input_layernorm, [attn.q_proj, attn.k_proj, attn.v_proj], s_attn)
        awq_ref.scale_ln_fcs(layer.post_attention_layernorm, [mlp.gate_proj, mlp.up_proj], s_mlp)
        awq_ref.scale_fc_fc(mlp.up_proj, mlp.down_proj, s_down)
        for names, scales in (
            (["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], s_attn),
            (["mlp.gate_proj", "mlp.up_proj"], s_mlp),
            (["mlp.down_proj"], s_down),
        ):
            for name in names:
                feat[name].div_(scales.view(1, -1))
        clips = {}
        for name, module in named.items():
            if "q_" in name or "k_" in name:
                continue
            clips[name] = awq_ref.auto_clip_layer(
                module.weight,
                feat[name],
                4,
                {"zero_point": True, "q_group_size": GROUP},
                n_sample_token=N_TOKENS,
                quantize_func=clip_quantize,
            )
        for name, max_val in clips.items():
            awq_ref.apply_clip(named[name], max_val)
            out[f"clip.{index}.{name.replace('self_attn.', 'attn.')}"] = max_val.clone()
    return out


def count_zero_excluded_groups(model) -> int:
    total = 0
    for layer in model.model.layers:
        for module in layer.modules():
            if isinstance(module, torch.nn.Linear):
                groups = module.weight.reshape(-1, GROUP)
                total += int(((groups.min(1).values > 0) | (groups.max(1).values < 0)).sum())
    return total


@compute
@pytest.mark.parametrize("kind", ["official", "effeval"])
def test_awq_matches_the_official_loop_on_tiny_llama(register_architecture, windows, kind):
    model = build_tiny_llama()
    register_architecture(model)
    official_model = copy.deepcopy(model)
    base = copy.deepcopy(model)
    if kind == "official":
        assert count_zero_excluded_groups(model) == 0, "the official quantizer would differ by construction"

    expected = run_official_awq(official_model, windows, kind)
    cache = ArtifactCache()
    applied, _ = apply_to(model, awq_preprocess(CLIP), downstream=W4_RULES, cache=cache)

    assert max_abs_diff(model, base) > 1e-2
    assert max_abs_diff(model, official_model) <= WEIGHT_ATOL
    tensors = cached_tensors(applied, cache)
    assert set(tensors) == set(expected)
    for name, value in expected.items():
        if name.startswith("ratio"):
            assert tensors[name].item() == value.item(), name
        else:
            assert torch.allclose(tensors[name], value, atol=WEIGHT_ATOL, rtol=0), name


def cached_tensors(applied, cache) -> dict[str, torch.Tensor]:
    return cache.get_or_compute(applied.artifacts["awq.scales"].key, lambda: pytest.fail("not cached"))


@compute
def test_awq_selected_ratios_are_not_all_the_same(register_architecture, windows):
    model = build_tiny_llama()
    register_architecture(model)
    cache = ArtifactCache()
    applied, _ = apply_to(model, awq_preprocess(CLIP), downstream=W4_RULES, cache=cache)
    tensors = cached_tensors(applied, cache)
    ratios = {name: value.item() for name, value in tensors.items() if name.startswith("ratio")}
    assert len(ratios) == 6 and len(set(ratios.values())) > 1


# ------------------------------------------------------------ AWQ pieces


@compute
@pytest.mark.parametrize("kind", ["official", "effeval"])
def test_search_clip_matches_the_official_auto_clip_layer(kind):
    generator = torch.Generator().manual_seed(5)
    weight = torch.randn(32, 64, generator=generator) * 0.05
    x = torch.randn(4, 24, 64, generator=generator)
    _, clip_quantize = official_quantizers(kind)
    expected = awq_ref.auto_clip_layer(
        weight,
        x,
        4,
        {"zero_point": True, "q_group_size": GROUP},
        n_sample_token=N_TOKENS,
        quantize_func=clip_quantize,
    )
    got = scaling.search_clip(weight, x, W4_FORMAT, 20, 0.5, N_TOKENS)
    assert got.shape == (32, 4, 1)
    assert torch.equal(got, expected)
    assert (got < weight.reshape(32, 4, GROUP).abs().amax(-1, keepdim=True)).any(), "no group was clipped"

    reference_weight = weight.clone()
    reference_layer = torch.nn.Linear(64, 32, bias=False)
    reference_layer.weight.data.copy_(reference_weight)
    awq_ref.apply_clip(reference_layer, expected)
    scaling.apply_clip(weight, got)
    assert torch.equal(weight, reference_layer.weight.data)


@compute
def test_search_clip_stops_at_max_shrink():
    generator = torch.Generator().manual_seed(9)
    weight = torch.randn(4, GROUP, generator=generator) * 0.02
    weight[:, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])  # one outlier per row, whose input channel is silent
    x = torch.randn(1, 64, GROUP, generator=generator)
    x[..., 0] = 0
    got = scaling.search_clip(weight, x, W4_FORMAT, 20, 0.5, 8)
    assert torch.allclose(got.reshape(4), torch.full((4,), 0.55), atol=1e-6), (
        "the smallest candidate is 1 - 9/20"
    )
    expected = awq_ref.auto_clip_layer(
        weight, x, 4, {"zero_point": True, "q_group_size": GROUP}, n_sample_token=8
    )
    assert torch.allclose(expected.reshape(4), got.reshape(4), atol=1e-6)


@compute
def test_effeval_asymmetric_grid_equals_awq_unless_a_group_excludes_zero():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(8, 64, generator=generator)
    official = awq_ref.pseudo_quantize_tensor(weight, n_bit=4, zero_point=True, q_group_size=GROUP)
    ours = fake_quantize(weight, W4_FORMAT, "weight")
    assert torch.allclose(official, ours, atol=1e-6, rtol=0)

    positive = weight.abs() + 0.5  # every group has min > 0: AWQ's range excludes zero, effeval's includes it
    assert not torch.allclose(
        awq_ref.pseudo_quantize_tensor(positive, n_bit=4, zero_point=True, q_group_size=GROUP),
        fake_quantize(positive, W4_FORMAT, "weight"),
        atol=1e-3,
    )


@compute
def test_search_scales_formula_uses_mean_abs_and_the_sqrt_normalisation():
    generator = torch.Generator().manual_seed(2)
    linear = torch.nn.Linear(16, 8, bias=False)
    x = torch.rand(3, 6, 16, generator=generator) * torch.linspace(0.2, 5, 16)
    scales, ratio = scaling.search_scales(
        linear, [linear], [lambda w: fake_quantize(w, W4_FORMAT, "weight")], x, {}, n_grid=20
    )
    expected = x.abs().reshape(-1, 16).mean(0).pow(ratio).clamp(min=1e-4)
    expected = expected / (expected.max() * expected.min()).sqrt()
    assert 0 <= ratio < 1 and ratio * 20 == round(ratio * 20)
    assert torch.allclose(scales, expected, atol=1e-6, rtol=0)
    assert float(scales.max() * scales.min()) == pytest.approx(1.0, rel=1e-5)


# ------------------------------------------------------------ propagation, errors, wiring


@compute
def test_scaling_errors_are_reported_before_any_edit(windows):
    model = MODELS["qwen3_wide"]()
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(engine.EngineError, match="quantizer rules"):
        apply_to(model, awq_preprocess(CLIP))
    with pytest.raises(engine.EngineError, match="GQA"):
        apply_to(model, awq_preprocess(CLIP, positions=("linear.attn.o_proj.input",)), downstream=W4_RULES)
    fp_rules = [
        QuantRule.model_validate(
            {
                "sites": ["linear.*.weight"],
                "quant": {"type": "fp", "encoding": "e4m3", "granularity": "per_channel"},
            }
        )
    ]
    with pytest.raises(engine.EngineError, match="integer weight quantizers"):
        apply_to(model, awq_preprocess(CLIP), downstream=fp_rules)
    online = copy.deepcopy(smoothquant_preprocess())
    online.steps[0].entries[0].mode = "online"
    with pytest.raises(engine.EngineError, match="online scaling"):
        apply_to(model, online)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


@compute
def test_hf_prepare_hands_the_quantizer_rules_to_the_preprocessing(monkeypatch, windows):
    seen = {}
    real = engine.make_context

    def spy(*args, **kwargs):
        seen["downstream"] = kwargs.get("downstream")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "make_context", spy)
    model = MODELS["qwen3_wide"]()
    monkeypatch.setattr(
        hf, "load", lambda _spec: InProcess(model=model, tokenizer=SimpleNamespace(name_or_path="t"))
    )
    rules = [r.model_dump(mode="json") for r in W4_RULES]
    spec = SimpleNamespace(
        preprocess=awq_preprocess(None),
        model=SimpleNamespace(checkpoint="tiny", revision=None),
        method=SimpleNamespace(config={"rules": rules}),
    )
    handle = hf.prepare(spec, builtin_catalog().manifests["rtn"])
    assert seen["downstream"] == W4_RULES
    assert handle.model is model

    other = MODELS["qwen3_square"]()
    monkeypatch.setattr(
        hf, "load", lambda _spec: InProcess(model=other, tokenizer=SimpleNamespace(name_or_path="t"))
    )
    spec.preprocess = smoothquant_preprocess()
    spec.method = SimpleNamespace(config={})
    hf.prepare(spec, builtin_catalog().manifests["base"])
    assert seen["downstream"] is None


# ------------------------------------------------------------ presets


def preset_steps(name: str) -> list[dict]:
    return builtin_preprocess_presets()[name].values["preprocess.steps"].value


@compute
def test_smoothquant_quarot_steps_keep_logits_on_both_tiny_qwen3(qwen3, windows):
    steps = copy.deepcopy(preset_steps("smoothquant_quarot"))
    assert [s["kind"] for s in steps] == ["rotation", "scaling"]
    steps[1]["calibration"] = CALIBRATION
    base = logits_of(qwen3)
    applied, _ = apply_to(qwen3, resolve_preprocess(Preprocess.model_validate({"steps": steps})))
    assert len(applied.results) == 2 and applied.online_ops is not None
    assert torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL)


@pytest.mark.parametrize(
    ("preprocess", "quantizer", "bits"),
    [("smoothquant", "smoothquant_w8a8", 8), ("awq", "awq_w4g128", 4), ("awq", "awq_w3g128", 3)],
)
def test_presets_validate_on_qwen3_0_6b(preprocess, quantizer, bits):
    raw = {
        "field": "quantization",
        "model": {"name": "qwen3-0.6b"},
        "method": {"name": "rtn", "preset": quantizer},
        "preprocess": {"preset": preprocess},
    }
    res = validate_module.resolve(raw, builtin_catalog())
    assert res.ok, [i.message for i in res.issues]
    rules = res.spec.method.config["rules"]
    assert rules[0]["quant"]["bits"] == bits
    assert res.spec.preprocess.steps[0].calibration.dataset == "pile_val"
    assert not res.fidelity.deviations


def test_smoothquant_quarot_validates_and_orders_the_rotation_step_first():
    raw = {
        "field": "quantization",
        "model": {"name": "qwen3-0.6b"},
        "method": {"name": "rtn", "preset": "quarot_w4a4kv4"},
        "preprocess": {"preset": "smoothquant_quarot"},
    }
    res = validate_module.resolve(raw, builtin_catalog())
    assert res.ok, [i.message for i in res.issues]
    assert [s.kind for s in res.spec.preprocess.steps] == ["rotation", "scaling"]
    assert "preprocess.order" not in [i.code for i in res.issues]
    steps = [s.model_dump(mode="json") for s in reversed(res.spec.preprocess.steps)]
    raw["preprocess"] = {"preset": "smoothquant_quarot", "steps": steps}
    swapped = validate_module.resolve(raw, builtin_catalog())
    assert "preprocess.order" in [i.code for i in swapped.issues]


# ------------------------------------------------------------ order of residual rotation and norm scaling


RESIDUAL_ROTATION = {
    "kind": "rotation",
    "entries": [
        {
            "position": "residual",
            "mode": "offline",
            "source": {"type": "random_hadamard", "seed": 0},
            "options": {"untie_embeddings": True, "center_embeddings": False},
        }
    ],
}


def norm_scaling_step() -> dict:
    return {
        "kind": "scaling",
        "calibration": CALIBRATION,
        "entries": [
            {
                "position": position,
                "mode": "offline",
                "source": {"type": "smoothquant", "alpha": 0.5, "weight_reduce": "group_max"},
            }
            for position in SQ_POSITIONS
        ],
    }


def apply_steps(model, steps: list[dict]) -> None:
    apply_to(model, resolve_preprocess(Preprocess.model_validate({"steps": copy.deepcopy(steps)})))


def parameters_of(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


@compute
def test_scaling_before_the_residual_rotation_is_cancelled_by_it(qwen3, windows):
    rotated_only = copy.deepcopy(qwen3)
    apply_steps(rotated_only, [RESIDUAL_ROTATION])
    apply_steps(qwen3, [norm_scaling_step(), RESIDUAL_ROTATION])
    expected = parameters_of(rotated_only)
    for name, value in parameters_of(qwen3).items():
        assert torch.allclose(value, expected[name], atol=1e-5, rtol=1e-4), name


@compute
def test_scaling_after_the_residual_rotation_changes_the_weights_and_keeps_the_logits(qwen3, windows):
    base = logits_of(qwen3)
    rotated_only = copy.deepcopy(qwen3)
    apply_steps(rotated_only, [RESIDUAL_ROTATION])
    apply_steps(qwen3, [RESIDUAL_ROTATION, norm_scaling_step()])
    rotated = parameters_of(rotated_only)
    for name in ("model.layers.0.self_attn.q_proj.weight", "model.layers.1.mlp.up_proj.weight"):
        assert (parameters_of(qwen3)[name] - rotated[name]).abs().max() > 1e-2, name
    gain = dict(qwen3.named_parameters())["model.layers.0.input_layernorm.weight"]
    assert (gain - 1).abs().max() > 1e-2
    assert torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL)


@compute
def test_scaling_calibration_sees_the_online_ops_of_an_earlier_rotation_step(qwen3, windows):
    online_rotation = {
        "kind": "rotation",
        "entries": [
            {"position": "linear.mlp.down_proj.input", "layers": "all", "mode": "online", "source": {"type": "hadamard"}}
        ],
    }
    seen: dict[str, int] = {}
    real = scaling.act_stats

    def spy(model, data, probes, stat):
        seen["pending"] = sum(bool(h) for h in model.model.layers[0].mlp.down_proj._forward_pre_hooks.values())
        return real(model, data, probes, stat)

    base = logits_of(qwen3)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(scaling, "act_stats", spy)
        apply_steps(qwen3, [online_rotation, norm_scaling_step()])
    assert seen["pending"] == 1
    assert torch.allclose(logits_of(qwen3), base, atol=LOGITS_ATOL)
