import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_llama, build_tiny_qwen3  # noqa: E402
from effeval import validate as validate_module  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.preprocess import (  # noqa: E402
    Preprocess,
    ResolvedPreprocess,
    builtin_preprocess_presets,
    preprocess_label,
    preprocess_title,
    resolve_preprocess,
)
from effeval.core.types import InProcess  # noqa: E402
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.ops import online, sites  # noqa: E402
from effeval.runtime import hf  # noqa: E402
from reference import quarot_rotation_official as official  # noqa: E402
from test_sites import padded_batch  # noqa: E402

compute = pytest.mark.compute
LOGITS_ATOL = 1e-4

RESIDUAL = {
    "position": "residual",
    "mode": "offline",
    "source": {"type": "random_hadamard", "seed": 0},
    "options": {"untie_embeddings": True},
}
V_HEAD = {"position": "attn.v_head", "mode": "offline", "source": {"type": "hadamard"}}
O_PROJ = {"position": "linear.attn.o_proj.input", "mode": "online", "source": {"type": "hadamard"}}
DOWN_PROJ = {"position": "linear.mlp.down_proj.input", "mode": "online", "source": {"type": "hadamard"}}
QK_ROPE = {"position": "attn.qk_rope", "mode": "online", "source": {"type": "hadamard"}}
QUAROT_ENTRIES = {
    "residual": RESIDUAL,
    "attn.v_head": V_HEAD,
    "linear.attn.o_proj.input": O_PROJ,
    "linear.mlp.down_proj.input": DOWN_PROJ,
    "attn.qk_rope": QK_ROPE,
}
MODELS = {
    "qwen3_wide": lambda: build_tiny_qwen3(head_dim=32, tie=True),
    "qwen3_square": lambda: build_tiny_qwen3(head_dim=16, tie=False),
}


def make_preprocess(*entries: dict) -> ResolvedPreprocess:
    step = {"kind": "rotation", "entries": [copy.deepcopy(e) for e in entries]}
    return resolve_preprocess(Preprocess.model_validate({"steps": [step]}))


def apply_to(model, preprocess: ResolvedPreprocess) -> engine.Applied:
    return engine.apply(model, preprocess, engine.make_context(model, None, {"checkpoint": "tiny"}))


@torch.no_grad()
def logits_of(model) -> torch.Tensor:
    input_ids, attention_mask = padded_batch()
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[attention_mask.bool()]


def snapshot(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def changed_parameters(model, before: dict[str, torch.Tensor]) -> list[str]:
    return [
        name for name, p in model.named_parameters() if name not in before or not torch.equal(p, before[name])
    ]


@pytest.fixture(params=sorted(MODELS))
def model(request):
    return MODELS[request.param]()


@compute
@pytest.mark.parametrize("position", sorted(QUAROT_ENTRIES))
def test_each_quarot_position_alone_keeps_logits(model, position):
    base = logits_of(model)
    before = snapshot(model)
    applied = apply_to(model, make_preprocess(QUAROT_ENTRIES[position]))
    assert applied.edits
    assert changed_parameters(model, before) or applied.online_ops is not None
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    if position == "attn.qk_rope":
        # Q and K share one orthogonal matrix, so QK^T is invariant and removing the ops changes nothing.
        modules = applied.online_ops.attention_modules
        assert len(modules) == len(model.model.layers)
        assert all(set(getattr(m, sites.ATTN_OPS_ATTR)) == {"attn.q", "attn.k"} for m in modules)
    elif applied.online_ops is not None:
        assert changed_parameters(model, before)
        applied.online_ops.remove()
        assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL), (
            "removing the online op must break the folded model, else the op does nothing"
        )


@compute
def test_all_quarot_positions_together_keep_logits(model):
    base = logits_of(model)
    applied = apply_to(model, make_preprocess(*QUAROT_ENTRIES.values()))
    got = logits_of(model)
    print(
        "all positions max abs logit diff",
        (got - base).abs().max().item(),
        "of scale",
        base.abs().max().item(),
    )
    assert torch.allclose(got, base, atol=LOGITS_ATOL)
    applied.online_ops.remove()
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
def test_only_the_attention_head_mix_runs_online_when_v_head_is_rotated():
    model = build_tiny_qwen3(head_dim=32)
    applied = apply_to(model, make_preprocess(V_HEAD, O_PROJ))
    ops = [op for result in applied.results for op in result.online]
    assert [type(op.fn) for op in ops] == [online.HeadMix]
    alone = apply_to(build_tiny_qwen3(head_dim=32), make_preprocess(O_PROJ))
    assert [type(op.fn) for r in alone.results for op in r.online] == [online.RightMultiply]


@compute
def test_residual_rotation_folds_every_norm_gain_and_untied_embeddings_differ():
    model = build_tiny_qwen3(head_dim=32, tie=True)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    apply_to(model, make_preprocess(RESIDUAL))
    hidden_gains = [
        p
        for n, p in model.named_parameters()
        if n.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", "model.norm.weight"))
    ]
    assert len(hidden_gains) == 2 * 2 + 1
    assert all(torch.equal(g, torch.ones_like(g)) for g in hidden_gains)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    assert not torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)
    assert model.config.tie_word_embeddings is False


@compute
def test_centering_embeddings_breaks_equivalence_like_quarot():
    model = build_tiny_qwen3(head_dim=32)
    base = logits_of(model)
    centred = copy.deepcopy(RESIDUAL)
    centred["options"]["center_embeddings"] = True
    apply_to(model, make_preprocess(centred))
    diff = (logits_of(model) - base).abs().max().item()
    print("centred embeddings max abs logit diff", diff)
    assert diff > 100 * LOGITS_ATOL


@compute
def test_tied_embeddings_without_untie_option_is_refused():
    model = build_tiny_qwen3(head_dim=32, tie=True)
    entry = copy.deepcopy(RESIDUAL)
    entry["options"] = {}
    with pytest.raises(engine.EngineError, match="untie_embeddings"):
        apply_to(model, make_preprocess(entry))


@compute
def test_unavailable_positions_and_duplicates_are_refused():
    model = build_tiny_qwen3(head_dim=32)
    online_residual = {**RESIDUAL, "mode": "online"}
    with pytest.raises(engine.EngineError, match="not available"):
        apply_to(model, make_preprocess(online_residual))
    with pytest.raises(engine.EngineError, match="not available"):
        apply_to(model, make_preprocess({**V_HEAD, "position": "norm.input_layernorm.out"}))
    with pytest.raises(engine.EngineError, match="twice"):
        apply_to(model, make_preprocess(DOWN_PROJ, DOWN_PROJ))


@compute
def test_frozen_source_version_must_match_the_installed_code():
    model = build_tiny_qwen3(head_dim=32)
    preprocess = make_preprocess(V_HEAD)
    stale = ResolvedPreprocess(preset=None, steps=preprocess.steps, versions={"rotation.hadamard": "0-stale"})
    with pytest.raises(engine.SourceVersionMismatch, match=r"rotation.hadamard"):
        apply_to(model, stale)


@compute
def test_unsupported_hadamard_size_is_refused_and_block_fallback_keeps_logits():
    model = build_tiny_qwen3(head_dim=32, intermediate=1216)
    with pytest.raises(engine.EngineError, match="no usable Hadamard"):
        apply_to(model, make_preprocess(DOWN_PROJ))
    fallback = {**DOWN_PROJ, "source": {"type": "random_hadamard", "fallback": "block_hadamard"}}
    base = logits_of(model)
    apply_to(model, make_preprocess(fallback))
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
def test_residual_with_random_orthogonal_source_keeps_logits():
    model = build_tiny_qwen3(head_dim=32)
    base = logits_of(model)
    entry = {**RESIDUAL, "source": {"type": "random_orthogonal", "seed": 3}}
    apply_to(model, make_preprocess(entry))
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
def test_layer_selection_limits_online_rotation_to_the_chosen_layers():
    model = build_tiny_qwen3(head_dim=32, layers=3)
    base = logits_of(model)
    before = snapshot(model)
    only_layer_one = {**DOWN_PROJ, "layers": "1"}
    applied = apply_to(model, make_preprocess(only_layer_one))
    changed = changed_parameters(model, before)
    assert changed == ["model.layers.1.mlp.down_proj.weight"]
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    applied.online_ops.remove()


@compute
def test_rotated_weights_equal_the_official_quarot_code_on_tiny_llama(register_architecture):
    """QuaRot fuse_layer_norms + rotate_model (vendored; Llama: head_dim = hidden / heads) vs this engine."""
    ours = build_tiny_llama()
    theirs = copy.deepcopy(ours)
    register_architecture(ours)

    official.fuse_layer_norms(theirs)
    torch.manual_seed(0)
    official.rotate_model(theirs)

    centred = {**RESIDUAL, "options": {"untie_embeddings": True, "center_embeddings": True}}
    apply_to(ours, make_preprocess(centred, V_HEAD, O_PROJ, DOWN_PROJ))

    expected = {n: p for n, p in theirs.named_parameters() if "norm" not in n}
    got = dict(ours.named_parameters())
    assert expected.keys() <= got.keys() and len(expected) == 2 * 7 + 2
    worst = max((got[n] - p).abs().max().item() for n, p in expected.items())
    print("max abs weight difference to the official QuaRot code", worst)
    assert worst < 1e-6
    for name, param in expected.items():
        assert torch.allclose(got[name], param, atol=1e-6), name


@compute
def test_official_comparison_is_sensitive_to_the_rotation_seed(register_architecture):
    ours = build_tiny_llama()
    theirs = copy.deepcopy(ours)
    register_architecture(ours)
    official.fuse_layer_norms(theirs)
    torch.manual_seed(0)
    official.rotate_model(theirs)
    other_seed = {**RESIDUAL, "source": {"type": "random_hadamard", "seed": 1}}
    other_seed["options"] = {"untie_embeddings": True, "center_embeddings": True}
    apply_to(ours, make_preprocess(other_seed, V_HEAD, O_PROJ, DOWN_PROJ))
    assert (
        dict(ours.named_parameters())["model.embed_tokens.weight"] - theirs.model.embed_tokens.weight
    ).abs().max() > 1e-3


def quarot_preset_steps() -> list[dict]:
    return copy.deepcopy(builtin_preprocess_presets()["quarot"].values["preprocess.steps"].value)


def resolve_quarot(method: str = "gptq", model: str = "qwen3-0.6b", steps: list[dict] | None = None):
    preprocess: dict = {"preset": "quarot"}
    if steps is not None:
        preprocess["steps"] = steps
    raw = {
        "field": "quantization",
        "model": {"name": model},
        "method": {"name": method, "preset": "quarot_w4a4kv4"},
        "preprocess": preprocess,
    }
    return validate_module.resolve(raw, builtin_catalog())


@compute
def test_the_quarot_preset_steps_keep_logits_on_both_tiny_qwen3(model):
    base = logits_of(model)
    preset = Preprocess.model_validate({"preset": "quarot", "steps": quarot_preset_steps()})
    applied = apply_to(model, resolve_preprocess(preset))
    assert len(applied.results) == 1 and applied.online_ops is not None
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@pytest.mark.parametrize("method", ["gptq", "rtn"])
def test_quarot_with_its_w4a4kv4_quantizer_preset_validates_on_qwen3_0_6b(method):
    res = resolve_quarot(method)
    assert res.ok, [i.message for i in res.issues]
    assert [e.position for e in res.spec.preprocess.steps[0].entries] == [
        "residual",
        "attn.v_head",
        "linear.attn.o_proj.input",
        "linear.mlp.down_proj.input",
        "attn.qk_rope",
    ]
    assert res.fidelity.deviations == []


def test_label_is_preset_title_plus_method_title_and_marks_deviations():
    titles = {name: m.title for name, m in builtin_catalog().manifests.items()}
    assert titles["gptq"] == "GPTQ"
    res = resolve_quarot("gptq")
    assert preprocess_label(res.spec.preprocess, titles["gptq"]) == "QuaRot + GPTQ"
    assert preprocess_label(None, titles["gptq"]) == "GPTQ"

    steps = quarot_preset_steps()
    steps[0]["entries"][0]["source"]["seed"] = 1
    deviating = resolve_quarot("gptq", steps=steps)
    assert deviating.ok
    assert preprocess_label(deviating.spec.preprocess, titles["gptq"]) == "QuaRot* + GPTQ"
    assert preprocess_title(deviating.spec.preprocess) == "QuaRot*"

    custom = Preprocess.model_validate({"steps": steps})
    assert preprocess_label(custom, titles["rtn"]) == "Custom preprocessing + RTN"


def test_hadamard_size_rule_checks_each_entry_against_the_model_dimension():
    steps = quarot_preset_steps()
    unsupported = resolve_quarot("gptq", model="qwen3-4b", steps=steps)
    errors = [i for i in unsupported.issues if i.code == "preprocess.hadamard_size"]
    assert [e.paths for e in errors] == [["preprocess.steps[0].entries[3].source"]]
    assert "9728" in errors[0].message

    steps[0]["entries"][3]["source"] = {"type": "random_hadamard", "seed": 0, "fallback": "block_hadamard"}
    assert resolve_quarot("gptq", model="qwen3-4b", steps=steps).ok

    steps[0]["entries"][3]["source"] = {"type": "random_orthogonal", "seed": 0}
    assert resolve_quarot("gptq", model="qwen3-4b", steps=steps).ok


@compute
def test_hf_prepare_applies_the_jobs_preprocess_and_leaves_jobs_without_one_alone(monkeypatch):
    base_manifest = builtin_catalog().manifests["base"]
    spec = SimpleNamespace(
        preprocess=make_preprocess(DOWN_PROJ),
        model=SimpleNamespace(checkpoint="tiny", revision=None),
        method=SimpleNamespace(config={}),
    )
    model = build_tiny_qwen3(head_dim=32)
    before = snapshot(model)
    monkeypatch.setattr(hf, "load", lambda _spec: InProcess(model=model, tokenizer=None))
    handle = hf.prepare(spec, base_manifest)
    assert handle.model is model
    assert "model.layers.0.mlp.down_proj.weight" in changed_parameters(model, before)

    untouched = build_tiny_qwen3(head_dim=32)
    before = snapshot(untouched)
    spec.preprocess = None
    monkeypatch.setattr(hf, "load", lambda _spec: InProcess(model=untouched, tokenizer=None))
    hf.prepare(spec, base_manifest)
    assert changed_parameters(untouched, before) == []


def _with(entry: dict, **changes) -> dict:
    return {**entry, **changes}


VALUE_PATH_CASES = {
    "v_head_random_orthogonal_o_proj_hadamard": (
        _with(V_HEAD, source={"type": "random_orthogonal", "seed": 4}),
        O_PROJ,
    ),
    "v_head_hadamard_o_proj_random_hadamard": (
        V_HEAD,
        _with(O_PROJ, source={"type": "random_hadamard", "seed": 5}),
    ),
    "v_head_on_layer_0_only_with_o_proj_everywhere": (
        _with(V_HEAD, source={"type": "random_orthogonal", "seed": 4}, layers="0"),
        O_PROJ,
    ),
    "v_head_random_orthogonal_without_o_proj_entry": (
        _with(V_HEAD, source={"type": "random_orthogonal", "seed": 4}),
        None,
    ),
    "v_head_on_layer_0_o_proj_on_layer_1": (
        _with(V_HEAD, source={"type": "random_orthogonal", "seed": 4}, layers="0"),
        _with(O_PROJ, layers="1"),
    ),
}


@compute
@pytest.mark.parametrize("case", sorted(VALUE_PATH_CASES))
def test_value_head_and_o_proj_rotations_with_non_symmetric_matrices_keep_logits(case):
    """Neither matrix is a symmetric Hadamard pair, so the o_proj online op is the general matrix path."""
    model = build_tiny_qwen3(head_dim=32, layers=3)
    base = logits_of(model)
    entries = [e for e in VALUE_PATH_CASES[case] if e is not None]
    applied = apply_to(model, make_preprocess(*entries))
    got = logits_of(model)
    assert torch.allclose(got, base, atol=LOGITS_ATOL), (got - base).abs().max().item()
    if applied.online_ops is not None:
        applied.online_ops.remove()
        assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
