from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_llama, build_tiny_qwen3  # noqa: E402
from effeval.core.preprocess import Preprocess, resolve_preprocess  # noqa: E402
from effeval.methods.preprocess import engine, smoothattention  # noqa: E402
from effeval.ops import online  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from test_sites import padded_batch  # noqa: E402

compute = pytest.mark.compute
LOGITS_ATOL = 1e-4
OPERAND_ATOL = 1e-5
CALIBRATION = {"dataset": "wikitext2", "n_samples": 8, "seqlen": 32, "seed": 0}
MODELS = {"qwen3": lambda: build_tiny_qwen3(head_dim=32), "llama": build_tiny_llama}
PRE_ROPE = "attn.qk_pre_rope"


@pytest.fixture(params=sorted(MODELS))
def model(request, register_architecture):
    built = MODELS[request.param]()
    register_architecture(built)
    return built


@pytest.fixture
def windows(monkeypatch):
    generator = torch.Generator().manual_seed(3)
    data = [torch.randint(0, 97, (1, 32), generator=generator) for _ in range(8)]
    monkeypatch.setattr(engine, "calibration_windows", lambda _calibration, _tokenizer_id: data)
    return data


def smooth_entry(position: str, mode: str, **source) -> dict:
    return {"position": position, "mode": mode, "source": {"type": "smoothattention", **source}}


def make_preprocess(*steps: tuple[str, list[dict]]):
    built = [{"kind": kind, "calibration": CALIBRATION, "entries": entries} for kind, entries in steps]
    return resolve_preprocess(Preprocess.model_validate({"steps": built}))


def make_context(model) -> engine.Context:
    return engine.make_context(
        model, SimpleNamespace(name_or_path="tiny"), {"checkpoint": "tiny"}, artifacts=ArtifactCache()
    )


def apply_to(model, preprocess) -> engine.Applied:
    return engine.apply(model, preprocess, make_context(model))


@torch.no_grad()
def logits_of(model) -> torch.Tensor:
    input_ids, attention_mask = padded_batch()
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[attention_mask.bool()]


@torch.no_grad()
def operand_of(model, site: str, ops: list[online.OnlineOp] = (), layer: int = 0) -> torch.Tensor:
    """The operand at `site` of `layer` on the padded batch, after `ops` (registered in one call)."""
    seen = []

    def observe(x):
        seen.append(x.detach().clone())
        return x

    attachment = online.register(model, [*ops, online.OnlineOp(site, observe, [layer])])
    try:
        input_ids, attention_mask = padded_batch()
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        attachment.remove()
    return seen[0]


def detach(applied: engine.Applied) -> None:
    """Remove the engine's own attachment of the online ops (they are re-registered by the caller)."""
    if applied.online_ops is not None:
        applied.online_ops.remove()
        applied.online_ops = None


def operand_after(model, applied: engine.Applied, site: str, layer: int = 0) -> torch.Tensor:
    """Operand at `site` with the applied online ops attached (replacing the engine's own attachment)."""
    detach(applied)
    return operand_of(model, site, [op for result in applied.results for op in result.online], layer)


@torch.no_grad()
def absmax_of(model, windows, site: str, ops: list[online.OnlineOp] = (), layer: int = 0) -> torch.Tensor:
    """max |operand| over all windows, [heads, D]; independent of ops.calibration."""
    best = None

    def observe(x):
        nonlocal best
        current = x.detach().abs().float().amax(dim=(0, 2))
        best = current if best is None else torch.maximum(best, current)
        return x

    attachment = online.register(model, [*ops, online.OnlineOp(site, observe, [layer])])
    try:
        for window in windows:
            model(window, use_cache=False)
    finally:
        attachment.remove()
    return best


def tied(a: torch.Tensor) -> torch.Tensor:
    half = a.shape[-1] // 2
    both = torch.maximum(a[..., :half], a[..., half:])
    return torch.cat([both, both], dim=-1)


def lam_of(a: torch.Tensor, alpha: float = 0.5, shared: bool = False) -> torch.Tensor:
    if shared:
        a = a.amax(dim=0)
    return a.clamp(min=1e-5).pow(alpha)


def by_head(lam: torch.Tensor) -> torch.Tensor:
    """[Hkv, D] or [D] scale as a broadcastable [1, Hkv, 1, D] / [1, 1, 1, D]."""
    return lam[None, :, None, :] if lam.ndim == 2 else lam[None, None, None, :]


def snapshot(model) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def changed_names(model, before: dict[str, torch.Tensor]) -> set[str]:
    return {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}


# ------------------------------------------------------------ pure helpers


def test_pair_tie_takes_the_max_of_each_rope_pair():
    a = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    assert torch.equal(smoothattention.pair_tie(a), torch.tensor([[3.0, 5.0, 3.0, 5.0]]))


def test_expand_to_query_heads_repeats_each_kv_row_for_consecutive_query_heads():
    t = torch.tensor([[1.0], [2.0]])
    assert smoothattention.expand_to_query_heads(t, 4).flatten().tolist() == [1.0, 1.0, 2.0, 2.0]


# ------------------------------------------------------------ attn.qk_pre_rope


@compute
@pytest.mark.parametrize("granularity", ["per_head", "shared"])
def test_pre_rope_scaling_keeps_logits_and_scales_the_k_cache(model, windows, granularity):
    base = logits_of(model)
    k_before = operand_of(model, "attn.k")
    a_ref = absmax_of(model, windows, "attn.k")
    before = snapshot(model)
    applied = apply_to(
        model,
        make_preprocess(("scaling", [smooth_entry(PRE_ROPE, "offline", alpha=1.0, granularity=granularity)])),
    )
    assert applied.online_ops is None and len(applied.edits) == 1
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    lam = lam_of(tied(a_ref), alpha=1.0, shared=granularity == "shared")
    assert lam.max() / lam.min() > 1.5
    k_after = operand_of(model, "attn.k")
    assert torch.allclose(k_after * by_head(lam), k_before, atol=OPERAND_ATOL)
    assert not torch.allclose(k_after, k_before, atol=1e-3)
    has_norms = any(n.endswith("q_norm.weight") for n in before)
    suffixes = ("q_norm.weight", "k_norm.weight") if has_norms else ("q_proj.weight", "k_proj.weight")
    assert changed_names(model, before) == {n for n in before if n.endswith(suffixes)}


@compute
def test_pre_rope_per_head_on_qwen3_installs_headwise_gains(windows):
    model = build_tiny_qwen3(head_dim=32)
    apply_to(model, make_preprocess(("scaling", [smooth_entry(PRE_ROPE, "offline", granularity="per_head")])))
    attention = model.model.layers[0].self_attn
    assert tuple(attention.k_norm.weight.shape) == (2, 32) and tuple(attention.q_norm.weight.shape) == (4, 32)
    shared = build_tiny_qwen3(head_dim=32)
    apply_to(shared, make_preprocess(("scaling", [smooth_entry(PRE_ROPE, "offline", granularity="shared")])))
    assert tuple(shared.model.layers[0].self_attn.k_norm.weight.shape) == (32,)


@compute
def test_fold_with_an_untied_lambda_breaks_the_model(model):
    base = logits_of(model)
    net = make_context(model).net
    heads, dim = net.shape.kv_heads, net.shape.head_dim
    generator = torch.Generator().manual_seed(5)
    lam = torch.rand(heads, dim, generator=generator) + 0.5
    assert not torch.allclose(lam[:, : dim // 2], lam[:, dim // 2 :])
    for layer in range(len(net.layers)):
        smoothattention.fold_qk_pre_rope(net, layer, lam)
    assert (logits_of(model) - base).abs().max() > 1e-3


@compute
def test_fold_with_a_tied_lambda_keeps_the_model(model):
    base = logits_of(model)
    net = make_context(model).net
    generator = torch.Generator().manual_seed(5)
    lam = smoothattention.pair_tie(
        torch.rand(net.shape.kv_heads, net.shape.head_dim, generator=generator) + 0.5
    )
    for layer in range(len(net.layers)):
        smoothattention.fold_qk_pre_rope(net, layer, lam)
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
def test_gqa_mapping_is_checked_by_a_modulo_head_mapping_changing_the_logits(model, windows, monkeypatch):
    base = logits_of(model)
    step = ("scaling", [smooth_entry(PRE_ROPE, "offline", alpha=1.0, granularity="per_head")])
    monkeypatch.setattr(
        smoothattention, "expand_to_query_heads", lambda t, heads: t.repeat(heads // t.shape[0], 1)
    )
    apply_to(model, make_preprocess(step))
    assert (logits_of(model) - base).abs().max() > 1e-3


# ------------------------------------------------------------ attn.qk_rope (online)


@compute
@pytest.mark.parametrize("granularity", ["per_head", "shared"])
def test_qk_rope_online_scaling_scales_k_and_q_operands(model, windows, granularity):
    base = logits_of(model)
    k_before, q_before = operand_of(model, "attn.k"), operand_of(model, "attn.q")
    a_ref = absmax_of(model, windows, "attn.k")
    applied = apply_to(
        model, make_preprocess(("scaling", [smooth_entry("attn.qk_rope", "online", granularity=granularity)]))
    )
    assert applied.online_ops is not None
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    lam = lam_of(a_ref, shared=granularity == "shared")
    q_lam = lam if lam.ndim == 1 else lam.repeat_interleave(2, dim=0)
    assert torch.allclose(operand_after(model, applied, "attn.k"), k_before / by_head(lam), atol=OPERAND_ATOL)
    assert torch.allclose(
        operand_after(model, applied, "attn.q"), q_before * by_head(q_lam), atol=OPERAND_ATOL
    )
    assert (k_before / by_head(lam) - k_before).abs().max() > 1e-3


@compute
def test_rotation_then_qk_rope_scaling_chains_in_step_order(model, windows):
    base = logits_of(model)
    k_before = operand_of(model, "attn.k")
    rotation = {
        "position": "attn.qk_rope",
        "mode": "online",
        "source": {"type": "random_hadamard", "seed": 1},
    }
    preprocess = make_preprocess(
        ("rotation", [rotation]),
        ("scaling", [smooth_entry("attn.qk_rope", "online", granularity="per_head")]),
    )
    applied = apply_to(model, preprocess)
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    detach(applied)
    matrix = applied.results[0].online[0].fn.matrix
    rotate = online.OnlineOp("attn.k", lambda x: x @ matrix)
    lam = lam_of(absmax_of(model, windows, "attn.k", [rotate]))
    k_ops = [op for result in applied.results for op in result.online if op.site == "attn.k"]
    assert len(k_ops) == 1 + len(model.model.layers)
    expected = (k_before @ matrix) / by_head(lam)
    assert torch.allclose(operand_after(model, applied, "attn.k"), expected, atol=OPERAND_ATOL)
    assert (expected - (k_before / by_head(lam)) @ matrix).abs().max() > 1e-3


# ------------------------------------------------------------ attn.v_head


@compute
@pytest.mark.parametrize("granularity", ["per_head", "shared"])
def test_v_head_scaling_keeps_logits_and_scales_the_v_cache(model, windows, granularity):
    base = logits_of(model)
    v_before = operand_of(model, "attn.v")
    a_ref = absmax_of(model, windows, "attn.v")
    before = snapshot(model)
    applied = apply_to(
        model,
        make_preprocess(
            ("scaling", [smooth_entry("attn.v_head", "offline", alpha=1.0, granularity=granularity)])
        ),
    )
    assert applied.online_ops is None
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    lam = lam_of(a_ref, alpha=1.0, shared=granularity == "shared")
    assert lam.max() / lam.min() > 1.5
    assert torch.allclose(operand_of(model, "attn.v") * by_head(lam), v_before, atol=OPERAND_ATOL)
    assert changed_names(model, before) == {
        n for n in before if n.endswith(("v_proj.weight", "o_proj.weight"))
    }


@compute
def test_v_head_scaling_after_an_online_v_rotation_step_is_refused(model, windows):
    rotation = {"position": "attn.v_head", "mode": "online", "source": {"type": "random_hadamard", "seed": 1}}
    preprocess = make_preprocess(
        ("rotation", [rotation]),
        ("scaling", [smooth_entry("attn.v_head", "offline", granularity="per_head")]),
    )
    with pytest.raises(engine.EngineError, match="put the scaling step first"):
        apply_to(model, preprocess)


@compute
def test_v_head_scaling_is_refused_after_an_online_op_on_the_o_proj_input(model, windows):
    ctx = make_context(model)
    ctx.step = make_preprocess(("scaling", [smooth_entry("attn.v_head", "offline")])).steps[0]
    ctx.pending_online = [online.OnlineOp("linear.attn.o_proj.input", lambda x: x)]
    with pytest.raises(engine.EngineError, match="put the scaling step first"):
        smoothattention.apply_smoothattention(model, ctx.net.graph, ctx.step.entries, ctx)


# ------------------------------------------------------------ validation of entries


@compute
def test_entries_are_checked(model, windows):
    with pytest.raises(engine.EngineError, match="not available"):
        apply_to(model, make_preprocess(("scaling", [smooth_entry(PRE_ROPE, "online")])))
    with pytest.raises(engine.EngineError, match="not available"):
        apply_to(model, make_preprocess(("scaling", [smooth_entry("attn.qk_rope", "offline")])))
    with pytest.raises(engine.EngineError, match="appears twice"):
        apply_to(
            model,
            make_preprocess(
                ("scaling", [smooth_entry(PRE_ROPE, "offline"), smooth_entry(PRE_ROPE, "offline")])
            ),
        )
    with pytest.raises(engine.EngineError, match="smoothattention scales the K/V cache"):
        apply_to(model, make_preprocess(("scaling", [smooth_entry("linear.mlp.down_proj.input", "offline")])))
