import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_llama, build_tiny_qwen3  # noqa: E402
from effeval.core.preprocess import Preprocess, ResolvedPreprocess, resolve_preprocess  # noqa: E402
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.methods.preprocess import rotation as rotation_module  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.online import HeadMix, HeadwiseRightMultiply, RightMultiply  # noqa: E402
from test_sites import padded_batch  # noqa: E402

compute = pytest.mark.compute
LOGITS_ATOL = 1e-4
HEADS, KV_HEADS = 4, 2

MODELS = {
    "qwen3": lambda: build_tiny_qwen3(head_dim=32, tie=False),
    "llama": lambda: build_tiny_llama(),
}
SEEDED_SOURCES = ["random_hadamard", "random_orthogonal"]


def v_head(mode: str, source: str = "random_hadamard", **options) -> dict:
    return {
        "position": "attn.v_head",
        "mode": mode,
        "source": {"type": source, "seed": 3},
        "options": options,
    }


def qk_rope(source: str = "random_hadamard", **options) -> dict:
    return {
        "position": "attn.qk_rope",
        "mode": "online",
        "source": {"type": source, "seed": 3},
        "options": options,
    }


O_PROJ_ONLINE = {
    "position": "linear.attn.o_proj.input",
    "mode": "online",
    "source": {"type": "random_hadamard", "seed": 5},
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
    return [name for name, p in model.named_parameters() if not torch.equal(p, before[name])]


def online_ops(applied: engine.Applied) -> list:
    return [op for result in applied.results for op in result.online]


def modulo_mapping(matrices: torch.Tensor, heads: int) -> torch.Tensor:
    """Wrong GQA mapping: query head h takes the matrix of KV head h % kv_heads."""
    return matrices[torch.arange(heads) % matrices.shape[0]]


@pytest.fixture(params=sorted(MODELS))
def model(request, register_architecture):
    model = MODELS[request.param]()
    register_architecture(model)
    return model


@compute
@pytest.mark.parametrize("source", SEEDED_SOURCES)
def test_v_head_online_keeps_logits_and_leaves_v_proj_alone(model, source):
    base = logits_of(model)
    before = snapshot(model)
    applied = apply_to(model, make_preprocess(v_head("online", source)))
    changed = changed_parameters(model, before)
    assert changed
    assert all(name.endswith("o_proj.weight") for name in changed), changed
    assert [(op.site, type(op.fn)) for op in online_ops(applied)] == [("attn.v", RightMultiply)]
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    applied.online_ops.remove()
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL), (
        "removing the online V rotation must break the folded o_proj, else the op does nothing"
    )


@compute
def test_v_head_online_matches_offline_o_proj_fold(model):
    offline = copy.deepcopy(model)
    apply_to(offline, make_preprocess(v_head("offline")))
    apply_to(model, make_preprocess(v_head("online")))
    for (name, online_param), offline_param in zip(
        model.named_parameters(), offline.parameters(), strict=True
    ):
        if name.endswith("o_proj.weight"):
            assert torch.allclose(online_param, offline_param, atol=1e-6), name


@compute
def test_v_head_online_with_o_proj_online_keeps_logits(model):
    base = logits_of(model)
    applied = apply_to(model, make_preprocess(v_head("online"), O_PROJ_ONLINE))
    sites_and_types = sorted((op.site, type(op.fn).__name__) for op in online_ops(applied))
    assert sites_and_types == [
        ("attn.v", "RightMultiply"),
        ("linear.attn.o_proj.input", "RightMultiply"),
    ]
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    applied.online_ops.remove()
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
@pytest.mark.parametrize("source", SEEDED_SOURCES)
@pytest.mark.parametrize("mode", ["offline", "online"])
def test_v_head_per_kv_head_keeps_logits(model, mode, source):
    base = logits_of(model)
    applied = apply_to(model, make_preprocess(v_head(mode, source, per_kv_head=True)))
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    ops = online_ops(applied)
    if mode == "online":
        (op,) = ops
        assert isinstance(op.fn, HeadwiseRightMultiply)
        assert op.fn.matrices.shape[0] == KV_HEADS
        assert not torch.allclose(op.fn.matrices[0], op.fn.matrices[1])
        applied.online_ops.remove()
        assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)
    else:
        assert ops == []


@compute
@pytest.mark.parametrize("mode", ["offline", "online"])
def test_v_head_per_kv_head_with_o_proj_online_keeps_logits(model, mode):
    base = logits_of(model)
    apply_to(model, make_preprocess(v_head(mode, per_kv_head=True), O_PROJ_ONLINE))
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
@pytest.mark.parametrize("source", SEEDED_SOURCES)
def test_qk_rope_per_kv_head_keeps_logits_with_distinct_matrices(model, source):
    base = logits_of(model)
    applied = apply_to(model, make_preprocess(qk_rope(source, per_kv_head=True)))
    by_site = {op.site: op.fn for op in online_ops(applied)}
    assert set(by_site) == {"attn.q", "attn.k"}
    assert all(isinstance(fn, HeadwiseRightMultiply) for fn in by_site.values())
    k_matrices, q_matrices = by_site["attn.k"].matrices, by_site["attn.q"].matrices
    assert k_matrices.shape[0] == KV_HEADS and q_matrices.shape[0] == HEADS
    assert not torch.allclose(k_matrices[0], k_matrices[1])
    assert torch.equal(q_matrices[0], k_matrices[0]) and torch.equal(q_matrices[1], k_matrices[0])
    assert torch.equal(q_matrices[2], k_matrices[1]) and torch.equal(q_matrices[3], k_matrices[1])
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
def test_qk_rope_per_kv_head_needs_both_sides(model):
    base = logits_of(model)
    applied = apply_to(model, make_preprocess(qk_rope(per_kv_head=True)))
    for module in applied.online_ops.attention_modules:
        getattr(module, sites.ATTN_OPS_ATTR).pop("attn.q")
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
@pytest.mark.parametrize(
    "entry",
    [
        v_head("offline", per_kv_head=True),
        v_head("online", per_kv_head=True),
        qk_rope(per_kv_head=True),
    ],
    ids=["v_offline", "v_online", "qk_rope"],
)
def test_wrong_query_head_mapping_breaks_logits(model, monkeypatch, entry):
    base = logits_of(model)
    monkeypatch.setattr(rotation_module, "expand_to_query_heads", modulo_mapping)
    apply_to(model, make_preprocess(entry))
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL), (
        "the equivalence check must be sensitive to which KV head a query head belongs to"
    )


@compute
def test_wrong_query_head_mapping_breaks_o_proj_online_composition(model, monkeypatch):
    base = logits_of(model)
    monkeypatch.setattr(rotation_module, "expand_to_query_heads", modulo_mapping)
    apply_to(model, make_preprocess(v_head("offline", per_kv_head=True), O_PROJ_ONLINE))
    assert not torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
@pytest.mark.parametrize(
    "entry",
    [v_head("offline", per_kv_head=True), qk_rope(per_kv_head=True)],
    ids=["attn.v_head", "attn.qk_rope"],
)
def test_per_kv_head_with_hadamard_raises(model, entry):
    entry["source"] = {"type": "hadamard"}
    with pytest.raises(engine.EngineError, match="hadamard is deterministic"):
        apply_to(model, make_preprocess(entry))


@compute
def test_unknown_kv_option_raises(model):
    with pytest.raises(engine.EngineError, match="unknown options"):
        apply_to(model, make_preprocess(v_head("offline", bogus=True)))


@compute
def test_shared_matrix_path_is_unchanged_by_the_option_default(model):
    base = logits_of(model)
    explicit = copy.deepcopy(model)
    apply_to(model, make_preprocess(v_head("offline")))
    apply_to(explicit, make_preprocess(v_head("offline", per_kv_head=False)))
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), explicit.parameters(), strict=True))
    assert torch.allclose(logits_of(model), base, atol=LOGITS_ATOL)


@compute
def test_plain_kron_path_still_uses_head_mix():
    model = build_tiny_qwen3(head_dim=32)
    hadamard_v = {"position": "attn.v_head", "mode": "offline", "source": {"type": "hadamard"}}
    hadamard_o = {**O_PROJ_ONLINE, "source": {"type": "hadamard"}}
    applied = apply_to(model, make_preprocess(hadamard_v, hadamard_o))
    assert [type(op.fn) for op in online_ops(applied)] == [HeadMix]
