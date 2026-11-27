"""Factored q/k/v layout, its positions, the prepare stage in `_apply`, and validation against the layout."""

from types import SimpleNamespace
from typing import get_args

import pytest

torch = pytest.importorskip("torch")

from catalogs import CATALOG  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core import registry  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.graph import input_width, positions  # noqa: E402
from effeval.core.lowrank import (  # noqa: E402
    ARCHITECTURE_ATTR,
    ARCHITECTURE_TRANSFORMS,
    QKV_SHARED,
    factor_qkv_architecture,
    qkv_budget,
    transform_architecture,
)
from effeval.core.manifest import MethodManifest  # noqa: E402
from effeval.methods.quantization.gptq import sequential_groups  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.runtime import hf  # noqa: E402
from effeval.validate import validate  # noqa: E402

pytestmark = pytest.mark.compute

DENSE = builtin_catalog().architectures["Qwen3ForCausalLM"]
LATENT = "the input is the latent of {}, whose width the decomposition chooses per layer"


def test_factored_linears_insert_the_down_factor_before_q_and_move_the_projections_to_up():
    assert factor_qkv_architecture(DENSE).linears == {
        "self_attn.qkv_down": "attn.qkv_down",
        "self_attn.q_proj.up": "attn.q_proj",
        "self_attn.k_proj.up": "attn.k_proj",
        "self_attn.v_proj.up": "attn.v_proj",
        "self_attn.o_proj": "attn.o_proj",
        "mlp.gate_proj": "mlp.gate_proj",
        "mlp.up_proj": "mlp.up_proj",
        "mlp.down_proj": "mlp.down_proj",
    }
    assert next(iter(factor_qkv_architecture(DENSE).linears)) == "self_attn.qkv_down"


def test_factored_graph_routes_the_norm_through_qkv_down_with_unchanged_projection_sites():
    dense, factored = DENSE.graph, factor_qkv_architecture(DENSE)
    graph = factored.graph
    assert (
        graph.node("qkv_down").canon == "attn.qkv_down"
        and graph.node("qkv_down").module == "self_attn.qkv_down"
    )
    assert [graph.node(n).module for n in ("q", "k", "v")] == [
        "self_attn.q_proj.up",
        "self_attn.k_proj.up",
        "self_attn.v_proj.up",
    ]
    new = [e for e in graph.edges if e not in dense.edges]
    assert new == [
        ("input_layernorm", "qkv_down", "linear.attn.qkv_down.input"),
        ("qkv_down", "q", "linear.attn.q_proj.input"),
        ("qkv_down", "k", "linear.attn.k_proj.input"),
        ("qkv_down", "v", "linear.attn.v_proj.input"),
    ]
    assert [e for e in dense.edges if e not in graph.edges] == [
        ("input_layernorm", "q", "linear.attn.q_proj.input"),
        ("input_layernorm", "k", "linear.attn.k_proj.input"),
        ("input_layernorm", "v", "linear.attn.v_proj.input"),
    ]
    assert factored.name == DENSE.name and factored.graph_overrides == {}


def test_sequential_groups_gain_a_leading_qkv_down_group():
    factored = factor_qkv_architecture(DENSE).graph
    assert sequential_groups(factored) == [["attn.qkv_down"], *sequential_groups(DENSE.graph)]
    assert sequential_groups(factored)[1] == ["attn.q_proj", "attn.k_proj", "attn.v_proj"]


def test_transform_dispatches_by_name_and_rejects_unknown_names():
    assert transform_architecture(DENSE, QKV_SHARED) == factor_qkv_architecture(DENSE)
    with pytest.raises(ValueError, match="unknown architecture transform"):
        transform_architecture(DENSE, "nope")


def test_every_manifest_transform_name_has_an_implementation():
    optional_literal = MethodManifest.model_fields["architecture_transform"].annotation
    (literal,) = (arg for arg in get_args(optional_literal) if arg is not type(None))
    assert set(ARCHITECTURE_TRANSFORMS) == set(get_args(literal))


def test_a_value_projection_reading_another_norm_is_rejected():
    llama_like = DENSE.model_copy(
        update={
            "graph_overrides": {
                "edges_remove": [["input_layernorm", "v", "linear.attn.v_proj.input"]],
                "edges_add": [["post_attention_layernorm", "v", "linear.attn.v_proj.input"]],
            }
        }
    )
    with pytest.raises(ValueError, match="one shared norm"):
        factor_qkv_architecture(llama_like)


def test_a_graph_without_support_is_rejected():
    with pytest.raises(ValueError, match="no supported layer graph"):
        factor_qkv_architecture(DENSE.model_copy(update={"graph": None}))


@pytest.mark.parametrize(
    ("code", "kept", "param", "latent"),
    [("qwen3-0.6b", 21504, 0.9375, 0.375), ("qwen3-4b", 69120, 1.0625, 0.9375)],
)
def test_qkv_budget_matches_the_released_formula(code, kept, param, latent):
    budget = qkv_budget(CATALOG.models[code].shape, 1.5)
    assert budget.kept == kept
    assert budget.mean_rank == kept / CATALOG.models[code].shape.num_layers
    assert budget.param_ratio == pytest.approx(param, abs=1e-12)
    assert budget.latent_ratio == pytest.approx(latent, abs=1e-12)


def test_positions_report_qkv_down_like_a_norm_consumer_and_the_latent_as_unavailable():
    shape = CATALOG.models["qwen3-0.6b"].shape
    table = positions(factor_qkv_architecture(DENSE).graph, shape)
    down = table["linear.attn.qkv_down.input"]
    assert down["rotation"]["offline"] == "same as the norm output: rotate the residual instead"
    assert down["rotation"]["online"] == "ok"
    assert down["scaling"]["offline"] == "ok"
    assert table["residual"]["rotation"]["offline"] == "ok"
    for canon in ("q", "k", "v"):
        cell = table[f"linear.attn.{canon}_proj.input"]
        reason = LATENT.format("attn.qkv_down")
        assert all(
            reason in text and "latent rotation in the method config" in text
            for kinds in cell.values()
            for text in kinds.values()
        )
    assert table["attn.v_head"]["rotation"]["offline"] == "ok"
    assert positions(DENSE.graph, shape)["linear.attn.q_proj.input"]["rotation"]["online"] == "ok"


def test_input_width_rejects_a_latent_and_accepts_the_down_factor():
    shape = CATALOG.models["qwen3-0.6b"].shape
    graph = factor_qkv_architecture(DENSE).graph
    assert input_width(graph, shape, graph.node("qkv_down")) == shape.hidden
    with pytest.raises(ValueError, match="latent"):
        input_width(graph, shape, graph.node("q"))


def test_architecture_prefers_the_attached_layout():
    model = build_tiny_qwen3()
    assert sites.architecture(model) == DENSE
    factored = factor_qkv_architecture(DENSE)
    setattr(model, ARCHITECTURE_ATTR, factored)
    assert sites.architecture(model) is factored
    assert "self_attn.qkv_down" in sites.architecture(model).linears


def probe_prepare(model, tokenizer, config):
    CALLS.append("prepare")


def probe_method(model, tokenizer, config):
    CALLS.append("fn")
    return model


CALLS: list[str] = []


@pytest.fixture
def probe(monkeypatch):
    """A throwaway registered method `lowrank_probe` and fake preprocess entries recording into CALLS."""
    CALLS.clear()
    monkeypatch.setattr(registry, "_METHODS", dict(registry._METHODS))
    registry.method("lowrank_probe", version="1", field="quantization", prepare=probe_prepare)(probe_method)
    manifest = registry.manifest_from_impl(registry._METHODS["lowrank_probe"], builtin=False)

    real = hf.resolve_entry
    fakes = {
        "effeval.methods.preprocess.engine:make_context": lambda *a, **k: "context",
        "effeval.methods.preprocess.engine:apply": lambda *a, **k: CALLS.append("preprocess") or "applied",
    }
    monkeypatch.setattr(hf, "resolve_entry", lambda entry: fakes.get(entry) or real(entry))
    monkeypatch.setattr(hf, "preprocess_summary", lambda preprocess, applied: {"applied": applied})
    manifest = manifest.model_copy(update={"entry": f"{__name__}:probe_method"})
    handle = hf.InProcess(model=build_tiny_qwen3(), tokenizer=None)
    model = SimpleNamespace(checkpoint="ckpt", revision=None)
    return manifest, handle, model


def run_apply(probe, preprocess):
    manifest, handle, model = probe
    return hf._apply(handle, model, SimpleNamespace(config={}), preprocess, manifest)


def test_prepare_runs_before_preprocess_and_the_method(probe):
    run_apply(probe, SimpleNamespace(steps=[]))
    assert CALLS == ["prepare", "preprocess", "fn"]


def test_prepare_runs_without_preprocess(probe):
    run_apply(probe, None)
    assert CALLS == ["prepare", "fn"]


def method_manifest(transform="qkv_shared"):
    """The RTN manifest renamed `lowrank_probe`, with a `decomposition` section and the transform set."""
    rtn = CATALOG.manifests["rtn"]
    schema = {
        **rtn.config_schema,
        "properties": {**rtn.config_schema["properties"], "decomposition": {"type": "object"}},
    }
    return rtn.model_copy(
        update={"name": "lowrank_probe", "config_schema": schema, "architecture_transform": transform}
    )


def probe_spec(model="qwen3-0.6b", rules=None, rank_ratio=None, preprocess=None):
    config = {"rules": rules if rules is not None else DENSE_RULE}
    if rank_ratio is not None:
        config["decomposition"] = {"rank_ratio": rank_ratio}
    spec = {
        "model": model,
        "method": {"name": "lowrank_probe", "preset": None, "config": config},
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
    }
    return spec if preprocess is None else {**spec, "preprocess": preprocess}


def issues_of(spec, transform="qkv_shared"):
    return validate(spec, CATALOG.with_manifests([method_manifest(transform)]))


DENSE_RULE = [
    {
        "sites": ["linear.*.weight"],
        "quant": {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_channel"},
    }
]
QKV_DOWN_RULE = [
    {
        "sites": ["linear.attn.qkv_down.weight"],
        "quant": {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_channel"},
    }
]


def test_validate_accepts_the_down_factor_site_only_with_the_transform():
    assert [i.code for i in issues_of(probe_spec(rules=QKV_DOWN_RULE)) if i.severity.value == "error"] == []
    rejected = [
        i for i in issues_of(probe_spec(rules=QKV_DOWN_RULE), transform=None) if i.severity.value == "error"
    ]
    assert [i.code for i in rejected] == ["site.unknown"]


def test_validate_reports_the_latent_position_as_unavailable_for_rotation():
    steps = [
        {
            "kind": "rotation",
            "entries": [
                {"position": "linear.attn.q_proj.input", "mode": "online", "source": {"type": "hadamard"}}
            ],
        }
    ]
    errors = [i for i in issues_of(probe_spec(preprocess={"steps": steps})) if i.severity.value == "error"]
    assert [i.code for i in errors] == ["preprocess.mode_unavailable"]
    assert LATENT.format("attn.qkv_down") in errors[0].message
    dense = [
        i
        for i in issues_of(probe_spec(preprocess={"steps": steps}), transform=None)
        if i.severity.value == "error"
    ]
    assert dense == []


def test_validate_warns_when_the_rank_ratio_saves_no_parameters():
    warned = [
        i
        for i in issues_of(probe_spec(model="qwen3-4b", rank_ratio=1.5))
        if i.code == "decomposition.no_saving"
    ]
    (issue,) = warned
    assert issue.severity.value == "warning" and issue.paths == ["method.config.decomposition.rank_ratio"]
    assert "106.2%" in issue.message and "93.8%" in issue.message and "1920 of 2560" in issue.message
    assert "rank_ratio below 1.4" in issue.hint
    assert not [
        i
        for i in issues_of(probe_spec(model="qwen3-0.6b", rank_ratio=1.5))
        if i.code == "decomposition.no_saving"
    ]
    assert not [
        i
        for i in issues_of(probe_spec(model="qwen3-4b", rank_ratio=1.5), transform=None)
        if i.code == "decomposition.no_saving"
    ]


def test_validate_reports_an_unsupported_layout_as_an_error(monkeypatch):
    monkeypatch.setitem(
        ARCHITECTURE_TRANSFORMS, QKV_SHARED, lambda arch: (_ for _ in ()).throw(ValueError("no shared norm"))
    )
    errors = [i for i in issues_of(probe_spec()) if i.severity.value == "error"]
    assert [i.code for i in errors] == ["method.architecture_unsupported"]
    assert errors[0].message == "no shared norm"
