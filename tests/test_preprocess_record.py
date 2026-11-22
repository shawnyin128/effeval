"""What a run records about its preprocessing, and what its cached artifacts are keyed by."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core import bundle as bundle_io  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.preprocess import Preprocess, builtin_preprocess_presets, resolve_preprocess  # noqa: E402
from effeval.core.types import InProcess  # noqa: E402
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from effeval.runtime import hf  # noqa: E402
from test_end_to_end import lab, run  # noqa: E402, F401
from test_evaluators import tiny_model_dir  # noqa: E402, F401
from test_preprocess_scaling import smoothquant_preprocess  # noqa: E402

compute = pytest.mark.compute


def prepare_with(monkeypatch, model, preprocess):
    monkeypatch.setattr(engine, "default_cache", ArtifactCache)
    monkeypatch.setattr(
        engine, "calibration_windows", lambda _calibration, _tokenizer_id: [torch.arange(32).reshape(1, 32)]
    )
    monkeypatch.setattr(
        hf, "load", lambda _spec: InProcess(model=model, tokenizer=SimpleNamespace(name_or_path="t"))
    )
    spec = SimpleNamespace(
        preprocess=preprocess,
        model=SimpleNamespace(checkpoint="tiny", revision=None),
        method=SimpleNamespace(config={}),
    )
    return hf.prepare(spec, builtin_catalog().manifests["base"])


@compute
def test_artifact_keys_depend_on_the_model_dtype(monkeypatch):
    preprocess = smoothquant_preprocess()
    in_float32 = prepare_with(monkeypatch, build_tiny_qwen3(), preprocess).preprocess_applied["artifacts"]
    again = prepare_with(monkeypatch, build_tiny_qwen3(), preprocess).preprocess_applied["artifacts"]
    in_bfloat16 = prepare_with(monkeypatch, build_tiny_qwen3().to(torch.bfloat16), preprocess)
    assert in_float32 and in_float32 == again
    assert set(in_float32) == set(in_bfloat16.preprocess_applied["artifacts"])
    assert all(in_float32[name] != key for name, key in in_bfloat16.preprocess_applied["artifacts"].items())


@compute
def test_prepare_returns_a_summary_of_what_the_preprocessing_applied(monkeypatch):
    steps = copy.deepcopy(builtin_preprocess_presets()["quarot"].values["preprocess.steps"].value)
    preprocess = resolve_preprocess(Preprocess.model_validate({"preset": "quarot", "steps": steps}))
    handle = prepare_with(monkeypatch, build_tiny_qwen3(head_dim=32), preprocess)
    summary = handle.preprocess_applied
    assert [step["kind"] for step in summary["steps"]] == [step.kind for step in preprocess.steps]
    entries = [e for step in summary["steps"] for e in step["entries"]]
    assert entries == [
        {
            "position": e.position,
            "layers": e.layers,
            "mode": e.mode,
            "source": e.source.model_dump(mode="json"),
        }
        for step in preprocess.steps
        for e in step.entries
    ]
    assert summary["versions"] == preprocess.versions
    assert summary["edits"] and all(isinstance(line, str) for line in summary["edits"])
    assert summary["artifacts"] == {}


@compute
def test_prepare_without_preprocess_returns_no_summary(monkeypatch):
    assert prepare_with(monkeypatch, build_tiny_qwen3(), None).preprocess_applied is None


@pytest.mark.network
def test_the_run_record_keeps_the_applied_preprocessing(lab, tiny_model_dir, tmp_path):  # noqa: F811
    from effeval.ops.introspect import model_entry

    ui, node = lab
    entry, _architecture = model_entry(tiny_model_dir, None)
    ui.post("/api/models", json=entry.model_dump(mode="json")).raise_for_status()
    raw = {
        "model": {"name": entry.code, "dtype": "float32"},
        "method": {"name": "base"},
        "preprocess": {"preset": "quarot"},
        "field": "quantization",
        "tasks": [{"name": "native/wikitext2_ppl", "params": {"limit": 1, "seqlen": 64}}],
    }
    plan = ui.post("/api/render", json={"spec": raw}).json()["plan"]
    outcome, _ = run(plan, node, tmp_path / "out", online=False)
    records = bundle_io.loads(Path(outcome.bundle_path).read_text()).records
    by_role = {r.role: r for r in records}
    assert "preprocess_applied" not in by_role["reference"].protocol
    applied = by_role["main"].protocol["preprocess_applied"]
    assert applied["steps"][0]["kind"] == "rotation"
    assert applied["versions"] and applied["edits"]
