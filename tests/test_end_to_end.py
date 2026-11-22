"""UI render -> compute node run -> push -> comparisons -> reuse, and the offline bundle path."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from effeval.api import execute  # noqa: E402
from effeval.core import bundle as bundle_io  # noqa: E402
from effeval.server.app import create_app, init_admin  # noqa: E402
from effeval.store.http import HttpClient  # noqa: E402
from effeval.store.sqlite import SqliteStore  # noqa: E402
from test_evaluators import tiny_model_dir  # noqa: E402, F401

pytestmark = [pytest.mark.compute, pytest.mark.network]


def spec(model_code):
    return {
        "model": {"name": model_code, "dtype": "float32"},
        "method": {
            "name": "rtn",
            "preset": None,
            "config": {
                "rules": [
                    {
                        "sites": ["linear.*.weight"],
                        "quant": {"type": "int", "bits": 4, "symmetric": False, "granularity": "per_channel"},
                    },
                    {
                        "sites": ["attn.q", "attn.k"],
                        "layers": "1",
                        "quant": {"type": "fp", "encoding": "e4m3", "granularity": "per_token"},
                    },
                ]
            },
        },
        "field": "quantization",
        "tasks": [{"name": "lm_eval/arc_challenge", "mode": "loglikelihood", "params": {"limit": 4}}],
        "generation": {"batch_size": 2},
    }


@pytest.fixture
def lab(tmp_path):
    init_admin(tmp_path / "server", "alice", "alice-password")
    app = create_app(tmp_path / "server")
    token = (
        TestClient(app)
        .post("/api/auth/token", json={"username": "alice", "password": "alice-password"})
        .json()
    )
    ui = TestClient(app, headers={"Authorization": f"Bearer {token['token']}"})
    node = HttpClient("http://testserver", token["token"])
    node._http = ui  # route the node's HTTP client through the in-process server
    return ui, node


@pytest.fixture
def tiny_code(lab, tiny_model_dir):  # noqa: F811
    """Register the tiny checkpoint like `effeval model add` would; returns its global code."""
    from effeval.ops.introspect import model_entry

    ui, _node = lab
    entry, _architecture = model_entry(tiny_model_dir, None)
    ui.post("/api/models", json=entry.model_dump(mode="json")).raise_for_status()
    return entry.code


def run(plan_doc, node, out_dir, online=True, keep_going=False, log=lambda line: None):
    report = execute.Doctor(
        server="http://testserver", online=online, reason="test", username="alice", gpu=None, gpu_count=0
    )
    outcome, issues = execute.run_document(
        plan_doc,
        doctor_report=report,
        client=node if online else None,
        out_dir=out_dir,
        shard_spec=None,
        confirm=lambda plan, issues: True,
        log=log,
        keep_going=keep_going,
    )
    assert outcome is not None, [i.model_dump() for i in issues]
    return outcome, issues


def test_render_run_push_compare_then_reuse(lab, tiny_code, tmp_path):
    ui, node = lab
    rendered = ui.post("/api/render", json={"spec": spec(tiny_code)}).json()
    plan = rendered["plan"]
    assert sorted(j["role"] for j in plan["jobs"]) == ["main", "reference"]
    assert [j["spec"]["evaluator"]["name"] for j in plan["jobs"]] == ["lm_eval", "lm_eval"]

    outcome, issues = run(plan, node, tmp_path / "out")
    assert len(outcome.pushed) == 2 and outcome.bundle_path is None
    hits = [i.message for i in issues if i.code == "site.hits"]
    assert any("14 locations" in h for h in hits) and any("['attn.k', 'attn.q']" in h for h in hits)

    rows = {r["role"]: r for r in ui.get("/api/records?scope=mine").json()}
    main = rows["main"]
    assert main["record"]["env"]["attn_implementation"] == "effeval_sdpa"
    stored = SqliteStore(tmp_path / "server").comparisons(main["record_id"])["values"]
    assert set(stored) >= {"flip_rate", "kl", "kl.n"} and stored["kl.n"] == "4"
    assert main["comparisons"]["values"] == {}

    again = ui.post("/api/render", json={"spec": spec(tiny_code)}).json()
    assert again["plan"]["jobs"] == [] and len(again["plan"]["reused"]) == 2
    assert again["commands"] == []


def test_offline_run_writes_bundle_that_pastes_into_the_ui(lab, tiny_code, tmp_path):
    ui, node = lab
    plan = ui.post("/api/render", json={"spec": spec(tiny_code)}).json()["plan"]
    outcome, _ = run(plan, node, tmp_path / "out", online=False)
    assert outcome.pushed == []
    text = Path(outcome.bundle_path).read_text()
    bundle = bundle_io.loads(text)
    assert len(bundle.records) == 2
    assert all(r.artifacts["per_sample"].startswith("file://") for r in bundle.records)
    preview = ui.post("/api/bundles/preview", content=text, headers={"Content-Type": "text/plain"}).json()
    assert {r["role"] for r in preview["rows"]} == {"main", "reference"}
    imported = ui.post("/api/bundles", content=text, headers={"Content-Type": "text/plain"}).json()
    assert len(imported["imported"]) == 2
    main = next(r for r in ui.get("/api/records?scope=mine").json() if r["role"] == "main")
    assert SqliteStore(tmp_path / "server").comparisons(main["record_id"])["values"] == {}  # per-sample files stayed on the node

    pushed = execute.push_file(Path(outcome.bundle_path), node)
    assert pushed["imported"] == [] and sorted(pushed["duplicates"]) == sorted(
        r.job_id for r in bundle.records
    )


def test_push_accepts_several_bundles(lab, tiny_code, tmp_path, monkeypatch, capsys):
    from effeval import cli

    ui, node = lab
    plan = ui.post("/api/render", json={"spec": spec(tiny_code)}).json()["plan"]
    outcome, _ = run(plan, node, tmp_path / "out", online=False)
    first = Path(outcome.bundle_path)
    bundle = bundle_io.loads(first.read_text())
    second = tmp_path / "second.json"
    second.write_text(bundle_io.dumps(bundle_io.make_bundle(bundle.records[:1])))
    monkeypatch.setattr(cli, "_client", lambda args: node)
    assert cli.main(["push", str(first), str(second)]) == 0
    out = capsys.readouterr().out
    assert f"{first}: imported 2, already stored 0" in out
    assert f"{second}: imported 0, already stored 1" in out
    assert "imported 2 record(s) from 2 bundle(s)" in out


def test_rerun_with_the_same_out_dir_reuses_finished_jobs(lab, tiny_code, tmp_path, monkeypatch):
    from effeval.evaluators import lm_eval

    ui, node = lab
    plan = ui.post("/api/render", json={"spec": spec(tiny_code)}).json()["plan"]
    first, _ = run(plan, node, tmp_path / "out", online=False)
    assert len(bundle_io.loads(Path(first.bundle_path).read_text()).records) == 2

    def boom(*args, **kwargs):
        raise AssertionError("a finished job was evaluated again")

    monkeypatch.setattr(lm_eval.EVALUATOR, "run", boom)
    lines = []
    second, _ = run(plan, node, tmp_path / "out", log=lines.append)
    assert len(second.pushed) == 2 and second.failed == []
    assert sum("finished in an earlier run" in line for line in lines) == 2


def test_keep_going_runs_the_rest_and_lists_failures(lab, tiny_code, tmp_path, monkeypatch):
    from effeval.evaluators import lm_eval

    ui, node = lab
    two_tasks = {
        **spec(tiny_code),
        "tasks": [{"name": "lm_eval/arc_challenge", "mode": "loglikelihood"}, "native/wikitext2_ppl"],
    }
    plan = ui.post("/api/render", json={"spec": two_tasks}).json()["plan"]

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(lm_eval.EVALUATOR, "run", boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        run(plan, node, tmp_path / "stop", online=False)
    outcome, _ = run(plan, node, tmp_path / "go", online=False, keep_going=True)
    assert len(outcome.failed) == 2 and all("evaluate: CUDA out of memory" in f for f in outcome.failed)
    done = {r.spec["task"]["name"] for r in bundle_io.loads(Path(outcome.bundle_path).read_text()).records}
    assert done == {"native/wikitext2_ppl"}
