import json

import yaml

from effeval.cli import main


def write(tmp_path, doc):
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(doc))
    return str(path)


def test_conflicting_mode_exits_with_code_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("EFFEVAL_HOME", str(tmp_path))
    monkeypatch.delenv("EFFEVAL_SERVER", raising=False)
    path = write(
        tmp_path,
        {
            "model": "qwen3-8b-base",
            "field": "quantization",
            "tasks": [{"name": "native/wikitext2_ppl", "mode": "generation"}],
        },
    )
    assert main(["validate", "--config", path, "--baseline", "rtn"]) == 2
    assert "error[mode.task_unsupported]" in capsys.readouterr().err


def test_baseline_flag_applies_the_paper_preset_and_field(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("EFFEVAL_SERVER", raising=False)
    path = write(tmp_path, {"model": "qwen3-0.6b-base"})
    assert main(["validate", "--config", path, "--baseline", "gptq", "--json"]) == 0
    issues = json.loads(capsys.readouterr().out)
    assert [i for i in issues if i["severity"] == "error"] == []


def test_baseline_conflicting_with_config_method_is_rejected(tmp_path, capsys):
    path = write(tmp_path, {"model": "qwen3-8b-base", "method": {"name": "rtn"}})
    assert main(["validate", "--config", path, "--baseline", "gptq"]) == 2
    assert "conflicts with method.name" in capsys.readouterr().err


def test_offline_render_prints_a_self_contained_command(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("EFFEVAL_HOME", str(tmp_path))
    monkeypatch.delenv("EFFEVAL_SERVER", raising=False)
    path = write(tmp_path, {"model": "qwen3-0.6b-base"})
    assert main(["render", "--config", path, "--baseline", "gptq"]) == 0
    out = capsys.readouterr().out
    first, rest = out.split("# 2. run")
    second, third = rest.split("# 3.")
    assert first.startswith("# 1. write the plan\ncat > effeval-") and "<<'EFFEVAL_PLAN_" in first
    assert (
        second.strip().startswith("effeval run --config effeval-")
        and "effeval push effeval-results/" in third
    )
    body = first.split("\n", 2)[2].rsplit("EFFEVAL_PLAN_", 1)[0]
    plan = yaml.safe_load(body)
    assert plan["plan_version"] == 1 and len(plan["jobs"]) == 2


def _announce_fake(monkeypatch):
    import importlib.metadata

    from effeval.core import evaluator

    point = importlib.metadata.EntryPoint(
        name="fake", value="fake_evaluator:FAKE", group=evaluator.ENTRY_POINT_GROUP
    )
    monkeypatch.setattr(evaluator.importlib.metadata, "entry_points", lambda group: [point])


def test_task_snapshot_writes_a_listing_the_catalog_accepts(tmp_path, monkeypatch):
    from effeval.core.catalog import builtin_catalog
    from effeval.core.listing import TaskListing

    _announce_fake(monkeypatch)
    assert main(["task", "snapshot", "--out", str(tmp_path / "out")]) == 0
    listing = TaskListing.model_validate_json((tmp_path / "out" / "fake.json").read_text(encoding="utf-8"))
    assert "fake/echo" in builtin_catalog().with_task_listings([listing]).tasks


def test_sync_uploads_each_listing_once(monkeypatch):
    from effeval.api import tasks

    _announce_fake(monkeypatch)
    uploaded = []

    class Client:
        def register_task_listing(self, listing):
            uploaded.append(listing.evaluator.name)

    assert [listing.evaluator.name for listing in tasks.sync_tasks(Client())] == ["fake"]
    assert uploaded == ["fake"]


def test_task_list_filters_by_evaluator(capsys):
    assert main(["task", "list", "--evaluator", "native"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("native/c4_ppl") and "native/wikitext2_ppl" in out and "kernel/" not in out
