"""Server tests without torch: auth, roles, permissions, bundles, comparisons."""

import gzip
import threading
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from effeval.core import bundle as bundle_io
from effeval.core.types import Artifact, Fidelity, Record, Sample
from effeval.server.app import create_app, init_admin


@pytest.fixture
def server(tmp_path):
    init_admin(tmp_path, "admin", "admin-password")
    app = create_app(tmp_path)
    client = TestClient(app)
    token = client.post("/api/auth/token", json={"username": "admin", "password": "admin-password"}).json()[
        "token"
    ]
    admin = TestClient(app, headers={"Authorization": f"Bearer {token}"})
    admin.post(
        "/api/admin/users", json={"username": "alice", "password": "alice-password"}
    ).raise_for_status()
    admin.post("/api/admin/users", json={"username": "bob", "password": "bob-password"}).raise_for_status()

    def as_user(name, can_publish=True):
        tok = client.post(
            "/api/auth/token",
            json={
                "username": name,
                "password": f"{name}-password",
                "can_publish": can_publish,
                "label": "cli" if can_publish else "agent",
            },
        ).json()["token"]
        return TestClient(app, headers={"Authorization": f"Bearer {tok}"})

    from catalogs import LISTINGS

    for listing in LISTINGS:
        admin.post("/api/task-catalogs", json=listing.model_dump(mode="json")).raise_for_status()
    return app, admin, as_user


def record(
    job_id, run_key, metrics, *, role="main", reference_key=None, compare=(), artifacts=None, produces=()
):
    return Record(
        record_id="rec_" + job_id,
        job_id=job_id,
        run_key=run_key,
        reference_key=reference_key,
        role=role,
        compare=list(compare),
        spec={
            "model": {"name": "m"},
            "method": {"name": "rtn"},
            "task": {"name": "lm_eval/mmlu", "mode": "loglikelihood"},
        },
        metrics=metrics,
        protocol={},
        env=None,
        fidelity=Fidelity(),
        artifacts=artifacts or {},
        produces=list(produces),
        created_at="2026-11-20T00:00:00+00:00",
    )


def upload(client, samples):
    payload = gzip.compress("\n".join(s.model_dump_json() for s in samples).encode())
    return client.post("/api/artifacts", files={"file": ("s.jsonl.gz", payload)}).json()["uri"]


def push(client, *records):
    return client.post(
        "/api/bundles",
        content=bundle_io.dumps(bundle_io.make_bundle(list(records))),
        headers={"Content-Type": "text/plain"},
    )


def test_login_required_and_no_self_registration(server):
    app, _admin, _ = server
    anonymous = TestClient(app)
    assert anonymous.get("/api/records").status_code == 401
    assert (
        anonymous.post("/api/admin/users", json={"username": "eve", "password": "x" * 8}).status_code == 401
    )
    assert (
        anonymous.post("/api/auth/login", json={"username": "alice", "password": "wrong"}).status_code == 401
    )
    session = TestClient(app)
    assert (
        session.post("/api/auth/login", json={"username": "alice", "password": "alice-password"}).status_code
        == 200
    )
    assert session.get("/api/me").json()["username"] == "alice"


def test_only_admins_manage_users_and_disabled_users_lose_access(server):
    _app, admin, as_user = server
    alice = as_user("alice")
    assert alice.post("/api/admin/users", json={"username": "eve", "password": "x" * 8}).status_code == 403
    admin.post("/api/admin/users/alice/disable").raise_for_status()
    assert alice.get("/api/me").status_code == 401


def login_status(app, username, password):
    response = TestClient(app).post("/api/auth/login", json={"username": username, "password": password})
    return response.status_code


def session_for(app, username):
    session = TestClient(app)
    session.post(
        "/api/auth/login", json={"username": username, "password": f"{username}-password"}
    ).raise_for_status()
    return session


def test_admins_delete_only_users_without_data_and_never_themselves(server):
    app, admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    bob_session = session_for(app, "bob")
    assert alice.delete("/api/admin/users/bob").status_code == 403
    push(alice, record("job_1", "rk_a", {"accuracy": "0.5"})).raise_for_status()
    refused = admin.delete("/api/admin/users/alice")
    assert refused.status_code == 409 and "records: 1" in refused.json()["detail"]
    assert login_status(app, "alice", "alice-password") == 200

    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    alice.post("/api/projects/p/members", json={"username": "bob"}).raise_for_status()
    admin.delete("/api/admin/users/bob").raise_for_status()
    assert bob.get("/api/me").status_code == 401
    assert bob_session.get("/api/me").status_code == 401
    assert login_status(app, "bob", "bob-password") == 401
    assert [u["username"] for u in admin.get("/api/admin/users").json()] == ["admin", "alice"]
    assert admin.delete("/api/admin/users/admin").status_code == 400
    assert admin.delete("/api/admin/users/bob").status_code == 404


def test_rename_moves_logins_records_and_projects_to_the_new_name(server):
    app, admin, as_user = server
    alice = as_user("alice")
    alice_session = session_for(app, "alice")
    push(alice, record("job_1", "rk_a", {"accuracy": "0.5"})).raise_for_status()
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    assert admin.patch("/api/admin/users/alice", json={"username": "bob"}).status_code == 409
    assert admin.patch("/api/admin/users/alice", json={"username": "Not Valid"}).status_code == 422

    admin.patch("/api/admin/users/alice", json={"username": "carol"}).raise_for_status()
    assert login_status(app, "alice", "alice-password") == 401
    assert login_status(app, "carol", "alice-password") == 200
    assert alice.get("/api/me").json()["username"] == "carol"
    assert alice_session.get("/api/me").json()["username"] == "carol"
    assert ids(alice, "mine") == ["rec_job_1"]
    assert alice.get("/api/records/rec_job_1").json()["owner"] == "carol"
    assert [p["name"] for p in alice.get("/api/projects").json()] == ["p"]
    assert admin.delete("/api/admin/users/carol").status_code == 409

    admin.patch("/api/admin/users/admin", json={"username": "root"}).raise_for_status()
    assert admin.get("/api/me").json()["username"] == "root"


def test_role_changes_take_effect_at_once_but_admins_cannot_change_their_own(server):
    _app, admin, as_user = server
    alice = as_user("alice")
    assert alice.get("/api/admin/users").status_code == 403
    admin.patch("/api/admin/users/alice", json={"role": "admin"}).raise_for_status()
    assert alice.get("/api/admin/users").status_code == 200
    assert admin.patch("/api/admin/users/admin", json={"role": "member"}).status_code == 400
    assert alice.patch("/api/admin/users/nobody", json={"role": "admin"}).status_code == 404
    alice.patch("/api/admin/users/admin", json={"role": "member"}).raise_for_status()
    assert admin.get("/api/admin/users").status_code == 403


def test_bundle_import_is_private_idempotent_and_tamper_checked(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    rec = record("job_1", "rk_a", {"accuracy": "0.6523178947368421"})
    first = push(alice, rec).json()
    assert first["imported"] == ["rec_job_1"]
    assert push(alice, rec).json()["duplicates"] == ["job_1"]
    assert [r["metrics"]["accuracy"] for r in alice.get("/api/records?scope=mine").json()] == [
        "0.6523178947368421"
    ]
    assert bob.get("/api/records?scope=pool").json() == []
    assert bob.get("/api/records/rec_job_1").status_code == 404
    text = bundle_io.dumps(bundle_io.make_bundle([rec])).replace("0.6523178947368421", "0.99")
    assert alice.post("/api/bundles", content=text, headers={"Content-Type": "text/plain"}).status_code == 400


def test_publish_needs_rights_and_retract_keeps_record_visible_but_unreusable(server):
    _app, _admin, as_user = server
    alice, agent, bob = as_user("alice"), as_user("alice", can_publish=False), as_user("bob")
    push(alice, record("job_1", "rk_a", {"accuracy": "0.5"}, role="reference"))
    assert agent.post("/api/records/rec_job_1/publish").status_code == 403
    assert bob.post("/api/records/rec_job_1/publish").status_code == 403
    alice.post("/api/records/rec_job_1/publish").raise_for_status()
    assert [r["record_id"] for r in bob.get("/api/records?scope=pool").json()] == ["rec_job_1"]
    assert bob.get("/api/reuse", params={"run_key": "rk_a"}).json()["record"]["record_id"] == "rec_job_1"
    alice.post("/api/records/rec_job_1/retract").raise_for_status()
    assert bob.get("/api/records?scope=pool").json()[0]["visibility"] == "retracted"
    assert bob.get("/api/reuse", params={"run_key": "rk_a"}).json()["record"] is None


def test_comparisons_computed_on_import_in_either_order(server, tmp_path):
    _app, _admin, as_user = server
    alice = as_user("alice")
    ref_samples = [
        Sample(id=f"s{i}", scores={"correct": float(i % 2)}, choice_loglikelihoods=[-1.0, -2.0])
        for i in range(4)
    ]
    main_samples = [
        Sample(id=f"s{i}", scores={"correct": 1.0}, choice_loglikelihoods=[-1.0, -1.5]) for i in range(4)
    ]
    per_sample = [Artifact.PER_SAMPLE, Artifact.CHOICE_LOGLIKELIHOODS]
    main = record(
        "job_m",
        "rk_main",
        {"accuracy": "1.0"},
        reference_key="rk_ref",
        compare=["flip_rate", "kl"],
        artifacts={"per_sample": upload(alice, main_samples)},
        produces=per_sample,
    )
    push(alice, main)  # main first: nothing to compare yet
    assert stored_comparisons(tmp_path, "rec_job_m") == {}
    ref = record(
        "job_r",
        "rk_ref",
        {"accuracy": "0.5"},
        role="reference",
        artifacts={"per_sample": upload(alice, ref_samples)},
        produces=per_sample,
    )
    result = push(alice, ref).json()
    assert result["comparisons_updated"] == ["rec_job_m"]
    values = stored_comparisons(tmp_path, "rec_job_m")
    assert values["flip_rate"] == "0.5" and values["flip_rate.n"] == "4"
    assert float(values["kl"]) > 0


def test_paste_import_without_server_artifacts_gets_no_comparisons(server, tmp_path):
    _app, _admin, as_user = server
    alice = as_user("alice")
    push(
        alice,
        record(
            "job_r",
            "rk_ref",
            {"accuracy": "0.25"},
            role="reference",
            artifacts={"per_sample": "file:///node/path.jsonl.gz"},
            produces=[Artifact.PER_SAMPLE],
        ),
    )
    push(
        alice,
        record(
            "job_m",
            "rk_main",
            {"accuracy": "0.75"},
            reference_key="rk_ref",
            compare=["flip_rate"],
            artifacts={"per_sample": "file:///node/m.jsonl.gz"},
            produces=[Artifact.PER_SAMPLE],
        ),
    )
    assert stored_comparisons(tmp_path, "rec_job_m") == {}
    need = ",".join([Artifact.PER_SAMPLE.value])
    assert alice.get("/api/reuse", params={"run_key": "rk_ref", "need": need}).json()["record"] is None


def test_validate_and_render_endpoints(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    bad = {
        "model": "qwen3-8b-base",
        "method": {"name": "rtn"},
        "field": "quantization",
        "tasks": [{"name": "evalscope/mmlu", "mode": "loglikelihood"}],
    }
    issues = alice.post("/api/validate", json={"spec": bad}).json()["issues"]
    assert [i["code"] for i in issues if i["severity"] == "error"] == ["mode.task_unsupported"]
    good = {
        "model": "qwen3-0.6b-base",
        "method": {"name": "gptq"},
        "field": "quantization",
        "hardware": {"gpu": "H100"},
    }
    rendered = alice.post("/api/render", json={"spec": good, "shards": 2}).json()
    assert len(rendered["commands"]) == 2  # gptq and its reference are separate model groups
    step = rendered["commands"][0]
    assert step["write"].startswith("cat > effeval-")
    assert step["run"].startswith("effeval run --config effeval-") and "--shard 1/2" in step["run"]
    assert step["push"].startswith(
        f"effeval push effeval-results/{rendered['plan']['plan_id']}.results.tar.gz"
    )
    assert rendered["estimate"]["total_s"] > 0
    assert alice.get("/api/runs").json()[0]["plan_id"] == rendered["plan"]["plan_id"]


METHOD_SOURCE = """
from typing import Literal
from pydantic import BaseModel, Field
from effeval.core.fragments import QuantRule
from effeval.core.registry import method

class OursConfig(BaseModel):
    model_config = {"extra": "forbid"}
    rules: list[QuantRule]
    rotation: Literal["none", "hadamard"] = "hadamard"
    iters: int = Field(ge=1, description="optimization steps")

@method("ours_quant", version="0.1", field="quantization", config=OursConfig, uses=("quant_rules",),
        display=("rotation",), presets={"fast": {"iters": 10}})
def apply(model, config, ctx):
    return model

def plain(model, config, ctx):
    return model
"""


def test_a_method_written_to_the_interface_is_discovered_and_gets_its_form(server, tmp_path, monkeypatch):
    import importlib.metadata
    import sys

    from effeval.core import registry

    (tmp_path / "ours_pkg.py").write_text(METHOD_SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "ours_pkg", raising=False)
    points = [importlib.metadata.EntryPoint("ours", "ours_pkg:apply", registry.ENTRY_POINT_GROUP)]
    monkeypatch.setattr(
        registry.importlib.metadata,
        "entry_points",
        lambda group: points if group == registry.ENTRY_POINT_GROUP else [],
    )
    [impl] = registry.discover()
    manifest = registry.manifest_from_impl(impl, builtin=False)
    assert manifest.entry == "ours_pkg:apply" and manifest.custom_presets == {"fast": {"iters": 10}}
    assert set(manifest.config_schema["properties"]) == {"rules", "rotation", "iters"}

    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    alice.post("/api/methods", json=manifest.model_dump(mode="json")).raise_for_status()
    described = alice.get("/api/describe").json()["methods"]["ours_quant"]
    assert described["config_schema"]["properties"]["iters"]["description"] == "optimization steps"
    rules = [
        {
            "sites": ["linear.*.weight"],
            "quant": {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_channel"},
        }
    ]
    spec = {
        "model": "qwen3-8b-base",
        "method": {"name": "ours_quant", "config": {"rules": rules}},
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
    }
    codes = {i["code"] for i in alice.post("/api/validate", json={"spec": spec}).json()["issues"]}
    assert "method.config_invalid" in codes  # iters has no default
    spec["method"]["preset"] = "fast"
    issues = alice.post("/api/validate", json={"spec": spec}).json()["issues"]
    assert [i for i in issues if i["severity"] == "error"] == []
    assert "method.unknown" in {
        i["code"] for i in bob.post("/api/validate", json={"spec": spec}).json()["issues"]
    }

    points[0] = importlib.metadata.EntryPoint("plain", "ours_pkg:plain", registry.ENTRY_POINT_GROUP)
    with pytest.raises(ValueError, match="not decorated"):
        registry.discover()


def test_models_are_lab_wide_and_codes_cannot_be_redefined(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    entry = {
        "code": "my-ckpt",
        "checkpoint": "/data/My-Ckpt",
        "revision": None,
        "architecture": "Qwen3ForCausalLM",
        "dtype": "bfloat16",
        "params": 1000,
        "tie_word_embeddings": False,
    }
    alice.post("/api/models", json=entry).raise_for_status()
    assert "my-ckpt" in {m["model"]["code"] for m in bob.get("/api/models").json()}
    clash = {**entry, "checkpoint": "/other/My-Ckpt"}
    assert bob.post("/api/models", json=clash).status_code == 409
    wrong_code = {**entry, "code": "mine"}
    assert alice.post("/api/models", json=wrong_code).status_code == 422
    builtin = {**entry, "code": "qwen3-8b", "checkpoint": "someone/Qwen3-8B"}
    assert alice.post("/api/models", json=builtin).status_code == 409
    unknown_arch = {**entry, "code": "x-1", "checkpoint": "/d/X-1", "architecture": "FooForCausalLM"}
    assert alice.post("/api/models", json=unknown_arch).status_code == 400
    spec = {"model": "my-ckpt", "method": {"name": "rtn"}, "field": "quantization"}
    issues = bob.post("/api/validate", json={"spec": spec}).json()["issues"]
    assert "model.unpinned" in {i["code"] for i in issues}


def test_language_is_per_user(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    assert alice.get("/api/me").json()["locale"] == "en"
    alice.put("/api/me/locale", json={"locale": "zh"}).raise_for_status()
    assert alice.get("/api/me").json()["locale"] == "zh"
    assert bob.get("/api/me").json()["locale"] == "en"
    assert alice.put("/api/me/locale", json={"locale": "fr"}).status_code == 422


def push_to(client, to, *records):
    return client.post(
        "/api/bundles",
        params={"to": to},
        content=bundle_io.dumps(bundle_io.make_bundle(list(records))),
        headers={"Content-Type": "text/plain"},
    )


def ids(client, scope):
    return sorted(r["record_id"] for r in client.get("/api/records", params={"scope": scope}).json())


def test_projects_share_results_only_with_members(server):
    _app, admin, as_user = server
    admin.post(
        "/api/admin/users", json={"username": "carol", "password": "carol-password"}
    ).raise_for_status()
    alice, bob, carol = as_user("alice"), as_user("bob"), as_user("carol")
    assert alice.post("/api/projects", json={"name": "Paper A"}).status_code == 400
    alice.post("/api/projects", json={"name": "paper-a"}).raise_for_status()
    assert alice.post("/api/projects", json={"name": "paper-a"}).status_code == 409
    assert bob.post("/api/projects/paper-a/members", json={"username": "bob"}).status_code == 403
    alice.post("/api/projects/paper-a/members", json={"username": "bob"}).raise_for_status()

    assert push_to(carol, "project:paper-a", record("job_c", "rk_c", {"accuracy": "0.1"})).status_code == 403
    push_to(alice, "project:paper-a", record("job_1", "rk_a", {"accuracy": "0.5"})).raise_for_status()
    assert ids(bob, "project:paper-a") == ["rec_job_1"]
    assert bob.get("/api/records/rec_job_1").json()["projects"] == ["paper-a"]
    assert ids(bob, "pool") == [] and ids(alice, "mine") == ["rec_job_1"]
    assert carol.get("/api/records", params={"scope": "project:paper-a"}).status_code == 403
    assert carol.get("/api/records/rec_job_1").status_code == 404

    push(carol, record("job_2", "rk_b", {"accuracy": "0.7"}))
    assert bob.post("/api/projects/paper-a/records", json={"record_ids": ["rec_job_2"]}).status_code == 404
    carol.post("/api/records/rec_job_2/publish").raise_for_status()
    bob.post("/api/projects/paper-a/records", json={"record_ids": ["rec_job_2"]}).raise_for_status()
    assert ids(alice, "project:paper-a") == ["rec_job_1", "rec_job_2"]
    assert [p["name"] for p in bob.get("/api/projects").json()] == ["paper-a"]
    assert carol.get("/api/projects").json() == []

    push_to(alice, "pool", record("job_3", "rk_d", {"accuracy": "0.2"})).raise_for_status()
    assert "rec_job_3" in ids(carol, "pool")
    agent = as_user("alice", can_publish=False)
    assert push_to(agent, "pool", record("job_4", "rk_e", {"accuracy": "0.3"})).status_code == 403


def test_trash_hides_records_flags_project_rows_and_can_restore_or_purge(server, tmp_path):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    alice.post("/api/projects/p/members", json={"username": "bob"}).raise_for_status()
    samples = [Sample(id="0", scores={"correct": 1.0})]
    ref = record(
        "job_r",
        "rk_r",
        {"accuracy": "0.5"},
        role="reference",
        artifacts={"per_sample": upload(alice, samples)},
        produces=[Artifact.PER_SAMPLE],
    )
    main = record(
        "job_m",
        "rk_m",
        {"accuracy": "0.4"},
        reference_key="rk_r",
        compare=["flip_rate"],
        artifacts={"per_sample": upload(alice, samples)},
        produces=[Artifact.PER_SAMPLE],
    )
    push_to(alice, "project:p", ref, main).raise_for_status()

    assert bob.delete("/api/records/rec_job_r").status_code == 403
    deps = alice.get("/api/records/rec_job_r/dependents").json()
    assert deps == {
        "comparisons": [{"record_id": "rec_job_m", "owner": "alice", "model": "m"}],
        "projects": ["p"],
    }
    alice.delete("/api/records/rec_job_r").raise_for_status()
    assert ids(alice, "mine") == ["rec_job_m"]
    assert alice.get("/api/reuse", params={"run_key": "rk_r"}).json()["record"] is None
    problems = {
        r["record_id"]: r["problems"] for r in bob.get("/api/records", params={"scope": "project:p"}).json()
    }
    assert problems == {"rec_job_r": ["trashed"], "rec_job_m": ["reference_trashed"]}
    assert [r["record_id"] for r in alice.get("/api/trash").json()] == ["rec_job_r"]
    assert bob.get("/api/trash").json() == []

    alice.post("/api/records/rec_job_r/restore").raise_for_status()
    assert ids(alice, "mine") == ["rec_job_m", "rec_job_r"]
    assert alice.post("/api/records/rec_job_r/restore").status_code == 409
    assert alice.delete("/api/trash/rec_job_r").status_code == 409
    alice.delete("/api/records/rec_job_r").raise_for_status()
    alice.delete("/api/trash/rec_job_r").raise_for_status()
    assert alice.get("/api/trash").json() == []
    assert ids(alice, "project:p") == ["rec_job_m"]
    assert stored_comparisons(tmp_path, "rec_job_m") == {}


def test_redo_greys_the_record_and_a_matching_import_replaces_it(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    spec = {"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}
    plan = alice.post("/api/render", json={"spec": spec}).json()["plan"]
    job = next(j for j in plan["jobs"] if j["role"] == "main")

    def result(job_id, accuracy):
        return Record(
            record_id="rec_" + job_id,
            job_id=job_id,
            run_key=job["run_key"],
            reference_key=job["reference_key"],
            role="main",
            compare=job["compare"],
            spec=job["spec"],
            metrics={"ppl": accuracy},
            protocol={},
            env=None,
            fidelity=Fidelity(),
            created_at="2026-11-20T00:00:00+00:00",
        )

    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    push_to(alice, "project:p", result("job_old", "99.0")).raise_for_status()
    alice.post("/api/records/rec_job_old/publish").raise_for_status()
    assert bob.post("/api/records/rec_job_old/redo").status_code == 403

    redo = alice.post("/api/records/rec_job_old/redo", json={"to": "project:p"}).json()
    assert [j["run_key"] for j in redo["plan"]["jobs"] if j["role"] == "main"] == [job["run_key"]]
    assert {j["role"] for j in redo["plan"]["jobs"]} == {
        "main"
    }  # no stored reference: nothing to re-run it from
    assert redo["commands"][0]["run"].rstrip().endswith("--to project:p")
    assert redo["commands"][0]["push"].rstrip().endswith("--to project:p")
    row = alice.get("/api/records/rec_job_old").json()
    assert row["state"] == "redo" and row["problems"] == ["redo"]
    assert alice.get("/api/reuse", params={"run_key": job["run_key"]}).json()["record"] is None

    preview = alice.post(
        "/api/bundles/preview",
        content=bundle_io.dumps(bundle_io.make_bundle([result("job_new", "27.6")])),
        headers={"Content-Type": "text/plain"},
    ).json()
    assert preview["rows"][0]["replaces"] == ["rec_job_old"]
    imported = push(alice, result("job_new", "27.6")).json()
    assert imported["redone"] == ["rec_job_old"]
    assert ids(alice, "mine") == ["rec_job_new"]
    assert ids(alice, "project:p") == ["rec_job_new", "rec_job_old"]
    assert ids(bob, "pool") == ["rec_job_new"]
    trash = alice.get("/api/trash").json()
    assert [(t["record_id"], t["trashed"]["superseded_by"]) for t in trash] == [
        ("rec_job_old", "rec_job_new")
    ]

    alice.post("/api/records/rec_job_new/redo").raise_for_status()
    alice.delete("/api/records/rec_job_new/redo").raise_for_status()
    assert alice.get("/api/records/rec_job_new").json()["state"] == "active"
    assert alice.delete("/api/records/rec_job_new/redo").status_code == 409


def test_redo_and_recompute_ignore_removed_comparator_names(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    spec = {"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}
    plan = alice.post("/api/render", json={"spec": spec}).json()["plan"]
    job = next(j for j in plan["jobs"] if j["role"] == "main")
    old = Record(
        record_id="rec_job_old",
        job_id="job_old",
        run_key=job["run_key"],
        reference_key=job["reference_key"],
        role="main",
        compare=["delta", "paired_delta", "flip_rate"],
        spec=job["spec"],
        metrics={"ppl": "99.0"},
        protocol={},
        env=None,
        fidelity=Fidelity(),
        created_at="2026-11-20T00:00:00+00:00",
    )
    push(alice, old).raise_for_status()

    redo = alice.post("/api/records/rec_job_old/redo")
    redo.raise_for_status()
    assert [j["compare"] for j in redo.json()["plan"]["jobs"] if j["role"] == "main"] == [["flip_rate"]]


def job_record(job, job_id, metrics, owner=None):
    return Record(
        record_id="rec_" + job_id,
        job_id=job_id,
        run_key=job["run_key"],
        reference_key=job["reference_key"],
        role=job["role"],
        compare=job["compare"],
        spec=job["spec"],
        metrics=metrics,
        protocol={},
        env=None,
        fidelity=Fidelity(),
        owner=owner,
        created_at="2026-11-20T00:00:00+00:00",
    )


TABLE = {
    "name": "main results",
    "field": "quantization",
    "models": ["qwen3-0.6b-base"],
    "tasks": ["native/wikitext2_ppl", "lm_eval/mmlu"],
    "rows": [
        {"label": "RTN", "method": {"name": "rtn", "preset": "gptq_paper"}},
        {"label": "GPTQ", "method": {"name": "gptq"}},
    ],
}


def cells(view):
    return {row["label"]: row["cells"]["native/wikitext2_ppl"] for row in view["blocks"][0]["rows"]}


def test_a_task_that_cannot_run_only_invalidates_its_own_cells(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    view = alice.post(
        "/api/projects/p/tables", json={**TABLE, "tasks": ["native/wikitext2_ppl", "kernel/gemm_w4a16"]}
    ).json()
    rows = {r["label"]: r for r in view["blocks"][0]["rows"]}
    assert "base" in rows
    for label in ("RTN", "GPTQ"):
        assert rows[label]["cells"]["native/wikitext2_ppl"]["status"] == "pending"
        invalid = rows[label]["cells"]["kernel/gemm_w4a16"]
        assert invalid["status"] == "invalid"
        assert "kernel.hardware_required" in {i["code"] for i in invalid["issues"]}
    assert view["counts"]["invalid"] == 2


def test_experiment_table_cells_track_claims_runs_and_results(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    alice.post("/api/projects/p/members", json={"username": "bob"}).raise_for_status()
    assert alice.post("/api/projects/p/tables", json={**TABLE, "rows": []}).status_code == 400
    view = alice.post("/api/projects/p/tables", json=TABLE).json()
    table_id = view["table_id"]
    assert [r["label"] for r in view["blocks"][0]["rows"]] == ["base", "RTN", "GPTQ"]
    assert set(view["blocks"][0]["rows"][0]["cells"]) == {
        "native/wikitext2_ppl",
        "lm_eval/mmlu",
    }  # one baseline row across evaluators
    assert view["blocks"][0]["rows"][0]["baseline"] is True
    assert {label: c["status"] for label, c in cells(view).items()} == dict.fromkeys(
        ["base", "RTN", "GPTQ"], "pending"
    )
    gptq, base = cells(view)["GPTQ"]["run_key"], cells(view)["base"]["run_key"]
    shared = {**TABLE, "name": "fp32", "shared": {"model": {"dtype": "float32"}}}
    fp32 = alice.post("/api/projects/p/tables", json=shared).json()
    assert cells(fp32)["GPTQ"]["run_key"] != gptq
    alice.delete(f"/api/projects/p/tables/{fp32['table_id']}").raise_for_status()

    alice.post("/api/projects/p/claims", json={"run_keys": [gptq]}).raise_for_status()
    assert bob.post("/api/projects/p/claims", json={"run_keys": [gptq]}).status_code == 409
    assert bob.delete(f"/api/projects/p/claims/{gptq}").status_code == 403
    assert cells(bob.get("/api/projects/p/tables").json()[0])["GPTQ"]["claim"]["by"] == "alice"

    rendered = alice.post(f"/api/projects/p/tables/{table_id}/command", json={"run_keys": [gptq]}).json()
    assert sorted(j["role"] for j in rendered["plan"]["jobs"]) == ["main", "reference"]
    assert rendered["commands"][0]["run"].rstrip().endswith("--to project:p")
    assert rendered["issues"] == []
    main = next(j for j in rendered["plan"]["jobs"] if j["role"] == "main")
    alice.post(
        "/api/heartbeat",
        json={"plan_id": rendered["plan"]["plan_id"], "job_id": main["job_id"], "status": "running"},
    )
    gptq_cell = cells(bob.get("/api/projects/p/tables").json()[0])["GPTQ"]
    assert gptq_cell["status"] == "running" and gptq_cell["activity"]["owner"] == "alice"
    warned = bob.post(f"/api/projects/p/tables/{table_id}/command", json={"run_keys": [gptq]}).json()
    assert {i["code"] for i in warned["issues"]} == {"run.claimed", "run.in_progress"}
    compose = bob.post(
        "/api/render",
        json={
            "spec": {
                "model": "qwen3-0.6b-base",
                "field": "quantization",
                "tasks": ["native/wikitext2_ppl"],
                "method": {"name": "gptq"},
            }
        },
    ).json()
    assert "run.in_progress" in {i["code"] for i in compose["issues"]}
    alice.post(
        "/api/heartbeat",
        json={"plan_id": rendered["plan"]["plan_id"], "job_id": main["job_id"], "status": "failed"},
    )
    assert cells(alice.get("/api/projects/p/tables").json()[0])["GPTQ"]["status"] == "failed"

    reference = next(j for j in rendered["plan"]["jobs"] if j["role"] == "reference")
    assert reference["run_key"] == base
    push(bob, job_record(reference, "job_ref", {"ppl": "20.9"}))
    assert cells(alice.get("/api/projects/p/tables").json()[0])["base"]["status"] == "pending"
    bob.post("/api/records/rec_job_ref/publish").raise_for_status()
    base_cell = cells(alice.get("/api/projects/p/tables").json()[0])["base"]
    assert (base_cell["status"], base_cell["record_id"], base_cell["owner"]) == (
        "reusable",
        "rec_job_ref",
        "bob",
    )
    reused = alice.post(f"/api/projects/p/tables/{table_id}/command", json={"run_keys": [base]}).json()
    assert reused["plan"]["jobs"] == [] and reused["commands"] == []
    alice.post("/api/projects/p/records", json={"record_ids": ["rec_job_ref"]}).raise_for_status()
    done_cell = cells(alice.get("/api/projects/p/tables").json()[0])["base"]
    assert (done_cell["status"], done_cell["visibility"]) == ("done", "public")
    rerun = alice.post(
        f"/api/projects/p/tables/{table_id}/command", json={"run_keys": [base], "force": True}
    ).json()
    assert [j["run_key"] for j in rerun["plan"]["jobs"]] == [base]

    alice.delete(f"/api/projects/p/claims/{gptq}").raise_for_status()
    assert "claim" not in cells(alice.get("/api/projects/p/tables").json()[0])["GPTQ"]
    assert (
        bob.post(f"/api/projects/p/tables/{table_id}/command", json={"run_keys": ["rk_nope"]}).status_code
        == 400
    )


def test_server_backfills_requested_derived_metrics_and_flags_undeclared_ones(server, tmp_path):
    _app, _admin, as_user = server
    alice = as_user("alice")
    samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 0.0, 1.0, 1.0])]
    held = record(
        "job_1",
        "rk_a",
        {"accuracy": "0.75"},
        artifacts={"per_sample": upload(alice, samples)},
        produces=[Artifact.PER_SAMPLE],
    )
    held.spec["metrics"] = ["accuracy_stderr"]
    loose = record("job_2", "rk_b", {"accuracy": "0.5", "bleu": "12.0"})
    loose.spec["metrics"] = ["accuracy_stderr"]
    result = push(alice, held, loose).json()
    assert result["unknown_metrics"] == ["bleu"]
    rows = {r["record_id"]: r for r in alice.get("/api/records?scope=mine").json()}
    assert rows["rec_job_1"]["metrics"] == {"accuracy": "0.75"} and rows["rec_job_1"]["derived"] == {}
    assert rows["rec_job_2"]["metrics"] == {"accuracy": "0.5", "bleu": "12.0"}

    from effeval.server.app import create_app
    from effeval.store.sqlite import SqliteStore

    store = SqliteStore(tmp_path)
    with store._write() as db:
        db.execute("DELETE FROM derived_metrics")
    create_app(tmp_path)  # backfills on start
    assert set(store.derived("rec_job_1")) == {"accuracy_stderr"}


def stored_comparisons(data_dir, record_id):
    """Comparison values the server stores for a record; the API shows only the speed ones, in table cells."""
    from effeval.store.sqlite import SqliteStore

    return SqliteStore(data_dir).comparisons(record_id)["values"]


def gz(samples):
    return gzip.compress("\n".join(s.model_dump_json() for s in samples).encode())


def test_late_per_sample_files_attach_after_checks_and_unlock_comparisons(server, tmp_path):
    _app, _admin, as_user = server
    alice = as_user("alice")
    ref_samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 1.0, 0.0, 1.0])]
    main_samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 0.0, 0.0, 1.0])]
    ref = record(
        "job_r",
        "rk_r",
        {"accuracy": "0.75"},
        role="reference",
        artifacts={"per_sample": upload(alice, ref_samples)},
        produces=[Artifact.PER_SAMPLE],
    )
    payload = gz(main_samples)
    main = record(
        "job_m",
        "rk_m",
        {"accuracy": "0.5"},
        reference_key="rk_r",
        compare=["flip_rate"],
        artifacts={
            "per_sample": "file:///node/job_m/per_sample.jsonl.gz",
            "per_sample_sha256": bundle_io.file_sha256(payload),
        },
        produces=[Artifact.PER_SAMPLE],
    )
    push(alice, ref, main).raise_for_status()
    assert "flip_rate" not in stored_comparisons(tmp_path, "rec_job_m")

    wrong = alice.post("/api/artifacts", files={"file": ("s.jsonl.gz", gz(ref_samples))}).json()["uri"]
    bad = main.model_copy(update={"artifacts": {**main.artifacts, "per_sample": wrong}})
    assert push(alice, bad).status_code == 400
    right = alice.post("/api/artifacts", files={"file": ("s.jsonl.gz", payload)}).json()["uri"]
    good = main.model_copy(update={"artifacts": {**main.artifacts, "per_sample": right}})
    result = push(alice, good).json()
    assert result["duplicates"] == ["job_m"] and result["evidence_attached"] == ["rec_job_m"]
    values = stored_comparisons(tmp_path, "rec_job_m")
    assert values["flip_rate"] == "0.25"
    assert (
        alice.post("/api/records/rec_job_m/evidence", files={"file": ("s.jsonl.gz", payload)}).status_code
        == 409
    )


def test_records_without_a_checksum_must_reproduce_their_metric(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 0.0, 1.0, 1.0])]
    push(
        alice, record("job_1", "rk_a", {"accuracy": "0.5"}, produces=[Artifact.PER_SAMPLE])
    ).raise_for_status()
    rejected = alice.post("/api/records/rec_job_1/evidence", files={"file": ("s.jsonl.gz", gz(samples))})
    assert rejected.status_code == 400 and "0.75" in rejected.json()["detail"]
    push(
        alice, record("job_2", "rk_b", {"accuracy": "0.75"}, produces=[Artifact.PER_SAMPLE])
    ).raise_for_status()
    alice.post(
        "/api/records/rec_job_2/evidence", files={"file": ("s.jsonl.gz", gz(samples))}
    ).raise_for_status()


def test_results_package_imports_records_with_their_per_sample_files(server, tmp_path):
    _app, _admin, as_user = server
    alice = as_user("alice")
    samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 0.0, 1.0, 1.0])]
    local = tmp_path / "job_p" / "per_sample.jsonl.gz"
    local.parent.mkdir()
    local.write_bytes(gz(samples))
    rec = record(
        "job_p",
        "rk_p",
        {"accuracy": "0.75"},
        artifacts={
            "per_sample": local.as_uri(),
            "per_sample_sha256": bundle_io.file_sha256(local.read_bytes()),
        },
        produces=[Artifact.PER_SAMPLE],
    )
    rec.spec["metrics"] = ["accuracy_stderr"]
    package = tmp_path / "plan.results.tar.gz"
    bundle_io.write_package(package, bundle_io.make_bundle([rec]))
    preview = alice.post("/api/packages/preview", files={"file": ("p.tar.gz", package.read_bytes())}).json()
    assert preview["with_samples"] == 1 and preview["rows"][0]["record_id"] == "rec_job_p"
    result = alice.post(
        "/api/packages", params={"to": "private"}, files={"file": ("p.tar.gz", package.read_bytes())}
    )
    assert result.json()["imported"] == ["rec_job_p"]
    row = alice.get("/api/records/rec_job_p").json()
    assert row["record"]["artifacts"]["per_sample"].startswith("server:artifacts/")
    assert "accuracy_stderr" not in row["metrics"]
    assert alice.post("/api/packages", files={"file": ("p.tar.gz", b"not a tarball")}).status_code == 400


def test_sample_diff_lists_flips_by_subtask_with_coverage(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    ref_samples = [
        Sample(id="algebra:0", target=1, scores={"correct": 1.0}, choice_loglikelihoods=[-2.0, -0.1]),
        Sample(id="algebra:1", target=0, scores={"correct": 0.0}, choice_loglikelihoods=[-2.0, -0.1]),
        Sample(id="history:0", target=0, scores={"correct": 1.0}, choice_loglikelihoods=[-0.1, -2.0]),
        Sample(id="history:9", target=0, scores={"correct": 1.0}, choice_loglikelihoods=[-0.1, -2.0]),
    ]
    main_samples = [
        Sample(id="algebra:0", target=1, scores={"correct": 0.0}, choice_loglikelihoods=[-0.1, -2.0]),
        Sample(id="algebra:1", target=0, scores={"correct": 1.0}, choice_loglikelihoods=[-0.1, -2.0]),
        Sample(id="history:0", target=0, scores={"correct": 1.0}, choice_loglikelihoods=[-0.1, -2.0]),
    ]
    ref = record(
        "job_r",
        "rk_r",
        {"accuracy": "0.75"},
        role="reference",
        artifacts={"per_sample": upload(alice, ref_samples)},
        produces=[Artifact.PER_SAMPLE],
    )
    main = record(
        "job_m",
        "rk_m",
        {"accuracy": "0.6667"},
        reference_key="rk_r",
        compare=["flip_rate"],
        artifacts={"per_sample": upload(alice, main_samples)},
        produces=[Artifact.PER_SAMPLE],
    )
    push(alice, ref, main).raise_for_status()
    diff = alice.get("/api/records/rec_job_m/samples").json()
    assert diff["against"] == "rec_job_r"
    assert diff["coverage"] == {"main": 3, "reference": 4, "aligned": 3}
    assert diff["counts"] == {"to_wrong": 1, "to_right": 1, "both_right": 1, "both_wrong": 0}
    assert [(s["id"], s["flip"], s["main"], s["reference"]) for s in diff["samples"]] == [
        ("algebra:0", "to_wrong", 0, 1),
        ("algebra:1", "to_right", 0, 1),
    ]
    only = alice.get("/api/records/rec_job_m/samples", params={"flip": "to_wrong"}).json()
    assert [s["id"] for s in only["samples"]] == ["algebra:0"]
    assert (
        alice.get("/api/records/rec_job_m/samples", params={"subtask": "history", "flip": "all"}).json()[
            "total"
        ]
        == 1
    )
    assert diff["subtasks"]["algebra"]["to_wrong"] == 1
    assert bob.get("/api/records/rec_job_m/samples").status_code == 404
    push(alice, record("job_x", "rk_x", {"accuracy": "0.5"}, reference_key="rk_r", compare=["flip_rate"]))
    assert alice.get("/api/records/rec_job_x/samples", params={"against": "rec_job_r"}).status_code == 409


def test_frozen_table_snapshot_keeps_its_numbers(server):
    _app, admin, as_user = server
    admin.post(
        "/api/admin/users", json={"username": "carol", "password": "carol-password"}
    ).raise_for_status()
    alice, carol = as_user("alice"), as_user("carol")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    table = alice.post("/api/projects/p/tables", json=TABLE).json()
    base = cells(table)["base"]["run_key"]
    plan = alice.post(
        f"/api/projects/p/tables/{table['table_id']}/command", json={"run_keys": [base]}
    ).json()["plan"]
    reference = next(j for j in plan["jobs"] if j["role"] == "reference")
    push_to(alice, "project:p", job_record(reference, "job_ref", {"ppl": "20.9"})).raise_for_status()
    snap = alice.post(f"/api/projects/p/tables/{table['table_id']}/snapshots").json()
    frozen = next(r for r in snap["body"]["blocks"][0]["rows"] if r["label"] == "base")["cells"][
        "native/wikitext2_ppl"
    ]
    assert (frozen["status"], frozen["metrics"]) == ("done", {"ppl": "20.9"})
    assert snap["body"]["metrics"]["ppl"]["format"]["decimals"] == 2
    alice.delete("/api/records/rec_job_ref").raise_for_status()
    live = cells(alice.get("/api/projects/p/tables").json()[0])["base"]
    assert live["status"] != "done"
    again = alice.get(f"/api/projects/p/snapshots/{snap['snapshot_id']}").json()
    assert (
        next(r for r in again["body"]["blocks"][0]["rows"] if r["label"] == "base")["cells"][
            "native/wikitext2_ppl"
        ]
        == frozen
    )
    assert [s["snapshot_id"] for s in alice.get("/api/projects/p/snapshots").json()] == [snap["snapshot_id"]]
    assert snap["body"]["columns"] == [
        {
            "task": "native/wikitext2_ppl",
            "metric": "ppl",
            "category": "language_modeling",
            "band": "PPL",
            "order": 0,
        },
        {"task": "lm_eval/mmlu", "metric": "acc", "category": "qa", "band": "General tasks", "order": 1},
    ]
    assert carol.get(f"/api/projects/p/snapshots/{snap['snapshot_id']}").status_code == 403


def test_resume_runs_only_jobs_without_results_and_keeps_their_ids(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    spec = {
        "model": "qwen3-0.6b-base",
        "method": {"name": "gptq"},
        "field": "quantization",
        "tasks": ["native/wikitext2_ppl", {"name": "lm_eval/arc_challenge", "mode": "loglikelihood"}],
        "reuse": "never",
    }
    plan = alice.post("/api/render", json={"spec": spec}).json()["plan"]
    done = plan["jobs"][0]
    push(
        alice,
        job_record(
            done, "x", {"ppl" if done["spec"]["task"]["name"] == "native/wikitext2_ppl" else "accuracy": "1"}
        ).model_copy(update={"job_id": done["job_id"], "record_id": "rec_done"}),
    ).raise_for_status()
    assert bob.post(f"/api/runs/{plan['plan_id']}/resume").status_code == 403
    resumed = alice.post(f"/api/runs/{plan['plan_id']}/resume", json={"to": "private"}).json()
    assert [j["job_id"] for j in resumed["plan"]["jobs"]] == [j["job_id"] for j in plan["jobs"][1:]]
    assert resumed["plan"]["plan_id"] != plan["plan_id"] and len(resumed["commands"]) == 1
    for job in plan["jobs"][1:]:
        push(
            alice,
            job_record(job, "y" + job["job_id"][-6:], {"accuracy": "1"}).model_copy(
                update={"job_id": job["job_id"]}
            ),
        )
    assert alice.post(f"/api/runs/{plan['plan_id']}/resume").json()["commands"] == []


def test_frozen_version_can_be_renamed_by_members_and_deleted_by_its_creator_or_an_admin(server):
    _app, admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    alice.post("/api/projects/p/members", json={"username": "bob"}).raise_for_status()
    table = alice.post("/api/projects/p/tables", json=TABLE).json()
    first = alice.post(f"/api/projects/p/tables/{table['table_id']}/snapshots").json()["snapshot_id"]
    second = alice.post(f"/api/projects/p/tables/{table['table_id']}/snapshots").json()["snapshot_id"]

    def titles():
        return {s["snapshot_id"]: s["title"] for s in alice.get("/api/projects/p/snapshots").json()}

    bob.patch(f"/api/projects/p/snapshots/{first}", json={"title": "  camera-ready  "}).raise_for_status()
    assert titles() == {first: "camera-ready", second: None}
    alice.patch(f"/api/projects/p/snapshots/{first}", json={"title": " "}).raise_for_status()
    assert titles()[first] is None
    assert alice.patch("/api/projects/p/snapshots/snap_nope", json={"title": "x"}).status_code == 404

    assert bob.delete(f"/api/projects/p/snapshots/{first}").status_code == 403
    alice.delete(f"/api/projects/p/snapshots/{first}").raise_for_status()
    assert list(titles()) == [second]
    assert alice.get(f"/api/projects/p/snapshots/{first}").status_code == 404
    admin.post("/api/projects/p/members", json={"username": "admin"}).raise_for_status()
    admin.delete(f"/api/projects/p/snapshots/{second}").raise_for_status()
    assert titles() == {}


def test_snapshot_body_does_not_follow_later_catalog_changes(server, monkeypatch):
    from effeval.core import catalog as catalog_module

    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    table = alice.post("/api/projects/p/tables", json=TABLE).json()
    base = cells(table)["base"]["run_key"]
    plan = alice.post(
        f"/api/projects/p/tables/{table['table_id']}/command", json={"run_keys": [base]}
    ).json()["plan"]
    reference = next(j for j in plan["jobs"] if j["role"] == "reference")
    push_to(alice, "project:p", job_record(reference, "job_ref", {"ppl": "20.9"})).raise_for_status()
    snap = alice.post(f"/api/projects/p/tables/{table['table_id']}/snapshots").json()
    real = catalog_module.builtin_catalog()
    metrics = dict(real.metrics)
    metrics["ppl"] = metrics["ppl"].model_copy(update={"format": {"style": "percent", "decimals": 1}})
    changed = real.model_copy(update={"metrics": metrics})
    monkeypatch.setattr("effeval.api.service.builtin_catalog", lambda: changed)
    again = alice.get(f"/api/projects/p/snapshots/{snap['snapshot_id']}").json()
    assert again["body"] == snap["body"]
    assert again["body"]["metrics"]["ppl"]["format"] == {"style": "fixed", "decimals": 2}


def test_a_table_may_name_a_task_the_catalog_no_longer_has_and_can_still_be_frozen(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    gone = "speed/removed_set"  # e.g. a user prompt set whose file was deleted after the table was saved
    view = alice.post("/api/projects/p/tables", json={**TABLE, "tasks": ["native/wikitext2_ppl", gone]})
    view.raise_for_status()
    rows = {r["label"]: r for r in view.json()["blocks"][0]["rows"]}
    assert rows["RTN"]["cells"][gone]["status"] == "invalid"
    assert alice.get("/api/projects/p/tables").json()[0]["counts"]["invalid"] >= 2

    snap = alice.post(f"/api/projects/p/tables/{view.json()['table_id']}/snapshots")
    assert snap.status_code == 200
    body = snap.json()["body"]
    assert body["missing_tasks"] == [gone]
    assert [c["task"] for c in body["columns"]] == ["native/wikitext2_ppl"]
    frozen_rows = {r["label"]: r for r in body["blocks"][0]["rows"]}
    assert frozen_rows["RTN"]["cells"][gone]["status"] == "invalid"


def test_runs_have_titles_link_their_results_and_can_be_removed(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    spec = {"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}
    plan = alice.post("/api/render", json={"spec": spec}).json()["plan"]
    [run] = alice.get("/api/runs").json()
    assert run["title"] is None and run["summary"] == "qwen3-0.6b-base · gptq · native/wikitext2_ppl"
    main = next(j for j in plan["jobs"] if j["role"] == "main")
    push(
        alice, job_record(main, "x", {"ppl": "30.1"}).model_copy(update={"job_id": main["job_id"]})
    ).raise_for_status()
    assert alice.get("/api/runs").json()[0]["records"] == {main["job_id"]: "rec_x"}
    assert bob.patch(f"/api/runs/{plan['plan_id']}", json={"title": "mine"}).status_code == 403
    assert bob.delete(f"/api/runs/{plan['plan_id']}").status_code == 403
    alice.patch(f"/api/runs/{plan['plan_id']}", json={"title": "W4 sanity"}).raise_for_status()
    assert alice.get("/api/runs").json()[0]["title"] == "W4 sanity"
    command = alice.post(f"/api/runs/{plan['plan_id']}/command").json()
    assert plan["plan_id"] in command["write"] and command["run"].startswith("effeval run --config")
    alice.delete(f"/api/runs/{plan['plan_id']}").raise_for_status()
    assert alice.get("/api/runs").json() == []
    assert alice.get("/api/records/rec_x").status_code == 200


def test_models_register_from_the_hub_by_lookup_or_by_hand(server, monkeypatch):
    import httpx

    from effeval.api import service

    _app, _admin, as_user = server
    alice = as_user("alice")
    answers = {
        "/api/models/Org/My-Model-7B": {"sha": "a" * 40, "safetensors": {"total": 7_000_000_000}},
        f"/Org/My-Model-7B/resolve/{'a' * 40}/config.json": {
            "architectures": ["Qwen3ForCausalLM"],
            "torch_dtype": "bfloat16",
            "tie_word_embeddings": False,
        },
        "/api/models/Org/Odd": {"sha": "b" * 40, "safetensors": {"total": 1}},
        f"/Org/Odd/resolve/{'b' * 40}/config.json": {
            "architectures": ["MambaForCausalLM"],
            "torch_dtype": "bfloat16",
        },
    }
    monkeypatch.setattr(service, "_hub_get", lambda path: answers[path])
    entry = alice.get("/api/models/lookup", params={"repo": "Org/My-Model-7B"}).json()
    assert (entry["code"], entry["revision"], entry["params"], entry["tie_word_embeddings"]) == (
        "my-model-7b",
        "a" * 40,
        7_000_000_000,
        False,
    )
    alice.post("/api/models", json=entry).raise_for_status()
    assert "my-model-7b" in alice.get("/api/describe").json()["models"]
    assert alice.get("/api/models/lookup", params={"repo": "Org/Odd"}).status_code == 400

    manual = {
        **entry,
        "code": "my-finetune",
        "checkpoint": "/data/ckpt/My-Finetune",
        "revision": None,
        "description": "",
    }
    alice.post("/api/models", json=manual).raise_for_status()

    monkeypatch.undo()

    def unreachable(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(service.httpx, "get", unreachable)
    offline = alice.get("/api/models/lookup", params={"repo": "Org/My-Model-7B"})
    assert offline.status_code == 502 and "cannot reach huggingface.co" in offline.json()["detail"]


def test_storage_paths_place_private_and_published_files_and_move_them(server, tmp_path):
    _app, admin, as_user = server
    alice = as_user("alice")
    samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 0.0])]
    mine, lab = tmp_path / "alice", tmp_path / "lab"
    assert alice.put("/api/me/storage", json={"path": "relative/dir"}).status_code == 400
    assert alice.put("/api/settings/storage/shared", json={"path": str(lab)}).status_code == 403
    assert alice.put("/api/me/storage", json={"path": str(mine)}).json() == {"moved": 0}
    uri = upload(alice, samples)
    name = uri.split("/")[-1]
    assert (mine / name).exists()
    push(
        alice,
        record(
            "job_1",
            "rk_a",
            {"accuracy": "0.5"},
            artifacts={"per_sample": uri},
            produces=[Artifact.PER_SAMPLE],
        ),
    )
    usage = alice.get("/api/settings/storage").json()
    assert (
        usage["personal"]["files"] == 1
        and usage["personal"]["path"] == str(mine)
        and usage["shared"]["editable"] is False
    )

    admin.put("/api/settings/storage/shared", json={"path": str(lab)}).raise_for_status()
    alice.post("/api/records/rec_job_1/publish").raise_for_status()
    assert (lab / name).exists() and not (mine / name).exists()

    moved_to = tmp_path / "lab2"
    assert admin.put("/api/settings/storage/shared", json={"path": str(moved_to)}).json() == {"moved": 1}
    assert (moved_to / name).exists()
    rows = alice.get("/api/records?scope=mine").json()
    assert rows[0]["record"]["artifacts"]["per_sample"] == uri
    assert (
        alice.get("/api/records/rec_job_1/samples", params={"against": "rec_job_1", "flip": "all"}).json()[
            "coverage"
        ]["aligned"]
        == 2
    )


def test_login_tokens_are_one_per_machine_expire_and_are_managed_by_admins(server, monkeypatch):
    from effeval.server import app as app_module

    app, admin, as_user = server
    client = TestClient(app)

    def login(label):
        body = {"username": "alice", "password": "alice-password", "label": label}
        return client.post("/api/auth/token", json=body).json()

    first, second = login("cli@node1"), login("cli@node1")
    headers = lambda tok: {"Authorization": f"Bearer {tok['token']}"}  # noqa: E731
    assert client.get("/api/me", headers=headers(first)).status_code == 401
    assert client.get("/api/me", headers=headers(second)).status_code == 200
    other = login("cli@node2")
    tokens = admin.get("/api/admin/tokens").json()
    alice_tokens = sorted(t["label"] for t in tokens if t["username"] == "alice")
    assert alice_tokens == ["cli@node1", "cli@node2"]
    assert next(t for t in tokens if t["token_id"] == second["token_id"])["last_used"] is not None
    assert as_user("bob").get("/api/admin/tokens").status_code == 403
    admin.delete(f"/api/admin/tokens/{other['token_id']}").raise_for_status()
    assert client.get("/api/me", headers=headers(other)).status_code == 401

    monkeypatch.setattr(app_module, "TOKEN_DAYS", -1)
    expired = login("cli@node3")
    response = client.get("/api/me", headers=headers(expired))
    assert response.status_code == 401 and "expired" in response.json()["detail"]


def test_server_answers_other_requests_during_a_slow_upload(server):
    app, admin, _ = server
    store = app.state.store
    inside = threading.Event()
    save = store.save_artifact

    def slow_save(payload, owner):
        inside.set()
        time.sleep(1.5)
        return save(payload, owner)

    store.save_artifact = slow_save
    payload = gzip.compress(Sample(id="s1", scores={"acc": 1.0}).model_dump_json().encode())
    with TestClient(app, headers=admin.headers) as client:
        uploader = threading.Thread(
            target=lambda: client.post("/api/artifacts", files={"file": ("s.jsonl.gz", payload)})
        )
        uploader.start()
        inside.wait(5)
        started = time.monotonic()
        assert client.get("/api/me").status_code == 200
        elapsed = time.monotonic() - started
        uploader.join()
    assert elapsed < 1.0


def test_uploaded_task_listings_reach_describe_and_validation(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    described = alice.get("/api/describe").json()
    assert "lm_eval/mmlu" in described["tasks"] and "evalscope/gsm8k" in described["tasks"]
    assert described["evaluators"]["lm_eval"]["library_version"] == "0.4.13"
    assert {c["evaluator"]: c["tasks"] for c in described["task_catalogs"]} == {"lm_eval": 8, "evalscope": 3}
    spec = {
        "model": "qwen3-8b-base",
        "method": {"name": "rtn"},
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
    }
    issues = alice.post("/api/validate", json={"spec": spec}).json()["issues"]
    assert [i for i in issues if i["severity"] == "error"] == []


def test_only_admins_upload_task_listings_and_builtin_names_are_reserved(server):
    from catalogs import raw_listing

    _app, admin, as_user = server
    alice = as_user("alice")
    assert alice.post("/api/task-catalogs", json=raw_listing("lm_eval")).status_code == 403
    native = raw_listing("lm_eval")
    native["evaluator"]["name"] = "native"
    assert admin.post("/api/task-catalogs", json=native).status_code == 409
    malformed = raw_listing("lm_eval")
    malformed["tasks"][0]["subtasks"] = ["nope"]
    assert admin.post("/api/task-catalogs", json=malformed).status_code in {400, 422}


def test_reuploading_a_listing_replaces_the_evaluators_tasks(server):
    from catalogs import raw_listing

    _app, admin, as_user = server
    alice = as_user("alice")
    narrowed = raw_listing("lm_eval")
    narrowed["tasks"] = [t for t in narrowed["tasks"] if t["name"] == "arc_challenge"]
    admin.post("/api/task-catalogs", json=narrowed).raise_for_status()
    tasks = alice.get("/api/describe").json()["tasks"]
    assert "lm_eval/gsm8k" not in tasks and "lm_eval/arc_challenge" in tasks and "evalscope/mmlu" in tasks
    assert [c["uploaded_by"] for c in alice.get("/api/task-catalogs").json()] == ["admin", "admin"]


def test_describe_is_gzip_encoded_for_clients_that_accept_it(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    response = alice.get("/api/describe", headers={"Accept-Encoding": "gzip"})
    assert response.headers["content-encoding"] == "gzip"


def test_common_tasks_follow_the_uploaded_catalogs(server):
    from catalogs import raw_listing

    _app, admin, as_user = server
    alice = as_user("alice")
    assert "lm_eval/mmlu" in alice.get("/api/describe").json()["common_tasks"]
    narrowed = raw_listing("lm_eval")
    narrowed["tasks"] = [t for t in narrowed["tasks"] if t["name"] == "gsm8k"]
    admin.post("/api/task-catalogs", json=narrowed).raise_for_status()
    assert "lm_eval/mmlu" not in alice.get("/api/describe").json()["common_tasks"]


REPEAT_TABLE = {
    **TABLE,
    "tasks": ["native/wikitext2_ppl"],
    "rows": TABLE["rows"][:1],
    "shared": {"repeats": {"seeds": [3, 5, 7]}},
}
REPEAT_SEEDS = [3, 5, 7]


def repeat_cell(view):
    row = next(r for r in view["blocks"][0]["rows"] if r["label"] == "RTN")
    return row["cells"]["native/wikitext2_ppl"]


def run_repeats(client, view, values):
    """Upload one result per seed of the RTN cell; `values` lists each seed's ppl (None: not run)."""
    keys = [p["run_key"] for p in repeat_cell(view)["repeats"]]
    plan = client.post(f"/api/projects/p/tables/{view['table_id']}/command", json={"run_keys": keys}).json()[
        "plan"
    ]
    for job in (j for j in plan["jobs"] if j["role"] == "main"):
        seed = job["spec"]["determinism"]["seed"]
        value = values[REPEAT_SEEDS.index(seed)]
        if value is not None:
            push_to(client, "project:p", job_record(job, f"seed{seed}", {"ppl": value})).raise_for_status()


def test_repeated_cell_shows_exact_mean_and_sample_std_over_its_repeats(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    view = alice.post("/api/projects/p/tables", json=REPEAT_TABLE).json()
    cell = repeat_cell(view)
    assert cell["status"] == "pending" and len(cell["repeats"]) == 3
    assert next(p["run_key"] for p in cell["repeats"]) == cell["run_key"]
    assert len({p["run_key"] for p in cell["repeats"]}) == 3

    run_repeats(alice, view, ["10", "12", None])
    cell = repeat_cell(alice.get("/api/projects/p/tables").json()[0])
    assert cell["status"] == "pending"
    assert cell["metrics"] == {"ppl": "11"} and cell["spread"]["ppl"]["n"] == 2

    run_repeats(alice, view, [None, None, "14"])
    cell = repeat_cell(alice.get("/api/projects/p/tables").json()[0])
    assert cell["status"] == "done"
    assert cell["metrics"] == {"ppl": "12"}
    assert cell["spread"] == {"ppl": {"std": "2", "n": 3}}


def test_a_count_repeat_cell_has_one_run_per_repeat_with_the_same_seed(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    table = {**REPEAT_TABLE, "shared": {"determinism": {"seed": 2}, "repeats": {"count": 2}}}
    view = alice.post("/api/projects/p/tables", json=table).json()
    cell = repeat_cell(view)
    assert len(cell["repeats"]) == 2 and len({p["run_key"] for p in cell["repeats"]}) == 2


def test_a_seed_list_with_a_different_count_makes_the_table_cells_invalid(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    table = {**REPEAT_TABLE, "shared": {"repeats": {"count": 2, "seeds": [3, 5, 7]}}}
    cell = repeat_cell(alice.post("/api/projects/p/tables", json=table).json())
    assert cell["status"] == "invalid" and "lists 3 seeds" in cell["issues"][0]["message"]


def test_a_table_without_repeats_has_a_single_run_cell_without_spread(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    view = alice.post("/api/projects/p/tables", json={**REPEAT_TABLE, "shared": {}}).json()
    assert "repeats" not in repeat_cell(view)


def test_reuse_preview_lists_stored_results_per_repeat_and_saves_nothing(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    spec = {
        "field": "quantization",
        "model": {"name": "qwen3-0.6b-base"},
        "method": {"name": "rtn", "preset": "gptq_paper"},
        "tasks": [{"name": "native/wikitext2_ppl"}],
        "repeats": {"seeds": [4, 5]},
    }
    plan = alice.post("/api/render", json={"spec": spec}).json()["plan"]
    first = next(j for j in plan["jobs"] if j["role"] == "main" and j["spec"]["determinism"]["seed"] == 4)
    push_to(alice, "private", job_record(first, "stored", {"ppl": "9.5"})).raise_for_status()
    runs_before = alice.get("/api/runs").json()

    preview = alice.post("/api/reuse-preview", json={"spec": spec}).json()
    mains = sorted((j for j in preview["jobs"] if j["role"] == "main"), key=lambda j: j["seed"])
    assert [(j["seed"], j["status"]) for j in mains] == [(4, "reused"), (5, "run")]
    assert mains[0]["record"]["record_id"] == "rec_stored" and mains[0]["record"]["metrics"] == {"ppl": "9.5"}
    assert mains[1]["record"] is None
    assert alice.get("/api/runs").json() == runs_before

    never = alice.post("/api/reuse-preview", json={"spec": {**spec, "reuse": "never"}}).json()
    stored = next(j for j in never["jobs"] if j["role"] == "main" and j["seed"] == 4)
    assert stored["status"] == "not_reused" and "Never" in stored["reason"]


def test_table_cells_report_the_sample_limit_stored_in_their_record(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    table = alice.post("/api/projects/p/tables", json=TABLE).json()
    key = cells(table)["GPTQ"]["run_key"]
    plan = alice.post(f"/api/projects/p/tables/{table['table_id']}/command", json={"run_keys": [key]}).json()[
        "plan"
    ]
    main = next(j for j in plan["jobs"] if j["role"] == "main")
    assert main["spec"]["task"]["limit"] is None
    stored = {**main, "spec": {**main["spec"], "task": {**main["spec"]["task"], "limit": 50}}}
    push_to(alice, "project:p", job_record(stored, "lim", {"ppl": "9.5"})).raise_for_status()
    done = cells(alice.get("/api/projects/p/tables").json()[0])["GPTQ"]
    assert (done["status"], done["limit"]) == ("done", 50)
    assert "limit" not in cells(alice.get("/api/projects/p/tables").json()[0])["RTN"]


def quarot_spec(**preprocess):
    return {
        "model": "qwen3-0.6b-base",
        "method": {"name": "gptq", "preset": "quarot_w4a4kv4"},
        "field": "quantization",
        "tasks": ["native/wikitext2_ppl"],
        "preprocess": {"preset": "quarot", **preprocess},
    }


def test_describe_exposes_the_decoder_depth_of_each_model(server):
    _app, _admin, as_user = server
    models = as_user("alice").get("/api/describe").json()["models"]
    assert models["qwen3-0.6b"]["shape"]["num_layers"] == 28
    assert models["qwen3-32b"]["shape"]["num_layers"] == 64


def test_describe_lists_architecture_graph_position_tables_and_preprocess_presets(server):
    from effeval.core.preprocess import builtin_preprocess_presets

    _app, _admin, as_user = server
    described = as_user("alice").get("/api/describe").json()
    model = described["models"]["qwen3-0.6b"]
    graph = described["architectures"][model["architecture"]]["graph"]
    assert {n["id"] for n in graph["nodes"]} >= {"q", "down", "input_layernorm"} and graph["edges"]
    positions = model["positions"]
    assert set(positions) >= {"residual", "attn.v_head", "attn.qk_rope", "linear.mlp.down_proj.input"}
    assert positions["residual"]["rotation"]["offline"] == "ok"
    assert positions["residual"]["rotation"]["online"] != "ok"
    assert positions["linear.mlp.down_proj.input"]["rotation"]["online"] == "ok"

    presets = described["preprocess_presets"]
    assert set(presets) == set(builtin_preprocess_presets()) and "quarot" in presets
    quarot = presets["quarot"]
    assert quarot["title"] == "QuaRot" and quarot["paper"]["arxiv"] == "2404.00456"
    (step,) = quarot["steps"]
    first = step["entries"][0]
    assert step["kind"] == "rotation" and (first["position"], first["mode"], first["layers"]) == (
        "residual",
        "offline",
        "all",
    )
    assert first["source"] == {"type": "random_hadamard", "seed": 0, "fallback": None}
    assert quarot["versions"]["rotation.random_hadamard"] == "1"
    assert quarot["quantizers"]["gptq"] == ["quarot_w4a16", "quarot_w4a4kv4"]

    sources = described["preprocess_sources"]
    assert sources["rotation"]["trained"]["implemented"] is False
    assert sources["rotation"]["hadamard"]["implemented"] is True
    assert (
        sources["scaling"]["awq"]["needs_data"] is True
        and sources["rotation"]["hadamard"]["needs_data"] is False
    )
    assert sources["scaling"]["awq"]["schema"]["properties"]["n_grid"]["default"] == 20


def test_runs_and_records_carry_the_preprocess_label(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    plain = alice.post("/api/render", json={"spec": {**quarot_spec(), "preprocess": None}}).json()["plan"]
    labelled = alice.post("/api/render", json={"spec": quarot_spec()}).json()["plan"]
    preview = alice.post("/api/reuse-preview", json={"spec": quarot_spec()}).json()
    assert {j["label"] for j in preview["jobs"] if j["role"] == "main"} == {"QuaRot + GPTQ (quarot_w4a4kv4)"}
    summaries = {r["plan_id"]: r["summary"] for r in alice.get("/api/runs").json()}
    assert summaries[labelled["plan_id"]].split(" · ")[1] == "QuaRot + GPTQ (quarot_w4a4kv4)"
    assert summaries[plain["plan_id"]].split(" · ")[1] == "gptq"

    main = next(j for j in labelled["jobs"] if j["role"] == "main")
    plain_main = next(j for j in plain["jobs"] if j["role"] == "main")
    odd = {**main, "spec": {**main["spec"], "preprocess": {**main["spec"]["preprocess"], "preset": None}}}
    tweaked = main["spec"]["preprocess"]["steps"][0]["entries"][0]["source"]["seed"] + 1
    edited = {**main["spec"]["preprocess"]}
    edited["steps"] = [dict(edited["steps"][0])]
    edited["steps"][0]["entries"] = [
        {**edited["steps"][0]["entries"][0], "source": {"type": "random_hadamard", "seed": tweaked}},
        *edited["steps"][0]["entries"][1:],
    ]
    deviating = {**main, "spec": {**main["spec"], "preprocess": edited}}
    push_to(
        alice,
        "project:p",
        job_record(main, "q1", {"ppl": "9.5"}),
        job_record({**odd, "run_key": "rk_odd"}, "q2", {"ppl": "9.6"}),
        job_record({**deviating, "run_key": "rk_dev"}, "q3", {"ppl": "9.7"}),
        job_record(plain_main, "q4", {"ppl": "9.8"}),
    ).raise_for_status()
    rows = {r["record_id"]: r for r in alice.get("/api/records?scope=mine").json()}
    assert rows["rec_q1"]["label"] == "QuaRot + GPTQ (quarot_w4a4kv4)"
    assert rows["rec_q1"]["display"]["preprocess"] == "QuaRot"
    assert rows["rec_q1"]["preprocess"]["preset"] == "quarot"
    assert rows["rec_q2"]["label"] == "Custom preprocessing + GPTQ (quarot_w4a4kv4)"
    assert rows["rec_q3"]["label"] == "QuaRot* + GPTQ (quarot_w4a4kv4)"
    assert rows["rec_q3"]["display"]["preprocess"] == "QuaRot*"
    assert rows["rec_q4"]["label"] == "GPTQ" and rows["rec_q4"]["preprocess"] is None
    assert "preprocess" not in rows["rec_q4"]["display"]


def test_table_rows_take_a_preprocess_and_are_labelled_with_it(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    jobs = alice.post("/api/render", json={"spec": quarot_spec()}).json()["plan"]["jobs"]
    custom = next(j for j in jobs if j["role"] == "main")["spec"]["preprocess"]
    view = alice.post(
        "/api/projects/p/tables",
        json={
            **TABLE,
            "tasks": ["native/wikitext2_ppl"],
            "rows": [
                {"method": {"name": "gptq", "preset": "quarot_w4a4kv4"}, "preprocess": "quarot"},
                {
                    "label": "mine",
                    "method": {"name": "gptq", "preset": "paper"},
                    "preprocess": {"steps": custom["steps"]},
                },
                {"label": "RTN", "method": {"name": "rtn", "preset": "gptq_paper"}},
            ],
        },
    )
    assert view.status_code == 200
    rows = {r["key"]: r for r in view.json()["blocks"][0]["rows"]}
    assert (rows["row0"]["label"], rows["row0"]["title"]) == 2 * ("QuaRot + GPTQ (quarot_w4a4kv4)",)
    assert rows["row0"]["preprocess"] == "quarot"
    assert (rows["row1"]["label"], rows["row1"]["title"]) == ("mine", "Custom preprocessing + GPTQ (paper)")
    assert (rows["row2"]["label"], rows["row2"]["title"], rows["row2"]["preprocess"]) == ("RTN", "RTN", None)
    assert rows["base0"]["title"] == "Base" and rows["base0"]["preprocess"] is None
    assert rows["row0"]["cells"]["native/wikitext2_ppl"]["status"] == "pending"
    implicit = {**quarot_spec(), "method": {"name": "gptq"}}
    issues = alice.post("/api/validate", json={"spec": implicit}).json()["issues"]
    assert "preset.needs_choice" in {i["code"] for i in issues if i["severity"] == "error"}


SPECULATIVE_SPEC = {
    "model": {"name": "qwen3-8b"},
    "method": {"name": "base"},
    "field": "speculative_decoding",
    "tasks": [{"name": "lm_eval/gsm8k"}],
    "decoding": {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": 4},
}


@pytest.fixture
def speed_comparatives(monkeypatch):
    """The speed comparatives are implemented elsewhere; the field names them, so register stand-ins."""
    from effeval.metrics.quality import COMPARATIVE, Comparative

    for name in ("speedup", "output_match"):
        monkeypatch.setitem(
            COMPARATIVE,
            name,
            Comparative(name, frozenset({Artifact.PER_SAMPLE}), frozenset(), "stand-in", lambda *_: {}),
        )


def test_table_rows_take_a_decoding_shown_as_config_columns_not_in_the_label(server, speed_comparatives):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()

    def decoding(k):
        return {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": k}

    view = alice.post(
        "/api/projects/p/tables",
        json={
            "name": "spec",
            "field": "speculative_decoding",
            "models": ["qwen3-8b"],
            "tasks": ["lm_eval/gsm8k"],
            "rows": [
                {"label": "draft k=2", "method": {"name": "base"}, "decoding": decoding(2)},
                {"label": "draft k=4", "method": {"name": "base"}, "decoding": decoding(4)},
            ],
        },
    )
    assert view.status_code == 200, view.text
    rows = {r["key"]: r for r in view.json()["blocks"][0]["rows"]}
    assert sorted(rows) == ["base0", "row0", "row1"]
    keys = {r["cells"]["lm_eval/gsm8k"]["run_key"] for r in rows.values()}
    assert len(keys) == 3
    assert (rows["base0"]["draft"], rows["base0"]["decoding"], rows["base0"]["runtime"]) == (None, None, "hf")
    assert rows["row0"]["draft"] == rows["row1"]["draft"] == "qwen3-0.6b"
    assert [rows[k]["decoding"] for k in ("row0", "row1")] == [
        {"k": 2, "verification": "greedy"},
        {"k": 4, "verification": "greedy"},
    ]
    assert rows["row0"]["runtime"] == "hf"
    unlabelled = alice.post(
        "/api/projects/p/tables",
        json={
            "name": "spec2",
            "field": "speculative_decoding",
            "models": ["qwen3-8b"],
            "tasks": ["lm_eval/gsm8k"],
            "rows": [{"method": {"name": "base"}, "decoding": decoding(3)}],
        },
    ).json()
    row = next(r for r in unlabelled["blocks"][0]["rows"] if r["key"] == "row0")
    assert "qwen3-0.6b" not in row["label"] and "k=" not in row["label"] and row["decoding"]["k"] == 3


def speed_table(shared=None, rows=None):
    def decoding(k):
        return {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": k}

    return {
        "name": "spec",
        "field": "speculative_decoding",
        "models": ["qwen3-8b"],
        "tasks": ["speed/spec_bench"],
        "rows": rows or [{"label": "k4", "method": {"name": "base"}, "decoding": decoding(4)}],
        "shared": {"hardware": {"gpu": "H100-SXM"}, **(shared or {})},
    }


def speed_jobs(client, view):
    """Plan jobs of the speed cells of the table's rows, as (row key -> list of jobs, in seed order)."""
    task = "speed/spec_bench"
    rows = {r["key"]: r for r in view["blocks"][0]["rows"]}
    keys = [p["run_key"] for r in rows.values() for p in r["cells"][task].get("repeats", [r["cells"][task]])]
    url = f"/api/projects/p/tables/{view['table_id']}/command"
    return client.post(url, json={"run_keys": keys}).json()["plan"]["jobs"]


def spec_cells(view):
    return {r["key"]: r["cells"]["speed/spec_bench"] for r in view["blocks"][0]["rows"]}


def test_table_cells_carry_the_speed_comparisons_of_their_record(server, tmp_path):
    from effeval.store.sqlite import SqliteStore

    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    view = alice.post("/api/projects/p/tables", json=speed_table()).json()
    jobs = speed_jobs(alice, view)
    main = next(j for j in jobs if j["role"] == "main")
    reference = next(j for j in jobs if j["role"] == "reference")
    push_to(
        alice,
        "project:p",
        job_record(main, "m1", {"throughput_tok_s": "30"}),
        job_record(reference, "r1", {"throughput_tok_s": "20"}),
    ).raise_for_status()
    values = {"speedup": "1.5", "speedup.n": "2", "output_match": "1", "flip_rate": "0.1"}
    SqliteStore(tmp_path).replace_comparisons("rec_m1", "rec_r1", values)
    cells = spec_cells(alice.get("/api/projects/p/tables").json()[0])
    assert cells["row0"]["comparisons"] == {"speedup": "1.5", "speedup.n": "2", "output_match": "1"}
    assert cells["base0"]["comparisons"] == {}


def test_repeated_table_cells_average_the_speed_comparisons(server, tmp_path):
    from effeval.store.sqlite import SqliteStore

    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    shared = {"determinism": {"seed": 2}, "repeats": {"count": 2}}
    view = alice.post("/api/projects/p/tables", json=speed_table(shared)).json()
    jobs = speed_jobs(alice, view)
    mains = [j for j in jobs if j["role"] == "main"]
    assert len(mains) == 2
    reference = next(j for j in jobs if j["role"] == "reference")
    push_to(alice, "project:p", job_record(reference, "ref", {"throughput_tok_s": "20"})).raise_for_status()
    store = SqliteStore(tmp_path)
    for i, (job, speedup) in enumerate(zip(mains, ("1.0", "2.0"), strict=True)):
        push_to(alice, "project:p", job_record(job, f"m{i}", {"throughput_tok_s": "30"})).raise_for_status()
        store.replace_comparisons(f"rec_m{i}", "rec_ref", {"speedup": speedup})
    cell = spec_cells(alice.get("/api/projects/p/tables").json()[0])["row0"]
    assert cell["comparisons"]["speedup"] == "1.5"
    assert cell["comparison_spread"]["speedup"]["n"] == 2


def test_frozen_table_keeps_the_speed_comparisons(server, tmp_path):
    from effeval.store.sqlite import SqliteStore

    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    view = alice.post("/api/projects/p/tables", json=speed_table()).json()
    jobs = speed_jobs(alice, view)
    main = next(j for j in jobs if j["role"] == "main")
    reference = next(j for j in jobs if j["role"] == "reference")
    push_to(
        alice,
        "project:p",
        job_record(main, "m1", {"throughput_tok_s": "30"}),
        job_record(reference, "r1", {"throughput_tok_s": "20"}),
    ).raise_for_status()
    SqliteStore(tmp_path).replace_comparisons("rec_m1", "rec_r1", {"speedup": "1.5"})
    snap = alice.post(f"/api/projects/p/tables/{view['table_id']}/snapshots").json()
    assert "speedup" in snap["body"]["metrics"]
    alice.delete("/api/records/rec_m1").raise_for_status()
    again = alice.get(f"/api/projects/p/snapshots/{snap['snapshot_id']}").json()
    assert spec_cells(again["body"])["row0"]["comparisons"]["speedup"] == "1.5"


def test_table_rows_take_a_runtime_and_each_provider_gets_its_own_baseline(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    decoding = {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": 4}
    rows = [
        {"label": "hf", "method": {"name": "base"}, "decoding": decoding},
        {"label": "vllm", "method": {"name": "base"}, "decoding": decoding, "runtime": {"name": "vllm"}},
    ]
    view = alice.post("/api/projects/p/tables", json=speed_table(rows=rows))
    assert view.status_code == 200, view.text
    block = {r["key"]: r for r in view.json()["blocks"][0]["rows"]}
    assert (block["row0"]["runtime"], block["row1"]["runtime"]) == ("hf", "vllm")
    baselines = [r for r in block.values() if r["baseline"]]
    assert sorted(r["runtime"] for r in baselines) == ["hf", "vllm"]
    assert len({r["cells"]["speed/spec_bench"]["run_key"] for r in block.values()}) == 4


def test_a_provider_has_one_baseline_row_across_speed_and_accuracy_tasks(server):
    _app, _admin, as_user = server
    alice = as_user("alice")
    alice.post("/api/projects", json={"name": "p"}).raise_for_status()
    decoding = {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": 4}
    tasks = ["speed/spec_bench", "lm_eval/gsm8k"]
    hf_row = {"label": "hf", "method": {"name": "base"}, "decoding": decoding}
    vllm_row = {**hf_row, "label": "vllm", "runtime": {"name": "vllm"}}

    def baselines(rows):
        body = {**speed_table(rows=rows), "tasks": tasks}
        view = alice.post("/api/projects/p/tables", json=body)
        assert view.status_code == 200, view.text
        return [r for r in view.json()["blocks"][0]["rows"] if r["baseline"]]

    (only,) = baselines([hf_row])
    assert only["runtime"] == "hf" and set(only["cells"]) == set(tasks)
    both = baselines([hf_row, vllm_row])
    assert sorted(r["runtime"] for r in both) == ["hf", "vllm"]
    assert all(r["cells"]["speed/spec_bench"] for r in both)


def test_records_carry_the_draft_k_and_provider_outside_the_label(server, speed_comparatives):
    _app, _admin, as_user = server
    alice = as_user("alice")
    plan = alice.post("/api/render", json={"spec": SPECULATIVE_SPEC}).json()["plan"]
    main = next(j for j in plan["jobs"] if j["role"] == "main")
    reference = next(j for j in plan["jobs"] if j["role"] == "reference")
    records = [job_record(main, "s1", {"n": "1"}), job_record(reference, "s2", {"n": "2"})]
    assert push_to(alice, "private", *records).status_code == 200
    rows = {r["record_id"]: r for r in alice.get("/api/records?scope=mine").json()}
    assert rows["rec_s1"]["draft"] == "qwen3-0.6b"
    assert rows["rec_s1"]["decoding"] == {"k": 4, "verification": "greedy"}
    assert rows["rec_s1"]["runtime"] == "hf" and rows["rec_s1"]["label"] == "Base"
    assert (rows["rec_s2"]["draft"], rows["rec_s2"]["decoding"]) == (None, None)
    drafted = alice.get("/api/records?scope=mine&draft=qwen3-0.6b").json()
    assert [r["record_id"] for r in drafted] == ["rec_s1"]


def test_describe_lists_the_runtimes_and_whether_an_engine_python_exists(server, monkeypatch, tmp_path):
    _app, _admin, as_user = server
    alice = as_user("alice")
    monkeypatch.delenv("EFFEVAL_VLLM_PYTHON", raising=False)
    monkeypatch.setenv("EFFEVAL_SGLANG_PYTHON", str(tmp_path / "missing" / "python"))
    runtimes = alice.get("/api/describe").json()["runtimes"]
    assert [(n, r["label"]) for n, r in runtimes.items()] == [
        ("hf", "transformers"),
        ("vllm", "vLLM"),
        ("sglang", "SGLang"),
    ]
    assert [r["available"] for r in runtimes.values()] == [True, False, False]
    python = tmp_path / "python"
    python.write_text("")
    monkeypatch.setenv("EFFEVAL_VLLM_PYTHON", str(python))
    runtimes = alice.get("/api/describe").json()["runtimes"]
    assert runtimes["vllm"]["available"] is True and runtimes["vllm"]["library"].startswith("vllm==")
    assert runtimes["sglang"]["available"] is False


def test_describe_fields_list_their_model_roles(server):
    _app, _admin, as_user = server
    fields = as_user("alice").get("/api/describe").json()["fields"]
    assert fields["quantization"]["roles"] == [{"name": "target", "path": "model"}]
    assert fields["speculative_decoding"]["roles"] == [
        {"name": "target", "path": "model"},
        {"name": "draft", "path": "decoding.draft.model"},
    ]
