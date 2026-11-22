"""Renaming and deleting projects: who may, what moves along, what stays."""

import pytest

pytest.importorskip("fastapi")
from test_server import push, record, server  # noqa: F401 (server is the fixture)


def setup_project(server):
    _app, _admin, as_user = server
    alice, bob = as_user("alice"), as_user("bob")
    alice.post("/api/projects", json={"name": "paper-a", "description": "first"}).raise_for_status()
    alice.post("/api/projects/paper-a/members", json={"username": "bob"}).raise_for_status()
    push(alice, record("job_1", "rk_a", {"accuracy": "0.5"})).raise_for_status()
    alice.post("/api/projects/paper-a/records", json={"record_ids": ["rec_job_1"]}).raise_for_status()
    alice.post("/api/projects/paper-a/claims", json={"run_keys": ["rk_b"]}).raise_for_status()
    task = next(iter(alice.get("/api/describe").json()["tasks"]))
    table = {"name": "t", "field": "quantization", "models": ["qwen3-0.6b"], "tasks": [task],
             "rows": [{"label": "rtn", "method": {"name": "rtn"}}]}
    alice.post("/api/projects/paper-a/tables", json=table).raise_for_status()
    return alice, bob


def test_rename_moves_members_records_and_claims_and_frees_the_old_name(server):
    alice, bob = setup_project(server)
    assert bob.patch("/api/projects/paper-a", json={"name": "paper-b"}).status_code == 403
    assert alice.patch("/api/projects/paper-a", json={"name": "Paper B"}).status_code == 400
    alice.post("/api/projects", json={"name": "taken"}).raise_for_status()
    assert alice.patch("/api/projects/paper-a", json={"name": "taken"}).status_code == 409

    renamed = alice.patch("/api/projects/paper-a", json={"name": "paper-b", "description": "second"}).json()
    assert renamed["name"] == "paper-b" and renamed["description"] == "second"
    assert renamed["members"] == ["alice", "bob"] and renamed["records"] == 1
    assert bob.get("/api/projects/paper-a").status_code == 404
    assert [r["record_id"] for r in bob.get("/api/records", params={"scope": "project:paper-b"}).json()] == [
        "rec_job_1"
    ]
    assert bob.get("/api/records/rec_job_1").json()["projects"] == ["paper-b"]
    assert sorted(p["name"] for p in alice.get("/api/projects").json()) == ["paper-b", "taken"]
    assert len(bob.get("/api/projects/paper-b/tables").json()) == 1


def test_delete_removes_the_project_but_keeps_the_records(server):
    alice, bob = setup_project(server)
    assert bob.delete("/api/projects/paper-a").status_code == 403
    alice.delete("/api/projects/paper-a").raise_for_status()
    assert alice.get("/api/projects/paper-a").status_code == 404
    assert bob.get("/api/projects").json() == []
    assert [r["record_id"] for r in alice.get("/api/records", params={"scope": "mine"}).json()] == ["rec_job_1"]
    assert alice.get("/api/records/rec_job_1").json()["projects"] == []
    alice.post("/api/projects", json={"name": "paper-a"}).raise_for_status()
    assert alice.get("/api/projects/paper-a").json()["records"] == 0
    assert alice.get("/api/projects/paper-a/tables").json() == []
