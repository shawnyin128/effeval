import gzip
import hashlib
import sqlite3
import threading
import time

import pytest

from effeval.core.types import Sample
from effeval.store.sqlite import SqliteStore


@pytest.fixture
def store(tmp_path):
    return SqliteStore(tmp_path)


def test_concurrent_writes_from_many_threads_all_land(store):
    errors = []

    def write(n):
        try:
            for i in range(40):
                store.heartbeat(f"plan_{n}", f"job_{i}", "running", "")
        except Exception as exc:  # collected so the assertion shows it
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with sqlite3.connect(store.data_dir / "effeval.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM heartbeats").fetchone()[0] == 8 * 40


def test_reads_do_not_wait_for_an_open_write(store):
    store.create_user("alice", "hash", "member")
    inside, release = threading.Event(), threading.Event()

    def hold_write():
        with store._write():
            inside.set()
            release.wait(5)

    holder = threading.Thread(target=hold_write)
    holder.start()
    inside.wait(5)
    started = time.monotonic()
    user = store.user("alice")
    elapsed = time.monotonic() - started
    release.set()
    holder.join()
    assert user["username"] == "alice"
    assert elapsed < 1.0


def test_write_waits_for_another_process_and_sees_its_commit(store):
    store.create_user("bob", "hash", "member")
    store.create_project("p", "", "bob")
    other = sqlite3.connect(store.data_dir / "effeval.sqlite3", isolation_level=None)
    other.execute("BEGIN IMMEDIATE")
    other.execute("INSERT INTO claims VALUES ('p', 'rk_1', 'bob', 'now')")
    outcome = []

    def claim():
        try:
            store.claim("p", ["rk_1"], "alice")
            outcome.append("claimed")
        except Exception as exc:  # the outcome under test
            outcome.append(exc)

    worker = threading.Thread(target=claim)
    worker.start()
    time.sleep(0.3)
    other.execute("COMMIT")
    other.close()
    worker.join(5)
    assert len(outcome) == 1
    assert isinstance(outcome[0], ValueError) and "claimed by bob" in str(outcome[0])


def test_interrupted_artifact_write_leaves_nothing_visible(store, monkeypatch):
    payload = gzip.compress(Sample(id="s1", scores={"acc": 1.0}).model_dump_json().encode())
    uri = "server:artifacts/art_" + hashlib.sha256(payload).hexdigest()[:32] + ".jsonl.gz"

    def crash(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr("effeval.store.sqlite.os.replace", crash)
    with pytest.raises(OSError):
        store.save_artifact(payload, "alice")
    assert store.artifact_path(uri) is None
    monkeypatch.undo()
    assert store.save_artifact(payload, "alice") == uri
    assert store.artifact_path(uri) is not None


def test_writes_bump_the_change_counters_of_their_topics_only(store):
    store.create_user("alice", "hash", "member")
    store.create_token("alice", "cli", True, "2099-01-01T00:00:00+00:00")
    store.create_session("alice", "2099-01-01T00:00:00+00:00")
    assert store.change_versions() == {"users": 1}
    store.create_project("p", "", "alice")
    store.save_plan("plan_1", "alice", "{}")
    assert store.change_versions() == {"users": 1, "projects": 1, "runs": 1}
    with pytest.raises(ValueError):
        store.create_project("p", "", "alice")  # rolled back: counters unchanged
    assert store.change_versions()["projects"] == 1
    with pytest.raises(ValueError, match="unknown change topics"), store._write("nonsense"):
        pass


def test_changes_from_another_connection_are_visible(store, tmp_path):
    other = SqliteStore(tmp_path)  # what a second process (e.g. `effeval admin`) opens
    other.create_user("bob", "hash", "member")
    assert store.change_versions() == {"users": 1}


def test_task_listing_round_trip_replacement_and_summaries(store):
    from catalogs import LISTINGS

    lm_eval, evalscope = LISTINGS
    store.put_task_listing(lm_eval, "alice")
    store.put_task_listing(evalscope, "alice")
    assert store.task_listing("lm_eval") == lm_eval
    assert store.task_listing_digests() == {"lm_eval": lm_eval.digest(), "evalscope": evalscope.digest()}

    narrowed = lm_eval.model_copy(update={"tasks": [t for t in lm_eval.tasks if t.name == "gsm8k"]})
    store.put_task_listing(narrowed, "bob")
    assert store.task_listing("lm_eval") == narrowed
    assert store.task_listing_digests()["lm_eval"] == narrowed.digest() != lm_eval.digest()
    by_name = {row["evaluator"]: row for row in store.task_listing_summaries()}
    assert by_name["lm_eval"]["uploaded_by"] == "bob" and by_name["lm_eval"]["tasks"] == 1
    assert by_name["lm_eval"]["library_version"] == "0.4.13"
    assert by_name["evalscope"]["tasks"] == 3 and by_name["evalscope"]["created_at"]
    assert store.user_data("bob") == {"task_catalogs": 1}

    with pytest.raises(KeyError):
        store.task_listing("nope")
