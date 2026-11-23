import os
import signal
import sys
import time
from pathlib import Path

import pytest

from effeval.runtime.engines import client as engine_client
from effeval.runtime.engines.client import EngineClient, EngineError

pytestmark = pytest.mark.skipif(os.name != "posix", reason="engine workers run on POSIX hosts")

FAKE_WORKER = Path(__file__).with_name("fake_engine_worker.py")


@pytest.fixture(autouse=True)
def fast_fake(monkeypatch):
    monkeypatch.setattr(engine_client, "WORKER_PATH", FAKE_WORKER)
    monkeypatch.setattr(engine_client, "READY_TIMEOUT_S", 30.0)
    monkeypatch.setattr(engine_client, "REQUEST_TIMEOUT_S", 30.0)
    monkeypatch.setattr(engine_client, "EXIT_TIMEOUT_S", 5.0)
    monkeypatch.setattr(engine_client, "TERM_GRACE_S", 2.0)
    monkeypatch.setattr(engine_client, "REAP_TIMEOUT_S", 3.0)


STOP = 999


def config(fake="ok", draft=True, **extra):
    return {
        **extra,
        "fake": fake,
        "engine": "vllm",
        "library": "vllm==0.32.0",
        "stop_ids": [STOP],
        "draft": {"model": "d", "revision": "r"} if draft else None,
        "num_speculative_tokens": 4 if draft else None,
    }


def start(tmp_path, **kwargs):
    return EngineClient.start(sys.executable, config(**kwargs), tmp_path, expect_library="vllm==0.32.0")


def pids(tmp_path):
    rows = (tmp_path / "pids.txt").read_text().splitlines()
    return {row.split()[0]: int(row.split()[1]) for row in rows}


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_gone(*targets, timeout=10.0):
    deadline = time.monotonic() + timeout
    while any(alive(p) for p in targets) and time.monotonic() < deadline:
        time.sleep(0.05)
    return [p for p in targets if alive(p)]


def test_ready_generate_and_close(tmp_path):
    engine = start(tmp_path)
    assert engine.ready["library"] == "vllm==0.32.0+fake"
    assert (engine.ready["torch"], engine.ready["cuda"]) == ("0.0.0+fake", "0.0")
    reply = engine.generate([5, 6, 7], 10)
    assert reply.output_ids == [8, 8, STOP]
    assert reply.seconds == 0.25
    assert (reply.rounds, reply.proposed, reply.accepted) == (1, 4, 3)
    assert engine.generate([5], 2).output_ids == [6, 6]
    worker = pids(tmp_path)["worker"]
    engine.close()
    engine.close()
    assert wait_gone(worker) == []
    assert engine._proc.returncode == 0
    with pytest.raises(EngineError, match="closed"):
        engine.generate([1], 1)


def test_autoregressive_reply_has_no_stats(tmp_path):
    engine = start(tmp_path, draft=False)
    try:
        reply = engine.generate([1, 2], 5)
    finally:
        engine.close()
    assert (reply.rounds, reply.proposed, reply.accepted) == (None, None, None)


def test_a_wrong_library_version_is_rejected(tmp_path):
    with pytest.raises(EngineError) as info:
        start(tmp_path, fake="wrong_library")
    assert "0.31.0" in str(info.value) and "0.32.0" in str(info.value)
    assert wait_gone(pids(tmp_path)["worker"]) == []


def test_an_error_before_ready_carries_the_message_and_the_log_tail(tmp_path):
    with pytest.raises(EngineError) as info:
        start(tmp_path, fake="error_before_ready")
    assert "fake startup failure" in str(info.value)
    assert "fake engine worker started in mode error_before_ready" in str(info.value)


def test_a_late_log_line_after_an_error_before_ready_is_in_the_message_and_the_worker_is_reaped(tmp_path):
    with pytest.raises(EngineError) as info:
        start(tmp_path, fake="late_error_before_ready")
    assert "fake startup failure" in str(info.value)
    assert "late-log-marker" in str(info.value)
    assert wait_gone(pids(tmp_path)["worker"]) == []


def test_a_crash_during_a_request_reports_the_exit_code_and_the_log_tail(tmp_path):
    engine = start(tmp_path, fake="crash_on_request")
    try:
        with pytest.raises(EngineError) as info:
            engine.generate([1, 2, 3], 4)
    finally:
        engine.close()
    assert "boom-marker" in str(info.value)
    assert "code 7" in str(info.value)


def test_a_worker_that_is_never_ready_times_out_and_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_client, "READY_TIMEOUT_S", 1.0)
    with pytest.raises(EngineError, match="not ready"):
        start(tmp_path, fake="never_ready")
    assert wait_gone(pids(tmp_path)["worker"]) == []


def test_close_kills_the_whole_process_group_of_a_worker_that_ignores_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_client, "EXIT_TIMEOUT_S", 1.0)
    engine = start(tmp_path, fake="ignore_exit")
    ids = pids(tmp_path)
    assert alive(ids["grandchild"])
    engine.close()
    assert wait_gone(ids["worker"], ids["grandchild"]) == []


def test_a_reply_with_cache_hits_is_an_error(tmp_path):
    engine = start(tmp_path, fake="cache_hit")
    try:
        with pytest.raises(EngineError, match="prefix caching"):
            engine.generate([1, 2], 3)
    finally:
        engine.close()


def test_a_non_json_line_is_an_error_and_the_worker_is_killed(tmp_path):
    with pytest.raises(EngineError, match="unexpected line"):
        start(tmp_path, fake="garbage")
    assert wait_gone(pids(tmp_path)["worker"]) == []


@pytest.mark.parametrize(
    ("fake", "draft"),
    [("stats_always", False), ("stats_never", True)],
    ids=["ar-with-stats", "spec-without-stats"],
)
def test_stats_presence_must_match_the_draft_config(tmp_path, fake, draft):
    engine = start(tmp_path, fake=fake, draft=draft)
    try:
        with pytest.raises(EngineError, match="stats"):
            engine.generate([1, 2], 3)
    finally:
        engine.close()


def test_the_worker_does_not_inherit_cublas_workspace_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("EFFEVAL_TEST_MARKER", "1")
    engine = start(tmp_path)
    try:
        keys = engine.ready["env_keys"]
    finally:
        engine.close()
    assert "EFFEVAL_TEST_MARKER" in keys
    assert "CUBLAS_WORKSPACE_CONFIG" not in keys


def test_an_error_reply_is_an_engine_error_with_the_workers_message(tmp_path):
    engine = start(tmp_path, fake="error_on_request")
    try:
        with pytest.raises(EngineError, match="fake request failure"):
            engine.generate([1, 2], 3)
    finally:
        engine.close()


def test_a_reply_with_the_wrong_id_is_an_error_and_the_worker_is_killed(tmp_path):
    engine = start(tmp_path, fake="wrong_id")
    try:
        with pytest.raises(EngineError, match="wanted id 1"):
            engine.generate([1, 2], 3)
        assert wait_gone(pids(tmp_path)["worker"]) == []
    finally:
        engine.close()


def test_close_escalates_to_sigkill_for_a_group_that_ignores_sigterm(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_client, "EXIT_TIMEOUT_S", 1.0)
    monkeypatch.setattr(engine_client, "TERM_GRACE_S", 1.0)
    engine = start(tmp_path, fake="ignore_term")
    ids = pids(tmp_path)
    assert alive(ids["worker"]) and alive(ids["grandchild"])
    engine.close()
    assert wait_gone(ids["worker"], ids["grandchild"]) == []
    assert engine._proc.returncode == -signal.SIGKILL


def test_a_worker_that_hangs_during_a_request_times_out_and_is_killed(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_client, "REQUEST_TIMEOUT_S", 1.0)
    engine = start(tmp_path, fake="hang_on_request")
    try:
        with pytest.raises(EngineError, match="not answering"):
            engine.generate([1, 2], 3)
        assert wait_gone(pids(tmp_path)["worker"]) == []
    finally:
        engine.close()


def test_a_worker_that_survives_sigkill_is_reported(tmp_path, monkeypatch):
    engine = start(tmp_path)
    real_killpg = os.killpg
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: None)
    monkeypatch.setattr(engine_client, "TERM_GRACE_S", 0.1)
    monkeypatch.setattr(engine_client, "REAP_TIMEOUT_S", 0.3)
    try:
        with pytest.raises(EngineError, match="still running"):
            engine._kill_group()
    finally:
        real_killpg(engine._proc.pid, signal.SIGKILL)
        engine._proc.wait(timeout=10)
        engine._closed = True


def test_the_group_is_not_signalled_once_the_worker_is_reaped(tmp_path, monkeypatch):
    engine = start(tmp_path)
    engine.close()
    assert engine._proc.returncode is not None
    calls = []
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append((pgid, sig)))
    engine._kill_group()
    assert calls == []


def test_a_reply_shorter_than_the_budget_without_a_stop_id_is_an_error(tmp_path):
    engine = start(tmp_path, fake="truncate")
    try:
        with pytest.raises(EngineError, match="truncated"):
            engine.generate([1, 2], 5)
    finally:
        engine.close()


def scripted(tmp_path, output_ids, rounds, accepted, max_new):
    engine = start(
        tmp_path,
        fake="scripted",
        scripted={
            "output_ids": output_ids,
            "stats": {"rounds": rounds, "proposed": 4 * rounds, "accepted": accepted},
        },
    )
    try:
        return engine, engine.generate([1, 2], max_new)
    finally:
        engine.close()


def test_accepted_drafts_beyond_the_budget_are_subtracted_when_the_output_ends_by_length(tmp_path):
    engine, reply = scripted(tmp_path, list(range(10, 20)), rounds=3, accepted=8, max_new=10)
    assert reply.accepted == 6 and reply.raw_accepted == 8
    assert 1 + reply.accepted + reply.rounds == len(reply.output_ids)
    assert (reply.rounds, reply.proposed) == (3, 12)
    assert engine.raw_counters() == {
        "requests": 1,
        "rounds": 3,
        "proposed": 12,
        "accepted": 8,
        "discarded_accepted": 2,
    }


def test_nothing_is_subtracted_when_the_output_ends_in_a_stop_id(tmp_path):
    engine, reply = scripted(tmp_path, [*range(10, 19), STOP], rounds=3, accepted=8, max_new=20)
    assert reply.accepted == 8 and reply.raw_accepted == 8
    assert engine.raw_counters()["discarded_accepted"] == 0


def test_normalization_never_increases_the_accepted_count(tmp_path):
    _, reply = scripted(tmp_path, list(range(10, 20)), rounds=2, accepted=3, max_new=10)
    assert reply.accepted == 3 and reply.raw_accepted == 3
