import asyncio
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from effeval.server.app import create_app
from effeval.server.events import change_stream
from effeval.store.sqlite import SqliteStore


def collect(store, act, *, polls, keepalive_s=60.0):
    """Messages of a change stream over `polls` polls; `act(store)` runs after the stream has started."""

    async def main():
        count = {"n": 0}

        async def disconnected():
            count["n"] += 1
            if count["n"] == 2:
                act(store)
            return count["n"] > polls

        return [m async for m in change_stream(store, disconnected, poll_s=0.01, keepalive_s=keepalive_s)]

    return asyncio.run(main())


def test_stream_names_the_changed_topics_and_carries_no_data(tmp_path):
    store = SqliteStore(tmp_path)
    store.create_user("alice", "hash", "member")  # before the stream: not reported
    messages = collect(store, lambda s: s.create_project("p", "secret description", "alice"), polls=3)
    assert len(messages) == 1
    assert messages[0].startswith("data: ") and messages[0].endswith("\n\n")
    assert json.loads(messages[0][len("data: ") :]) == {"topics": ["projects"]}


def test_stream_sends_keepalives_when_nothing_changes(tmp_path):
    messages = collect(SqliteStore(tmp_path), lambda s: None, polls=4, keepalive_s=0.02)
    assert messages and all(m == ": keepalive\n\n" for m in messages)


def test_events_need_a_login(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/events").status_code == 401
