"""Server-sent events that tell open pages which kinds of data changed, so they refetch through the API."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable

from starlette.concurrency import run_in_threadpool

from effeval.store.sqlite import SqliteStore

POLL_S = 1.0
KEEPALIVE_S = 15.0


async def change_stream(
    store: SqliteStore,
    disconnected: Callable[[], Awaitable[bool]],
    poll_s: float = POLL_S,
    keepalive_s: float = KEEPALIVE_S,
) -> AsyncIterator[str]:
    """SSE messages `data: {"topics": [...]}` naming the topics whose counters moved since the last poll.

    Changes made before the stream started are not reported. A comment line is sent after `keepalive_s` of
    silence so proxies keep the connection open. Ends once `disconnected()` returns True.
    """
    seen = await run_in_threadpool(store.change_versions)
    quiet = 0.0
    while not await disconnected():
        await asyncio.sleep(poll_s)
        current = await run_in_threadpool(store.change_versions)
        moved = sorted(t for t, v in current.items() if seen.get(t) != v)
        seen = current
        if moved:
            quiet = 0.0
            yield f"data: {json.dumps({'topics': moved})}\n\n"
        else:
            quiet += poll_s
            if quiet >= keepalive_s:
                quiet = 0.0
                yield ": keepalive\n\n"
