"""Task listings from the installed evaluators: upload to the server, or write as the shipped snapshot.

Loads evaluator entry points, so it runs on compute nodes and for maintainers, never on the server.
"""

from __future__ import annotations

import json
from pathlib import Path

from effeval.core.evaluator import load_all
from effeval.core.listing import TaskListing
from effeval.store.http import HttpClient


def task_listings() -> list[TaskListing]:
    """The listing of every installed evaluator that lists its own tasks, sorted by evaluator name."""
    return [listing for evaluator in load_all() if (listing := evaluator.list_tasks()) is not None]


def sync_tasks(client: HttpClient) -> list[TaskListing]:
    """Upload every installed evaluator's listing to the server; returns what was uploaded."""
    listings = task_listings()
    for listing in listings:
        client.register_task_listing(listing)
    return listings


def write_snapshot(listings: list[TaskListing], out_dir: Path) -> list[Path]:
    """Write each listing as `<evaluator>.json` under `out_dir`; returns the files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for listing in listings:
        path = out_dir / f"{listing.evaluator.name}.json"
        text = json.dumps(listing.model_dump(mode="json"), indent=1, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")
        paths.append(path)
    return paths
