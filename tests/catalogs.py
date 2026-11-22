"""Catalogs for tests independent of the shipped task snapshot: builtin data plus fixture listings."""

import json
from pathlib import Path

from effeval.core.catalog import builtin_catalog
from effeval.core.listing import TaskListing

LISTING_DIR = Path(__file__).parent / "data" / "listings"
LISTINGS = [
    TaskListing.model_validate_json((LISTING_DIR / f"{name}.json").read_text(encoding="utf-8"))
    for name in ("lm_eval", "evalscope")
]


def raw_listing(name: str) -> dict:
    return json.loads((LISTING_DIR / f"{name}.json").read_text(encoding="utf-8"))


def _without_snapshot():
    catalog = builtin_catalog()
    for evaluator in [e.name for e in catalog.evaluators.values() if e.source == "listing"]:
        catalog = catalog.model_copy(
            update={
                "tasks": {n: t for n, t in catalog.tasks.items() if t.evaluator != evaluator},
                "evaluators": {n: e for n, e in catalog.evaluators.items() if n != evaluator},
            }
        )
    return catalog


CATALOG = _without_snapshot().with_task_listings(LISTINGS)
