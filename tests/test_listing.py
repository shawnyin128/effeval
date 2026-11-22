import json
from pathlib import Path

import pytest

from effeval.core.listing import TaskListing

DATA = Path(__file__).parent / "data" / "listings"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


def listing_with(**changes):
    raw = load("lm_eval")
    tasks = {t["name"]: t for t in raw["tasks"]}
    for name, update in changes.items():
        tasks[name] = {**tasks[name], **update}
    raw["tasks"] = list(tasks.values())
    return raw


@pytest.mark.parametrize("name", ["lm_eval", "evalscope"])
def test_fixture_listings_validate_and_digest_survives_a_round_trip(name):
    listing = TaskListing.model_validate(load(name))
    again = TaskListing.model_validate_json(listing.model_dump_json())
    assert again == listing
    assert again.digest() == listing.digest()
    assert len(listing.digest()) == 64


def test_digest_changes_with_content():
    base = TaskListing.model_validate(load("lm_eval"))
    other = TaskListing.model_validate(listing_with(gsm8k={"version": "4"}))
    assert base.digest() != other.digest()


def test_duplicate_names_are_rejected_with_the_task_name():
    raw = load("lm_eval")
    raw["tasks"].append(raw["tasks"][0])
    with pytest.raises(ValueError, match=r"mmlu.*twice"):
        TaskListing.model_validate(raw)


def test_whitespace_in_a_name_is_rejected():
    raw = listing_with()
    raw["tasks"][0]["name"] = "bad name"
    with pytest.raises(ValueError, match="bad name"):
        TaskListing.model_validate(raw)


def test_unknown_subtask_is_rejected_with_the_task_name():
    with pytest.raises(ValueError, match=r"gsm8k.*nope"):
        TaskListing.model_validate(listing_with(gsm8k={"subtasks": ["nope"]}))


def test_subtask_cycle_is_rejected():
    with pytest.raises(ValueError, match="cycle"):
        TaskListing.model_validate(listing_with(mmlu_anatomy={"subtasks": ["mmlu"]}))


def test_fewshot_allowed_must_contain_the_default():
    with pytest.raises(ValueError, match=r"gsm8k.*fewshot_allowed"):
        TaskListing.model_validate(listing_with(gsm8k={"fewshot_allowed": [0]}))


def test_dangling_stderr_of_is_rejected():
    raw = load("lm_eval")
    next(t for t in raw["tasks"] if t["name"] == "arc_challenge")["metrics"][1]["stderr_of"] = "nope"
    with pytest.raises(ValueError, match=r"arc_challenge.*nope"):
        TaskListing.model_validate(raw)


def test_mode_outside_the_capabilities_is_rejected():
    raw = load("evalscope")
    raw["tasks"][0]["mode"] = "loglikelihood"
    with pytest.raises(ValueError, match=r"mmlu.*capabilities"):
        TaskListing.model_validate(raw)
