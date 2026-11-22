"""service._listed converts a stored task listing once per digest and keeps only the latest digest."""

from effeval.api import service


class StubStore:
    def __init__(self, digests):
        self.digests = digests

    def task_listing_digests(self):
        return self.digests

    def task_listing(self, name):
        return f"listing:{name}"


def test_only_the_latest_digest_of_each_evaluator_stays_cached(monkeypatch):
    converted = []

    def convert(listing, metrics, categories):
        converted.append(listing)
        return (listing, {})

    monkeypatch.setattr(service, "task_defs_from_listing", convert)
    monkeypatch.setattr(service, "_LISTED", {})
    store = StubStore({"lm_eval": "d1", "evalscope": "e1"})

    service._listed(store)
    service._listed(store)
    assert sorted(converted) == ["listing:evalscope", "listing:lm_eval"]

    store.digests = {"lm_eval": "d2", "evalscope": "e1"}
    service._listed(store)
    assert len(converted) == 3
    assert len(service._LISTED) == 2

    store.digests = {"lm_eval": "d1", "evalscope": "e1"}
    service._listed(store)
    assert len(converted) == 4
