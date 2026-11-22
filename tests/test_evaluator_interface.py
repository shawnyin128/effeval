import importlib.metadata

import pytest

from catalogs import CATALOG
from effeval.core import evaluator as ev


def point(name, value="fake_evaluator:FAKE"):
    return importlib.metadata.EntryPoint(name=name, value=value, group=ev.ENTRY_POINT_GROUP)


def announce(monkeypatch, *points):
    monkeypatch.setattr(ev.importlib.metadata, "entry_points", lambda group: list(points))


def test_load_returns_the_announced_object(monkeypatch):
    from fake_evaluator import FAKE

    announce(monkeypatch, point("fake"))
    assert ev.load("fake") is FAKE
    assert ev.load_all() == [FAKE]


def test_an_entry_point_name_must_equal_the_evaluator_name(monkeypatch):
    announce(monkeypatch, point("other"))
    with pytest.raises(ValueError, match="other"):
        ev.load("other")


def test_an_object_without_run_is_not_an_evaluator(monkeypatch):
    announce(monkeypatch, point("fake", "fake_evaluator:NOT_ONE"))
    import fake_evaluator

    monkeypatch.setattr(fake_evaluator, "NOT_ONE", type("N", (), {"name": "fake"})(), raising=False)
    with pytest.raises(TypeError):
        ev.load("fake")


def test_unknown_name_lists_the_known_ones(monkeypatch):
    announce(monkeypatch, point("fake"))
    with pytest.raises(LookupError, match="fake"):
        ev.load("nope")


def test_duplicate_names_are_rejected(monkeypatch):
    announce(monkeypatch, point("fake"), point("fake"))
    with pytest.raises(ValueError, match=r"twice|two"):
        ev.entry_points()


def test_library_issue(monkeypatch):
    monkeypatch.setattr(ev.importlib.metadata, "version", lambda name: "0.4.13")
    assert ev.library_issue("lm-eval==0.4.13") is None
    assert ev.library_issue(None) is None
    message = ev.library_issue("lm-eval==0.4.9")
    assert "0.4.13" in message and "0.4.9" in message

    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(ev.importlib.metadata, "version", missing)
    assert ev.library_issue("lm-eval==0.4.13") == "lm-eval is not installed"


def test_the_primary_metric_must_be_reported_and_extras_are_fine():
    task = CATALOG.tasks["lm_eval/arc_challenge"]
    ev.check_result_metrics(task, {"acc": "0.5", "acc_stderr": "0.1", "extra": "1"})
    with pytest.raises(ValueError, match="acc"):
        ev.check_result_metrics(task, {"acc_norm": "0.5"})


def test_builtin_entry_points_are_declared():
    assert set(ev.entry_points()) >= {"lm_eval", "evalscope", "native", "kernel"}
