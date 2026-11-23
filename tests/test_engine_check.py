"""Node check of the engine runtimes: the engine's Python must be configured."""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.compute

pytest.importorskip("torch")
pytest.importorskip("transformers")

from effeval import check as node_check  # noqa: E402
from effeval.core.engines import ENGINES  # noqa: E402
from effeval.validate import validate  # noqa: E402
from test_spec_schema import (  # noqa: E402,F401  (_speed_comparatives is an autouse fixture)
    _catalog,
    _speed_comparatives,
    frozen,
    speed_spec,
)

ENV = ENGINES["vllm"].python_env


@pytest.fixture(autouse=True)
def _no_role_checks(monkeypatch):
    monkeypatch.setattr(node_check, "_role_issues", lambda *args: [])


def unavailable(runtime, catalog=None):
    catalog = catalog or _catalog()
    plan, _ = frozen(speed_spec(runtime=runtime), catalog)
    return [i for i in node_check.check(plan, catalog) if i.code == "runtime.engine_unavailable"]


def test_an_unset_engine_python_is_reported_naming_the_variable(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    issues = unavailable({"name": "vllm"})
    assert len(issues) == 1 and ENV in issues[0].hint


def test_an_engine_python_that_is_not_a_file_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV, str(tmp_path / "missing-python"))
    assert len(unavailable({"name": "vllm"})) == 1


def test_an_existing_engine_python_is_accepted(monkeypatch):
    monkeypatch.setenv(ENV, sys.executable)
    assert unavailable({"name": "vllm"}) == []


def test_an_hf_plan_needs_no_engine_python(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert unavailable({"name": "hf"}) == []


def capacity_message(runtime):
    issues = validate(speed_spec(runtime=runtime, generation={"max_new_tokens": 4097}), _catalog())
    return next(i.message for i in issues if i.code == "runtime.static_cache_capacity")


def test_the_capacity_message_names_the_engine_context_for_an_engine_and_the_static_cache_for_hf():
    assert "vllm engine's context" in capacity_message({"name": "vllm"})
    assert "static KV cache" not in capacity_message({"name": "vllm"})
    assert "static KV cache" in capacity_message({"name": "hf"})
