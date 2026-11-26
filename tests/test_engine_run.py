"""Running speed plans on an engine runtime through the real run path, with the fake engine worker."""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.compute,
    pytest.mark.skipif(os.name != "posix", reason="engine workers run on POSIX hosts"),
]

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from effeval import run as run_module  # noqa: E402
from effeval.core import evaluator as evaluators  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.promptsets import ENV_VAR  # noqa: E402
from effeval.core.types import Sample  # noqa: E402
from effeval.evaluators import speed  # noqa: E402
from effeval.metrics.quality import COMPARATIVE  # noqa: E402
from effeval.ops.introspect import model_entry  # noqa: E402
from effeval.plan import freeze  # noqa: E402
from effeval.runtime.engines import client as engine_client  # noqa: E402
from effeval.runtime.engines import prepare as engine_prepare  # noqa: E402
from effeval.runtime.hf import stop_token_ids  # noqa: E402
from test_spec_run import save_tiny, tokenizer  # noqa: E402, F401

FAKE_WORKER = Path(__file__).with_name("fake_engine_worker.py")
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}assistant:{% endif %}"
)
PROMPTS = [
    {"id": "a", "category": "qa", "turns": ["hello there"]},
    {"id": "b", "category": "qa", "turns": ["one", "two"]},
]
EXTRA_STOP_ID = 7


@pytest.fixture(scope="module")
def model_dirs(tmp_path_factory, tokenizer):  # noqa: F811
    root = tmp_path_factory.mktemp("engine-run")
    target = save_tiny(root / "target", tokenizer, hidden=32, seed=0)
    draft = save_tiny(root / "draft", tokenizer, hidden=16, seed=1)
    for path in (target, draft):
        saved = transformers.AutoTokenizer.from_pretrained(path)
        saved.chat_template = CHAT_TEMPLATE
        saved.save_pretrained(path)
    eos = [tokenizer.eos_token_id, EXTRA_STOP_ID]
    transformers.GenerationConfig(eos_token_id=eos).save_pretrained(target)
    return {"target": target, "draft": draft}


@pytest.fixture
def engine_env(monkeypatch, tmp_path):
    """A user prompt set, the fake engine worker as the engine Python and a fake GPU name.

    Returns the dict that sets the fake worker's mode for the speculative and the autoregressive group.
    """
    sets = tmp_path / "sets"
    sets.mkdir()
    (sets / "tiny.jsonl").write_text("\n".join(json.dumps(p) for p in PROMPTS) + "\n", encoding="utf-8")
    monkeypatch.setenv(ENV_VAR, str(sets))
    monkeypatch.setenv("EFFEVAL_VLLM_PYTHON", sys.executable)
    monkeypatch.setattr(engine_client, "WORKER_PATH", FAKE_WORKER)
    monkeypatch.setattr(engine_client, "READY_TIMEOUT_S", 30.0)
    monkeypatch.setattr(engine_client, "REQUEST_TIMEOUT_S", 30.0)
    monkeypatch.setattr(engine_client, "EXIT_TIMEOUT_S", 5.0)
    monkeypatch.setattr(evaluators, "load", lambda name: speed.EVALUATOR)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Fake GPU")
    modes = {"speculative": "ok", "autoregressive": "ok"}
    real = engine_prepare.worker_config

    def with_mode(spec, stop_ids):
        config = real(spec, stop_ids)
        key = "autoregressive" if config["draft"] is None else "speculative"
        return {**config, "fake": modes[key]}

    monkeypatch.setattr(engine_prepare, "worker_config", with_mode)
    return modes


def plan_for(model_dirs, runtime=None, **extra):
    target, _ = model_entry(model_dirs["target"], None)
    draft, _ = model_entry(model_dirs["draft"], None)
    catalog = builtin_catalog()
    models = {**catalog.models, target.code: target, draft.code: draft}
    catalog = catalog.model_copy(update={"models": models})
    raw = {
        "model": {"name": target.code, "dtype": "float32"},
        "method": {"name": "base"},
        "field": "speculative_decoding",
        "tasks": [{"name": "speed/tiny", "params": {"max_new_tokens": 10, "warmup": 1}}],
        "decoding": {
            "mode": "speculative",
            "draft": {"model": {"name": draft.code, "dtype": "float32"}},
            "num_speculative_tokens": 3,
        },
        "hardware": {"gpu": "H100-SXM"},
        "runtime": runtime or {"name": "vllm"},
        **extra,
    }
    plan, issues = freeze(raw, catalog, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return catalog, plan


def run(catalog, plan, tmp_path, **kwargs):
    return run_module.run_plan(
        plan, catalog, client=None, out_dir=tmp_path / "out", owner=None, log=lambda line: None, **kwargs
    )


def pids(tmp_path):
    files = sorted((tmp_path / "out").glob("*/engine-*/pids.txt"))
    return files, [int(line.split()[1]) for f in files for line in f.read_text().splitlines()]


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def assert_all_gone(pid_list):
    deadline = time.monotonic() + 10.0
    while any(alive(p) for p in pid_list) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert [p for p in pid_list if alive(p)] == []


def samples_of(tmp_path, record):
    path = tmp_path / "out" / record.artifacts["per_sample"].rsplit("/out/", 1)[1]
    return [Sample.model_validate_json(line) for line in gzip.decompress(path.read_bytes()).splitlines()]


def test_main_and_reference_run_through_the_engine_and_close_it(engine_env, model_dirs, tmp_path):
    catalog, plan = plan_for(model_dirs)
    outcome = run(catalog, plan, tmp_path)
    assert outcome.failed == []
    records = {job.role: run_module.saved_record(tmp_path / "out", job.job_id) for job in plan.jobs}
    main, reference = records["main"], records["reference"]
    assert main.protocol["engine"]["name"] == reference.protocol["engine"]["name"] == "vllm"
    in_process_stops = stop_token_ids(
        transformers.AutoTokenizer.from_pretrained(model_dirs["target"]),
        transformers.AutoModelForCausalLM.from_pretrained(model_dirs["target"]).generation_config,
    )
    assert EXTRA_STOP_ID in in_process_stops and len(in_process_stops) == 2
    assert set(main.protocol["engine"]["config"]["stop_ids"]) == in_process_stops
    assert main.protocol["engine"]["config"]["draft"] is not None
    assert reference.protocol["engine"]["config"]["draft"] is None
    assert main.protocol["device"] == "Fake GPU"
    assert (main.protocol["engine"]["torch"], main.protocol["engine"]["cuda"]) == ("0.0.0+fake", "0.0")
    assert "raw_counters" in main.protocol["engine"]["counters"]
    assert main.protocol["engine"]["raw_counters"] == {
        "requests": 4,
        "rounds": 4,
        "proposed": 12,
        "accepted": 8,
        "discarded_accepted": 0,
    }
    assert reference.protocol["engine"]["raw_counters"] == {
        "requests": 4,
        "rounds": 0,
        "proposed": 0,
        "accepted": 0,
        "discarded_accepted": 0,
    }
    assert float(main.metrics["acceptance_rate"]) == pytest.approx(2 / 3)
    assert "acceptance_rate" not in reference.metrics
    assert all("ttft_ms" not in r.metrics and "peak_memory_gb" not in r.metrics for r in records.values())
    speedup = COMPARATIVE["speedup"].compute(
        main.metrics, reference.metrics, samples_of(tmp_path, main), samples_of(tmp_path, reference)
    )
    assert float(speedup["speedup"]) == pytest.approx(1.0)
    files, pid_list = pids(tmp_path)
    assert len(files) == 2 and len(pid_list) == 2
    assert_all_gone(pid_list)


def test_a_worker_crash_fails_that_group_with_the_log_tail_and_the_other_group_still_runs(
    engine_env, model_dirs, tmp_path
):
    engine_env["speculative"] = "crash_on_request"
    catalog, plan = plan_for(model_dirs)
    outcome = run(catalog, plan, tmp_path, keep_going=True)
    main = next(job for job in plan.jobs if job.role == "main")
    reference = next(job for job in plan.jobs if job.role == "reference")
    assert len(outcome.failed) == 1
    assert main.job_id in outcome.failed[0] and "boom-marker" in outcome.failed[0]
    assert run_module.saved_record(tmp_path / "out", main.job_id) is None
    assert run_module.saved_record(tmp_path / "out", reference.job_id) is not None
    assert_all_gone(pids(tmp_path)[1])


def test_a_failure_with_a_live_worker_still_closes_it_when_swallowed(engine_env, model_dirs, tmp_path):
    engine_env["speculative"] = "cache_hit"
    catalog, plan = plan_for(model_dirs)
    outcome = run(catalog, plan, tmp_path, keep_going=True)
    assert len(outcome.failed) == 1 and "prefix caching" in outcome.failed[0]
    assert_all_gone(pids(tmp_path)[1])


def test_a_failure_that_propagates_still_closes_the_engine(engine_env, model_dirs, tmp_path):
    engine_env["speculative"] = "cache_hit"
    engine_env["autoregressive"] = "cache_hit"
    catalog, plan = plan_for(model_dirs)
    with pytest.raises(engine_client.EngineError, match="prefix caching"):
        run(catalog, plan, tmp_path, keep_going=False)
    files, pid_list = pids(tmp_path)
    assert len(files) == 1 and len(pid_list) == 1
    assert_all_gone(pid_list)


@pytest.fixture
def close_raises_permission_error(monkeypatch):
    """Make `EngineClient.close` shut the worker down as usual and then raise PermissionError."""
    real = engine_client.EngineClient.close

    def close(self):
        real(self)
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(engine_client.EngineClient, "close", close)


def test_a_failing_close_does_not_mask_the_error_that_ended_the_run(
    engine_env, close_raises_permission_error, model_dirs, tmp_path
):
    engine_env["speculative"] = "cache_hit"
    engine_env["autoregressive"] = "cache_hit"
    catalog, plan = plan_for(model_dirs)
    with pytest.raises(engine_client.EngineError, match="prefix caching"):
        run(catalog, plan, tmp_path, keep_going=False)


def test_a_failing_close_after_a_successful_group_is_raised(
    engine_env, close_raises_permission_error, model_dirs, tmp_path
):
    catalog, plan = plan_for(model_dirs)
    with pytest.raises(PermissionError):
        run(catalog, plan, tmp_path, keep_going=False)


def test_worker_config_follows_the_job_spec(engine_env, model_dirs):
    runtime = {"name": "vllm", "static_cache_tokens": 512, "cuda_graphs": False}
    _, plan = plan_for(model_dirs, runtime=runtime)
    main = next(job for job in plan.jobs if job.role == "main").spec
    reference = next(job for job in plan.jobs if job.role == "reference").spec
    config = engine_prepare.worker_config(main, frozenset({9, 3}))
    assert config["context_length"] == 513
    assert config["cuda_graphs"] is False
    assert config["stop_ids"] == [3, 9]
    assert config["num_speculative_tokens"] == 3
    assert config["draft"] == {
        "model": main.decoding.draft.model.checkpoint,
        "revision": main.decoding.draft.model.revision,
    }
    assert config["seed"] == main.determinism.seed
    assert config["sampling"] is None
    assert config["library"] == "vllm==0.32.0"
    ar = engine_prepare.worker_config(reference, frozenset({3}))
    assert ar["draft"] is None and ar["num_speculative_tokens"] is None
    assert ar["context_length"] == 513
    _, sampled_plan = plan_for(
        model_dirs,
        runtime=runtime,
        generation={"do_sample": True, "temperature": 0.8, "top_p": 0.9, "top_k": 5},
    )
    sampled = next(job for job in sampled_plan.jobs if job.role == "main").spec
    assert engine_prepare.worker_config(sampled, frozenset({3}))["sampling"] == {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 5,
    }


def test_sampled_engine_runs_start_one_engine_per_seed(engine_env, model_dirs, tmp_path):
    catalog, plan = plan_for(
        model_dirs, generation={"do_sample": True, "top_p": 0.9}, repeats={"seeds": [3, 5]}
    )
    assert len(plan.jobs) == 4
    outcome = run(catalog, plan, tmp_path)
    assert outcome.failed == []
    files, pid_list = pids(tmp_path)
    assert len(files) == 4 and len(pid_list) == 4
    assert len(list((tmp_path / "out").glob("*/engine-*"))) == 4
    for job in plan.jobs:
        record = run_module.saved_record(tmp_path / "out", job.job_id)
        assert record.protocol["engine"]["engine_seed"] == job.spec.determinism.seed
        assert record.protocol["engine"]["config"]["sampling"]["top_p"] == 0.9
        assert record.protocol["sampling"]["seed"] == job.spec.determinism.seed
    assert sorted(j.spec.determinism.seed for j in plan.jobs) == [3, 3, 5, 5]
    assert_all_gone(pid_list)
