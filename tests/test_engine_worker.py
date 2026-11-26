"""The engine worker: kwargs mapping, per-request statistics, timing, protocol, isolation and import rules."""

import ast
import io
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from effeval.runtime.engines import worker

SRC = Path(__file__).resolve().parents[1] / "src" / "effeval"
WORKER_FILE = SRC / "runtime" / "engines" / "worker.py"

STOP_IDS = [151643, 151645]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": None}
DRAFT = {"model": "Qwen/Qwen3-0.6B", "revision": "draft-sha"}


def make_config(engine, speculative, cuda_graphs=True, sampling=None):
    return {
        "engine": engine,
        "library": f"{engine}==1.0.0",
        "model": "Qwen/Qwen3-8B",
        "revision": "target-sha",
        "dtype": "bfloat16",
        "seed": 3,
        "context_length": 4097,
        "cuda_graphs": cuda_graphs,
        "gpu_memory_fraction": 0.8,
        "stop_ids": STOP_IDS,
        "draft": DRAFT if speculative else None,
        "num_speculative_tokens": 4 if speculative else None,
        "sampling": sampling,
    }


def test_vllm_kwargs_ar_and_speculative():
    base = {
        "model": "Qwen/Qwen3-8B",
        "revision": "target-sha",
        "dtype": "bfloat16",
        "seed": 3,
        "gpu_memory_utilization": 0.8,
        "max_model_len": 4097,
        "enforce_eager": False,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
    }
    assert worker.vllm_kwargs(make_config("vllm", False)) == base
    speculative = worker.vllm_kwargs(make_config("vllm", True, cuda_graphs=False))
    assert speculative == {
        **base,
        "enforce_eager": True,
        "speculative_config": {
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "revision": "draft-sha",
            "num_speculative_tokens": 4,
            "draft_sample_method": "greedy",
        },
    }
    sampled = worker.vllm_kwargs(make_config("vllm", True, sampling=SAMPLING))
    assert sampled["speculative_config"]["draft_sample_method"] == "probabilistic"
    assert {k: v for k, v in sampled.items() if k != "speculative_config"} == base


def test_sglang_kwargs_ar_and_speculative():
    base = {
        "model_path": "Qwen/Qwen3-8B",
        "revision": "target-sha",
        "dtype": "bfloat16",
        "random_seed": 3,
        "mem_fraction_static": 0.8,
        "context_length": 4097 + 6,
        "max_total_tokens": 4 * 4097,
        "disable_radix_cache": True,
        "disable_cuda_graph": False,
        "log_level": "info",
    }
    assert worker.sglang_kwargs(make_config("sglang", False)) == base
    speculative = worker.sglang_kwargs(make_config("sglang", True, cuda_graphs=False))
    assert speculative == {
        **base,
        "disable_cuda_graph": True,
        "speculative_algorithm": "STANDALONE",
        "speculative_draft_model_path": "Qwen/Qwen3-0.6B",
        "speculative_draft_model_revision": "draft-sha",
        "speculative_num_steps": 4,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 5,
    }


class Counter:
    def __init__(self, name, value, labels=None):
        self.name = name
        self.value = value
        self.labels = labels or {}


class FakeVllm:
    """A stand-in for the `vllm` module whose LLM serves scripted cumulative counter snapshots."""

    def __init__(self, snapshots, events=None, token_ids=(5, 6, 151645)):
        self.snapshots = list(snapshots)
        self.events = events if events is not None else []
        self.token_ids = token_ids
        self.sampling_calls = []
        self.generate_calls = []
        self.llm_kwargs = None
        outer = self

        class SamplingParams:
            def __init__(self, **kwargs):
                outer.sampling_calls.append(kwargs)

        class LLM:
            def __init__(self, **kwargs):
                outer.llm_kwargs = kwargs
                self.llm_engine = types.SimpleNamespace(vllm_config="CONFIG TEXT")

            def get_metrics(self):
                outer.events.append("metrics")
                return outer.snapshots.pop(0)

            def generate(self, prompts, params, use_tqdm):
                outer.events.append("generate")
                outer.generate_calls.append((prompts, params, use_tqdm))
                completion = types.SimpleNamespace(token_ids=tuple(outer.token_ids))
                return [types.SimpleNamespace(outputs=[completion])]

        self.SamplingParams = SamplingParams
        self.LLM = LLM


def snapshot(drafts, draft_tokens, accepted, hits, spec=True):
    metrics = [Counter("vllm:prefix_cache_hits", hits), Counter("vllm:prefix_cache_queries", 99)]
    if spec:
        metrics += [
            Counter("vllm:spec_decode_num_drafts", drafts),
            Counter("vllm:spec_decode_num_draft_tokens", draft_tokens),
            Counter("vllm:spec_decode_num_accepted_tokens", accepted),
        ]
    return metrics


def install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def test_vllm_counters_become_per_request_deltas(monkeypatch):
    split = [
        Counter("vllm:spec_decode_num_drafts", 1, {"engine": "0"}),
        Counter("vllm:spec_decode_num_drafts", 1, {"engine": "1"}),
    ]
    first_after = snapshot(0, 8, 7, 4)[:2] + split + snapshot(0, 8, 7, 4)[3:]
    fake = FakeVllm(
        [
            snapshot(0, 0, 0, 0),
            first_after,
            first_after,
            snapshot(5, 20, 15, 4),
        ]
    )
    install(monkeypatch, "vllm", fake)
    engine = worker.VllmEngine(make_config("vllm", True))
    first = engine.generate([1, 2], 16, lambda: 0.0)
    second = engine.generate([1, 2], 16, lambda: 0.0)
    assert first["stats"] == {"rounds": 2, "proposed": 8, "accepted": 7}
    assert second["stats"] == {"rounds": 3, "proposed": 12, "accepted": 8}
    assert first["cache_hits"] == 4
    assert second["cache_hits"] == 0

    ar_fake = FakeVllm([snapshot(0, 0, 0, 0, spec=False), snapshot(0, 0, 0, 0, spec=False)])
    install(monkeypatch, "vllm", ar_fake)
    ar = worker.VllmEngine(make_config("vllm", False)).generate([1], 4, lambda: 0.0)
    assert ar["stats"] is None

    broken = FakeVllm([snapshot(0, 0, 0, 0, spec=False), snapshot(0, 0, 0, 0, spec=False)])
    install(monkeypatch, "vllm", broken)
    with pytest.raises(RuntimeError, match="speculative counters"):
        worker.VllmEngine(make_config("vllm", True)).generate([1], 4, lambda: 0.0)


def test_vllm_sampling_params_and_prompt_shape(monkeypatch):
    fake = FakeVllm([snapshot(0, 0, 0, 0), snapshot(1, 4, 4, 0)], token_ids=(7, 8, 151645))
    install(monkeypatch, "vllm", fake)
    engine = worker.VllmEngine(make_config("vllm", True))
    reply = engine.generate([11, 12, 13], 64, lambda: 0.0)
    assert fake.sampling_calls == [
        {"max_tokens": 64, "temperature": 0.0, "stop_token_ids": STOP_IDS, "detokenize": False}
    ]
    prompts, params, use_tqdm = fake.generate_calls[0]
    assert prompts == [{"prompt_token_ids": [11, 12, 13]}]
    assert use_tqdm is False
    assert isinstance(params, fake.SamplingParams)
    assert reply["output_ids"] == [7, 8, 151645]
    assert fake.llm_kwargs == worker.vllm_kwargs(make_config("vllm", True))
    assert engine.resolved() == "CONFIG TEXT"

    sampled_fake = FakeVllm([snapshot(0, 0, 0, 0), snapshot(1, 4, 4, 0)])
    install(monkeypatch, "vllm", sampled_fake)
    sampled = worker.VllmEngine(make_config("vllm", True, sampling=SAMPLING))
    sampled.generate([11, 12, 13], 64, lambda: 0.0)
    assert sampled_fake.sampling_calls == [
        {
            "max_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "seed": 3,
            "stop_token_ids": STOP_IDS,
            "detokenize": False,
        }
    ]


class FakeSglang:
    def __init__(self, meta, events=None):
        self.meta = meta
        self.events = events if events is not None else []
        self.calls = []
        outer = self

        class Engine:
            def __init__(self, **kwargs):
                outer.kwargs = kwargs
                self.server_args = "SERVER ARGS"

            def generate(self, **kwargs):
                outer.events.append("generate")
                outer.calls.append(kwargs)
                return {"output_ids": [4, 5, 151645], "meta_info": dict(outer.meta)}

            def shutdown(self):
                outer.events.append("shutdown")

        self.Engine = Engine


def test_sglang_meta_info_mapping(monkeypatch):
    spec_meta = {
        "spec_verify_ct": 2,
        "spec_num_proposed_drafts": 8,
        "spec_num_correct_drafts": 6,
        "cached_tokens": 0,
    }
    fake = FakeSglang(spec_meta)
    install(monkeypatch, "sglang", fake)
    engine = worker.SglangEngine(make_config("sglang", True))
    reply = engine.generate([1, 2, 3], 32, lambda: 0.0)
    assert reply["stats"] == {"rounds": 2, "proposed": 8, "accepted": 6}
    assert reply["cache_hits"] == 0
    assert reply["output_ids"] == [4, 5, 151645]
    assert fake.calls == [
        {
            "input_ids": [1, 2, 3],
            "sampling_params": {"max_new_tokens": 32, "temperature": 0.0, "stop_token_ids": STOP_IDS},
        }
    ]
    assert engine.resolved() == "SERVER ARGS"

    sampled_fake = FakeSglang(spec_meta)
    install(monkeypatch, "sglang", sampled_fake)
    sampled = worker.SglangEngine(make_config("sglang", True, sampling={**SAMPLING, "top_k": 20}))
    sampled.generate([1, 2, 3], 32, lambda: 0.0)
    (call,) = sampled_fake.calls
    assert call["sampling_params"] == {
        "max_new_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "stop_token_ids": STOP_IDS,
    }
    assert "sampling_seed" not in call["sampling_params"]

    fake.meta = {"cached_tokens": 3}
    reply = engine.generate([1], 8, lambda: 0.0)
    assert reply["stats"] == {"rounds": 0, "proposed": 0, "accepted": 0}
    assert reply["cache_hits"] == 3

    ar_fake = FakeSglang({"cached_tokens": 0})
    install(monkeypatch, "sglang", ar_fake)
    reply = worker.SglangEngine(make_config("sglang", False)).generate([1], 8, lambda: 0.0)
    assert reply["stats"] is None


def test_timing_covers_only_the_engine_call(monkeypatch):
    events = []
    ticks = iter([10.0, 12.5])

    def clock():
        events.append("clock")
        return next(ticks)

    fake = FakeVllm([snapshot(0, 0, 0, 0), snapshot(1, 4, 4, 0)], events=events)
    install(monkeypatch, "vllm", fake)
    reply = worker.VllmEngine(make_config("vllm", True)).generate([1], 4, clock)
    assert events == ["metrics", "clock", "generate", "clock", "metrics"]
    assert reply["seconds"] == 2.5

    events.clear()
    ticks = iter([1.0, 4.0])
    sglang = FakeSglang({"cached_tokens": 0}, events=events)
    install(monkeypatch, "sglang", sglang)
    reply = worker.SglangEngine(make_config("sglang", False)).generate([1], 4, clock)
    assert events == ["clock", "generate", "clock"]
    assert reply["seconds"] == 3.0


class EchoEngine:
    def generate(self, prompt_ids, max_new_tokens, clock):
        return {
            "output_ids": prompt_ids[-1:] * max_new_tokens,
            "seconds": 0.5,
            "stats": None,
            "cache_hits": 0,
        }


def run_serve(lines):
    out = io.StringIO()
    code = worker.serve(EchoEngine(), io.StringIO("".join(line + "\n" for line in lines)), out)
    return code, [json.loads(line) for line in out.getvalue().splitlines()]


def test_serve_protocol():
    code, replies = run_serve(
        [
            json.dumps({"id": 7, "prompt_ids": [1, 2, 9], "max_new_tokens": 2}),
            json.dumps({"id": 8, "max_new_tokens": 2}),
            "not json",
            json.dumps({"id": 9, "prompt_ids": [4], "max_new_tokens": 1}),
            json.dumps({"cmd": "exit"}),
            json.dumps({"id": 10, "prompt_ids": [4], "max_new_tokens": 1}),
        ]
    )
    assert code == 0
    assert replies[0] == {"id": 7, "output_ids": [9, 9], "seconds": 0.5, "stats": None, "cache_hits": 0}
    assert replies[1]["id"] == 8 and "KeyError" in replies[1]["error"]
    assert replies[2]["id"] is None and "error" in replies[2]
    assert replies[3]["id"] == 9 and replies[3]["output_ids"] == [4]
    assert replies[4] == {"bye": True}
    assert len(replies) == 5

    code, replies = run_serve([json.dumps({"id": 1, "prompt_ids": [2], "max_new_tokens": 1})])
    assert code == 0 and len(replies) == 1


def test_library_mismatch():
    assert worker.library_mismatch("vllm==0.32.0", lambda dist: "0.32.0+cu129") is None
    message = worker.library_mismatch("vllm==0.32.0", lambda dist: "0.31.1")
    assert "0.31.1" in message and "0.32.0" in message
    assert worker.library_mismatch("sglang==0.5.12", lambda dist: "0.5.12") is None


def test_worker_starts_isolated_and_reports_a_missing_engine(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(make_config("vllm", False) | {"library": "vllm==0.32.0"}))
    result = subprocess.run(
        [sys.executable, "-I", str(WORKER_FILE), "--config", str(config)],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=60,
    )
    assert result.returncode != 0
    lines = result.stdout.splitlines()
    assert len(lines) == 1 and "error" in json.loads(lines[0])
    assert "effeval engine worker:" in result.stderr
    assert "effeval engine worker:" not in result.stdout


NATIVE_WRITER = r"""
import json, os, subprocess, sys, types
sys.path.insert(0, sys.argv[1])
cuda = types.SimpleNamespace(cuda="12.9")
sys.modules["torch"] = types.SimpleNamespace(__version__="2.9.1+cu129", version=cuda)
from effeval.runtime.engines import worker


class Noisy:
    engine_args = {}

    def __init__(self, config):
        os.write(1, b"native log from the engine\n")
        subprocess.run(["echo", "child log from the engine"], check=True)

    def resolved(self):
        return "noisy"

    def generate(self, prompt_ids, max_new_tokens, clock):
        return {"output_ids": [1], "seconds": 0.1, "stats": None, "cache_hits": 0}

    def close(self):
        pass


worker.ADAPTERS["fake"] = Noisy
worker.library_mismatch = lambda library: None
worker.importlib.metadata.version = lambda dist: "1.0.0"
sys.exit(worker.main(["--config", sys.argv[2]]))
"""


def test_native_and_child_writes_to_stdout_stay_off_the_protocol_channel(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(make_config("fake", False) | {"library": "fake==1.0.0"}))
    request = json.dumps({"id": 1, "prompt_ids": [2], "max_new_tokens": 1})
    result = subprocess.run(
        [sys.executable, "-I", "-c", NATIVE_WRITER, str(SRC.parent), str(config)],
        input=f"{request}\n{json.dumps({'cmd': 'exit'})}\n",
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert [sorted(r) for r in replies] == [
        ["cuda", "engine_args", "library", "ready", "resolved", "torch"],
        ["cache_hits", "id", "output_ids", "seconds", "stats"],
        ["bye"],
    ]
    assert (replies[0]["torch"], replies[0]["cuda"]) == ("2.9.1+cu129", "12.9")
    assert "native log from the engine" in result.stderr
    assert "child log from the engine" in result.stderr


CLOSE_TRACKER = r"""
import json, sys, types
sys.path.insert(0, sys.argv[1])
sys.modules["torch"] = types.SimpleNamespace(__version__="2.9.1", version=types.SimpleNamespace(cuda=None))
from effeval.runtime.engines import worker


class Tracked:
    engine_args = {}

    def __init__(self, config):
        self.marker = config["marker"]

    def resolved(self):
        if json.load(open(sys.argv[2]))["resolved_fails"]:
            raise RuntimeError("resolved failed")
        return "tracked"

    def generate(self, prompt_ids, max_new_tokens, clock):
        return {"output_ids": [1], "seconds": 0.1, "stats": None, "cache_hits": 0}

    def close(self):
        open(self.marker, "w").write("closed")


worker.ADAPTERS["fake"] = Tracked
worker.library_mismatch = lambda library: None
worker.importlib.metadata.version = lambda dist: "1.0.0"
sys.exit(worker.main(["--config", sys.argv[2]]))
"""


@pytest.mark.parametrize("resolved_fails", [False, True], ids=["serve-then-exit", "error-before-serving"])
def test_main_closes_the_engine_however_it_ends(tmp_path, resolved_fails):
    marker = tmp_path / "closed.txt"
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            make_config("fake", False)
            | {"library": "fake==1.0.0", "marker": str(marker), "resolved_fails": resolved_fails}
        )
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", CLOSE_TRACKER, str(SRC.parent), str(config)],
        input=json.dumps({"cmd": "exit"}) + chr(10),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (result.returncode != 0) == resolved_fails, result.stderr
    assert marker.read_text() == "closed"


def imports_of(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield (node.module or "").split(".")[0], node


def test_worker_imports_only_stdlib_and_its_engine():
    tree = ast.parse(WORKER_FILE.read_text(encoding="utf-8"))
    top_level = {
        name for node in tree.body if not isinstance(node, ast.ClassDef) for name, _ in imports_of(node)
    }
    assert top_level <= set(sys.stdlib_module_names)
    inside = {cls.name: {n for n, _ in imports_of(cls)} for cls in tree.body if isinstance(cls, ast.ClassDef)}
    assert inside["VllmEngine"] == {"vllm"}
    assert inside["SglangEngine"] == {"sglang"}
    assert all(not names for cls, names in inside.items() if cls not in ("VllmEngine", "SglangEngine"))
    everywhere = {name for name, _ in imports_of(tree)}
    assert everywhere & {"effeval", "torch", "transformers"} == set()

    offenders = []
    for path in SRC.rglob("*.py"):
        if path == WORKER_FILE:
            continue
        names = {name for name, _ in imports_of(ast.parse(path.read_text(encoding="utf-8")))}
        if names & {"vllm", "sglang"}:
            offenders.append(str(path.relative_to(SRC)))
    assert offenders == []
