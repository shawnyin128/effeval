"""Speed evaluator: sample format, prompt sets (built-in and user), run identity, backend interface,
comparatives and a CPU run on a tiny model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from catalogs import CATALOG
from effeval.core import promptsets
from effeval.core.catalog import builtin_catalog
from effeval.core.prompt_converters import spec_bench
from effeval.core.promptsets import ENV_VAR, parse_prompt_set
from effeval.core.spec import run_key
from effeval.core.task_search import search_tasks
from effeval.core.types import EngineHandle, Sample
from effeval.evaluators import speed
from effeval.evaluators.speed_backend import Generation
from effeval.metrics.quality import COMPARATIVE
from effeval.plan import freeze

SPEC_BENCH_ROWS = [
    {"question_id": 81, "category": "writing", "turns": ["alpha beta", "gamma delta"]},
    {"question_id": 82, "category": "writing", "turns": ["beta gamma", "alpha delta"]},
    {"question_id": 91, "category": "roleplay", "turns": ["delta alpha", "beta beta"]},
    {"question_id": 161, "category": "translation", "turns": ["alpha"]},
    {"question_id": 162, "category": "translation", "turns": ["beta"]},
    {"question_id": 241, "category": "summarization", "turns": ["gamma gamma alpha"]},
    {"question_id": 242, "category": "summarization", "turns": ["delta"]},
    {"question_id": 321, "category": "qa", "turns": ["alpha beta gamma"]},
    {"question_id": 322, "category": "qa", "turns": ["beta delta"]},
    {"question_id": 401, "category": "math_reasoning", "turns": ["alpha alpha"]},
    {"question_id": 402, "category": "math_reasoning", "turns": ["gamma"]},
    {"question_id": 481, "category": "rag", "turns": ["delta delta beta"]},
]
CATEGORIES = list(spec_bench.CATEGORIES)


def jsonl(rows) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


@pytest.fixture
def user_sets(tmp_path, monkeypatch):
    """An empty user prompt set directory named by the environment variable."""
    directory = tmp_path / "sets"
    directory.mkdir()
    monkeypatch.setenv(ENV_VAR, str(directory))
    return directory


def write_set(directory, name, rows):
    path = directory / f"{name}.jsonl"
    path.write_text(jsonl(rows), encoding="utf-8")
    return path


# Sample format

GOOD = [
    {"id": "a", "category": "qa", "turns": ["hi"]},
    {"id": "b", "category": "qa", "turns": ["hi", "again"], "system": "be brief", "max_new_tokens": 5},
]


def test_a_good_file_parses_with_optional_keys():
    prompts = parse_prompt_set(jsonl(GOOD) + "\n  \n", "good.jsonl")
    assert [p.id for p in prompts] == ["a", "b"]
    assert prompts[0].system is None and prompts[0].max_new_tokens is None
    assert prompts[1].turns == ("hi", "again") and prompts[1].system == "be brief"
    assert prompts[1].max_new_tokens == 5


def line(record) -> str:
    return json.dumps(record)


BAD = {
    "invalid json": ([line(GOOD[0]), "{not json"], 2, "invalid JSON"),
    "not an object": ([line(GOOD[0]), "[1]"], 2, "must be a JSON object"),
    "missing id": ([line({"category": "qa", "turns": ["x"]})], 1, "missing required keys ['id']"),
    "missing turns": ([line(GOOD[0]), line({"id": "z", "category": "qa"})], 2, "['turns']"),
    "unknown key": ([line(GOOD[0]), line({**GOOD[0], "id": "z", "extra": 1})], 2, "unknown keys ['extra']"),
    "empty id": ([line({**GOOD[0], "id": " "})], 1, "'id' must be a non-empty string"),
    "non-string category": ([line({**GOOD[0], "category": 3})], 1, "'category' must be"),
    "empty turns": ([line({**GOOD[0], "turns": []})], 1, "non-empty list"),
    "turns not a list": ([line({**GOOD[0], "turns": "hi"})], 1, "non-empty list"),
    "non-string turn": ([line({**GOOD[0], "turns": ["ok", 7]})], 1, "'turns[]' must be"),
    "empty system": ([line({**GOOD[0], "system": ""})], 1, "'system' must be"),
    "zero max_new_tokens": ([line({**GOOD[0], "max_new_tokens": 0})], 1, "'max_new_tokens' must be"),
    "bool max_new_tokens": ([line({**GOOD[0], "max_new_tokens": True})], 1, "'max_new_tokens' must be"),
    "string max_new_tokens": ([line({**GOOD[0], "max_new_tokens": "5"})], 1, "'max_new_tokens' must be"),
    "duplicate id": (
        [line(GOOD[0]), line(GOOD[1]), line(GOOD[0])],
        3,
        "duplicate id 'a', first used at bad.jsonl:1",
    ),
}


@pytest.mark.parametrize(("lines", "number", "message"), BAD.values(), ids=BAD.keys())
def test_each_bad_file_raises_with_the_line_number(lines, number, message):
    with pytest.raises(ValueError, match=rf"^bad\.jsonl:{number}: .*{re.escape(message)}"):
        parse_prompt_set("\n".join(lines), "bad.jsonl")


def test_an_empty_file_raises():
    with pytest.raises(ValueError, match="at least one sample"):
        parse_prompt_set("\n\n", "empty.jsonl")


# Spec-Bench manifest and converter


def test_the_converter_maps_categories_and_keeps_turns_and_order():
    records = spec_bench.convert(jsonl(SPEC_BENCH_ROWS).encode())
    assert [r["id"] for r in records][:5] == [
        "writing:81",
        "writing:82",
        "roleplay:91",
        "translation:161",
        "translation:162",
    ]
    expected = ["mt_bench"] * 3 + [c for c in CATEGORIES[1:5] for _ in range(2)] + ["rag"]
    assert [r["category"] for r in records] == expected
    assert records[0]["turns"] == ["alpha beta", "gamma delta"] and records[3]["turns"] == ["alpha"]
    assert set(records[0]) == {"id", "category", "turns"}
    with pytest.raises(ValueError, match="unknown category 'astronomy'"):
        spec_bench.convert(jsonl([{"question_id": 1, "category": "astronomy", "turns": ["x"]}]).encode())


def test_the_spec_bench_manifest_is_pinned_and_loads_through_the_converter(tmp_path, monkeypatch):
    manifest = promptsets.builtin_manifests()["spec_bench"]
    assert manifest.download.sha256 == "4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"
    assert manifest.download.commit == "fd2c1cd7d2201ef71db4c5f4e455008f017967bf"
    assert manifest.download.commit in manifest.download.url and manifest.converter == "spec_bench"
    assert (
        manifest.download.records_sha256 == "3f1c18c2248e415758c5c3c1280dcbe919ee893d3ac71fa38ce4e18d63664d4f"
    )

    source = tmp_path / "question.jsonl"
    raw = jsonl(SPEC_BENCH_ROWS).encode()
    source.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    records = converted_sha(raw)
    pinned = manifest.model_copy(
        update={
            "download": manifest.download.model_copy(
                update={"url": source.as_uri(), "sha256": sha, "records_sha256": records}
            )
        }
    )
    monkeypatch.setattr(promptsets, "builtin_manifests", lambda: {"spec_bench": pinned})
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    loaded = promptsets.load_prompt_set("spec_bench")
    assert len(loaded.prompts) == 12 and loaded.prompts[0].category == "mt_bench"
    assert loaded.info.version == f"1+sha256:{sha}+records:{records}" and loaded.source["sha256"] == sha
    assert loaded.sha256 == records
    cached = tmp_path / "cache" / "effeval" / "specbench" / "fd2c1cd7" / "question.jsonl"
    assert cached.read_bytes() == raw
    with pytest.raises(ValueError, match="sha256"):
        promptsets.fetch(tmp_path / "other.jsonl", source.as_uri(), "0" * 64)
    assert not (tmp_path / "other.jsonl").exists()


def converted_sha(raw: bytes) -> str:
    """sha256 of the canonical records `spec_bench.convert` makes of `raw`."""
    records = [promptsets.check_record(r, "x") for r in spec_bench.convert(raw)]
    return hashlib.sha256(promptsets._canonical_jsonl(records)).hexdigest()


def pinned_spec_bench(tmp_path, monkeypatch):
    """The Spec-Bench manifest pointed at a small local file, its converted records pinned as they are now."""
    raw = jsonl(SPEC_BENCH_ROWS).encode()
    source = tmp_path / "question.jsonl"
    source.write_bytes(raw)
    manifest = promptsets.builtin_manifests()["spec_bench"]
    download = manifest.download.model_copy(
        update={
            "url": source.as_uri(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "records_sha256": converted_sha(raw),
        }
    )
    monkeypatch.setattr(
        promptsets,
        "builtin_manifests",
        lambda: {"spec_bench": manifest.model_copy(update={"download": download})},
    )
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


def test_a_changed_converter_fails_the_load_with_a_clear_error(tmp_path, monkeypatch):
    pinned_spec_bench(tmp_path, monkeypatch)
    assert len(promptsets.load_prompt_set("spec_bench").prompts) == len(SPEC_BENCH_ROWS)

    original = spec_bench.convert

    def edited(raw):
        records = original(raw)
        records[0]["turns"] = [*records[0]["turns"][:-1], "a different last turn"]
        return records

    monkeypatch.setattr(spec_bench, "convert", edited)
    with pytest.raises(ValueError, match=r"converter produced records.*manifest pins.*converter changed"):
        promptsets.load_prompt_set("spec_bench")


def test_the_converted_records_hash_is_part_of_the_task_version(tmp_path, monkeypatch):
    manifest = promptsets.builtin_manifests()["spec_bench"]
    other = manifest.download.model_copy(update={"records_sha256": "0" * 64})
    changed = manifest.model_copy(update={"download": other})
    before = promptsets._builtin_info(manifest).version
    after = promptsets._builtin_info(changed).version
    assert before != after
    assert before.endswith(f"+records:{manifest.download.records_sha256}")


def test_an_inline_manifest_loads_its_shipped_file(monkeypatch):
    manifest = promptsets.PromptSetManifest(
        name="inline", version="3", description="d", samples=2, file="x.jsonl"
    )
    raw = jsonl(GOOD).encode()
    monkeypatch.setattr(promptsets, "builtin_manifests", lambda: {"inline": manifest})
    monkeypatch.setattr(promptsets, "_builtin_file", lambda m: raw)
    loaded = promptsets.load_prompt_set("inline")
    assert [p.id for p in loaded.prompts] == ["a", "b"]
    assert loaded.info.version == f"3+sha256:{hashlib.sha256(raw).hexdigest()}"


def test_a_manifest_needs_exactly_one_source_and_a_converter_only_with_a_download():
    base = {"name": "m", "version": "1", "description": "d", "samples": 1}
    with pytest.raises(ValueError, match="exactly one"):
        promptsets.PromptSetManifest(**base)
    with pytest.raises(ValueError, match="converter"):
        promptsets.PromptSetManifest(**base, file="f.jsonl", converter="spec_bench")


# User sets


def test_a_user_set_is_discovered_as_a_task_and_searchable(user_sets):
    write_set(user_sets, "mine", GOOD)
    catalog = builtin_catalog()
    task = catalog.tasks["speed/mine"]
    assert task.evaluator == "speed" and task.target == "mine" and task.reference
    assert {"limit", "categories", "max_new_tokens", "warmup"} <= set(task.params_schema["properties"])
    assert "speed/spec_bench" in catalog.tasks
    dumped = {n: t.model_dump(mode="json") for n, t in catalog.tasks.items()}
    assert "speed/mine" in search_tasks(dumped, "mine")


def test_a_missing_directory_means_no_user_sets_and_no_error(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "absent"))
    speed_tasks = {n for n in builtin_catalog().tasks if n.startswith("speed/")}
    assert speed_tasks == {"speed/spec_bench"}
    monkeypatch.delenv(ENV_VAR)
    assert "speed/spec_bench" in builtin_catalog().tasks


def test_a_user_set_cannot_reuse_a_built_in_name_or_an_invalid_name(user_sets):
    write_set(user_sets, "spec_bench", GOOD)
    with pytest.raises(ValueError, match="built-in"):
        builtin_catalog()
    (user_sets / "spec_bench.jsonl").unlink()
    write_set(user_sets, "has space", GOOD)
    with pytest.raises(ValueError, match="must match"):
        builtin_catalog()


def test_a_new_user_file_shows_up_without_a_restart(user_sets):
    assert "speed/later" not in builtin_catalog().tasks
    write_set(user_sets, "later", GOOD)
    assert "speed/later" in builtin_catalog().tasks


def planned(name, warmup=1, runtime=None):
    """The main and reference jobs of a speculative speed plan; `runtime` defaults to the dynamic cache."""
    raw = {
        "model": "qwen3-8b",
        "method": {"name": "base"},
        "field": "speculative_decoding",
        "tasks": [{"name": name, "params": {"max_new_tokens": 10, "warmup": warmup}}],
        "decoding": {"mode": "speculative", "draft": {"model": "qwen3-0.6b"}, "num_speculative_tokens": 3},
        "hardware": {"gpu": "H100-SXM"},
        "runtime": {"cuda_graphs": False} if runtime is None else runtime,
    }
    plan, issues = freeze(raw, builtin_catalog(), owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return next(j for j in plan.jobs if j.role == "main"), next(j for j in plan.jobs if j.role == "reference")


def test_editing_a_user_set_changes_the_run_key(user_sets):
    path = write_set(user_sets, "mine", GOOD)
    first_main, first_ref = planned("speed/mine")
    again_main, _ = planned("speed/mine")
    assert run_key(first_main.spec) == run_key(again_main.spec)

    path.write_text(jsonl([*GOOD, {"id": "c", "category": "qa", "turns": ["new"]}]), encoding="utf-8")
    edited_main, edited_ref = planned("speed/mine")
    assert run_key(edited_main.spec) != run_key(first_main.spec)
    assert run_key(edited_ref.spec) != run_key(first_ref.spec)
    assert edited_main.spec.task.version != first_main.spec.task.version


def test_a_set_edited_after_planning_is_refused_at_run_time(user_sets):
    path = write_set(user_sets, "mine", GOOD)
    _, ref = planned("speed/mine")
    path.write_text(jsonl(GOOD[:1]), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since the job was planned"):
        speed.evaluate(FakeBackend(), ref.spec, builtin_catalog().tasks["speed/mine"])


# Backend interface: a fake backend shows the evaluator does not depend on any decoding library


class FakeTokenizer:
    chat_template = "fake template"

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        assert add_generation_prompt and not tokenize
        self.rendered.append(([dict(m) for m in messages], enable_thinking))
        return " ".join(f"{m['role']}:{m['content']}" for m in messages)

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return SimpleNamespace(input_ids=[len(word) for word in text.split()])

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return "out" + "".join(str(i) for i in ids)


class FakeBackend:
    def __init__(self, stats=False, peak=None, decode_ms=40.0):
        self.tokenizer = FakeTokenizer()
        self.sampled = False
        self.calls = []
        self.stats = stats
        self.peak = peak
        self.decode_ms = decode_ms
        self.synchronized = 0
        self.resets = 0

    def describe(self):
        return {
            "decoder": "fake",
            "device": "nowhere",
            "timing": "reported by the backend",
            "temperature": None,
        }

    def generate(self, prompt_ids, max_new_tokens):
        self.calls.append((prompt_ids, max_new_tokens))
        if not self.stats:
            return Generation([1, 2, 3], seconds=0.5)
        return Generation(
            [1, 2, 3],
            seconds=0.5,
            ttft_ms=7.0,
            rounds=2,
            proposed=6,
            accepted=3,
            draft_ms=10.0,
            decode_ms=self.decode_ms,
        )

    def reset(self):
        self.resets += 1

    def synchronize(self):
        self.synchronized += 1

    def reset_peak_memory(self):
        pass

    def peak_memory_gb(self):
        return self.peak


FAKE_ROWS = [
    {"id": "a", "category": "qa", "turns": ["hi"], "system": "be brief"},
    {"id": "b", "category": "qa", "turns": ["one", "two"], "max_new_tokens": 7},
    {"id": "c", "category": "chat", "turns": ["yo"]},
]


@pytest.fixture
def fake_set(user_sets):
    write_set(user_sets, "fake", FAKE_ROWS)
    return planned("speed/fake")


def run_fake(fake_set, backend, role=1):
    task = builtin_catalog().tasks["speed/fake"]
    return speed.evaluate(backend, fake_set[role].spec, task)


def test_the_evaluator_runs_any_backend_and_omits_what_it_does_not_report(fake_set):
    backend = FakeBackend()
    result = run_fake(fake_set, backend)

    assert [s.id for s in result.samples] == ["a", "b", "c"]
    assert [s.category for s in result.samples] == ["qa", "qa", "chat"]
    assert [s.output_ids for s in result.samples] == [[[1, 2, 3]], [[1, 2, 3]] * 2, [[1, 2, 3]]]
    scores = result.samples[1].scores
    assert scores["new_tokens"] == 6 and scores["tokens_per_s"] == pytest.approx(6.0)
    assert scores["latency_ms"] == pytest.approx(1000.0)
    assert result.samples[0].prediction == ["out123"]
    assert set(result.metrics) == {"throughput_tok_s", "latency_ms"}
    assert all("ttft_ms" not in s.scores and "rounds" not in s.scores for s in result.samples)
    assert set(result.protocol["categories"]) == {"qa", "chat"}
    assert result.protocol["categories"]["qa"]["samples"] == 2
    assert result.protocol["decoder"] == "fake" and result.protocol["timing"] == "reported by the backend"
    assert result.protocol["prompt_set"]["origin"] == "user"
    assert len(result.protocol["prompt_set"]["sha256"]) == 64
    assert backend.synchronized > 0


def test_prompts_budgets_and_conversation_reach_the_backend(fake_set):
    backend = FakeBackend()
    run_fake(fake_set, backend)
    # warmup=1 repeats the first prompt; then a (1 turn), b (2 turns), c (1 turn)
    assert [budget for _, budget in backend.calls] == [10, 10, 7, 7, 10]
    assert backend.calls[0][0] == [9, 5, 7]  # word lengths of "system:be brief user:hi"
    rendered = backend.tokenizer.rendered
    assert [m["role"] for m in rendered[1][0]] == ["system", "user"]
    assert rendered[1][0][0]["content"] == "be brief"
    assert [m["role"] for m in rendered[3][0]] == ["user", "assistant", "user"]
    assert rendered[3][0][1]["content"] == "out123" and rendered[3][1] is False


def test_backend_reported_counters_become_the_speculative_metrics(fake_set):
    result = run_fake(fake_set, FakeBackend(stats=True, peak=1.5), role=0)
    metrics = result.metrics
    assert float(metrics["acceptance_rate"]) == pytest.approx(0.5)
    assert float(metrics["mean_accepted_tokens"]) == pytest.approx(2.5)
    assert float(metrics["draft_time_share"]) == pytest.approx(0.25)
    assert float(metrics["ttft_ms"]) == pytest.approx(7.0)
    assert float(metrics["peak_memory_gb"]) == pytest.approx(1.5)
    two_turn = result.samples[1].scores
    assert two_turn["rounds"] == 4 and two_turn["accepted"] == 6 and two_turn["ttft_ms"] == 7.0


def test_categories_and_limit_select_samples_per_category(fake_set):
    prompts = promptsets.load_prompt_set("fake").prompts
    assert [p.id for p in speed.select(prompts, None, None)] == ["a", "b", "c"]
    assert [p.id for p in speed.select(prompts, ["chat", "qa"], 1)] == ["c", "a"]
    assert [p.id for p in speed.select(prompts, ["qa"], 0.5)] == ["a"]
    with pytest.raises(ValueError, match=r"unknown categories \['math'\]"):
        speed.select(prompts, ["math"], None)


def test_an_engine_handle_is_a_clear_not_implemented_stub_naming_wp8(fake_set):
    handle = EngineHandle(client=None, tokenizer=None, runtime="vllm")
    with pytest.raises(NotImplementedError, match="WP8"):
        speed.make_backend(handle, fake_set[0].spec)
    with pytest.raises(NotImplementedError, match="vllm"):
        speed.EVALUATOR.run(handle, fake_set[0].spec, builtin_catalog().tasks["speed/fake"], None)


# Timing contract and the backend call protocol


class OrderedTokenizer(FakeTokenizer):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def apply_chat_template(self, messages, add_generation_prompt, enable_thinking, tokenize):
        self.events.append("render")
        return super().apply_chat_template(messages, add_generation_prompt, enable_thinking, tokenize)

    def __call__(self, text, add_special_tokens):
        self.events.append("tokenize")
        return super().__call__(text, add_special_tokens)

    def decode(self, ids, skip_special_tokens):
        self.events.append("decode")
        return super().decode(ids, skip_special_tokens)


class OrderedBackend(FakeBackend):
    """Logs every call into `events` and reports no time of its own, so the evaluator times each call."""

    def __init__(self, events, clock):
        super().__init__()
        self.events = events
        self.clock = clock
        self.tokenizer = OrderedTokenizer(events)

    def generate(self, prompt_ids, max_new_tokens):
        self.events.append("generate")
        self.clock.now += 2.0
        return Generation([1, 2, 3])

    def reset(self):
        self.events.append("reset")

    def synchronize(self):
        self.events.append("sync")

    def reset_peak_memory(self):
        self.events.append("reset_peak")


TIMED_TURN = ["render", "tokenize", "reset", "sync", "clock", "generate", "sync", "clock", "decode"]


def test_each_turn_is_timed_between_two_syncs_with_tokenization_and_reset_outside(fake_set, monkeypatch):
    events = []
    clock = SimpleNamespace(now=100.0)

    def perf_counter():
        events.append("clock")
        return clock.now

    monkeypatch.setattr(speed, "time", SimpleNamespace(perf_counter=perf_counter))
    result = run_fake(fake_set, OrderedBackend(events, clock))

    # warmup=1: the first prompt once; then reset and peak-memory restart; then a, b (two turns), c
    assert events == [*TIMED_TURN, "reset", "reset_peak", *TIMED_TURN * 4]
    assert [s.scores["latency_ms"] for s in result.samples] == [2000.0, 4000.0, 2000.0]
    assert [s.scores["tokens_per_s"] for s in result.samples] == [1.5, 1.5, 1.5]


def test_the_evaluator_resets_the_backend_before_every_generation_and_after_warmup(fake_set):
    backend = FakeBackend()
    run_fake(fake_set, backend)
    assert len(backend.calls) == 5  # warmup, a, b1, b2, c
    assert backend.resets == len(backend.calls) + 1


def test_a_backend_that_times_itself_is_not_timed_by_the_evaluator(fake_set):
    backend = FakeBackend()
    result = run_fake(fake_set, backend)
    assert result.samples[0].scores["latency_ms"] == pytest.approx(500.0)


# Spec-Bench warmup and conversation protocol


class PaddedTokenizer(FakeTokenizer):
    def decode(self, ids, skip_special_tokens):
        return " \n" + super().decode(ids, skip_special_tokens) + "\n "


def test_warmup_runs_every_turn_of_the_first_sample_the_given_number_of_times(user_sets):
    write_set(
        user_sets,
        "two",
        [
            {"id": "x", "category": "qa", "turns": ["one", "two"]},
            {"id": "y", "category": "qa", "turns": ["solo"]},
        ],
    )
    _, ref = planned("speed/two", warmup=3)
    backend = FakeBackend()
    backend.tokenizer = PaddedTokenizer()
    speed.evaluate(backend, ref.spec, builtin_catalog().tasks["speed/two"])

    assert len(backend.calls) == 3 * 2 + 3  # 3 warmup rounds of x's two turns, then x (2 turns) and y
    rendered = backend.tokenizer.rendered
    for start in (0, 2, 4):  # each warmup round starts a fresh conversation with x
        assert [m["role"] for m in rendered[start][0]] == ["user"]
        assert [m["role"] for m in rendered[start + 1][0]] == ["user", "assistant", "user"]
    assert [m["content"] for m in rendered[1][0]] == ["one", "out123", "two"]  # turn 1 stripped for turn 2


def test_the_previous_answer_is_stripped_in_the_next_prompt_and_in_the_prediction(fake_set):
    backend = FakeBackend()
    backend.tokenizer = PaddedTokenizer()
    result = run_fake(fake_set, backend)
    second = next(r for r in backend.tokenizer.rendered if len(r[0]) == 3)
    assert second[0][1] == {"role": "assistant", "content": "out123"}
    assert result.samples[1].prediction == ["out123", "out123"]


# Metrics that need a backend's own numbers


class VaryingBackend(FakeBackend):
    """Reports a time to first token that grows with every call: 10, 20, 30 ms and so on."""

    def generate(self, prompt_ids, max_new_tokens):
        generation = super().generate(prompt_ids, max_new_tokens)
        return replace(generation, ttft_ms=10.0 * len(self.calls))


def test_ttft_is_the_first_turns_and_latency_the_total_over_turns(fake_set):
    result = run_fake(fake_set, VaryingBackend(stats=True))
    two_turn = result.samples[1].scores  # calls 3 and 4 (the first is the warmup, then a)
    assert two_turn["ttft_ms"] == 30.0
    assert two_turn["latency_ms"] == pytest.approx(1000.0)  # 0.5 s per turn, two turns
    assert CATALOG.metrics["ttft_ms"].description.endswith("that of the first turn.")
    assert "summed over its turns" in CATALOG.metrics["latency_ms"].description


def test_the_draft_time_share_is_left_out_when_no_decoding_time_was_reported(fake_set):
    result = run_fake(fake_set, FakeBackend(stats=True, decode_ms=0.0), role=0)
    assert "draft_time_share" not in result.metrics
    assert "acceptance_rate" in result.metrics


def test_the_spec_bench_deviations_are_recorded_in_the_protocol_of_spec_bench_runs_only(fake_set):
    result = run_fake(fake_set, FakeBackend())
    assert "deviations_from_spec_bench" not in result.protocol
    assert any("template" in d for d in speed.SPEC_BENCH_DEVIATIONS)
    assert {"dtype", "determinism", "draft length", "output match"} <= {
        d.split(":")[0] for d in speed.SPEC_BENCH_DEVIATIONS
    }


def test_spec_bench_runs_carry_the_deviations_in_their_protocol(fake_set, monkeypatch):
    loaded = promptsets.load_prompt_set("fake")
    renamed = replace(loaded, info=replace(loaded.info, name="spec_bench"))
    monkeypatch.setattr(speed, "load_prompt_set", lambda name: renamed)
    result = run_fake(fake_set, FakeBackend())
    assert result.protocol["deviations_from_spec_bench"] == list(speed.SPEC_BENCH_DEVIATIONS)


# The backend interface


def test_the_backend_interface_requires_reset():
    from effeval.evaluators.speed_backend import SpeedBackend

    doc = SpeedBackend.reset.__doc__
    assert "before each generation" in doc and "prefix caching" in doc and "vLLM" in doc


# Speed comparatives


def sample(id, tokens_per_s=None, ids=None, text=None, category=None):
    scores = {} if tokens_per_s is None else {"tokens_per_s": tokens_per_s}
    return Sample(id=id, prediction=text, scores=scores, output_ids=ids, category=category)


def compare(name, main, ref):
    return COMPARATIVE[name].compute({}, {}, main, ref)


def test_speedup_is_the_ratio_of_mean_per_sample_speeds_overall_and_per_category():
    ref = [
        sample("a", 10.0, category="mt"),
        sample("b", 20.0, category="mt"),
        sample("c", 100.0, category="rag"),
    ]
    main = [
        sample("a", 20.0, category="mt"),
        sample("b", 40.0, category="mt"),
        sample("c", 300.0, category="rag"),
    ]
    out = compare("speedup", main, ref)
    assert float(out["speedup"]) == pytest.approx(360 / 130)
    assert float(out["speedup.mt"]) == pytest.approx(2.0)
    assert float(out["speedup.rag"]) == pytest.approx(3.0)
    assert out["speedup.n"] == "3"
    assert "speedup.qa" not in out


def timed(id, tokens, seconds):
    return Sample(
        id=id, scores={"tokens_per_s": tokens / seconds, "new_tokens": tokens, "latency_ms": seconds * 1000}
    )


def test_speedup_is_the_mean_of_per_sample_speeds_not_total_tokens_over_total_time():
    # Reference: 100 tokens in 1 s and 10 tokens in 10 s; main: 100 tokens in 0.5 s and 10 tokens in 1 s.
    # Mean of speeds: (200 + 10) / (100 + 1) = 2.079; total tokens over total time: (110 / 1.5) / (110 / 11)
    # = 7.33.
    ref = [timed("qa:a", 100, 1.0), timed("qa:b", 10, 10.0)]
    main = [timed("qa:a", 100, 0.5), timed("qa:b", 10, 1.0)]
    assert float(compare("speedup", main, ref)["speedup"]) == pytest.approx(210 / 101)


def test_speedup_pairs_by_id_and_is_empty_without_the_score():
    ref = [sample("qa:a", 10.0), sample("qa:b", 10.0)]
    main = [sample("qa:b", 30.0), sample("qa:a", 10.0)]
    assert float(compare("speedup", main, ref)["speedup"]) == pytest.approx(2.0)
    assert compare("speedup", [sample("qa:a")], [sample("qa:a")]) == {}


def test_output_match_counts_equal_token_lists_by_id_and_falls_back_to_text():
    ref = [sample("qa:a", ids=[[1, 2]]), sample("qa:b", ids=[[3]])]
    main = [sample("qa:b", ids=[[3, 4]]), sample("qa:a", ids=[[1, 2]])]
    out = compare("output_match", main, ref)
    assert (out["output_match"], out["output_match.n"]) == ("0.5", "2")
    text_ref = [sample("qa:a", text=["x"]), sample("qa:b", text=["y"])]
    text_main = [sample("qa:a", text=["x"]), sample("qa:b", text=["z"])]
    assert compare("output_match", text_main, text_ref)["output_match"] == "0.5"


def test_output_match_looks_at_every_turn():
    ref = [sample("qa:a", ids=[[1, 2], [3]]), sample("qa:b", ids=[[5], [6]])]
    second_turn_differs = [sample("qa:a", ids=[[1, 2], [4]]), sample("qa:b", ids=[[5], [6]])]
    out = compare("output_match", second_turn_differs, ref)
    assert (out["output_match"], out["output_match.n"]) == ("0.5", "2")
    text_ref = [sample("qa:a", text=["x", "y"])]
    assert compare("output_match", [sample("qa:a", text=["x", "z"])], text_ref)["output_match"] == "0.0"


def test_multi_turn_prompt_match_is_the_share_of_multi_turn_samples_with_the_same_earlier_answers():
    ref = [
        sample("a", text=["x", "answer"]),
        sample("b", text=["p", "q"]),
        sample("c", text=["u", "v", "w"]),
        sample("d", text=["solo"]),
    ]
    main = [
        sample("a", text=["x", "other last answer"]),  # same later prompt: only the last answer differs
        sample("b", text=["P", "q"]),  # turn 1 answer differs, so turn 2 was prompted differently
        sample("c", text=["u", "v2", "w"]),  # the second answer differs, so turn 3 was
        sample("d", text=["different"]),  # single turn: not counted
    ]
    out = compare("multi_turn_prompt_match", main, ref)
    assert (
        float(out["multi_turn_prompt_match"]) == pytest.approx(1 / 3)
        and out["multi_turn_prompt_match.n"] == "3"
    )
    assert compare("multi_turn_prompt_match", [sample("d", text=["x"])], [sample("d", text=["y"])]) == {}


def test_multi_turn_prompt_match_is_declared_for_the_field_and_has_a_plain_description():
    assert "multi_turn_prompt_match" in CATALOG.fields["speculative_decoding"].compare
    declared = CATALOG.metrics["multi_turn_prompt_match"]
    assert declared.description and "later-turn prompts" in declared.description


def test_the_speed_comparatives_are_registered_for_the_field():
    assert {"speedup", "output_match"} <= set(COMPARATIVE)
    assert {"speedup", "output_match"} <= set(CATALOG.fields["speculative_decoding"].compare)


def test_the_spec_bench_task_is_in_the_catalog_with_its_settings():
    task = CATALOG.tasks["speed/spec_bench"]
    assert task.evaluator == "speed" and task.reference
    expected = {"limit", "categories", "max_new_tokens", "do_sample", "temperature", "warmup"}
    assert expected <= set(task.params_schema["properties"])
    assert task.settings({}).max_new_tokens == 1024 and task.settings({}).enable_thinking is False
    assert task.version == (
        "1+sha256:4b6d33e79484f9841c487ee87d1cf6aa8c6066f61d5d482ff09e5a007fafdf04"
        "+records:3f1c18c2248e415758c5c3c1280dcbe919ee893d3ac71fa38ce4e18d63664d4f"
    )
    assert "speed" in CATALOG.evaluators


# Tiny model on CPU


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    from conftest import _randomize_norm_gains

    words = ["alpha", "beta", "gamma", "delta", "user", "assistant"]
    vocab = {"[UNK]": 0, "<eos>": 1, **{w: i + 2 for i, w in enumerate(words)}}
    for i in range(len(vocab), 97):
        vocab[f"w{i}"] = i
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="<eos>", chat_template=template
    )

    def build(scale: float):
        torch.manual_seed(0)
        config = transformers.Qwen3Config(
            vocab_size=97,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=256,
            tie_word_embeddings=False,
            initializer_range=0.5,
        )
        model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
        _randomize_norm_gains(model, 1)
        model = model.double()
        if scale == 0.0:
            return model
        import copy

        draft = copy.deepcopy(model)
        generator = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for p in draft.parameters():
                p += torch.randn(p.shape, generator=generator, dtype=p.dtype) * scale * p.std()
        return draft

    return tokenizer, build(0.0), build(0.05)


def speculative_handles(tiny, k=3):
    from effeval.core.types import InProcess
    from effeval.runtime.speculative import GreedyAcceptance, ModelDrafter, SpeculativeDecoder

    tokenizer, target, draft = tiny
    plain = InProcess(model=target, tokenizer=tokenizer)
    decoder = SpeculativeDecoder(target, ModelDrafter(draft), k, GreedyAcceptance())
    return plain, replace(plain, decoder=decoder)


@pytest.fixture
def tiny_set(user_sets):
    """The 12 Spec-Bench-shaped questions as a user set `tiny`, converted by the Spec-Bench converter."""
    write_set(user_sets, "tiny", spec_bench.convert(jsonl(SPEC_BENCH_ROWS).encode()))
    return planned("speed/tiny")


@pytest.mark.compute
def test_speed_on_tiny_model_cpu(tiny, tiny_set):
    main_job, ref_job = tiny_set
    task = builtin_catalog().tasks["speed/tiny"]
    plain, spec_handle = speculative_handles(tiny)

    ref = speed.EVALUATOR.run(plain, ref_job.spec, task, None)
    main = speed.EVALUATOR.run(spec_handle, main_job.spec, task, None)

    assert len(main.samples) == len(ref.samples) == len(SPEC_BENCH_ROWS)
    assert [s.id for s in main.samples] == [s.id for s in ref.samples]
    assert [len(s.output_ids) for s in main.samples[:3]] == [2, 2, 2] and len(main.samples[3].output_ids) == 1
    assert [s.output_ids for s in main.samples] == [s.output_ids for s in ref.samples]
    assert [s.prediction for s in main.samples] == [s.prediction for s in ref.samples]
    assert {"throughput_tok_s", "latency_ms", "ttft_ms"} <= set(ref.metrics)
    assert "acceptance_rate" not in ref.metrics and "peak_memory_gb" not in ref.metrics  # CPU, autoregressive
    assert {"acceptance_rate", "mean_accepted_tokens", "draft_time_share"} <= set(main.metrics)
    assert float(main.metrics["mean_accepted_tokens"]) >= 1.0
    assert 0.0 <= float(main.metrics["acceptance_rate"]) <= 1.0
    assert all(s.scores["new_tokens"] >= 1 and s.scores["tokens_per_s"] > 0 for s in main.samples)
    assert set(main.protocol["categories"]) == set(CATEGORIES)
    assert main.protocol["categories"]["mt_bench"]["samples"] == 3
    assert main.protocol["prompt_set"]["name"] == "tiny" and main.protocol["enable_thinking"] is False
    assert main.protocol["decoder"]["num_speculative_tokens"] == 3
    assert ref.protocol["decoder"] == "autoregressive"
    assert main.protocol["chat_template_sha256"] == ref.protocol["chat_template_sha256"]

    out = COMPARATIVE["output_match"].compute({}, {}, main.samples, ref.samples)
    assert out["output_match"] == "1.0"
    speedup = COMPARATIVE["speedup"].compute({}, {}, main.samples, ref.samples)
    assert float(speedup["speedup"]) > 0 and {f"speedup.{c}" for c in CATEGORIES} <= set(speedup)


@pytest.mark.compute
def test_the_second_turn_continues_the_conversation(tiny, tiny_set):
    _, ref_job = tiny_set
    task = builtin_catalog().tasks["speed/tiny"]
    plain, _ = speculative_handles(tiny)
    seen = []
    tokenizer = tiny[0]
    original = tokenizer.apply_chat_template

    def spy(messages, **kwargs):
        seen.append(([dict(m) for m in messages], kwargs["enable_thinking"]))
        return original(messages, **kwargs)

    only_first = ref_job.spec.task.model_copy(
        update={"limit": 1, "params": {"categories": ["mt_bench"], "warmup": 0}}
    )
    spec = ref_job.spec.model_copy(update={"task": only_first})
    tokenizer.apply_chat_template = spy
    try:
        result = speed.EVALUATOR.run(plain, spec, task, None)
    finally:
        del tokenizer.apply_chat_template

    first, second = seen
    assert [m["role"] for m in first[0]] == ["user"]
    assert [m["role"] for m in second[0]] == ["user", "assistant", "user"]
    assert second[0][1]["content"] == result.samples[0].prediction[0]
    assert not first[1] and not second[1]


# InProcessBackend: eos handling, decode time, devices


def in_process_backend(tiny, tiny_set, *, eos=None, config_eos=None, speculative=False, monkeypatch=None):
    """An InProcessBackend over the tiny models with the given tokenizer and generation_config eos."""
    from effeval.evaluators._speed_inprocess import InProcessBackend

    plain, spec_handle = speculative_handles(tiny)
    handle = spec_handle if speculative else plain
    handle = replace(handle, tokenizer=SimpleNamespace(eos_token_id=eos))
    monkeypatch.setattr(handle.model.generation_config, "eos_token_id", config_eos, raising=False)
    return InProcessBackend(handle, tiny_set[1].spec)


PROMPT_IDS = [5, 9, 3, 7, 2, 4]


def first_new_tokens(tiny, tiny_set, monkeypatch, n=24):
    return in_process_backend(tiny, tiny_set, monkeypatch=monkeypatch).generate(PROMPT_IDS, n).output_ids


def eos_candidates(tokens):
    """(index, token) with the token's first occurrence at 3 <= index < 12, and a token that is absent."""
    inside = [(i, t) for i, t in enumerate(tokens) if 3 <= i < 12 and tokens.index(t) == i]
    absent = next(t for t in range(97) if t not in tokens)
    return inside, absent


@pytest.mark.compute
@pytest.mark.parametrize("speculative", [False, True])
def test_generation_ends_at_the_tokenizers_eos(tiny, tiny_set, monkeypatch, speculative):
    full = first_new_tokens(tiny, tiny_set, monkeypatch)
    assert len(full) == 24
    inside, _ = eos_candidates(full)
    assert len(inside) >= 2
    for index, eos in inside:
        backend = in_process_backend(
            tiny, tiny_set, eos=eos, speculative=speculative, monkeypatch=monkeypatch
        )
        assert backend.generate(PROMPT_IDS, 24).output_ids == full[: index + 1]


@pytest.mark.compute
@pytest.mark.parametrize("speculative", [False, True])
def test_generation_ends_at_the_generation_configs_eos_even_without_a_tokenizer_eos(
    tiny, tiny_set, monkeypatch, speculative
):
    full = first_new_tokens(tiny, tiny_set, monkeypatch)
    inside, absent = eos_candidates(full)
    for index, eos in inside:
        for config_eos in (eos, [absent, eos]):
            backend = in_process_backend(
                tiny, tiny_set, config_eos=config_eos, speculative=speculative, monkeypatch=monkeypatch
            )
            assert backend.generate(PROMPT_IDS, 24).output_ids == full[: index + 1]


@pytest.mark.compute
def test_both_eos_sources_are_stop_tokens_together(tiny, tiny_set, monkeypatch):
    full = first_new_tokens(tiny, tiny_set, monkeypatch)
    inside, _ = eos_candidates(full)
    (early_index, early), (late_index, late) = inside[0], inside[-1]
    assert early_index < late_index
    backend = in_process_backend(tiny, tiny_set, eos=late, config_eos=early, monkeypatch=monkeypatch)
    assert backend.generate(PROMPT_IDS, 24).output_ids == full[: early_index + 1]
    backend = in_process_backend(tiny, tiny_set, eos=early, config_eos=late, monkeypatch=monkeypatch)
    assert backend.generate(PROMPT_IDS, 24).output_ids == full[: early_index + 1]


class StubDecoder:
    """A decoder that returns fixed statistics, to check how the backend turns them into a Generation."""

    models: list = []  # noqa: RUF012
    prepared: dict = {}  # noqa: RUF012

    def generate(self, prompt_ids, *, max_new_tokens, sampler, eos_ids, stop=None):
        from effeval.runtime.speculative.loop import DecodeResult, DecodeStats

        stats = DecodeStats(
            rounds=2,
            proposed=6,
            accepted=3,
            prefill_ms=10.0,
            draft_ms=20.0,
            verify_ms=30.0,
            first_token_ms=7.0,
        )
        return DecodeResult([5, 6], stats, "length")


@pytest.mark.compute
def test_decode_ms_covers_prefill_draft_and_verify_and_ttft_is_the_first_token_time(
    tiny, tiny_set, monkeypatch
):
    from effeval.core.types import InProcess
    from effeval.evaluators._speed_inprocess import InProcessBackend

    handle = InProcess(model=tiny[1], tokenizer=SimpleNamespace(eos_token_id=None), decoder=StubDecoder())
    monkeypatch.setattr(handle.model.generation_config, "eos_token_id", None, raising=False)
    generation = InProcessBackend(handle, tiny_set[0].spec).generate(PROMPT_IDS, 8)
    assert generation.decode_ms == 60.0 and generation.draft_ms == 20.0 and generation.ttft_ms == 7.0
    assert (generation.rounds, generation.proposed, generation.accepted) == (2, 6, 3)
    assert generation.seconds is None


@pytest.mark.compute
def test_a_real_speculative_generation_reports_consistent_times(tiny, tiny_set, monkeypatch):
    backend = in_process_backend(tiny, tiny_set, speculative=True, monkeypatch=monkeypatch)
    generation = backend.generate(PROMPT_IDS, 24)
    assert generation.rounds >= 1 and generation.proposed >= generation.accepted
    assert 0 < generation.ttft_ms < generation.decode_ms and 0 < generation.draft_ms < generation.decode_ms
    plain = in_process_backend(tiny, tiny_set, monkeypatch=monkeypatch).generate(PROMPT_IDS, 24)
    assert plain.rounds is None and plain.decode_ms is None and plain.ttft_ms > 0
    assert plain.output_ids == generation.output_ids


class FakeParameter:
    def __init__(self, device):
        self.device = device


class FakeModel:
    """The parts of a model the backend reads: parameter devices, `hf_device_map`, eos and the embedding."""

    def __init__(self, *devices, device_map=None):
        import torch

        self._params = [FakeParameter(torch.device(d)) for d in devices]
        self.hf_device_map = device_map
        self.generation_config = SimpleNamespace(eos_token_id=None)

    def parameters(self):
        return iter(self._params)

    def buffers(self):
        return iter(())

    def get_input_embeddings(self):
        return SimpleNamespace(weight=self._params[0])


class DeviceDecoder:
    prepared: dict = {}  # noqa: RUF012

    def __init__(self, *models):
        self.models = list(models)


@pytest.fixture
def cuda_calls(monkeypatch):
    """Records torch.cuda synchronize / reset_peak_memory_stats calls and serves per-device peak memory."""
    torch = pytest.importorskip("torch")
    calls = SimpleNamespace(synchronize=[], reset=[])
    peaks = {0: 1e9, 1: 2e9, 2: 4e9, 5: 8e9}
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.synchronize.append(device.index))
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: calls.reset.append(device.index)
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: peaks[device.index])
    return calls


def device_backend(tiny_set, target, draft):
    from effeval.core.types import InProcess
    from effeval.evaluators._speed_inprocess import InProcessBackend

    handle = InProcess(
        model=target, tokenizer=SimpleNamespace(eos_token_id=None), decoder=DeviceDecoder(target, draft)
    )
    return InProcessBackend(handle, tiny_set[0].spec)


@pytest.mark.compute
def test_synchronize_reset_and_peak_cover_every_cuda_device_of_target_and_draft(tiny_set, cuda_calls):
    target = FakeModel("cuda:0", "cpu", device_map={"embed": 0, "layers": 1, "tail": "cpu", "off": "disk"})
    draft = FakeModel("cuda:2")
    backend = device_backend(tiny_set, target, draft)

    backend.synchronize()
    backend.reset_peak_memory()
    assert cuda_calls.synchronize == [0, 1, 2]
    assert cuda_calls.reset == [0, 1, 2]
    assert backend.peak_memory_gb() == pytest.approx((1e9 + 2e9 + 4e9) / 1e9)


@pytest.mark.compute
def test_a_model_split_over_devices_is_covered_without_a_device_map(tiny_set, cuda_calls):
    backend = device_backend(tiny_set, FakeModel("cuda:0", "cuda:1", "cuda:0"), FakeModel("cpu"))
    backend.synchronize()
    assert cuda_calls.synchronize == [0, 1]
    assert backend.peak_memory_gb() == pytest.approx(3.0)


@pytest.mark.compute
def test_a_device_map_alone_is_enough_to_find_a_device(tiny_set, cuda_calls):
    backend = device_backend(tiny_set, FakeModel("cpu", device_map={"a": "cuda:2", "b": 0}), FakeModel("cpu"))
    backend.synchronize()
    assert cuda_calls.synchronize == [0, 2]


@pytest.mark.compute
def test_a_draft_on_its_own_device_counts_even_when_the_target_is_on_one_device(tiny_set, cuda_calls):
    backend = device_backend(tiny_set, FakeModel("cuda:0"), FakeModel("cuda:5"))
    backend.synchronize()
    assert cuda_calls.synchronize == [0, 5]
    assert backend.peak_memory_gb() == pytest.approx(9.0)


@pytest.mark.compute
def test_without_cuda_devices_nothing_is_synchronized_and_peak_memory_is_unknown(tiny_set, cuda_calls):
    backend = device_backend(tiny_set, FakeModel("cpu"), FakeModel("cpu"))
    backend.synchronize()
    backend.reset_peak_memory()
    assert cuda_calls.synchronize == [] and cuda_calls.reset == []
    assert backend.peak_memory_gb() is None


@pytest.mark.compute
def test_the_in_process_backend_reset_is_a_no_op_that_exists(tiny, tiny_set, monkeypatch):
    backend = in_process_backend(tiny, tiny_set, monkeypatch=monkeypatch)
    before = backend.generate(PROMPT_IDS, 8).output_ids
    assert backend.reset() is None
    assert backend.generate(PROMPT_IDS, 8).output_ids == before


# GPU: real synchronization and peak memory (skipped without CUDA)


@pytest.fixture
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch


def cuda_backend(torch, tiny, tiny_set, draft_device=None):
    """A speculative backend over copies of the tiny models: target on cuda:0, draft on `draft_device`."""
    import copy

    from effeval.core.types import InProcess
    from effeval.evaluators._speed_inprocess import InProcessBackend
    from effeval.runtime.speculative import GreedyAcceptance, ModelDrafter, SpeculativeDecoder

    tokenizer, target, draft = tiny
    target = copy.deepcopy(target).to("cuda:0")
    draft = copy.deepcopy(draft).to(draft_device or "cuda:0")
    decoder = SpeculativeDecoder(target, ModelDrafter(draft), 3, GreedyAcceptance())
    handle = InProcess(model=target, tokenizer=tokenizer, decoder=decoder)
    return InProcessBackend(handle, tiny_set[0].spec)


def busy(torch, device):
    """Queue enough matmuls on `device` that they are still running when the call returns."""
    a = torch.randn(4096, 4096, device=device)
    for _ in range(30):
        a = (a @ a) / 4096
    with torch.cuda.device(device):
        event = torch.cuda.Event()
        event.record()
    return event


@pytest.mark.gpu
@pytest.mark.compute
def test_gpu_peak_memory_follows_resets_and_synchronize_waits_for_pending_work(cuda, tiny, tiny_set):
    torch = cuda
    backend = cuda_backend(torch, tiny, tiny_set)
    backend.reset_peak_memory()
    block = torch.empty(64 * 1024 * 1024, dtype=torch.float32, device="cuda:0")  # 256 MB
    del block
    assert backend.peak_memory_gb() >= 0.25
    backend.reset_peak_memory()
    # The peak restarts at what target, draft and decode cache still hold on cuda:0 (counted once).
    assert backend.peak_memory_gb() == pytest.approx(torch.cuda.memory_allocated("cuda:0") / 1e9)

    pending = busy(torch, "cuda:0")
    backend.synchronize()
    assert pending.query()
    assert backend.generate(PROMPT_IDS, 8).output_ids


@pytest.mark.gpu
@pytest.mark.compute
def test_gpu_peak_memory_and_synchronize_cover_a_draft_on_another_device(cuda, tiny, tiny_set):
    torch = cuda
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    backend = cuda_backend(torch, tiny, tiny_set, draft_device="cuda:1")
    backend.reset_peak_memory()
    block = torch.empty(
        64 * 1024 * 1024, dtype=torch.float32, device="cuda:1"
    )  # 256 MB on the draft's device
    del block
    assert backend.peak_memory_gb() >= 0.25

    pending = busy(torch, "cuda:1")
    backend.synchronize()
    assert pending.query()


# Decode cache: one settings function for both arms, prepare outside the timed region


def static_eager(runtime):
    """Stand-in for `hf.decode_settings` that gives the static cache without graphs (runs on CPU)."""
    from effeval.runtime.speculative import StaticCacheSettings

    return StaticCacheSettings(64, False) if runtime.cuda_graphs else None


def arm_handles(tiny, main_job, k=3):
    """A plain target handle and a speculative one, both taking settings from `hf.decode_settings`."""
    from effeval.core.types import InProcess
    from effeval.runtime import hf
    from effeval.runtime.speculative import GreedyAcceptance, ModelDrafter, SpeculativeDecoder

    tokenizer, target, draft = tiny
    settings = hf.decode_settings(main_job.spec.runtime)
    drafter = ModelDrafter(draft, settings)
    decoder = SpeculativeDecoder(target, drafter, k, GreedyAcceptance(), settings=settings)
    plain = InProcess(model=target, tokenizer=tokenizer)
    return plain, replace(plain, decoder=decoder)


@pytest.mark.compute
def test_both_arms_report_the_same_target_decode_cache_and_the_draft_matches(tiny, tiny_set, monkeypatch):
    from effeval.runtime import hf

    monkeypatch.setattr(hf, "decode_settings", static_eager)
    main_job, ref_job = planned("speed/tiny", runtime={})
    task = builtin_catalog().tasks["speed/tiny"]
    plain, spec_handle = arm_handles(tiny, main_job)

    ref = speed.EVALUATOR.run(plain, ref_job.spec, task, None)
    main = speed.EVALUATOR.run(spec_handle, main_job.spec, task, None)

    timing = {"graphs", "capture_ms"}
    ref_target = {k: v for k, v in ref.protocol["decode_cache"]["target"].items() if k not in timing}
    main_target = {k: v for k, v in main.protocol["decode_cache"]["target"].items() if k not in timing}
    main_draft = {k: v for k, v in main.protocol["decode_cache"]["draft"].items() if k not in timing}
    assert ref_target["kind"] == "static" and ref_target["capacity"] == 64
    assert set(ref.protocol["decode_cache"]) == {"target"}
    assert ref_target == main_target == main_draft


@pytest.mark.compute
def test_both_arms_record_the_attention_function_they_measured_with(tiny, tiny_set, monkeypatch):
    import copy

    from effeval.core.types import InProcess
    from effeval.run import attention_used
    from effeval.runtime import hf
    from effeval.runtime.speculative.static_attention import SDPA_NAME

    monkeypatch.setattr(hf, "decode_settings", static_eager)
    main_job, ref_job = planned("speed/tiny", runtime={})
    task = builtin_catalog().tasks["speed/tiny"]

    def own_models():
        models = (tiny[0], copy.deepcopy(tiny[1]), copy.deepcopy(tiny[2]))
        for model in models[1:]:
            model.set_attn_implementation("sdpa")
        return models

    plain = InProcess(model=own_models()[1], tokenizer=tiny[0])
    _, spec_handle = arm_handles(own_models(), main_job)
    before_ref = plain.model.config._attn_implementation  # what run.py reads before the backend exists
    before_main = spec_handle.model.config._attn_implementation  # the decoder is built before run.py reads it
    assert before_ref == "sdpa"

    ref = speed.EVALUATOR.run(plain, ref_job.spec, task, None)
    main = speed.EVALUATOR.run(spec_handle, main_job.spec, task, None)

    assert attention_used(ref.protocol, before_ref) == attention_used(main.protocol, before_main) == SDPA_NAME
    assert attention_used({}, "sdpa") == "sdpa"
    assert attention_used({"decode_cache": {"target": {"kind": "dynamic"}}}, "sdpa") == "sdpa"


@pytest.mark.compute
def test_the_dynamic_cache_is_reported_for_both_roles(tiny, tiny_set):
    main_job, ref_job = planned("speed/tiny")
    plain, spec_handle = arm_handles(tiny, main_job)
    from effeval.evaluators._speed_inprocess import InProcessBackend

    assert InProcessBackend(plain, ref_job.spec).describe()["decode_cache"] == {"target": {"kind": "dynamic"}}
    assert InProcessBackend(spec_handle, main_job.spec).describe()["decode_cache"] == {
        "target": {"kind": "dynamic"},
        "draft": {"kind": "dynamic"},
    }


@pytest.mark.compute
@pytest.mark.parametrize("speculative", [False, True])
def test_prepare_runs_once_before_the_first_generate(tiny, tiny_set, monkeypatch, speculative):
    from effeval.runtime.speculative import AutoregressiveDecoder, SpeculativeDecoder

    calls = []
    decoder_class = SpeculativeDecoder if speculative else AutoregressiveDecoder
    for name in ("prepare", "generate"):
        original = getattr(decoder_class, name)

        def spy(self, *args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(decoder_class, name, spy)
    main_job, ref_job = planned("speed/tiny")
    task = builtin_catalog().tasks["speed/tiny"]
    plain, spec_handle = arm_handles(tiny, main_job)
    handle, job = (spec_handle, main_job) if speculative else (plain, ref_job)

    backend = speed.make_backend(handle, job.spec)
    assert calls == ["prepare"]

    speed.evaluate(backend, job.spec, task)
    assert calls.count("prepare") == 1 and calls[0] == "prepare"
    assert calls.count("generate") > 1 and set(calls[1:]) == {"generate"}


@pytest.mark.compute
def test_a_prepared_decoder_is_not_prepared_again(tiny, tiny_set, monkeypatch):
    from effeval.runtime.speculative import SpeculativeDecoder

    main_job, _ = planned("speed/tiny")
    _, spec_handle = arm_handles(tiny, main_job)
    spec_handle.decoder.prepare()
    monkeypatch.setattr(SpeculativeDecoder, "prepare", lambda self: pytest.fail("prepared twice"))
    speed.make_backend(spec_handle, main_job.spec)


@pytest.mark.compute
def test_cuda_graphs_on_a_cpu_model_fail_fast(tiny, tiny_set):
    _, ref_job = planned("speed/tiny", runtime={})
    assert ref_job.spec.runtime.cuda_graphs
    from effeval.core.types import InProcess

    tokenizer, target, _ = tiny
    with pytest.raises(RuntimeError, match="CUDA"):
        speed.make_backend(InProcess(model=target, tokenizer=tokenizer), ref_job.spec)
