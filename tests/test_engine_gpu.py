"""Real-engine smoke tests: Qwen3-0.6B (target and draft) on one GPU through the engine worker and client.

Run with `-m gpu`, CUDA_VISIBLE_DEVICES set to one free GPU index, HF_HUB_OFFLINE=1 and the engine's
EFFEVAL_*_PYTHON variable naming its Python executable; a test is skipped when its variable is unset.
"""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from effeval.core.engines import ENGINES
from effeval.runtime.engines.client import EngineClient

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.compute,
    pytest.mark.skipif(os.name != "posix", reason="engine workers run on POSIX hosts"),
]

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
K = 4
CONTEXT = 4097
PROMPT = "Say hello in one short sentence."
FOLLOW_UP = "Now say it in French."
MAX_NEW = 64
IM_END = 151645
COUNTING_PROMPT = "Write the numbers from 1 to 1000 in words, separated by commas, without stopping."
STATIC_CACHE_TOKENS = 256
VARIANTS = [("ar", True), ("spec", True), ("spec", False)]


@dataclass
class Run:
    ready: dict
    log: str
    first: object
    second: object
    follow_up: object
    follow_up_prompt_len: int
    memory_before: int
    memory_after: int
    prompt_len: int
    stop_ids: set = field(default_factory=set)


def gpu_memory_used_mib() -> int:
    index = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", index],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip())


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


@pytest.fixture(scope="module")
def stop_ids(tokenizer):
    from transformers import GenerationConfig

    from effeval.runtime.hf import stop_token_ids

    return stop_token_ids(tokenizer, GenerationConfig.from_pretrained(MODEL, revision=REVISION))


def render(tokenizer, messages) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def engine_config(engine, mode, cuda_graphs, context_length, stop_ids, gpu_memory_fraction) -> dict:
    speculative = mode == "spec"
    return {
        "engine": engine,
        "library": ENGINES[engine].library,
        "model": MODEL,
        "revision": REVISION,
        "dtype": "bfloat16",
        "seed": 0,
        "context_length": context_length,
        "cuda_graphs": cuda_graphs,
        "gpu_memory_fraction": gpu_memory_fraction,
        "stop_ids": sorted(stop_ids),
        "draft": {"model": MODEL, "revision": REVISION} if speculative else None,
        "num_speculative_tokens": K if speculative else None,
    }


def run_engine(engine, mode, cuda_graphs, tokenizer, stop_ids, workdir: Path) -> Run:
    """Start one engine worker, exercise it, close it, and return everything the tests assert on."""
    env_var = ENGINES[engine].python_env
    python = os.environ.get(env_var)
    if not python:
        pytest.skip(f"{env_var} is not set")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("CUDA_VISIBLE_DEVICES must name one free GPU")
    config = engine_config(engine, mode, cuda_graphs, CONTEXT, stop_ids, 0.8)
    prompt = render(tokenizer, [{"role": "user", "content": PROMPT}])
    memory_before = gpu_memory_used_mib()
    client = EngineClient.start(python, config, workdir, expect_library=ENGINES[engine].library)
    try:
        first = client.generate(prompt, MAX_NEW)
        second = client.generate(prompt, MAX_NEW)
        reply_text = tokenizer.decode(first.output_ids, skip_special_tokens=True)
        follow_up_prompt = render(
            tokenizer,
            [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": reply_text},
                {"role": "user", "content": FOLLOW_UP},
            ],
        )
        follow_up = client.generate(follow_up_prompt, MAX_NEW)
        ready = client.ready
    finally:
        client.close()
    return Run(
        ready=ready,
        log=(workdir / "engine.log").read_text(errors="replace"),
        first=first,
        second=second,
        follow_up=follow_up,
        follow_up_prompt_len=len(follow_up_prompt),
        memory_before=memory_before,
        memory_after=gpu_memory_used_mib(),
        prompt_len=len(prompt),
        stop_ids=set(stop_ids),
    )


@pytest.fixture(scope="module")
def runs(request, tokenizer, stop_ids, tmp_path_factory):
    """Per engine, the three variants run one after another; each engine is closed before the next starts."""
    cache: dict = {}

    def get(engine, mode, cuda_graphs) -> Run:
        key = (engine, mode, cuda_graphs)
        if key not in cache:
            workdir = tmp_path_factory.mktemp(f"{engine}-{mode}-{'graphs' if cuda_graphs else 'eager'}")
            cache[key] = run_engine(engine, mode, cuda_graphs, tokenizer, stop_ids, workdir)
        return cache[key]

    return get


@pytest.fixture(params=["vllm", "sglang"])
def engine(request):
    return request.param


@pytest.fixture(params=VARIANTS, ids=lambda v: f"{v[0]}-{'graphs' if v[1] else 'eager'}")
def run(request, engine, runs) -> Run:
    mode, cuda_graphs = request.param
    return runs(engine, mode, cuda_graphs)


def test_ready_reports_the_pinned_library_and_a_resolved_config(engine, run):
    assert run.ready["library"].split("+")[0] == ENGINES[engine].library
    assert isinstance(run.ready["resolved"], str)
    assert len(run.ready["resolved"]) > 50
    assert run.ready["engine_args"]


def test_output_ends_at_the_end_of_turn_token_within_the_budget(run):
    ids = run.first.output_ids
    assert ids[-1] == IM_END
    assert len(ids) < MAX_NEW
    assert not (set(ids[:-1]) & run.stop_ids)


def test_repeated_prompt_gives_identical_output(run):
    assert run.second.output_ids == run.first.output_ids


def test_a_second_turn_extending_the_first_is_served_without_cache_reuse(run):
    assert run.follow_up_prompt_len > run.prompt_len
    assert run.follow_up.output_ids
    assert run.follow_up.output_ids[-1] == IM_END


def test_speculative_stats_are_consistent_and_ar_has_none(run, request):
    speculative = "spec" in request.node.callspec.id
    for reply in (run.first, run.second, run.follow_up):
        if not speculative:
            assert (reply.rounds, reply.proposed, reply.accepted) == (None, None, None)
            continue
        assert reply.rounds >= 1
        assert reply.proposed == K * reply.rounds  # far from the context limit every round drafts k tokens
        assert 0 <= reply.accepted <= reply.proposed


def test_seconds_are_positive_and_memory_is_released_on_close(run):
    for reply in (run.first, run.second, run.follow_up):
        assert reply.seconds > 0
    assert run.memory_after <= run.memory_before + 1024


def test_autoregressive_and_speculative_outputs_agree(engine, runs):
    ar = runs(engine, "ar", True)
    for graphs in (True, False):
        spec = runs(engine, "spec", graphs)
        assert spec.first.output_ids == ar.first.output_ids
        assert spec.follow_up.output_ids == ar.follow_up.output_ids


def test_cuda_graphs_false_reaches_the_engine(engine, runs):
    eager = runs(engine, "spec", False)
    graphs = runs(engine, "spec", True)
    if engine == "vllm":
        assert eager.ready["engine_args"]["enforce_eager"] is True
        assert graphs.ready["engine_args"]["enforce_eager"] is False
        assert "'cudagraph_mode': <CUDAGraphMode.NONE: 0>" in eager.ready["resolved"]
        assert "'mode': <CompilationMode.NONE: 0>" in eager.ready["resolved"]
        assert "'cudagraph_mode': <CUDAGraphMode.NONE: 0>" not in graphs.ready["resolved"]
        assert "Capturing CUDA graphs" not in eager.log
        assert "Capturing CUDA graphs" in graphs.log
    else:
        assert eager.ready["engine_args"]["disable_cuda_graph"] is True
        assert "Capturing batches" not in eager.log
        assert "Capturing batches" in graphs.log


@pytest.mark.parametrize("mode", ["ar", "spec"])
def test_a_generation_at_the_capacity_limit_gets_its_full_budget(engine, mode, tokenizer, stop_ids, tmp_path):
    """prompt + max_new_tokens - 1 == static_cache_tokens: the engine must return exactly max_new_tokens."""
    env_var = ENGINES[engine].python_env
    python = os.environ.get(env_var)
    if not python or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip(f"{env_var} and CUDA_VISIBLE_DEVICES must be set")
    prompt = render(tokenizer, [{"role": "user", "content": COUNTING_PROMPT}])
    config = engine_config(engine, mode, False, STATIC_CACHE_TOKENS + 1, stop_ids, 0.5)
    limit = STATIC_CACHE_TOKENS + 1 - len(prompt)
    client = EngineClient.start(python, config, tmp_path, expect_library=ENGINES[engine].library)
    try:
        replies = {budget: client.generate(prompt, budget) for budget in (limit, limit - 1, 20)}
    finally:
        client.close()
    for budget, reply in replies.items():
        assert len(reply.output_ids) == budget
        assert reply.output_ids[-1] not in stop_ids
        if mode == "spec":
            assert 1 + reply.accepted + reply.rounds == budget
            assert reply.accepted <= reply.raw_accepted
            assert reply.proposed <= K * reply.rounds
