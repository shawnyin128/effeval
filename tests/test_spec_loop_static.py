"""Decoders over the static cache: equality with the dynamic-cache loop, autoregressive decoding and HF."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from effeval.runtime.speculative import (  # noqa: E402
    AutoregressiveDecoder,
    GreedyAcceptance,
    ModelDrafter,
    SpeculativeDecoder,
    StaticCachedModel,
    StaticCacheSettings,
)
from test_spec_loop import (  # noqa: E402
    GREEDY,
    KS,
    NOISE_SCALES,
    first_occurrences,
    hf_assisted,
    noisy_copy,
    prompts,
    run_ar,
    run_spec,
    tiny_qwen3,
)

EAGER = StaticCacheSettings(capacity=128, cuda_graphs=False, min_bucket=16)
GRAPHS = StaticCacheSettings(capacity=512, cuda_graphs=True)

requires_gpu = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def gpu(fn):
    for mark in requires_gpu:
        fn = mark(fn)
    return fn


@pytest.fixture(scope="module")
def target():
    return tiny_qwen3(seed=0)


@pytest.fixture(scope="module")
def drafts(target):
    return {scale: noisy_copy(target, scale) for scale in NOISE_SCALES}


def spec_decoder(model, draft, k, settings, on_round=None):
    decoder = SpeculativeDecoder(
        model, ModelDrafter(draft, settings), k, GreedyAcceptance(), on_round=on_round, settings=settings
    )
    if settings is not None and settings.cuda_graphs:
        decoder.prepare()
    return decoder


def run_static_spec(model, draft, prompt, n, k, settings=EAGER, eos=frozenset(), stop=None, on_round=None):
    decoder = spec_decoder(model, draft, k, settings, on_round)
    return decoder.generate(prompt, max_new_tokens=n, sampler=GREEDY, eos_ids=eos, stop=stop)


def run_static_ar(model, prompt, n, settings=EAGER, eos=frozenset(), stop=None):
    decoder = AutoregressiveDecoder(model, settings)
    if settings.cuda_graphs:
        decoder.prepare()
    return decoder.generate(prompt, max_new_tokens=n, sampler=GREEDY, eos_ids=eos, stop=stop)


def assert_same_stats(got, expected):
    for name in ("rounds", "proposed", "accepted", "accept_lengths", "plain_steps", "new_tokens"):
        assert getattr(got.stats, name) == getattr(expected.stats, name), name
    assert got.finish == expected.finish


@pytest.mark.parametrize("scale", NOISE_SCALES)
@pytest.mark.parametrize("k", KS)
def test_static_speculative_equals_static_and_dynamic_autoregressive_and_hf(target, drafts, scale, k):
    for prompt in prompts(4):
        hf = target.generate(prompt, do_sample=False, max_new_tokens=40, pad_token_id=0)
        expected = hf[0, prompt.shape[1] :].tolist()
        assert run_ar(target, prompt, 40).output_ids == expected
        assert run_static_ar(target, prompt, 40).output_ids == expected
        got = run_static_spec(target, drafts[scale], prompt, 40, k)
        assert got.output_ids == expected
        assert got.finish == "length"
        assert_same_stats(got, run_spec(target, drafts[scale], prompt, 40, k))


@pytest.mark.parametrize("k", [2, 4])
def test_static_speculative_matches_hf_assisted_generation(target, drafts, k, monkeypatch):
    draft = drafts[0.5]
    for prompt in prompts(4):
        hf = hf_assisted(target, draft, prompt, k, 40, monkeypatch)
        ours = run_static_spec(target, draft, prompt, 40, k)
        assert ours.output_ids == hf[0, prompt.shape[1] :].tolist()


def test_eos_inside_a_block(target, drafts):
    prompt = prompts(1)[0]
    full = run_ar(target, prompt, 40).output_ids
    cases = first_occurrences(full, 2, 14)
    assert len(cases) >= 4
    for index, eos in cases:
        expected = run_ar(target, prompt, 40, eos=frozenset({eos}))
        assert expected.output_ids == full[: index + 1]
        assert run_static_ar(target, prompt, 40, eos=frozenset({eos})).output_ids == expected.output_ids
        for scale in (0.0, 0.5):
            got = run_static_spec(target, drafts[scale], prompt, 40, 5, eos=frozenset({eos}))
            assert got.output_ids == expected.output_ids
            assert got.finish == "eos"
            assert_same_stats(got, run_spec(target, drafts[scale], prompt, 40, 5, eos=frozenset({eos})))


def test_stop_criterion_inside_a_block(target, drafts):
    prompt = prompts(1)[0]
    length = prompt.shape[1]
    full = run_ar(target, prompt, 40).output_ids
    pairs = [
        (i, (full[i - 1], full[i]))
        for i in range(3, 14)
        if full[i - 1 : i + 1] not in [full[j - 1 : j + 1] for j in range(1, i)]
    ]
    assert len(pairs) >= 4
    for index, pair in pairs:

        def stop(seq, pair=pair):
            return seq.shape[1] - length >= 2 and tuple(seq[0, -2:].tolist()) == pair

        for scale in (0.0, 0.5):
            got = run_static_spec(target, drafts[scale], prompt, 40, 5, stop=stop)
            assert got.output_ids == full[: index + 1]
            assert got.finish == "stop"


@pytest.mark.parametrize("scale", [0.0, 0.5])
@pytest.mark.parametrize("budget", [1, 2, 7, 11])
def test_budgets_including_ones_that_are_not_multiples_of_k_plus_1(target, drafts, scale, budget):
    for prompt in prompts(3):
        got = run_static_spec(target, drafts[scale], prompt, budget, 4)
        assert len(got.output_ids) == budget
        assert got.output_ids == run_ar(target, prompt, budget).output_ids
        assert_same_stats(got, run_spec(target, drafts[scale], prompt, budget, 4))


def test_decoder_is_reusable_across_prompts(target, drafts):
    decoder = spec_decoder(target, drafts[0.5], 3, EAGER)
    for prompt in prompts(4):
        got = decoder.generate(prompt, max_new_tokens=20, sampler=GREEDY, eos_ids=frozenset())
        assert got.output_ids == run_ar(target, prompt, 20).output_ids


def test_on_round_counts_equal_the_dynamic_runs(target, drafts):
    seen = []
    for scale in (0.05, 0.5):
        for prompt in prompts(3):
            ours, theirs = [], []
            run_static_spec(target, drafts[scale], prompt, 40, 5, on_round=ours.append)
            run_spec(target, drafts[scale], prompt, 40, 5, on_round=theirs.append)
            assert ours == theirs
            seen.extend(ours)
    assert any(r["m"] < r["k"] for r in seen) and any(r["m"] == r["k"] for r in seen)


def test_generation_beyond_the_capacity_fails_before_any_forward(target, drafts):
    settings = StaticCacheSettings(capacity=40, cuda_graphs=False, min_bucket=16)
    calls = []
    handle = target.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="capacity is 40"):
            run_static_ar(target, prompts(1)[0], 40, settings=settings)
        with pytest.raises(ValueError, match="capacity is 40"):
            run_static_spec(target, drafts[0.5], prompts(1)[0], 36, 4, settings=settings)
        assert calls == []
        run_static_spec(target, drafts[0.5], prompts(1)[0], 20, 4, settings=settings)
        assert calls
    finally:
        handle.remove()


@pytest.mark.parametrize("arm", ["autoregressive", "speculative"])
def test_the_reserve_is_exact_prompt_plus_budget_minus_one_for_both_arms(target, drafts, arm):
    prompt, new = prompts(1)[0], 20
    length = prompt.shape[1]

    def run(settings):
        if arm == "autoregressive":
            return run_static_ar(target, prompt, new, settings=settings)
        return run_static_spec(target, drafts[0.5], prompt, new, 4, settings=settings)

    expected = run_ar(target, prompt, new).output_ids
    fits = StaticCacheSettings(capacity=length + new - 1, cuda_graphs=False, min_bucket=16)
    assert run(fits).output_ids == expected

    calls = []
    handles = [m.register_forward_pre_hook(lambda *_: calls.append(1)) for m in (target, drafts[0.5])]
    try:
        too_small = StaticCacheSettings(capacity=length + new - 2, cuda_graphs=False, min_bucket=16)
        message = rf"needs {length + new - 1} cache positions.*runtime\.static_cache_tokens"
        with pytest.raises(ValueError, match=message):
            run(too_small)
        assert calls == []
    finally:
        for handle in handles:
            handle.remove()


def test_prepare_describes_the_cache_of_each_model(target, drafts):
    dynamic = SpeculativeDecoder(target, ModelDrafter(drafts[0.5]), 3, GreedyAcceptance())
    assert dynamic.prepare() == {"target": {"kind": "dynamic"}, "draft": {"kind": "dynamic"}}
    assert dynamic.prepared == dynamic.prepare()
    assert AutoregressiveDecoder(target).prepare() == {"kind": "dynamic"}
    decoder = spec_decoder(target, drafts[0.5], 3, EAGER)
    prepared = decoder.prepare()
    assert prepared["target"]["kind"] == prepared["draft"]["kind"] == "static"
    assert prepared["target"]["capacity"] == 128 and prepared["target"]["graphs"] == 0
    assert prepared["target"]["cuda_graphs"] is False
    assert prepared["draft"]["buckets"] == [16, 32, 64, 128]
    assert isinstance(decoder.target, StaticCachedModel)
    assert isinstance(decoder.drafter.model, StaticCachedModel)
    assert decoder.target.settings is decoder.drafter.model.settings is EAGER


def test_static_decoders_need_no_prepare_without_graphs(target, drafts):
    prompt = prompts(1)[0]
    expected = run_ar(target, prompt, 20).output_ids
    decoder = SpeculativeDecoder(
        target, ModelDrafter(drafts[0.5], EAGER), 3, GreedyAcceptance(), settings=EAGER
    )
    assert decoder.prepared is None
    got = decoder.generate(prompt, max_new_tokens=20, sampler=GREEDY, eos_ids=frozenset())
    assert got.output_ids == expected


# -------------------------------------------------------------------------------------- GPU (graphs)
@pytest.fixture(scope="module")
def cuda_target():
    return tiny_qwen3(seed=0).to("cuda")


@pytest.fixture(scope="module")
def cuda_drafts(target):
    return {scale: noisy_copy(target, scale).to("cuda") for scale in (0.05, 0.5)}


def cuda_prompts(n):
    return [p.to("cuda") for p in prompts(n)]


@gpu
@pytest.mark.parametrize("scale", [0.05, 0.5])
@pytest.mark.parametrize("k", [2, 4])
def test_graph_speculative_equals_dynamic_speculative_ar_and_hf(cuda_target, cuda_drafts, scale, k):
    draft = cuda_drafts[scale]
    decoder = spec_decoder(cuda_target, draft, k, GRAPHS)
    for prompt in cuda_prompts(4):
        hf = cuda_target.generate(prompt, do_sample=False, max_new_tokens=40, pad_token_id=0)
        expected = hf[0, prompt.shape[1] :].tolist()
        dynamic = run_spec(cuda_target, draft, prompt, 40, k)
        got = decoder.generate(prompt, max_new_tokens=40, sampler=GREEDY, eos_ids=frozenset())
        assert got.output_ids == expected == dynamic.output_ids
        assert run_static_ar(cuda_target, prompt, 40, GRAPHS).output_ids == expected
        assert_same_stats(got, dynamic)
        assert got.stats.rounds > 1


@gpu
@pytest.mark.parametrize("k", [2, 4])
def test_graph_round_schedule_matches_the_dynamic_run(cuda_target, cuda_drafts, k):
    seen = []
    for draft in cuda_drafts.values():
        decoder = spec_decoder(cuda_target, draft, k, GRAPHS)
        for prompt in cuda_prompts(3):
            ours, theirs = [], []
            decoder.on_round = ours.append
            decoder.generate(prompt, max_new_tokens=40, sampler=GREEDY, eos_ids=frozenset())
            run_spec(cuda_target, draft, prompt, 40, k, on_round=theirs.append)
            assert ours == theirs
            seen.extend(ours)
    assert any(r["m"] < r["k"] for r in seen) and any(r["m"] == r["k"] for r in seen)


@gpu
def test_graph_decoders_capture_the_decode_shapes_and_require_prepare(cuda_target, cuda_drafts):
    k = 4
    draft = cuda_drafts[0.5]
    prompt = cuda_prompts(1)[0]
    decoder = SpeculativeDecoder(
        cuda_target, ModelDrafter(draft, GRAPHS), k, GreedyAcceptance(), settings=GRAPHS
    )
    with pytest.raises(RuntimeError, match="needs the decoder's prepare"):
        decoder.generate(prompt, max_new_tokens=10, sampler=GREEDY, eos_ids=frozenset())
    prepared = decoder.prepare()
    assert decoder.prepared is prepared
    assert prepared["target"]["graphs"] == (1 + k) * 2  # shapes (1, 1), (2, 2) .. (k + 1, k + 1) x 2 buckets
    assert prepared["draft"]["graphs"] == 2 * 2  # shapes (1, 1), (2, 1) x 2 buckets
    assert prepared["target"]["cuda_graphs"] is True and prepared["target"]["capture_ms"] > 0
    assert decoder.generate(prompt, max_new_tokens=10, sampler=GREEDY, eos_ids=frozenset()).output_ids

    autoregressive = AutoregressiveDecoder(cuda_target, GRAPHS)
    with pytest.raises(RuntimeError, match="needs the decoder's prepare"):
        autoregressive.generate(prompt, max_new_tokens=10, sampler=GREEDY, eos_ids=frozenset())
    assert autoregressive.prepare()["graphs"] == 2


@gpu
def test_graph_generation_beyond_the_capacity_fails_before_compute(cuda_target):
    settings = StaticCacheSettings(capacity=256, cuda_graphs=True)
    decoder = AutoregressiveDecoder(cuda_target, settings)
    decoder.prepare()
    with pytest.raises(ValueError, match="capacity is 256"):
        decoder.generate(cuda_prompts(1)[0], max_new_tokens=300, sampler=GREEDY, eos_ids=frozenset())


@gpu
def test_graph_eos_and_budgets_match_dynamic(cuda_target, cuda_drafts):
    draft = cuda_drafts[0.05]
    decoder = spec_decoder(cuda_target, draft, 4, GRAPHS)
    prompt = cuda_prompts(1)[0]
    full = run_ar(cuda_target, prompt, 40).output_ids
    for index, eos in first_occurrences(full, 2, 14)[:4]:
        got = decoder.generate(prompt, max_new_tokens=40, sampler=GREEDY, eos_ids=frozenset({eos}))
        assert got.output_ids == full[: index + 1] and got.finish == "eos"
    for budget in (1, 2, 7):
        got = decoder.generate(prompt, max_new_tokens=budget, sampler=GREEDY, eos_ids=frozenset())
        assert got.output_ids == full[:budget]
        assert_same_stats(got, run_spec(cuda_target, draft, prompt, budget, 4))


class CountingGraph:
    """Delegates `replay` to a captured graph and counts the calls."""

    def __init__(self, graph):
        self.graph = graph
        self.replays = 0

    def replay(self):
        self.replays += 1
        self.graph.replay()


def count_replays(static):
    """Count the replays of `static`'s captured graphs and spy on its eager forwards; (counters, eager)."""
    counters = {}
    for shape, captured in static._graphs.items():
        captured.graph = counters[shape] = CountingGraph(captured.graph)
    eager = []
    forward = static._forward

    def spy(ids, keep, bucket):
        eager.append((ids.shape[1], keep, bucket))
        return forward(ids, keep, bucket)

    static._forward = spy
    return counters, eager


def replays_by_tokens(counters):
    """Replay counts summed over buckets, keyed by (tokens fed, logits rows kept)."""
    totals = {}
    for (q, keep, _bucket), counter in counters.items():
        totals[(q, keep)] = totals.get((q, keep), 0) + counter.replays
    return {shape: n for shape, n in totals.items() if n}


@gpu
@pytest.mark.parametrize("k", [2, 4])
def test_graph_speculative_loop_replays_graphs_for_every_decode_step(cuda_target, cuda_drafts, k):
    decoder = spec_decoder(cuda_target, cuda_drafts[0.05], k, GRAPHS)
    target_replays, target_eager = count_replays(decoder.target)
    draft_replays, draft_eager = count_replays(decoder.drafter.model)
    prompt = cuda_prompts(1)[0]
    got = decoder.generate(prompt, max_new_tokens=40, sampler=GREEDY, eos_ids=frozenset())

    assert got.stats.rounds > 2
    assert [shape[:2] for shape in target_eager] == [(prompt.shape[1] + k, k + 1)]  # the fused prefill round
    assert [shape[:2] for shape in draft_eager] == [(prompt.shape[1], 1)]  # the draft's prefill
    target = replays_by_tokens(target_replays)
    draft = replays_by_tokens(draft_replays)
    assert sum(n for (q, _), n in target.items() if q >= 2) == got.stats.rounds - 1
    assert all(keep == q for (q, keep), _ in target.items())
    assert target.get((1, 1), 0) == got.stats.plain_steps
    assert set(draft) <= {(1, 1), (2, 1)} and sum(draft.values()) == got.stats.proposed - 1
    assert draft.get((1, 1), 0) > 0 and draft.get((2, 1), 0) > 0


@gpu
def test_graph_autoregressive_loop_replays_one_graph_per_token(cuda_target):
    decoder = AutoregressiveDecoder(cuda_target, GRAPHS)
    decoder.prepare()
    replays, eager = count_replays(decoder.target)
    prompt = cuda_prompts(1)[0]
    got = decoder.generate(prompt, max_new_tokens=20, sampler=GREEDY, eos_ids=frozenset())
    assert [shape[:2] for shape in eager] == [(prompt.shape[1], 1)]
    assert replays_by_tokens(replays) == {(1, 1): got.stats.new_tokens - 1}
