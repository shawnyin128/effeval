"""Greedy speculative decoding loop: cache handling, acceptance rule, equality with autoregressive and HF."""

from __future__ import annotations

import copy

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import _randomize_norm_gains  # noqa: E402
from effeval.runtime.speculative import (  # noqa: E402
    AutoregressiveDecoder,
    CachedModel,
    GreedyAcceptance,
    ModelDrafter,
    Proposal,
    SpeculativeDecoder,
    TokenSampler,
)

VOCAB = 97
GREEDY = TokenSampler(None)
NOISE_SCALES = [0.0, 0.05, 0.5]
KS = [1, 2, 3, 5, 8]


def tiny_qwen3(layers: int = 2, seed: int = 0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        initializer_range=0.5,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
    _randomize_norm_gains(model, seed + 1)
    return model.double()


def noisy_copy(model, scale: float, seed: int = 123):
    """Copy of `model` with Gaussian noise of std `scale` x (parameter std) added to every parameter."""
    draft = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in draft.parameters():
            if scale > 0:
                p += torch.randn(p.shape, generator=generator, dtype=p.dtype) * scale * p.std()
    return draft


@pytest.fixture(scope="module")
def target():
    return tiny_qwen3(seed=0)


@pytest.fixture(scope="module")
def drafts(target):
    return {scale: noisy_copy(target, scale) for scale in NOISE_SCALES}


def prompts(n: int = 8, length: int = 6):
    generator = torch.Generator().manual_seed(7)
    return [torch.randint(0, VOCAB, (1, length), generator=generator) for _ in range(n)]


def run_ar(model, prompt, n, eos=frozenset(), stop=None):
    return AutoregressiveDecoder(model).generate(
        prompt, max_new_tokens=n, sampler=GREEDY, eos_ids=eos, stop=stop
    )


def run_spec(model, draft, prompt, n, k, eos=frozenset(), stop=None, on_round=None, drafter=None):
    decoder = SpeculativeDecoder(
        model, drafter or ModelDrafter(draft), k, GreedyAcceptance(), on_round=on_round
    )
    return decoder.generate(prompt, max_new_tokens=n, sampler=GREEDY, eos_ids=eos, stop=stop)


class RejectingDrafter:
    """Stateless drafter that always proposes `token`."""

    cached = 0
    models: list = []  # noqa: RUF012

    def __init__(self, token: int):
        self.token = token

    def reset(self) -> None:
        pass

    def propose(self, sequence, k, sampler, with_probs):
        return Proposal([self.token] * k, None)

    def commit(self, accepted_length: int) -> None:
        pass


def test_cached_model_rollback_matches_fresh_forward(target):
    seq = torch.randint(0, VOCAB, (1, 8), generator=torch.Generator().manual_seed(1))
    cm = CachedModel(target)
    with torch.inference_mode():
        cm.feed(seq[:, :5], None, 1)
        assert cm.cached == 5
        cm.feed(seq[:, :8], None, 1)
        assert cm.cached == 8
        cm.rollback(6)
        assert cm.cache.get_seq_length() == 6
        assert cm.cached == 6
        logits = cm.feed(seq[:, :7], None, 1)
        fresh = target(input_ids=seq[:, :7]).logits[0, -1:]
    assert torch.allclose(logits, fresh, atol=1e-10)


def test_cached_model_extra_tokens_and_keep_rows(target):
    seq = torch.randint(0, VOCAB, (1, 7), generator=torch.Generator().manual_seed(2))
    cm = CachedModel(target)
    with torch.inference_mode():
        cm.feed(seq[:, :3], None, 1)
        logits = cm.feed(seq[:, :4], seq[:, 4:7], 4)
        fresh = target(input_ids=seq).logits[0, -4:]
    assert cm.cached == 7
    assert logits.shape == (4, VOCAB)
    assert logits.dtype == torch.float64
    assert torch.allclose(logits, fresh, atol=1e-10)
    with pytest.raises(ValueError):
        cm.rollback(8)
    with pytest.raises(ValueError):
        cm.feed(seq[:, :7], None, 1)


def test_cached_model_float32_output_for_low_precision():
    model = tiny_qwen3(seed=0).float()
    seq = torch.randint(0, VOCAB, (1, 4))
    with torch.inference_mode():
        assert CachedModel(model).feed(seq, None, 1).dtype == torch.float32


def test_greedy_acceptance_rows():
    rule = GreedyAcceptance()
    logits = torch.zeros(4, 5)
    for row, best in enumerate([3, 1, 4, 2]):
        logits[row, best] = 1.0
    assert rule.verify(Proposal([3, 1, 4]), logits, GREEDY) == (3, 2)
    assert rule.verify(Proposal([3, 0, 4]), logits, GREEDY) == (1, 1)
    assert rule.verify(Proposal([0, 1, 4]), logits, GREEDY) == (0, 3)
    assert rule.verify(Proposal([3]), logits[:2], GREEDY) == (1, 1)
    assert rule.verify(Proposal([0]), logits[:2], GREEDY) == (0, 3)
    with pytest.raises(ValueError):
        rule.verify(Proposal([3, 1]), logits, GREEDY)
    with pytest.raises(ValueError):
        rule.verify(Proposal([3]), logits[:2], TokenSampler(1.0))


def test_token_sampler():
    logits = torch.tensor([0.0, 2.0, 1.0])
    assert GREEDY.pick(logits) == 1
    with pytest.raises(ValueError):
        GREEDY.probs(logits)
    with pytest.raises(ValueError):
        TokenSampler(0.0)
    sampler = TokenSampler(0.5, torch.Generator().manual_seed(0))
    probs = sampler.probs(logits)
    assert torch.allclose(probs, torch.softmax(logits / 0.5, dim=-1))
    first = [TokenSampler(0.5, torch.Generator().manual_seed(3)).pick(logits) for _ in range(5)]
    assert len(set(first)) == 1


@pytest.mark.parametrize("scale", NOISE_SCALES)
@pytest.mark.parametrize("k", KS)
def test_greedy_matches_autoregressive(target, drafts, scale, k):
    for prompt in prompts():
        expected = run_ar(target, prompt, 40)
        got = run_spec(target, drafts[scale], prompt, 40, k)
        assert got.output_ids == expected.output_ids
        assert got.finish == "length"
        assert len(got.output_ids) == 40


def test_greedy_matches_autoregressive_one_layer_draft(target):
    draft = tiny_qwen3(layers=1, seed=5)
    for prompt in prompts(4):
        expected = run_ar(target, prompt, 40)
        for k in (2, 5):
            assert run_spec(target, draft, prompt, 40, k).output_ids == expected.output_ids


def test_greedy_matches_hf_generate(target):
    for prompt in prompts(4):
        hf = target.generate(prompt, do_sample=False, max_new_tokens=40, pad_token_id=0)
        assert run_ar(target, prompt, 40).output_ids == hf[0, prompt.shape[1] :].tolist()


def hf_assisted(target, draft, prompt, k, budget, monkeypatch):
    """transformers' greedy assisted generation with a constant draft length k and no confidence cut-off.

    The candidate generator reads these settings from the assistant's own generation_config, not from the
    arguments of `generate`.
    """
    config = draft.generation_config
    monkeypatch.setattr(config, "num_assistant_tokens", k, raising=False)
    monkeypatch.setattr(config, "num_assistant_tokens_schedule", "constant", raising=False)
    monkeypatch.setattr(config, "assistant_confidence_threshold", 0.0, raising=False)
    return target.generate(
        prompt, assistant_model=draft, do_sample=False, max_new_tokens=budget, pad_token_id=0
    )


@pytest.mark.parametrize("k", [2, 4])
def test_greedy_matches_hf_assisted_generation(target, drafts, k, monkeypatch):
    draft = drafts[0.5]
    for prompt in prompts(4):
        hf = hf_assisted(target, draft, prompt, k, 40, monkeypatch)
        ours = run_spec(target, draft, prompt, 40, k)
        assert ours.output_ids == hf[0, prompt.shape[1] :].tolist()


@pytest.mark.parametrize("k", [1, 3, 5])
def test_identical_draft_accepts_everything(target, drafts, k):
    rounds = 3
    n = rounds * (k + 1)
    result = run_spec(target, drafts[0.0], prompts(1)[0], n, k)
    stats = result.stats
    assert stats.rounds == rounds
    assert stats.acceptance_rate == 1.0
    assert stats.accept_lengths == [k + 1] * rounds
    assert stats.tau == k + 1
    # a budget that ends inside a round shortens only the last round
    odd = run_spec(target, drafts[0.0], prompts(1)[0], n + 2, k)
    assert odd.stats.acceptance_rate == 1.0
    assert odd.stats.accept_lengths[:rounds] == [k + 1] * rounds
    assert len(odd.output_ids) == n + 2


def test_middle_noise_exercises_accept_and_reject(target, drafts):
    draft = drafts[0.05]
    accepted = proposed = 0
    for prompt in prompts():
        stats = run_spec(target, draft, prompt, 40, 4).stats
        accepted += stats.accepted
        proposed += stats.proposed
    assert 0.1 < accepted / proposed < 0.95


def test_always_rejecting_drafter(target):
    prompt = prompts(1)[0]
    expected = run_ar(target, prompt, 30)
    absent = next(t for t in range(VOCAB) if t not in expected.output_ids)
    result = run_spec(target, None, prompt, 30, 4, drafter=RejectingDrafter(absent))
    assert result.output_ids == expected.output_ids
    assert result.stats.accepted == 0
    assert result.stats.acceptance_rate == 0.0
    assert result.stats.tau == 1.0
    assert result.stats.accept_lengths == [1] * result.stats.rounds


def test_cache_invariants(target, drafts):
    seen = []
    for scale in (0.05, 0.5):
        for prompt in prompts(3):
            run_spec(target, drafts[scale], prompt, 40, 5, on_round=seen.append)
    assert any(r["m"] < r["k"] for r in seen) and any(r["m"] == r["k"] for r in seen)
    for r in seen:
        assert r["target_cached"] == r["n"] + r["m"]
        assert r["draft_cached"] <= r["n"] + r["m"]
        assert 0 <= r["m"] <= r["k"]


def test_drafter_state_matches_fresh_drafter_after_rollback(target, drafts):
    """After a partial acceptance the reused drafter must propose what a fresh one proposes."""
    prompt = prompts(1)[0]
    reused = ModelDrafter(drafts[0.5])
    with torch.inference_mode():
        sequence = prompt
        for accept in (0, 2, 1, 3, 0):
            proposal = reused.propose(sequence, 4, GREEDY, False)
            fresh = ModelDrafter(drafts[0.5]).propose(sequence, 4, GREEDY, False)
            assert proposal.tokens == fresh.tokens
            nxt = (proposal.tokens[0] + 1) % VOCAB
            block = [*proposal.tokens[:accept], nxt]
            sequence = torch.cat([sequence, torch.tensor([block])], dim=1)
            reused.commit(sequence.shape[1])
            assert reused.cached <= sequence.shape[1] - 1


def first_occurrences(tokens, lo, hi):
    """(index, token) for tokens whose first occurrence in `tokens` lies in [lo, hi)."""
    return [(i, t) for i, t in enumerate(tokens) if lo <= i < hi and tokens.index(t) == i]


def test_eos_inside_block(target, drafts):
    prompt = prompts(1)[0]
    full = run_ar(target, prompt, 40).output_ids
    cases = first_occurrences(full, 2, 14)
    assert len(cases) >= 4
    mid_block = False
    for index, eos in cases:
        expected = run_ar(target, prompt, 40, eos=frozenset({eos}))
        assert expected.output_ids == full[: index + 1]
        assert expected.finish == "eos"
        for scale in (0.0, 0.5):
            got = run_spec(target, drafts[scale], prompt, 40, 5, eos=frozenset({eos}))
            assert got.output_ids == expected.output_ids
            assert got.finish == "eos"
            assert got.stats.new_tokens == index + 1
            if scale == 0.0 and sum(got.stats.accept_lengths) > got.stats.new_tokens:
                mid_block = True
    assert mid_block


def test_eos_as_first_token(target, drafts):
    prompt = prompts(1)[0]
    first = run_ar(target, prompt, 1).output_ids[0]
    got = run_spec(target, drafts[0.0], prompt, 40, 4, eos=frozenset({first}))
    assert got.output_ids == [first] and got.finish == "eos"
    assert got.stats.rounds == 1 and got.stats.new_tokens == 1  # the first round ran; a stop cut its block


def test_stop_criteria_inside_block(target, drafts):
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

        expected = run_ar(target, prompt, 40, stop=stop)
        assert expected.output_ids == full[: index + 1] and expected.finish == "stop"
        for scale in (0.0, 0.5):
            got = run_spec(target, drafts[scale], prompt, 40, 5, stop=stop)
            assert got.output_ids == expected.output_ids
            assert got.finish == "stop"


@pytest.mark.parametrize("scale", [0.0, 0.5])
def test_budget_not_multiple_of_k_plus_1(target, drafts, scale):
    for prompt in prompts(3):
        expected = run_ar(target, prompt, 7)
        got = run_spec(target, drafts[scale], prompt, 7, 4)
        assert len(got.output_ids) == 7
        assert got.output_ids == expected.output_ids


def test_budget_of_one_and_two(target, drafts):
    prompt = prompts(1)[0]
    one = run_spec(target, drafts[0.5], prompt, 1, 4)
    assert one.output_ids == run_ar(target, prompt, 1).output_ids
    assert (one.stats.rounds, one.stats.plain_steps) == (0, 1)
    assert one.stats.prefill_ms > 0 and one.stats.first_token_ms == one.stats.prefill_ms
    two = run_spec(target, drafts[0.5], prompt, 2, 4)
    assert two.output_ids == run_ar(target, prompt, 2).output_ids
    assert (two.stats.rounds, two.stats.proposed) == (1, 1)  # a round of one draft, even for two tokens
    assert sum(two.stats.accept_lengths) + two.stats.plain_steps == 2


@pytest.mark.parametrize("scale", [0.0, 0.05, 0.5])
def test_stats_consistency(target, drafts, scale):
    for prompt in prompts(3):
        for n in (7, 40):
            stats = run_spec(target, drafts[scale], prompt, n, 4).stats
            assert sum(stats.accept_lengths) + stats.plain_steps == stats.new_tokens == n
            assert stats.rounds == len(stats.accept_lengths)
            assert stats.accepted == sum(stats.accept_lengths) - stats.rounds
            assert stats.proposed >= stats.accepted
            assert stats.tau == pytest.approx(sum(stats.accept_lengths) / stats.rounds)
            assert stats.prefill_ms == 0.0 and stats.draft_ms > 0 and stats.verify_ms > 0
            assert 0 < stats.first_token_ms < stats.draft_ms + stats.verify_ms


def forward_lengths(model):
    """Token counts of the `input_ids` of every forward of `model`, and the hook handle that records them."""
    lengths = []

    def record(_module, _args, kwargs):
        lengths.append(kwargs["input_ids"].shape[1])

    return lengths, model.register_forward_pre_hook(record, with_kwargs=True)


def test_the_first_round_is_fused_with_the_target_prefill(target, drafts):
    prompt = prompts(1)[0]
    length, k = prompt.shape[1], 4
    draft = drafts[0.5]
    target_lengths, target_hook = forward_lengths(target)
    draft_lengths, draft_hook = forward_lengths(draft)
    try:
        result = run_spec(target, draft, prompt, 40, k)
    finally:
        target_hook.remove()
        draft_hook.remove()
    assert draft_lengths[0] == length  # the draft proposes from the bare prompt
    assert target_lengths[0] == length + k  # one target forward: prompt plus the first k drafts
    assert len(target_lengths) == result.stats.rounds + result.stats.plain_steps
    assert result.stats.accept_lengths[0] >= 1 and result.stats.first_token_ms > 0


def test_a_budget_of_one_prefills_without_drafting(target, drafts):
    prompt = prompts(1)[0]
    target_lengths, target_hook = forward_lengths(target)
    draft_lengths, draft_hook = forward_lengths(drafts[0.5])
    try:
        run_spec(target, drafts[0.5], prompt, 1, 4)
    finally:
        target_hook.remove()
        draft_hook.remove()
    assert target_lengths == [prompt.shape[1]] and draft_lengths == []


def test_autoregressive_first_token_time_is_the_prefill(target):
    stats = run_ar(target, prompts(1)[0], 5).stats
    assert stats.first_token_ms == stats.prefill_ms > 0


@pytest.mark.parametrize("k", [2, 4])
@pytest.mark.parametrize("scale", [0.05, 0.5])
@pytest.mark.parametrize("budget", [2, 7, 40])
def test_target_forward_schedule_matches_hf_assisted_generation(
    target, drafts, k, scale, budget, monkeypatch
):
    """The first target forward is the prompt plus the first drafts, as in transformers' assisted generation.

    The whole schedule of forward sizes is compared while it is short (budgets 2 and 7). Over a longer run
    it is not: transformers' own drafts (checked in 5.23.0 against a fresh greedy decode of the draft model)
    differ from the draft model's greedy continuation in many rounds, so acceptances, and with them the
    later forward sizes, diverge from the loop's, whose drafter state is tested against a fresh drafter.
    """
    draft = drafts[scale]
    for prompt in prompts(3):
        hf_target, hook_a = forward_lengths(target)
        try:
            hf = hf_assisted(target, draft, prompt, k, budget, monkeypatch)
        finally:
            hook_a.remove()
        our_target, hook_b = forward_lengths(target)
        try:
            ours = run_spec(target, draft, prompt, budget, k)
        finally:
            hook_b.remove()
        assert ours.output_ids == hf[0, prompt.shape[1] :].tolist()
        assert our_target[0] == hf_target[0] == prompt.shape[1] + min(k, budget - 1)
        if budget <= 7:
            assert our_target == hf_target


def test_decoder_is_reusable_across_prompts(target, drafts):
    decoder = SpeculativeDecoder(target, ModelDrafter(drafts[0.5]), 3, GreedyAcceptance())
    for prompt in prompts(3):
        got = decoder.generate(prompt, max_new_tokens=20, sampler=GREEDY, eos_ids=frozenset())
        assert got.output_ids == run_ar(target, prompt, 20).output_ids


def test_invalid_inputs(target, drafts):
    prompt = prompts(1)[0]
    with pytest.raises(ValueError):
        run_ar(target, prompt, 0)
    with pytest.raises(ValueError):
        run_spec(target, drafts[0.0], torch.cat([prompt, prompt]), 5, 2)
    with pytest.raises(ValueError):
        SpeculativeDecoder(target, ModelDrafter(drafts[0.0]), 0, GreedyAcceptance())
