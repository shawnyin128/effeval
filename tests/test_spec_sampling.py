"""Speculative sampling: warpers, the rejection rule's exact output distribution, drafter, end to end."""

from __future__ import annotations

import itertools

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from conftest import _randomize_norm_gains  # noqa: E402
from effeval.runtime.speculative import (  # noqa: E402
    AutoregressiveDecoder,
    ModelDrafter,
    Proposal,
    SpeculativeDecoder,
    SpeculativeSampling,
    StaticCacheSettings,
    TokenSampler,
    residual_distribution,
)


def tiny_qwen3(vocab: int, seed: int = 0):
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=vocab,
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
    _randomize_norm_gains(model, seed + 1)
    return model.double()


def kept(probs: torch.Tensor) -> list[int]:
    return torch.nonzero(probs).flatten().tolist()


def test_warp_hand_computed():
    ties = torch.tensor([3.0, 2.0, 2.0, 2.0, 0.0, -1.0])
    probs = TokenSampler(1.0, top_k=3).probs(ties)
    assert kept(probs) == [0, 1, 2, 3]
    assert torch.allclose(probs[:4], torch.softmax(ties[:4], dim=-1))

    logits = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, -2.0])
    probs = TokenSampler(1.0, top_p=0.6).probs(logits)
    assert kept(probs) == [0, 1]
    assert torch.allclose(probs[:2], torch.softmax(logits[:2], dim=-1))

    # top-k first: {0, 1, 2} renormalizes to [.6285, .2312, .1402], so top_p 0.62 keeps only token 0;
    # top-p first would keep {0, 1} on the full distribution
    probs = TokenSampler(1.0, top_k=3, top_p=0.62).probs(logits)
    assert kept(probs) == [0]
    assert probs[0] == 1.0

    # T=2 on doubled logits is the same distribution; top-k {0..3} renormalizes, top_p 0.9 drops token 3
    probs = TokenSampler(2.0, top_k=4, top_p=0.9).probs(2 * logits)
    assert kept(probs) == [0, 1, 2]
    assert torch.allclose(probs[:3], torch.softmax(logits[:3], dim=-1))

    rows = torch.stack([logits, ties])
    both = TokenSampler(1.0, top_k=3, top_p=0.62).probs(rows)
    assert both.shape == rows.shape
    assert kept(both[0]) == [0]


def test_residual_distribution():
    p = torch.tensor([0.4, 0.3, 0.2, 0.1])
    q = torch.tensor([0.1, 0.5, 0.1, 0.3])
    assert torch.allclose(residual_distribution(p, q), torch.tensor([0.75, 0.0, 0.25, 0.0]))
    assert torch.equal(residual_distribution(p, p), p)


def test_rule_rejects_bad_inputs():
    rule = SpeculativeSampling()
    sampler = TokenSampler(1.0, torch.Generator().manual_seed(0))
    logits = torch.randn(3, 5)
    probs = torch.full((2, 5), 0.2)
    with pytest.raises(ValueError):
        rule.verify(Proposal([0, 1], probs), logits, TokenSampler(None))
    with pytest.raises(ValueError):
        rule.verify(Proposal([0, 1], None), logits, sampler)
    with pytest.raises(ValueError):
        rule.verify(Proposal([0, 1], probs), logits[:2], sampler)
    with pytest.raises(ValueError):
        rule.verify(Proposal([0, 1], probs[:1]), logits, sampler)
    with pytest.raises(ValueError):
        rule.verify(Proposal([0, 1], probs[:, :4]), logits, sampler)


class ScriptedSampler(TokenSampler):
    """Returns preset uniforms and records the weights of every draw (always returns index 0)."""

    def __init__(self, temperature, top_p, top_k, uniforms):
        super().__init__(temperature, None, top_p, top_k)
        self.uniforms = uniforms
        self.drawn: list[torch.Tensor] = []

    def uniform(self, n):
        return self.uniforms[:n].clone()

    def draw(self, weights):
        self.drawn.append(weights / weights.sum())
        return 0


SETTINGS = [(1.0, None, None), (0.7, 3, None), (1.3, None, 0.8), (0.9, 4, 0.85)]


@pytest.mark.parametrize(("temperature", "top_k", "top_p"), SETTINGS)
def test_rule_emits_the_target_distribution_exactly(temperature, top_k, top_p):
    k, vocab = 3, 5
    target_logits = torch.randn(k + 1, vocab, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    draft_logits = torch.randn(k, vocab, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    sampler = ScriptedSampler(temperature, top_p, top_k, torch.zeros(k, dtype=torch.float64))
    p = sampler.probs(target_logits)
    q = sampler.probs(draft_logits)
    rule = SpeculativeSampling()
    # a draft token every position can start from: positive under both distributions
    anchors = [int((p[j] * (q[j] > 0)).argmax()) for j in range(k)]
    assert all(p[j, anchors[j]] > 0 and q[j, anchors[j]] > 0 for j in range(k))
    if top_k is not None or top_p is not None:
        assert ((q > 0) & (p[:k] == 0)).any()

    def run(tokens, uniforms):
        sampler.uniforms = torch.tensor(uniforms, dtype=torch.float64)
        sampler.drawn.clear()
        m, _ = rule.verify(Proposal(tokens, q), target_logits, sampler)
        return m

    for i in range(k):
        emitted = torch.zeros(vocab, dtype=torch.float64)
        for x in torch.nonzero(q[i]).flatten().tolist():
            tokens = [*anchors[:i], x, *anchors[i + 1 :]]
            uniforms = [0.0] * k

            def accepted(u, i=i, tokens=tokens, uniforms=uniforms):
                return run(tokens, [*uniforms[:i], u, *uniforms[i + 1 :]]) > i

            if not accepted(0.0):
                accept_mass = 0.0
                lo, hi = 0.0, 0.0
            elif accepted(1 - 1e-15):
                accept_mass = 1.0
            else:
                lo, hi = 0.0, 1 - 1e-15
                for _ in range(60):
                    mid = (lo + hi) / 2
                    lo, hi = (mid, hi) if accepted(mid) else (lo, mid)
                accept_mass = lo
            if accept_mass < 1.0:
                assert not accepted(hi)
                residual = sampler.drawn[-1]
                emitted += q[i, x] * (1 - accept_mass) * residual
            emitted[x] += q[i, x] * accept_mass
        assert torch.allclose(emitted, p[i], atol=1e-9)
        assert 0.5 * (emitted - q[i]).abs().sum() > 0.05

    run(anchors, [0.0] * k)
    assert len(sampler.drawn) == 1
    assert torch.allclose(sampler.drawn[0], p[k], atol=1e-12)


def test_drafter_samples_from_the_probs_it_reports():
    class RecordingSampler(TokenSampler):
        drawn: list[torch.Tensor]

        def draw(self, weights):
            self.drawn.append(weights.clone())
            return super().draw(weights)

    draft = tiny_qwen3(vocab=11, seed=3)
    sampler = RecordingSampler(0.8, torch.Generator().manual_seed(0), top_p=0.9, top_k=5)
    sampler.drawn = []
    sequence = torch.tensor([[1, 2, 3, 4]])
    k = 3
    proposal = ModelDrafter(draft).propose(sequence, k, sampler, True)
    assert proposal.probs.shape == (k, 11)
    assert len(sampler.drawn) == k
    full = torch.cat([sequence, torch.tensor([proposal.tokens])], dim=1)
    with torch.no_grad():
        for i in range(k):
            fresh = sampler.probs(draft(input_ids=full[:, : 4 + i]).logits[0, -1])
            assert torch.equal(sampler.drawn[i], proposal.probs[i])
            assert torch.allclose(proposal.probs[i], fresh, atol=1e-12)
            assert proposal.probs[i, proposal.tokens[i]] > 0
    assert (proposal.probs > 0).sum(-1).max() <= 5


# ------------------------------------------------------------------------------------ end to end
E2E_VOCAB = 4
E2E_PROMPT = torch.tensor([[0, 1, 2, 3]])
E2E_RUNS = 10_000
E2E_STEPS = 3
# scale keeping the 4-token distributions away from point masses (x3 collapses top_p 0.9 to one token)
E2E_HEAD_SCALE = 0.3


def scaled_pair():
    target, draft = tiny_qwen3(E2E_VOCAB, seed=0), tiny_qwen3(E2E_VOCAB, seed=1)
    for model in (target, draft):
        model.lm_head.weight.data *= E2E_HEAD_SCALE
    return target, draft


def exact_joint(model, sampler) -> torch.Tensor:
    """P(a, b, c) of the first three sampled tokens by enumerating every prefix."""
    joint = torch.zeros((E2E_VOCAB,) * E2E_STEPS, dtype=torch.float64)

    def step(prefix: tuple[int, ...]) -> torch.Tensor:
        ids = torch.cat([E2E_PROMPT, torch.tensor([prefix], dtype=torch.long).reshape(1, -1)], dim=1)
        with torch.no_grad():
            return sampler.probs(model(input_ids=ids).logits[0, -1])

    for tokens in itertools.product(range(E2E_VOCAB), repeat=E2E_STEPS):
        weight = 1.0
        for depth in range(E2E_STEPS):
            weight = weight * step(tokens[:depth])[tokens[depth]]
        joint[tokens] = weight
    return joint


def cell_groups(expected: torch.Tensor, minimum: float = 5.0) -> list[list[int]]:
    """Cells with expected count >= `minimum` stand alone; the rest are pooled (with the smallest standalone
    cell if the pool is still below `minimum`)."""
    order = expected.argsort(descending=True).tolist()
    groups = [[c] for c in order if expected[c] >= minimum]
    pool = [c for c in order if expected[c] < minimum]
    if pool and expected[pool].sum() < minimum and groups:
        pool.extend(groups.pop())
    return [*groups, pool] if pool else groups


def p_value(statistic: torch.Tensor, df: int) -> float:
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), statistic / 2))


def gof_p(counts: torch.Tensor, probs: torch.Tensor, groups: list[list[int]]) -> float:
    observed = torch.stack([counts[g].sum() for g in groups]).double()
    expected = torch.stack([probs[g].sum() for g in groups]) * counts.sum()
    if ((expected == 0) & (observed > 0)).any():
        return 0.0
    live = expected > 0
    return p_value(((observed - expected)[live] ** 2 / expected[live]).sum(), int(live.sum()) - 1)


def two_sample_p(a: torch.Tensor, b: torch.Tensor, groups: list[list[int]]) -> float:
    ga = torch.stack([a[g].sum() for g in groups]).double()
    gb = torch.stack([b[g].sum() for g in groups]).double()
    live = ga + gb > 0
    return p_value(((ga - gb)[live] ** 2 / (ga + gb)[live]).sum(), int(live.sum()) - 1)


def sample_counts(decoder, settings, seed) -> tuple[torch.Tensor, int, int]:
    temperature, top_k, top_p = settings
    generator = torch.Generator().manual_seed(seed)
    sampler = TokenSampler(temperature, generator, top_p, top_k)
    counts = torch.zeros(E2E_VOCAB**E2E_STEPS, dtype=torch.float64)
    proposed = accepted = 0
    for _ in range(E2E_RUNS):
        result = decoder.generate(E2E_PROMPT, max_new_tokens=E2E_STEPS, sampler=sampler, eos_ids=frozenset())
        a, b, c = result.output_ids
        counts[(a * E2E_VOCAB + b) * E2E_VOCAB + c] += 1
        proposed += result.stats.proposed
        accepted += result.stats.accepted
    return counts, proposed, accepted


@pytest.mark.parametrize("settings", [(0.8, None, None), (1.0, 3, 0.9)])
def test_sampling_matches_the_target_distribution_end_to_end(settings):
    temperature, top_k, top_p = settings
    target, draft = scaled_pair()
    warp = TokenSampler(temperature, None, top_p, top_k)
    target_joint = exact_joint(target, warp).flatten()
    draft_joint = exact_joint(draft, warp).flatten()
    groups = cell_groups(target_joint * E2E_RUNS)

    spec = SpeculativeDecoder(target, ModelDrafter(draft), 2, SpeculativeSampling())
    spec_counts, proposed, accepted = sample_counts(spec, settings, 0)
    ar_counts, _, _ = sample_counts(AutoregressiveDecoder(target), settings, 10**6)

    assert gof_p(spec_counts, target_joint, groups) > 1e-4
    assert gof_p(ar_counts, target_joint, groups) > 1e-4
    assert two_sample_p(spec_counts, ar_counts, groups) > 1e-4
    assert gof_p(spec_counts, draft_joint, cell_groups(draft_joint * E2E_RUNS)) < 1e-10
    assert 0.05 < accepted / proposed < 0.95


def spec_run(decoder, seed, prompt=None, steps=12):
    sampler = TokenSampler(1.0, torch.Generator().manual_seed(seed))
    return decoder.generate(
        E2E_PROMPT if prompt is None else prompt, max_new_tokens=steps, sampler=sampler, eos_ids=frozenset()
    )


def stats_key(result):
    s = result.stats
    return (result.output_ids, s.rounds, s.proposed, s.accepted, s.accept_lengths, s.plain_steps)


def test_sampling_is_seed_reproducible():
    target, draft = tiny_qwen3(97, seed=0), tiny_qwen3(97, seed=1)
    prompt = torch.tensor([[5, 9, 20, 41, 3]])
    decoder = SpeculativeDecoder(target, ModelDrafter(draft), 3, SpeculativeSampling())
    outputs = []
    for seed in range(5):
        first = spec_run(decoder, seed, prompt)
        assert stats_key(first) == stats_key(spec_run(decoder, seed, prompt))
        outputs.append(tuple(first.output_ids))
    assert len(set(outputs)) >= 2


def test_static_eager_sampling_equals_dynamic():
    target, draft = tiny_qwen3(97, seed=0), tiny_qwen3(97, seed=1)
    settings = StaticCacheSettings(capacity=128, cuda_graphs=False, min_bucket=16)
    dynamic = SpeculativeDecoder(target, ModelDrafter(draft), 3, SpeculativeSampling())
    static = SpeculativeDecoder(
        target, ModelDrafter(draft, settings), 3, SpeculativeSampling(), settings=settings
    )
    for prompt in ([5, 9, 20, 41, 3], [1, 2, 3], [60, 7, 7, 80, 12, 33, 2]):
        ids = torch.tensor([prompt])
        got, expected = spec_run(static, 4, ids, 20), spec_run(dynamic, 4, ids, 20)
        assert got.output_ids == expected.output_ids
        assert got.stats.accept_lengths == expected.stats.accept_lengths
