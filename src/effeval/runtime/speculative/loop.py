"""Autoregressive and speculative decoders (batch size 1) sharing prefill, cache handling and stop logic."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch

from effeval.runtime.speculative.drafters import Drafter
from effeval.runtime.speculative.rules import AcceptanceRule, TokenSampler
from effeval.runtime.speculative.state import make_cached_model, prepare_cached_model
from effeval.runtime.speculative.static import StaticCacheSettings

Finish = Literal["eos", "length", "stop"]


@dataclass
class DecodeStats:
    """Per-generation counters; round values are the untruncated ones even if a stop cut the output.

    The first round of a speculative generation runs the target over the prompt and the first drafts in one
    forward, like transformers' assisted generation, so there is no separate prefill. `accept_lengths` holds
    m + 1 per round (accepted drafts plus the correction/bonus token), so `rounds + accepted ==
    sum(accept_lengths)` and tau = 1 + accepted / rounds. `plain_steps` counts single target steps taken
    outside a round (a budget that leaves no room for a draft). Without a stop, `new_tokens ==
    sum(accept_lengths) + plain_steps`; it is the emitted count after truncation. Times are milliseconds per
    phase: `prefill_ms` is a target forward over the prompt alone (autoregressive decoding, or a budget of
    one token), `draft_ms` the drafter, `verify_ms` the target forwards of rounds and plain steps.
    `first_token_ms` is the time from the start to the first emitted token: the prefill, or the first
    round's draft and verify.
    """

    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    accept_lengths: list[int] = field(default_factory=list)
    plain_steps: int = 0
    new_tokens: int = 0
    prefill_ms: float = 0.0
    draft_ms: float = 0.0
    verify_ms: float = 0.0
    first_token_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        """accepted / proposed (0.0 when nothing was proposed)."""
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tau(self) -> float:
        """Mean tokens gained per round including the bonus: 1 + accepted / rounds (0.0 without rounds)."""
        return 1 + self.accepted / self.rounds if self.rounds else 0.0


@dataclass
class DecodeResult:
    """Emitted tokens (prompt excluded), their statistics and why decoding ended."""

    output_ids: list[int]
    stats: DecodeStats
    finish: Finish


class Decoder(Protocol):
    @property
    def models(self) -> list[torch.nn.Module]:
        """Every model the decoder runs, for device and memory accounting."""
        ...

    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        sampler: TokenSampler,
        eos_ids: frozenset[int],
        stop: Callable[[torch.Tensor], bool] | None = None,
    ) -> DecodeResult:
        """Generate up to `max_new_tokens` after `prompt_ids` [1, L].

        Ends after the first token in `eos_ids` (kept in the output) or the first token after which
        `stop(sequence)` is true; `sequence` is the full [1, L + emitted] tensor.
        """
        ...


class _Clock:
    """Accumulates elapsed milliseconds per phase: CUDA events (no sync until `totals`) or perf_counter."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.device = device
        self.spans: list[tuple[str, object, object]] = []

    @contextmanager
    def span(self, name: str) -> Iterator[None]:
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
        else:
            start = time.perf_counter()
            yield
            end = time.perf_counter()
        self.spans.append((name, start, end))

    def totals(self, upto: int | None = None) -> dict[str, float]:
        """Milliseconds per phase over the first `upto` spans (all by default)."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        totals: dict[str, float] = {}
        for name, start, end in self.spans[:upto]:
            ms = start.elapsed_time(end) if self.cuda else (end - start) * 1000.0
            totals[name] = totals.get(name, 0.0) + ms
        return totals


def _validate(prompt_ids: torch.Tensor, max_new_tokens: int) -> None:
    if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0:
        raise ValueError(f"prompt_ids must have shape [1, L] with L >= 1, got {tuple(prompt_ids.shape)}")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")


def _append(
    sequence: torch.Tensor,
    block: list[int],
    eos_ids: frozenset[int],
    stop: Callable[[torch.Tensor], bool] | None,
) -> tuple[torch.Tensor, int, Finish | None]:
    """Append the part of `block` up to the first eos/stop hit; returns (sequence, kept count, finish)."""
    extended = torch.cat(
        [sequence, torch.tensor([block], dtype=sequence.dtype, device=sequence.device)], dim=1
    )
    n = sequence.shape[1]
    for i, token in enumerate(block):
        if token in eos_ids:
            return extended[:, : n + i + 1], i + 1, "eos"
        if stop is not None and stop(extended[:, : n + i + 1]):
            return extended[:, : n + i + 1], i + 1, "stop"
    return extended, len(block), None


def cache_rows(prompt_length: int, max_new_tokens: int) -> int:
    """Target cache positions a generation of up to `max_new_tokens` tokens after the prompt needs.

    The same for both decoders: the last generated token is never fed and a speculative round drafts at most
    max_new_tokens - generated - 1 tokens.
    """
    return prompt_length + max_new_tokens - 1


def _require_prepared(settings: StaticCacheSettings | None, prepared: dict[str, Any] | None) -> None:
    if settings is not None and settings.cuda_graphs and prepared is None:
        raise RuntimeError("graph mode needs the decoder's prepare() before generate")


class AutoregressiveDecoder:
    """One target forward per token; shares prefill and cache code with the speculative loop.

    With `settings` the target uses a static cache (see `StaticCachedModel`); in graph mode `prepare` must
    run before `generate`. `prepared` holds the result of `prepare` (None before).
    """

    def __init__(self, model, settings: StaticCacheSettings | None = None):
        self.target = make_cached_model(model, settings)
        self.settings = settings
        self.prepared: dict[str, Any] | None = None

    def prepare(self) -> dict[str, Any]:
        """Prepare the target's decode shape outside any timed region; returns its cache description."""
        self.prepared = prepare_cached_model(self.target, [(1, 1)])
        return self.prepared

    @property
    def models(self) -> list[torch.nn.Module]:
        """The models this decoder runs."""
        return [self.target.model]

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        sampler: TokenSampler,
        eos_ids: frozenset[int],
        stop: Callable[[torch.Tensor], bool] | None = None,
    ) -> DecodeResult:
        _validate(prompt_ids, max_new_tokens)
        _require_prepared(self.settings, self.prepared)
        self.target.reserve(cache_rows(prompt_ids.shape[1], max_new_tokens))
        self.target.reset()
        clock = _Clock(self.target.device)
        stats = DecodeStats()
        sequence = prompt_ids
        output: list[int] = []
        finish: Finish = "length"
        with clock.span("prefill"):
            logits = self.target.feed(sequence, None, 1)
        token = sampler.pick(logits[-1])
        first_token_spans = len(clock.spans)
        while True:
            sequence, _, ended = _append(sequence, [token], eos_ids, stop)
            output.append(token)
            if ended:
                finish = ended
                break
            if len(output) >= max_new_tokens:
                break
            with clock.span("verify"):
                logits = self.target.feed(sequence, None, 1)
            token = sampler.pick(logits[-1])
            stats.plain_steps += 1
        stats.new_tokens = len(output)
        totals = clock.totals()
        stats.prefill_ms = totals.get("prefill", 0.0)
        stats.verify_ms = totals.get("verify", 0.0)
        stats.first_token_ms = sum(clock.totals(first_token_spans).values())
        return DecodeResult(output, stats, finish)


class SpeculativeDecoder:
    """Draft-and-verify decoding: a drafter proposes up to k tokens per round, the target verifies at once.

    The first round drafts from the prompt and the target runs the prompt plus those drafts in one forward,
    as transformers' assisted generation does; the first token comes out of that round.

    `on_round`, if given, receives {"n", "k", "m", "target_cached", "draft_cached"} after each round
    (n = sequence length before the round, k = drafted tokens, m = accepted drafts), before truncation.

    With `settings` the target uses a static cache (see `StaticCachedModel`); in graph mode `prepare` (target
    and drafter) must run before `generate`. `prepared` holds the result of `prepare` (None before).
    """

    def __init__(
        self,
        model,
        drafter: Drafter,
        k: int,
        rule: AcceptanceRule,
        on_round: Callable[[dict], None] | None = None,
        settings: StaticCacheSettings | None = None,
    ):
        if k < 1:
            raise ValueError("k must be at least 1")
        self.target = make_cached_model(model, settings)
        self.drafter = drafter
        self.k = k
        self.rule = rule
        self.on_round = on_round
        self.settings = settings
        self.prepared: dict[str, Any] | None = None

    def prepare(self) -> dict[str, Any]:
        """Prepare target and drafter decode shapes outside any timed region.

        The target runs 1 token, or 2 to k + 1 tokens keeping all their logits. Returns {"target": ...,
        "draft": ...} with the cache description of each.
        """
        shapes = [(1, 1), *((q, q) for q in range(2, self.k + 2))]
        self.prepared = {
            "target": prepare_cached_model(self.target, shapes),
            "draft": self.drafter.prepare(),
        }
        return self.prepared

    @property
    def models(self) -> list[torch.nn.Module]:
        """The models this decoder runs: the target, then the drafter's."""
        return [self.target.model, *self.drafter.models]

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        sampler: TokenSampler,
        eos_ids: frozenset[int],
        stop: Callable[[torch.Tensor], bool] | None = None,
    ) -> DecodeResult:
        _validate(prompt_ids, max_new_tokens)
        _require_prepared(self.settings, self.prepared)
        self.target.reserve(cache_rows(prompt_ids.shape[1], max_new_tokens))
        self.target.reset()
        self.drafter.reset()
        clock = _Clock(self.target.device)
        stats = DecodeStats()
        sequence = prompt_ids
        output: list[int] = []
        finish: Finish = "length"
        first_token_spans = None
        while finish == "length" and len(output) < max_new_tokens:
            n = sequence.shape[1]
            k_eff = min(self.k, max_new_tokens - len(output) - 1)
            if k_eff == 0:
                with clock.span("verify" if output else "prefill"):
                    logits = self.target.feed(sequence, None, 1)
                block = [sampler.pick(logits[-1])]
                stats.plain_steps += 1
            else:
                with clock.span("draft"):
                    proposal = self.drafter.propose(sequence, k_eff, sampler, self.rule.needs_draft_probs)
                if len(proposal.tokens) != k_eff:
                    raise RuntimeError(f"drafter returned {len(proposal.tokens)} tokens, expected {k_eff}")
                drafted = torch.tensor([proposal.tokens], dtype=sequence.dtype, device=sequence.device)
                with clock.span("verify"):
                    logits = self.target.feed(sequence, drafted, k_eff + 1)
                    m, bonus = self.rule.verify(proposal, logits, sampler)
                    if not 0 <= m <= k_eff:
                        raise RuntimeError(f"acceptance rule returned m={m} for k={k_eff}")
                    self.target.rollback(n + m)
                self.drafter.commit(n + m + 1)
                stats.rounds += 1
                stats.proposed += k_eff
                stats.accepted += m
                stats.accept_lengths.append(m + 1)
                block = [*proposal.tokens[:m], bonus]
                if self.on_round is not None:
                    self.on_round(
                        {
                            "n": n,
                            "k": k_eff,
                            "m": m,
                            "target_cached": self.target.cached,
                            "draft_cached": self.drafter.cached,
                        }
                    )
            sequence, kept, ended = _append(sequence, block, eos_ids, stop)
            output.extend(block[:kept])
            if first_token_spans is None:
                first_token_spans = len(clock.spans)
            if ended:
                finish = ended
        stats.new_tokens = len(output)
        totals = clock.totals()
        stats.prefill_ms = totals.get("prefill", 0.0)
        stats.draft_ms = totals.get("draft", 0.0)
        stats.verify_ms = totals.get("verify", 0.0)
        stats.first_token_ms = sum(clock.totals(first_token_spans).values())
        return DecodeResult(output, stats, finish)
