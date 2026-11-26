"""Token sampling and acceptance rules that decide how many drafted tokens the target keeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper


class TokenSampler:
    """Turns logits into tokens: argmax when `temperature` is None, else draws from the warped distribution.

    The warped distribution is softmax of the logits divided by `temperature`, then restricted by `top_k`
    and `top_p` with the transformers warpers in that order (the order `generate` uses). `top_p` and `top_k`
    require a temperature. Random numbers come from `generator` (the default RNG without one).
    """

    def __init__(
        self,
        temperature: float | None,
        generator: torch.Generator | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive or None, got {temperature}")
        if temperature is None and (top_p is not None or top_k is not None):
            raise ValueError("top_p and top_k need a temperature (a greedy sampler takes neither)")
        if top_p is not None and not 0.0 < top_p < 1.0:
            raise ValueError(f"top_p must be in (0, 1), got {top_p}")
        if top_k is not None and top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        self.temperature = temperature
        self.generator = generator
        self.top_p = top_p
        self.top_k = top_k

    @property
    def greedy(self) -> bool:
        return self.temperature is None

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Warped distribution of a [V] or [n, V] logits tensor, same shape, float32 or wider.

        Raises ValueError when greedy.
        """
        if self.temperature is None:
            raise ValueError("a greedy sampler has no distribution")
        scores = logits.to(torch.promote_types(logits.dtype, torch.float32)) / self.temperature
        rows = scores.reshape(-1, scores.shape[-1])
        if self.top_k is not None:
            rows = TopKLogitsWarper(self.top_k)(None, rows)
        if self.top_p is not None:
            rows = TopPLogitsWarper(self.top_p)(None, rows)
        return torch.softmax(rows, dim=-1).reshape(scores.shape)

    def draw(self, weights: torch.Tensor) -> int:
        """One index drawn from a [V] non-negative weight row (need not sum to 1) using the generator."""
        if self.generator is not None:
            weights = weights.to(self.generator.device)
        return int(torch.multinomial(weights, 1, generator=self.generator).item())

    def uniform(self, n: int) -> torch.Tensor:
        """`n` float64 uniforms in [0, 1) from the generator, on its device (CPU without a generator)."""
        device = "cpu" if self.generator is None else self.generator.device
        return torch.rand(n, generator=self.generator, device=device, dtype=torch.float64)

    def pick(self, logits: torch.Tensor) -> int:
        """Argmax of a [V] logits row when greedy, else a draw from `probs`."""
        if self.temperature is None:
            return int(logits.argmax(-1).item())
        return self.draw(self.probs(logits))


@dataclass
class Proposal:
    """Drafted tokens; `probs` [k, V] holds the distribution each token was drawn from (None if unused)."""

    tokens: list[int]
    probs: torch.Tensor | None = None


class AcceptanceRule(Protocol):
    """Decides how much of a proposal the target accepts."""

    needs_draft_probs: bool

    def verify(
        self, proposal: Proposal, target_logits: torch.Tensor, sampler: TokenSampler
    ) -> tuple[int, int]:
        """`target_logits` [k+1, V]: row i = target logits after the sequence plus `proposal.tokens[:i]`.

        Returns (accepted count m in 0..k, the next token: a correction after a rejection or the bonus token).
        """
        ...


class GreedyAcceptance:
    """Accept the longest prefix where the target argmax equals the draft; next token = argmax of row m."""

    needs_draft_probs = False

    def verify(
        self, proposal: Proposal, target_logits: torch.Tensor, sampler: TokenSampler
    ) -> tuple[int, int]:
        if not sampler.greedy:
            raise ValueError("GreedyAcceptance requires a greedy sampler")
        k = len(proposal.tokens)
        if target_logits.shape[0] != k + 1:
            raise ValueError(
                f"expected {k + 1} target rows for {k} drafted tokens, got {target_logits.shape[0]}"
            )
        best = target_logits.argmax(-1).tolist()
        m = 0
        while m < k and best[m] == proposal.tokens[m]:
            m += 1
        return m, best[m]


def residual_distribution(p_row: torch.Tensor, q_row: torch.Tensor) -> torch.Tensor:
    """Normalized max(p - q, 0); `p_row` itself when that is zero everywhere."""
    residual = (p_row - q_row).clamp_min(0)
    total = residual.sum()
    return residual / total if total > 0 else p_row


class SpeculativeSampling:
    """Lossless rejection sampling (Leviathan et al., Chen et al. 2023) on the sampler's warped distributions.

    Draft token x_i is kept with probability min(1, p_i(x_i) / q_i(x_i)); the first rejection at position m
    emits a token from normalize(max(p_m - q_m, 0)), and a fully accepted proposal emits a token from p_k.
    Random numbers per round: k uniforms, then one draw.
    """

    needs_draft_probs = True

    def verify(
        self, proposal: Proposal, target_logits: torch.Tensor, sampler: TokenSampler
    ) -> tuple[int, int]:
        if sampler.greedy:
            raise ValueError("SpeculativeSampling requires a sampling sampler")
        k = len(proposal.tokens)
        if proposal.probs is None:
            raise ValueError("SpeculativeSampling needs the draft distributions in the proposal")
        if target_logits.shape[0] != k + 1:
            raise ValueError(
                f"expected {k + 1} target rows for {k} drafted tokens, got {target_logits.shape[0]}"
            )
        expected = (k, target_logits.shape[-1])
        if tuple(proposal.probs.shape) != expected:
            raise ValueError(f"expected draft probs of shape {expected}, got {tuple(proposal.probs.shape)}")
        p = sampler.probs(target_logits)
        q = proposal.probs.to(device=p.device, dtype=p.dtype)
        tokens = torch.tensor(proposal.tokens, dtype=torch.long, device=p.device)
        rows = torch.arange(k, device=p.device)
        u = sampler.uniform(k).to(p.device)
        accept = u < p[rows, tokens] / q[rows, tokens]
        m = int(accept.long().cumprod(0).sum().item())
        if m == k:
            return m, sampler.draw(p[k])
        return m, sampler.draw(residual_distribution(p[m], q[m]))
