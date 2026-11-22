"""Token sampling and acceptance rules that decide how many drafted tokens the target keeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class TokenSampler:
    """Turns logits into tokens: argmax when `temperature` is None, else sampling at that temperature."""

    def __init__(self, temperature: float | None, generator: torch.Generator | None = None):
        if temperature is not None and temperature <= 0:
            raise ValueError(f"temperature must be positive or None, got {temperature}")
        self.temperature = temperature
        self.generator = generator

    @property
    def greedy(self) -> bool:
        return self.temperature is None

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """softmax(logits / temperature) in float32 or wider; raises ValueError when greedy."""
        if self.temperature is None:
            raise ValueError("a greedy sampler has no distribution")
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        return torch.softmax(logits / self.temperature, dim=-1)

    def pick(self, logits: torch.Tensor) -> int:
        """Argmax of a [V] logits row when greedy, else a draw from `probs` using the generator."""
        if self.temperature is None:
            return int(logits.argmax(-1).item())
        return int(torch.multinomial(self.probs(logits), 1, generator=self.generator).item())


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
