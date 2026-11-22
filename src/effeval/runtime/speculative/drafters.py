"""Draft-token proposers."""

from __future__ import annotations

from typing import Protocol

import torch

from effeval.runtime.speculative.rules import Proposal, TokenSampler
from effeval.runtime.speculative.state import CachedModel


class Drafter(Protocol):
    """Proposes continuations of the running sequence; may keep state between rounds.

    `cached` is the number of sequence tokens the drafter's state currently covers (0 if stateless).
    """

    cached: int

    @property
    def models(self) -> list[torch.nn.Module]:
        """The neural models the drafter runs (empty if none)."""
        ...

    def reset(self) -> None:
        """Forget all state before a new generation."""
        ...

    def propose(self, sequence: torch.Tensor, k: int, sampler: TokenSampler, with_probs: bool) -> Proposal:
        """Draft `k` tokens continuing `sequence` [1, n]; with `with_probs`, also their distributions."""
        ...

    def commit(self, accepted_length: int) -> None:
        """The sequence is now final up to `accepted_length` tokens; drop state beyond what is still valid."""
        ...


class ModelDrafter:
    """An independent draft model with its own KV cache.

    A proposal leaves the cache at n + k - 1 tokens: the last drafted token is never fed.
    """

    def __init__(self, model):
        self.model = CachedModel(model)

    @property
    def cached(self) -> int:
        return self.model.cached

    @property
    def models(self) -> list[torch.nn.Module]:
        return [self.model.model]

    def reset(self) -> None:
        self.model.reset()

    def propose(self, sequence: torch.Tensor, k: int, sampler: TokenSampler, with_probs: bool) -> Proposal:
        if k < 1:
            raise ValueError("k must be at least 1")
        current = sequence
        tokens: list[int] = []
        rows: list[torch.Tensor] = []
        for _ in range(k):
            logits = self.model.feed(current, None, 1)[-1]
            token = sampler.pick(logits)
            tokens.append(token)
            if with_probs:
                rows.append(sampler.probs(logits))
            step = torch.tensor([[token]], dtype=current.dtype, device=current.device)
            current = torch.cat([current, step], dim=1)
        return Proposal(tokens, torch.stack(rows) if with_probs else None)

    def commit(self, accepted_length: int) -> None:
        self.model.rollback(min(self.model.cached, accepted_length - 1))
