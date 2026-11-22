"""A causal LM with its own KV cache over a growing token sequence."""

from __future__ import annotations

import torch


class CachedModel:
    """A causal LM with its own DynamicCache over a growing token sequence (batch size 1).

    `cached` is the number of leading sequence tokens whose keys/values are in the cache.
    """

    def __init__(self, model):
        self.model = model
        self.cache = None
        self.cached = 0

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset(self) -> None:
        """Drop the cache; the next `feed` starts from an empty one."""
        self.cache = None
        self.cached = 0

    def feed(self, sequence: torch.Tensor, extra: torch.Tensor | None, keep: int) -> torch.Tensor:
        """Run `sequence[:, cached:]` followed by `extra` through the model and extend the cache.

        Returns the last `keep` logits rows, shape [keep, V], in float32 (float64 stays float64).
        `extra` tokens are cached too; they are not part of `sequence`. Raises ValueError when there is
        nothing new to run or `keep` exceeds the number of run tokens.
        """
        new = sequence[:, self.cached :]
        if extra is not None:
            new = torch.cat([new, extra.to(new.device)], dim=1)
        if new.shape[0] != 1 or new.shape[1] == 0:
            raise ValueError(f"feed needs a batch of 1 with an uncached token, got shape {tuple(new.shape)}")
        if keep > new.shape[1]:
            raise ValueError(f"keep={keep} exceeds the {new.shape[1]} tokens run")
        out = self.model(input_ids=new, past_key_values=self.cache, use_cache=True, logits_to_keep=keep)
        self.cache = out.past_key_values
        expected = self.cached + new.shape[1]
        if self.cache.get_seq_length() != expected:
            raise RuntimeError(f"cache holds {self.cache.get_seq_length()} tokens, expected {expected}")
        self.cached = expected
        logits = out.logits[0]
        return logits.to(torch.promote_types(logits.dtype, torch.float32))

    def rollback(self, length: int) -> None:
        """Keep the first `length` cached tokens (transformers' `crop` takes a negative removal count)."""
        if not 0 <= length <= self.cached:
            raise ValueError(f"cannot roll back to {length} with {self.cached} cached")
        if self.cached > length:
            self.cache.crop(-(self.cached - length))
            self.cached = length
