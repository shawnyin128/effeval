"""Norms with per-head gains, torch only."""

from __future__ import annotations

import torch
from torch import nn


class HeadwiseRMSNorm(nn.Module):
    """RMSNorm over head_dim with one gain per head, for inputs [B, T, heads, head_dim].

    Same computation as transformers' Qwen3RMSNorm (statistics in float32, gain applied in the input dtype);
    `weight` is [heads, head_dim] instead of [head_dim].
    """

    def __init__(self, weight: torch.Tensor, eps: float):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError(f"weight must be [heads, head_dim], got {tuple(weight.shape)}")
        self.weight = nn.Parameter(weight.detach().clone())
        self.variance_epsilon = eps

    @classmethod
    def from_shared(cls, norm: nn.Module, heads: int) -> HeadwiseRMSNorm:
        """Copy of a Qwen3RMSNorm (gain [head_dim]) with its gain repeated for every head."""
        gain = norm.weight.detach()
        return cls(gain[None, :].repeat(heads, 1), norm.variance_epsilon)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def install_headwise_norm(attention: nn.Module, name: str, heads: int) -> HeadwiseRMSNorm:
    """Replace `attention.<name>` (q_norm or k_norm) in place by a HeadwiseRMSNorm with `heads` equal gains.

    Raises ValueError when the module is already headwise. Returns the new norm.
    """
    old = getattr(attention, name)
    if isinstance(old, HeadwiseRMSNorm):
        raise ValueError(f"{name} is already a HeadwiseRMSNorm")
    new = HeadwiseRMSNorm.from_shared(old, heads)
    setattr(attention, name, new)
    return new
