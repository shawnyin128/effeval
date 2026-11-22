"""Per-channel scaling folds (float64), torch only.

A channel scale s rescales activations as x / s; the matching fold multiplies the consumer's input columns
by s so that (x / s) @ (W * s).T == x @ W.T. Each helper returns a new tensor in the input's dtype.
"""

from __future__ import annotations

import torch
from torch import nn


def fold_into_norm_gain(gain: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Gain of a norm whose output is divided by `scale` per channel: gain / scale."""
    return (gain.double() / scale.to(gain.device).double()).to(gain.dtype)


def scale_input_columns(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W * s along input columns for a consumer whose input is divided by s: W [out, in], s [in]."""
    return (weight.double() * scale.to(weight.device).double()[None, :]).to(weight.dtype)


def scale_output_rows(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """W / s along output rows for a producer whose output is divided by s: W [out, in], s [out]."""
    return (weight.double() / scale.to(weight.device).double()[:, None]).to(weight.dtype)


class ChannelDivide(nn.Module):
    """Online op x / s over the last dimension, computed in float32 and returned in x's dtype."""

    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / self.scale).to(x.dtype)


class HeadChannelScale(nn.Module):
    """Online op x * scale on an attention operand x [B, Hx, T, D], computed in float32.

    `scale` is [Hx, D] (one scale per head and channel) or [D] (shared by all heads). Raises ValueError when
    x does not match the scale's shape.
    """

    def __init__(self, scale: torch.Tensor):
        super().__init__()
        if scale.ndim not in (1, 2):
            raise ValueError(f"scale must be [Hx, D] or [D], got {tuple(scale.shape)}")
        self.register_buffer("scale", scale.detach().clone().float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(x.device)
        if (
            x.ndim != 4
            or x.shape[-1] != scale.shape[-1]
            or (scale.ndim == 2 and x.shape[1] != scale.shape[0])
        ):
            raise ValueError(f"scale {tuple(scale.shape)} does not match operand {tuple(x.shape)}")
        scale = scale[None, :, None, :] if scale.ndim == 2 else scale[None, None, None, :]
        return (x.float() * scale).to(x.dtype)
