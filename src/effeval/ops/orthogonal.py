"""Trainable orthogonal matrices (Cayley parametrization) and the activation quantization error."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def cayley_rotation(dim: int, seed: int, device: torch.device | str = "cpu") -> nn.Module:
    """A float32 `nn.Linear(dim, dim, bias=False)` under torch's Cayley orthogonal parametrization.

    Its initial weight is the orthogonalization of the default `nn.Linear` initialisation drawn after
    `torch.manual_seed(seed)` on the CPU (the global RNG state is restored). `module.weight` is the
    orthogonal matrix R applied to row vectors as `x @ R`; the trainable tensor is
    `module.parametrizations.weight.original`.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        linear = nn.Linear(dim, dim, bias=False, dtype=torch.float32)
    return nn.utils.parametrizations.orthogonal(linear, orthogonal_map="cayley").to(device)


def trainable(rotation: nn.Module) -> nn.Parameter:
    """The unconstrained parameter behind a `cayley_rotation` module."""
    return rotation.parametrizations.weight.original


def act_quant_mse(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Mean squared error between `x` and its per-token asymmetric min-max fake quantization.

    One range [min, max] per vector along the last axis, scale (max - min) / (2^bits - 1) clamped to at
    least 1e-5, offset min, rounding without gradient: gradients reach `x` through the range only.
    """
    flat = x.reshape(-1, x.shape[-1])
    low = flat.min(dim=-1, keepdim=True)[0]
    high = flat.max(dim=-1, keepdim=True)[0]
    levels = 2**bits - 1
    scale = ((high - low) / levels).clamp(min=1e-5)
    quantized = torch.round((flat - low) / scale).clamp(0, levels) * scale + low
    return F.mse_loss(flat, quantized)
