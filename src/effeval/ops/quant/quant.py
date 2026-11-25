"""Composition entry for fake quantization: granularity -> scale units -> number format."""

from __future__ import annotations

from typing import Literal

import torch

from effeval.core.fragments import CodebookFormat, FpFormat, IntFormat, MxFormat, sequence_options
from effeval.ops import codebook
from effeval.ops.quant.formats import fp_scaled, int_grid, mx_e4m3, mx_e8m0

TensorKind = Literal["weight", "activation"]


def fake_quantize(
    x: torch.Tensor, fmt: IntFormat | FpFormat | MxFormat | CodebookFormat, kind: TensorKind
) -> torch.Tensor:
    """Quantize-dequantize `x` in float32 and return it in x's dtype and shape.

    Granularity is the reduction of one scale unit. For weights [out, in], `per_channel` is one scale per
    output row. For activations [..., D], `per_token` is one scale per vector along D and `per_channel` is
    one scale per feature of D across all other positions. `per_group` splits D into consecutive groups of
    group_size.

    Formats with `group_axis` or `residual` act on activations [..., T, D] with T the token axis: the most
    recent `residual` tokens stay untouched, and `group_axis: token` makes a group the group_size consecutive
    tokens of one feature (the quantized span is cut to whole groups, the remaining tokens stay untouched).

    A `CodebookFormat` fits k-means codebooks on a weight in the dtype it configures (see `ops.codebook`).
    """
    if isinstance(fmt, CodebookFormat):
        if kind != "weight":
            raise ValueError("codebook applies to weights, not activations")
        return codebook.fake_quantize(x, fmt)
    if not sequence_options(fmt):
        return _quantize_units(x, fmt, kind)
    if kind != "activation":
        raise ValueError(f"{' and '.join(sequence_options(fmt))} apply to activations, not {kind}")
    quantized = max(x.shape[-2] - (fmt.residual or 0), 0)
    by_token = fmt.group_axis == "token"
    if by_token:
        quantized -= quantized % fmt.group_size
    if quantized == 0:
        return x
    head = x[..., :quantized, :]
    if by_token:
        head = _quantize_units(head.transpose(-1, -2), fmt, kind).transpose(-1, -2)  # groups run along T
    else:
        head = _quantize_units(head, fmt, kind)
    return torch.cat([head, x[..., quantized:, :]], dim=-2)


def _quantize_units(x: torch.Tensor, fmt: IntFormat | FpFormat | MxFormat, kind: TensorKind) -> torch.Tensor:
    work = x.float()
    if isinstance(fmt, MxFormat):
        blocks = _split_last_dim(work, fmt.block_size)  # [N, block_size]
        if fmt.scale == "e8m0":
            out = mx_e8m0(blocks, fmt.element)
        elif fmt.scale == "e4m3":
            out = mx_e4m3(blocks, fmt.element, work.abs().amax())
        else:
            raise ValueError(f"unsupported MX scale {fmt.scale!r}")
        return out.reshape(x.shape).to(x.dtype)

    units, transposed = _scale_units(work, fmt.granularity, fmt.group_size, kind)  # [N, K]
    if isinstance(fmt, IntFormat):
        if fmt.clip_search is not None and kind != "weight":
            raise ValueError("clip_search applies to weights only")
        grid = int_grid(
            units, fmt.bits, fmt.symmetric, fmt.clip_ratio, fmt.clip_search, fmt.zero_point == "float"
        )
        out = grid.apply(units)
    elif isinstance(fmt, FpFormat):
        out = fp_scaled(units, fmt.encoding, fmt.special, fmt.clip_ratio)
    else:
        raise TypeError(f"unsupported format {type(fmt).__name__}")
    if transposed:
        out = out.t()
    return out.reshape(x.shape).to(x.dtype)


def _split_last_dim(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] % size != 0:
        raise ValueError(f"last dimension {x.shape[-1]} is not divisible by {size}")
    return x.reshape(-1, size)


def _scale_units(
    x: torch.Tensor, granularity: str, group_size: int | None, kind: TensorKind
) -> tuple[torch.Tensor, bool]:
    """Reshape x into [N, K] scale units; returns (units, transposed)."""
    last = x.shape[-1]
    if granularity == "per_tensor":
        return x.reshape(1, -1), False
    if granularity == "per_group":
        return _split_last_dim(x, group_size), False
    if granularity == "per_channel" and kind == "weight":
        return x.reshape(-1, last), False
    if granularity == "per_token" and kind == "activation":
        return x.reshape(-1, last), False
    if granularity == "per_channel" and kind == "activation":
        return x.reshape(-1, last).t(), True  # [D, positions]
    if granularity == "per_token" and kind == "weight":
        raise ValueError("granularity 'per_token' applies to activations; use 'per_channel' for weights")
    raise ValueError(f"unsupported granularity {granularity!r} for {kind}")
