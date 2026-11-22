"""Number formats as quantize-dequantize operations on float32 tensors of scale units.

Shape symbols: N = number of scale units (rows, tokens, groups or blocks), K = elements per unit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from effeval.core.fragments import ClipSearch, parse_encoding


def resolve_special(encoding: str, special: str) -> str:
    """Map `special: auto` to the conventional code reservation of a float encoding."""
    if special != "auto":
        return special
    if encoding == "e5m2":
        return "ieee"
    if encoding == "e4m3":
        return "fn"
    return "none"


def fp_max(exponent_bits: int, mantissa_bits: int, special: str) -> float:
    """Largest finite value of an eXmY format.

    ieee: the all-ones exponent is reserved for inf/nan. fn: only all-ones exponent with all-ones mantissa is
    NaN. none: every code is finite.
    """
    bias = 2 ** (exponent_bits - 1) - 1
    if special == "ieee":
        return (2.0 - 2.0**-mantissa_bits) * 2.0 ** (2**exponent_bits - 2 - bias)
    if special == "fn":
        return (2.0 - 2.0 ** -(mantissa_bits - 1)) * 2.0 ** (2**exponent_bits - 1 - bias)
    if special == "none":
        return (2.0 - 2.0**-mantissa_bits) * 2.0 ** (2**exponent_bits - 1 - bias)
    raise ValueError(f"unknown special-value convention {special!r}")


def round_to_fp(x: torch.Tensor, exponent_bits: int, mantissa_bits: int, max_value: float) -> torch.Tensor:
    """Round float32 values to the nearest eXmY value (ties to even), saturating at +-max_value."""
    min_normal_exp = 2 - 2 ** (exponent_bits - 1)  # 1 - bias
    magnitude = x.abs()
    _, frexp_exp = torch.frexp(magnitude)  # magnitude = m * 2**frexp_exp with m in [0.5, 1)
    exp = (frexp_exp - 1).clamp(min=min_normal_exp)  # floor(log2 |x|); subnormals share the minimum exponent
    step = torch.ldexp(torch.ones_like(magnitude), exp - mantissa_bits)
    rounded = (torch.round(magnitude / step) * step).clamp(max=max_value)
    return torch.copysign(rounded, x)


@dataclass(frozen=True)
class IntGrid:
    """Per-unit integer grid: value = scale * (clamp(round(x / scale) + zero, qmin, qmax) - zero).

    With `offset` (float zero point) the grid is scale * clamp(round((x - offset) / scale), qmin, qmax)
    + offset instead, and `zero` is unused.
    """

    scale: torch.Tensor  # [N, 1]
    zero: torch.Tensor  # [N, 1]
    qmin: int
    qmax: int
    offset: torch.Tensor | None = None  # [N, 1]

    def apply(self, units: torch.Tensor) -> torch.Tensor:
        if self.offset is not None:
            q = torch.clamp(torch.round((units - self.offset) / self.scale), self.qmin, self.qmax)
            return self.scale * q + self.offset
        q = torch.clamp(torch.round(units / self.scale) + self.zero, self.qmin, self.qmax)
        return self.scale * (q - self.zero)


def int_grid(
    units: torch.Tensor,
    bits: int,
    symmetric: bool,
    clip_ratio: float,
    clip_search: ClipSearch | None = None,
    float_zero_point: bool = False,
) -> IntGrid:
    """Fit an integer grid to each row of units [N, K] (float32).

    Asymmetric: min-max range that always contains zero, op for op as the official GPTQ code
    (github.com/IST-DASLab/gptq quant.py `find_params`), so GPTQ/RTN weights are bit-identical to the paper.
    Symmetric: scale = absmax / (2^(bits-1) - 1), codes in [-2^(bits-1), 2^(bits-1) - 1].
    With `clip_search` the clip ratio of every row is chosen by `search_int_grid` and `clip_ratio` must be 1.
    With `float_zero_point` (asymmetric only) the range is exactly [min, max] of the row, not extended to
    contain zero, scale = (max - min) / (2^bits - 1) (1 for a constant row), and the offset is the row min,
    as in KIVI's `quant_and_pack_kcache`; `clip_ratio` must be 1 and `clip_search` is unsupported.
    """
    if float_zero_point:
        if symmetric or clip_ratio != 1.0 or clip_search is not None:
            raise ValueError("float_zero_point needs an asymmetric grid, clip_ratio 1.0 and no clip_search")
        maxq = 2**bits - 1
        low = units.min(1, keepdim=True)[0]  # [N, 1]
        scale = (units.max(1, keepdim=True)[0] - low) / maxq
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return IntGrid(scale=scale, zero=torch.zeros_like(scale), qmin=0, qmax=maxq, offset=low)
    if clip_search is not None:
        if clip_ratio != 1.0:
            raise ValueError("clip_search replaces clip_ratio; clip_ratio must be 1.0")
        return search_int_grid(units, bits, symmetric, clip_search)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        absmax = units.abs().amax(dim=1, keepdim=True) * clip_ratio  # [N, 1]
        scale = torch.where(absmax > 0, absmax / qmax, torch.ones_like(absmax))
        return IntGrid(scale=scale, zero=torch.zeros_like(scale), qmin=-qmax - 1, qmax=qmax)
    maxq = 2**bits - 1
    zeros = torch.zeros(units.shape[0], device=units.device)
    xmin = torch.minimum(units.min(1)[0], zeros) * clip_ratio  # [N]
    xmax = torch.maximum(units.max(1)[0], zeros) * clip_ratio  # [N]
    both_zero = (xmin == 0) & (xmax == 0)
    xmin[both_zero] = -1
    xmax[both_zero] = +1
    scale = ((xmax - xmin) / maxq).unsqueeze(1)  # [N, 1]
    zero = torch.round(-xmin.unsqueeze(1) / scale)  # [N, 1]
    return IntGrid(scale=scale, zero=zero, qmin=0, qmax=maxq)


def search_int_grid(units: torch.Tensor, bits: int, symmetric: bool, search: ClipSearch) -> IntGrid:
    """Per row of units [N, K], the grid with the clip ratio of `search` that minimises sum |q - x|^norm.

    Op for op as QuaRot's `WeightQuantizer.find_params` with `mse=True`: candidate i uses ratio 1 - i / grid,
    a candidate replaces the best one only when its error is strictly smaller.
    """
    best_error = torch.full((units.shape[0], 1), float("inf"), device=units.device)
    best: IntGrid | None = None
    for i in range(int(search.max_shrink * search.grid)):
        candidate = int_grid(units, bits, symmetric, 1 - i / search.grid)
        error = (candidate.apply(units) - units).abs().pow(search.norm).sum(1, keepdim=True)  # [N, 1]
        better = error < best_error
        best_error = torch.where(better, error, best_error)
        if best is None:
            best = candidate
        else:
            best = IntGrid(
                scale=torch.where(better, candidate.scale, best.scale),
                zero=torch.where(better, candidate.zero, best.zero),
                qmin=candidate.qmin,
                qmax=candidate.qmax,
            )
    if best is None:
        raise ValueError("clip_search tries no candidate: max_shrink * grid must be at least 1")
    return best


def fp_scaled(units: torch.Tensor, encoding: str, special: str, clip_ratio: float) -> torch.Tensor:
    """eXmY values with a float32 scale per unit mapping the unit's absmax to the format's max value."""
    exponent_bits, mantissa_bits = parse_encoding(encoding)
    max_value = fp_max(exponent_bits, mantissa_bits, resolve_special(encoding, special))
    absmax = units.abs().amax(dim=1, keepdim=True) * clip_ratio  # [N, 1]
    scale = torch.where(absmax > 0, absmax / max_value, torch.ones_like(absmax))
    return round_to_fp(units / scale, exponent_bits, mantissa_bits, max_value) * scale


def mx_e8m0(blocks: torch.Tensor, element: str) -> torch.Tensor:
    """OCP Microscaling: per block, scale = 2^(floor(log2(absmax)) - emax_elem); blocks: [N, block_size]."""
    exponent_bits, mantissa_bits = parse_encoding(element)
    max_value = fp_max(exponent_bits, mantissa_bits, resolve_special(element, "auto"))
    _, max_frexp = torch.frexp(torch.tensor(max_value))
    emax_elem = int(max_frexp) - 1  # exponent of the element format's largest normal
    absmax = blocks.abs().amax(dim=1, keepdim=True)  # [N, 1]
    _, absmax_frexp = torch.frexp(absmax)
    shared_exp = (absmax_frexp - 1 - emax_elem).clamp(-127, 127)
    scale = torch.where(absmax > 0, torch.ldexp(torch.ones_like(absmax), shared_exp), torch.ones_like(absmax))
    return round_to_fp(blocks / scale, exponent_bits, mantissa_bits, max_value) * scale


def mx_e4m3(blocks: torch.Tensor, element: str, tensor_absmax: torch.Tensor) -> torch.Tensor:
    """Two-level NVFP4-style scaling: float32 tensor scale x e4m3 block scale; blocks: [N, block_size]."""
    exponent_bits, mantissa_bits = parse_encoding(element)
    max_value = fp_max(exponent_bits, mantissa_bits, resolve_special(element, "auto"))
    e4m3_max = fp_max(4, 3, "fn")
    if float(tensor_absmax) == 0.0:
        return torch.zeros_like(blocks)
    tensor_scale = tensor_absmax / (e4m3_max * max_value)
    block_absmax = blocks.abs().amax(dim=1, keepdim=True)  # [N, 1]
    block_scale = round_to_fp(block_absmax / max_value / tensor_scale, 4, 3, e4m3_max)
    scale = block_scale * tensor_scale
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return round_to_fp(blocks / scale, exponent_bits, mantissa_bits, max_value) * scale
