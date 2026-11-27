"""Hybrid FFN ops: SVD/dense channel split of the MLP's gate and up projections.

Numerics follow the released LSVD code (`fake_quant/utils/hybrid_ffn_utils.py`): the Phase-1 greedy channel
selection `select_svd_channels_svd_aware`, the Phase-2 whitened SVD of `build_hybrid_ffn` /
`_make_svd_linear`, and the block-Hadamard latent rotation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from effeval.core.lowrank import (
    ARCHITECTURE_ATTR,
    FFN_CANONS,
    hybrid_ffn_architecture,
)
from effeval.ops import sites
from effeval.ops.lowrank import cholesky_lower

DOWN_CANON = "mlp.down_proj"
MAX_SELECTION_ROWS = 8192


class HybridProjection(nn.Module):
    """A gate or up projection whose output channels are `[svd_up(svd_down(x)), dense(x)]`.

    With no SVD branch it is `dense(x)`. `svd_down` and `svd_up` are both given or both None.
    """

    def __init__(self, dense: nn.Linear, svd_down: nn.Linear | None, svd_up: nn.Linear | None):
        super().__init__()
        if (svd_down is None) != (svd_up is None):
            raise ValueError("svd_down and svd_up must both be given or both be None")
        self.dense = dense
        if svd_down is not None:
            if svd_down.out_features != svd_up.in_features or svd_down.in_features != dense.in_features:
                raise ValueError("svd branch shapes do not match the dense branch")
            self.svd_down = svd_down
            self.svd_up = svd_up

    @property
    def in_features(self) -> int:
        return self.dense.in_features

    @property
    def out_features(self) -> int:
        extra = self.svd_up.out_features if hasattr(self, "svd_up") else 0
        return extra + self.dense.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "svd_up"):
            return self.dense(x)
        return torch.cat([self.svd_up(self.svd_down(x)), self.dense(x)], dim=-1)


def _power2_blocks(n: int) -> list[int]:
    blocks = []
    remaining = int(n)
    while remaining > 0:
        block = 1 << (remaining.bit_length() - 1)
        blocks.append(block)
        remaining -= block
    return blocks


def _hadamard_power2_lastdim(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 1:
        return x
    y = x.reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        y0 = y[:, :, 0, :].clone()
        y1 = y[:, :, 1, :].clone()
        y[:, :, 0, :] = y0 + y1
        y[:, :, 1, :] = y0 - y1
        y = y.reshape(-1, n)
        h *= 2
    return (y.reshape(x.shape) / math.sqrt(n)).to(x.dtype)


def block_hadamard_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal block-Hadamard transform of the last dimension.

    The dimension is split greedily into power-of-two blocks (24 -> 16 + 8) and each block gets a normalised
    Sylvester Hadamard transform, computed in float32 and returned in `x.dtype`.
    """
    n = x.shape[-1]
    if n <= 1:
        return x
    out = x.clone()
    start = 0
    for block in _power2_blocks(n):
        end = start + block
        out[..., start:end] = _hadamard_power2_lastdim(out[..., start:end])
        start = end
    return out


def _quantize_rows_sym(w: torch.Tensor, bits: int) -> torch.Tensor:
    maxq = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / maxq
    q = torch.clamp(torch.round(w / scale), -(maxq + 1), maxq)
    return q * scale


def _fake_quantize_tokens_sym(x: torch.Tensor, bits: int, clip_ratio: float) -> torch.Tensor:
    if bits >= 16:
        return x
    maxq = float(2 ** (bits - 1) - 1)
    flat = x.float().reshape(-1, x.shape[-1])
    zeros = torch.zeros(flat.shape[0], device=flat.device, dtype=flat.dtype)
    xmin = torch.minimum(flat.min(dim=1).values, zeros) * clip_ratio
    xmax = torch.maximum(flat.max(dim=1).values, zeros) * clip_ratio
    xmax = torch.maximum(xmin.abs(), xmax)
    scale = (xmax / maxq).unsqueeze(1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / scale), -(maxq + 1), maxq)
    return (q * scale).reshape(x.shape).to(x.dtype)


def strided_rows(total: int, max_rows: int) -> torch.Tensor:
    """Row indices `arange(total)[::total // max_rows][:max_rows]`, or every row when `total <= max_rows`."""
    rows = torch.arange(total)
    if total <= max_rows:
        return rows
    return rows[:: max(1, total // max_rows)][:max_rows]


def whitening_factor(x: torch.Tensor, damping: float, gain: torch.Tensor | None) -> torch.Tensor:
    """Lower Cholesky factor (float64) of the damped input Gram of `x [N, d]`.

    `H = x^T x / N`, `lambda = damping * mean(diag(H) / gain^2)` (floored at 1e-8) and the factor is that of
    `H + lambda * diag(gain^2)`. With `gain=None` this is the released `H + damping * mean(diag H) * I`.
    With a gain, `x` is the unnormalised input `x_n * gain`, and the result equals `diag(gain)` times the
    factor the same recipe gives on `x_n`. Raises ValueError for a zero gain or a non-positive-definite Gram.
    """
    x = x.reshape(-1, x.shape[-1]).double()
    gram = x.T @ x / x.shape[0]
    g = torch.ones(x.shape[-1], dtype=torch.float64, device=x.device) if gain is None else gain.double()
    if bool((g == 0).any()):
        raise ValueError("whitening needs non-zero gains")
    g2 = g * g
    damp = damping * (gram.diagonal() / g2).mean().clamp(min=1e-8)
    gram = gram + torch.diag(damp * g2)
    return cholesky_lower(gram)


def rotated_selection_inputs(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    x: torch.Tensor,
    gain: torch.Tensor,
    rotation: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The FFN weights and input as the rotated model stores them, in float32.

    With `rotation = R` (residual basis) and the norm gain folded into gate/up: `(W_gate * gain) @ R`,
    `(W_up * gain) @ R`, `R^T @ W_down` and `(x / gain) @ R`, each computed in float64 and rounded to `dtype`.
    Returns `(w_gate, w_up, w_down, x)`.
    """
    r = rotation.double()
    g = gain.double()
    gate = ((w_gate.double() * g) @ r).to(dtype).float()
    up = ((w_up.double() * g) @ r).to(dtype).float()
    down = (r.T @ w_down.double()).to(dtype).float()
    inputs = ((x.double() / g) @ r).to(dtype).float()
    return gate, up, down, inputs


def _split_sigma(
    U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor, rank: int, beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(A [n, rank], B [rank, d])` with the first `rank` triplets; `A = U S^(1-beta)`, `B = Vt S^beta`."""
    U_r, S_r, Vt_r = U[:, :rank], S[:rank], Vt[:rank, :]
    if beta == 0.0:
        return U_r.mul(S_r).contiguous(), Vt_r.contiguous()
    if beta == 0.5:
        return U_r.mul(S_r.sqrt()).contiguous(), Vt_r.mul(S_r.sqrt().view(-1, 1)).contiguous()
    raise ValueError(f"beta must be 0 or 0.5, got {beta}")


def _rotate_latent(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if a.shape[1] <= 1:
        return a, b
    return block_hadamard_lastdim(a), block_hadamard_lastdim(b.T).T


def _channel_margins(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    act_scale: torch.Tensor,
    rank: int,
    bits: int,
) -> torch.Tensor:
    down_norm = w_down.norm(dim=0)
    down_norm = down_norm / (down_norm.mean() + 1e-6)
    d_in = w_gate.shape[1]
    n_levels = 2 ** (bits - 1) - 1

    def proxy(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w = w * act_scale.to(w.device, dtype=w.dtype).unsqueeze(0)
        err_svd = w.square().sum(dim=1) * (1.0 - min(1.0, rank / d_in))
        step = w.abs().amax(dim=1).clamp(min=1e-9) / n_levels
        return err_svd, step.square() * (d_in / 12.0)

    svd_g, quant_g = proxy(w_gate)
    svd_u, quant_u = proxy(w_up)
    return down_norm * ((quant_g + quant_u) - (svd_g + svd_u))


def svd_aware_order(
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    x: torch.Tensor,
    rank: int,
    w_bits: int,
    a_bits: int,
    a_clip_ratio: float,
    beta: float,
    chol: torch.Tensor | None,
    latent_rotation: bool,
) -> list[int]:
    """Channels in the order the released Phase-1 `svd_aware` greedy moves them from SVD to RTN.

    Starts from the quantized-SVD branch of every channel (rank `rank`, whitened by the lower Cholesky factor
    `chol` when given), visits channels in ascending margin order and accepts a channel when switching it to
    the RTN-quantized dense branch strictly lowers the FFN output MSE on the first 8192 rows of `x [N, d]`.
    Weights are `[d_ffn, d]` (gate, up) and `[d_out, d_ffn]` (down). Returns the accepted channels in order.
    """
    d_in = w_gate.shape[1]
    d_out = w_down.shape[0]
    wg, wu, wd = w_gate.float(), w_up.float(), w_down.float()
    x = x.float().reshape(-1, d_in)[:MAX_SELECTION_ROWS].contiguous()
    n = x.shape[0]

    with torch.no_grad():
        y_ref = (F.silu(x @ wg.T) * (x @ wu.T)) @ wd.T
        x_q = _fake_quantize_tokens_sym(x, a_bits, a_clip_ratio)

        if chol is not None:
            l_d = chol.double()
            l_inv = torch.linalg.inv(l_d)
            decomposed = []
            for w in (wg, wu):
                U, S, Vt_w = torch.linalg.svd((w.double() @ l_d).float(), full_matrices=False)
                decomposed.append((U, S, (Vt_w.double() @ l_inv).float()))
        else:
            decomposed = [torch.linalg.svd(w, full_matrices=False) for w in (wg, wu)]

        hidden_parts = []
        for U, S, Vt in decomposed:
            a, b = _split_sigma(U, S, Vt, rank, beta)
            if latent_rotation:
                a, b = _rotate_latent(a, b)
            a_q, b_q = _quantize_rows_sym(a, w_bits), _quantize_rows_sym(b, w_bits)
            latent = _fake_quantize_tokens_sym(x_q @ b_q.T, a_bits, a_clip_ratio)
            hidden_parts.append(latent @ a_q.T)
        g_svd, u_svd = hidden_parts
        h_svd = F.silu(g_svd) * u_svd
        y_svd = h_svd @ wd.T
        baseline_mse = float(((y_svd - y_ref) ** 2).mean().item())

        g_q = x_q @ _quantize_rows_sym(wg, w_bits).T
        u_q = x_q @ _quantize_rows_sym(wu, w_bits).T
        delta_h = F.silu(g_q) * u_q - h_svd

        if chol is not None:
            wg_m = (wg.double() @ l_d).float()
            wu_m = (wu.double() @ l_d).float()
            act_scale = (x.double() @ l_d).float().abs().mean(dim=0) + 1e-6
        else:
            wg_m, wu_m = wg, wu
            act_scale = x.abs().mean(dim=0) + 1e-6
        margins = _channel_margins(wg_m, wu_m, wd, act_scale, rank, w_bits)
        candidates = torch.argsort(margins).tolist()

        current_mse = baseline_mse
        score = (y_svd - y_ref).contiguous() @ wd
        dh_sq = (delta_h**2).mean(dim=0)
        wd_sq = (wd**2).mean(dim=0)
        norm = float(n * d_out)
        accepted: list[int] = []
        for ch in candidates:
            dh = delta_h[:, ch]
            cross = float((dh @ score[:, ch]).item()) / norm
            trial_mse = current_mse + 2.0 * cross + float(dh_sq[ch].item()) * float(wd_sq[ch].item())
            if trial_mse < current_mse:
                accepted.append(ch)
                score.add_(dh.unsqueeze(1) * (wd[:, ch] @ wd).unsqueeze(0))
                current_mse = trial_mse
    return accepted


def selection_is_reused(order_len: int, d_ffn: int, n_svd: int, selection_n_svd: int) -> bool:
    """Whether the released code reuses its stored selection for `n_svd` SVD channels.

    The selection ran with `selection_n_svd` SVD channels and accepted `order_len` channels, so it stored
    `d_ffn - min(order_len, d_ffn - selection_n_svd)` SVD channels; it is reused when that equals `n_svd`.
    """
    return d_ffn - min(order_len, d_ffn - selection_n_svd) == n_svd


def partition_channels(
    order: Sequence[int], d_ffn: int, n_svd: int, selection_n_svd: int | None = None
) -> list[int]:
    """Channel permutation `svd_idx + quant_idx` for `n_svd` SVD channels out of `d_ffn`.

    The quantized channels are the first `d_ffn - n_svd` of `order` (padded with the channels `order` never
    lists, in index order, when it is shorter); the SVD channels are the rest in ascending order. The
    quantized channels keep the acceptance order, except when `selection_n_svd` is given and
    `selection_is_reused` holds: the released code then reuses its stored selection, whose quantized channels
    are sorted.
    Raises ValueError for an invalid `n_svd` or an `order` with duplicates or out-of-range channels.
    """
    if not 0 < n_svd < d_ffn:
        raise ValueError(f"n_svd must be in (0, {d_ffn}), got {n_svd}")
    order = list(order)
    if len(set(order)) != len(order) or any(not 0 <= c < d_ffn for c in order):
        raise ValueError("order must list distinct channels in range(d_ffn)")
    n_quant = d_ffn - n_svd
    if n_quant <= len(order):
        quant = order[:n_quant]
    else:
        listed = set(order)
        quant = order + [c for c in range(d_ffn) if c not in listed][: n_quant - len(order)]
    if selection_n_svd is not None and selection_is_reused(len(order), d_ffn, n_svd, selection_n_svd):
        quant = sorted(quant)
    quant_set = set(quant)
    return [c for c in range(d_ffn) if c not in quant_set] + quant


def svd_branch(
    w_rows: torch.Tensor,
    rank: int,
    chol: torch.Tensor,
    beta: float,
    latent_rotation: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank-`rank` whitened SVD factors `(up [n, rank], down [rank, d])` of `w_rows [n, d]`, in `dtype`.

    `chol` is the lower Cholesky factor `L` of the input Gram: the SVD is taken of `w_rows @ L` and the right
    factor is mapped back with `inv(L)^T`. `beta` splits the singular values (0: all into `up`, 0.5: square
    root into each); `latent_rotation` applies a block-Hadamard to the latent. Raises ValueError for another
    `beta` or a `rank` outside `[1, min(n, d)]`.
    """
    n, d = w_rows.shape
    if not 1 <= rank <= min(n, d):
        raise ValueError(f"rank must be in [1, {min(n, d)}], got {rank}")
    l32 = chol.float()
    whitened = (w_rows.float().double() @ l32.double()).float()
    U, S, Vt = torch.linalg.svd(whitened, full_matrices=False)
    V = Vt[:rank].T.contiguous()
    V = (torch.linalg.inv(l32).double().T @ V.double()).float()
    a, b = _split_sigma(U, S, V.T, rank, beta)
    if latent_rotation:
        a, b = _rotate_latent(a, b)
    return a.to(dtype).contiguous(), b.to(dtype).contiguous()


@dataclass(frozen=True)
class HybridLayer:
    """One layer's hybrid FFN: channel permutation, SVD channel count, `(up, down)` factors of gate and up."""

    perm: list[int]
    n_svd: int
    gate: tuple[torch.Tensor, torch.Tensor]
    up: tuple[torch.Tensor, torch.Tensor]


def _linear_like(ref: nn.Linear, weight: torch.Tensor) -> nn.Linear:
    out_features, in_features = weight.shape
    device, dtype = ref.weight.device, ref.weight.dtype
    linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=dtype)
    linear.weight.data.copy_(weight)
    return linear


def install_hybrid_ffn(model: nn.Module, layers: Sequence[HybridLayer | None]) -> None:
    """Replace every layer's gate and up by `HybridProjection`s and permute the hybrid layers' channels.

    `layers[i]` is None for a skipped layer, whose projections keep their original linear as `dense`.
    In a hybrid layer the gate/up rows follow `perm`, rows `n_svd:` form the dense branch, and the columns
    of down_proj follow `perm`. The attached layout is extended with `hybrid_ffn_architecture`.
    Raises ValueError if the layout already has FFN branches, if no layer is hybrid, if gate/up carry a
    bias, or if a `HybridLayer` does not fit the layer's shapes.
    """
    arch = sites.architecture(model)
    decoder = sites.decoder_layers(model)
    if len(layers) != len(decoder):
        raise ValueError(f"{len(layers)} entries for {len(decoder)} layers")
    hybrid_ids = [i for i, entry in enumerate(layers) if entry is not None]
    if not hybrid_ids:
        raise ValueError("no hybrid layer given")
    new_arch = hybrid_ffn_architecture(arch, hybrid_ids)
    paths = {canon: path for path, canon in arch.linears.items() if canon in (*FFN_CANONS, DOWN_CANON)}

    for index, (layer, entry) in enumerate(zip(decoder, layers, strict=True)):
        parents = {canon: layer.get_submodule(path.rsplit(".", 1)[0]) for canon, path in paths.items()}
        names = {canon: path.rsplit(".", 1)[1] for canon, path in paths.items()}
        projections = {canon: getattr(parents[canon], names[canon]) for canon in FFN_CANONS}
        for canon, linear in projections.items():
            if linear.bias is not None:
                raise ValueError(f"layer {index} {canon} has a bias; hybrid FFN does not support it")
        if entry is not None:
            d_ffn, d_in = projections[FFN_CANONS[0]].weight.shape
            down_parent, down_name = parents[DOWN_CANON], names[DOWN_CANON]
            down = getattr(down_parent, down_name)
            if sorted(entry.perm) != list(range(d_ffn)) or not 0 < entry.n_svd < d_ffn:
                raise ValueError(f"layer {index}: perm must permute range({d_ffn}) and 0 < n_svd < {d_ffn}")
            perm = torch.tensor(entry.perm, dtype=torch.long)
        for canon, linear in projections.items():
            if entry is None:
                hybrid = HybridProjection(linear, None, None)
            else:
                up_w, down_w = entry.gate if canon == FFN_CANONS[0] else entry.up
                rank = down_w.shape[0]
                if tuple(up_w.shape) != (entry.n_svd, rank) or down_w.shape[1] != d_in:
                    raise ValueError(
                        f"layer {index} {canon}: factors {tuple(up_w.shape)} / {tuple(down_w.shape)} do not "
                        f"fit n_svd={entry.n_svd}, d={d_in}"
                    )
                dense_rows = linear.weight.data[perm][entry.n_svd :]
                hybrid = HybridProjection(
                    _linear_like(linear, dense_rows),
                    _linear_like(linear, down_w),
                    _linear_like(linear, up_w),
                )
            setattr(parents[canon], names[canon], hybrid)
        if entry is not None:
            down.weight.data = down.weight.data[:, perm].contiguous()
    setattr(model, ARCHITECTURE_ATTR, new_arch)
