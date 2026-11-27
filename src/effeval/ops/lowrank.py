"""Low-rank q/k/v factorisation: SVD helpers, rank allocation, and installing the shared-down layout.

Numerics follow the released QSVD code (`fake_quant/svd_utils.py`, `grad_info_utils.py`). Installing the
layout relies on `Qwen3Attention.forward` (transformers modeling_qwen3.py, lines 244-246 on the host install)
calling `q_proj`, `k_proj` and `v_proj` exactly once each.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import torch
from torch import nn

from effeval.core.lowrank import (
    ARCHITECTURE_ATTR,
    DOWN_NAME,
    QKV_CANONS,
    factor_qkv_architecture,
)
from effeval.ops import sites


class LowRankProjection(nn.Module):
    """`up(down(x))` where `down` may be shared with sibling projections.

    `down` is held as a plain attribute, not a child module, so a shared factor's parameters appear once in
    `named_parameters()`; `up` is the registered child.
    """

    def __init__(self, down: nn.Linear, up: nn.Linear):
        super().__init__()
        object.__setattr__(self, "down", down)
        self.up = up

    @property
    def in_features(self) -> int:
        return self.down.in_features

    @property
    def out_features(self) -> int:
        return self.up.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def joint_whitened_svd(
    weights: Sequence[torch.Tensor], scale: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin SVD of the row-concatenated `weights`, columns multiplied by `scale` when given.

    Returns `(U, S, V)` in float32 with `W * scale = U diag(S) V^T`.
    """
    w = torch.cat([x.float() for x in weights], 0)
    if scale is not None:
        w = w * scale.view(1, -1)
    U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
    return U, S, Vt.T


def projected_gradient(
    U: torch.Tensor, V: torch.Tensor, grad: torch.Tensor, scale_inv: torch.Tensor | None
) -> torch.Tensor:
    """`diag(U^T (G * scale_inv) V)` as a vector, with the columns of `grad` multiplied by `scale_inv`."""
    g = grad.float()
    if scale_inv is not None:
        g = g * scale_inv.view(1, -1).to(torch.float32)
    return torch.sum(U * (g @ V), dim=0)


def global_top_k(scores: Sequence[torch.Tensor], k: int) -> list[list[int]]:
    """Indices of the `k` largest scores over all layers, per layer, in global descending-score order.

    Ties keep layer-major, index-ascending order (stable sort), matching the released code. Raises ValueError
    on NaN scores or when `k` exceeds the number of scores.
    """
    flat = torch.cat([s.detach().cpu().reshape(-1) for s in scores])
    if torch.isnan(flat).any():
        raise ValueError("scores contain NaN")
    if k > flat.numel():
        raise ValueError(f"k={k} exceeds the {flat.numel()} available scores")
    order = torch.sort(flat, descending=True, stable=True).indices[:k]
    starts = torch.tensor([0, *itertools.accumulate(s.numel() for s in scores)][:-1])
    layers = torch.bucketize(order, starts, right=True) - 1
    selected: list[list[int]] = [[] for _ in scores]
    for position, layer in zip(order.tolist(), layers.tolist(), strict=True):
        selected[layer].append(position - int(starts[layer]))
    return selected


def factor_weights(
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    indices: list[int],
    scale: torch.Tensor | None,
    beta: float,
    out_sizes: Sequence[int],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Factors of the selected singular triplets: `(down [r, d_in], ups [out_p, r])` cast to `dtype`.

    `up_p = U_p S^(1-beta)` and `down = S^beta V^T / scale`, so `up_p @ down` approximates the dense weight.
    `U` is split by rows into `out_sizes`. Raises ValueError for empty `indices`.
    """
    if not indices:
        raise ValueError("no singular values selected")
    idx = torch.as_tensor(indices, dtype=torch.long, device=U.device)
    u_sel, s_sel, v_sel = U[:, idx], S[idx], V[:, idx]
    if scale is not None:
        v_sel = v_sel / scale.view(-1, 1)

    def power(p: float) -> torch.Tensor:
        return s_sel.sqrt() if p == 0.5 else s_sel**p

    ups = [u.mul(power(1 - beta)).contiguous().to(dtype) for u in torch.split(u_sel, list(out_sizes), dim=0)]
    down = v_sel.t().mul(power(beta).view(-1, 1)).contiguous().to(dtype)
    return down, ups


def rotate_latent(down: nn.Linear, ups: Sequence[nn.Linear], matrix: torch.Tensor) -> None:
    """Rotate the latent space in place: `up <- up @ H`, `down <- H^T @ down`, computed in float64."""
    h = matrix.double().to(down.weight.device)
    for up in ups:
        up.weight.data = (up.weight.double() @ h).to(up.weight.dtype)
    down.weight.data = (h.T @ down.weight.double()).to(down.weight.dtype)


def install_shared_qkv(
    model: nn.Module, factors: Sequence[tuple[torch.Tensor, Sequence[torch.Tensor]]]
) -> None:
    """Replace each layer's q/k/v by `LowRankProjection`s sharing one `qkv_down` linear.

    `factors[l]` is `(down [r, d_in], [up_q, up_k, up_v])`. New linears take the original weights' device and
    dtype; an original bias moves to the up factor. Afterwards the model carries the factored layout.
    Raises ValueError if a layout is already attached or `factors` does not match the layers.
    """
    if getattr(model, ARCHITECTURE_ATTR, None) is not None:
        raise ValueError("model already carries a factored architecture")
    arch = sites.architecture(model)
    layers = sites.decoder_layers(model)
    if len(factors) != len(layers):
        raise ValueError(f"{len(factors)} factor sets for {len(layers)} layers")
    names = [canon.split(".")[1] for canon in QKV_CANONS]
    new_arch = factor_qkv_architecture(arch)
    for layer, (down_w, up_ws) in zip(layers, factors, strict=True):
        attention = layer.get_submodule(arch.attention_path)
        dense = [getattr(attention, name) for name in names]
        if len(up_ws) != len(dense):
            raise ValueError(f"expected {len(dense)} up factors, got {len(up_ws)}")
        ref = dense[0].weight
        rank, d_in = down_w.shape
        if d_in != dense[0].in_features:
            raise ValueError(f"down factor has {d_in} columns, the layer input is {dense[0].in_features}")
        down = nn.Linear(d_in, rank, bias=False, device=ref.device, dtype=ref.dtype)
        down.weight.data.copy_(down_w)
        projections = {}
        for name, proj, up_w in zip(names, dense, up_ws, strict=True):
            if tuple(up_w.shape) != (proj.out_features, rank):
                raise ValueError(f"{name} up factor {tuple(up_w.shape)} != ({proj.out_features}, {rank})")
            up = nn.Linear(
                rank, proj.out_features, bias=proj.bias is not None, device=ref.device, dtype=ref.dtype
            )
            up.weight.data.copy_(up_w)
            if proj.bias is not None:
                up.bias.data.copy_(proj.bias)
            projections[name] = LowRankProjection(down, up)
        setattr(attention, DOWN_NAME, down)
        for name, projection in projections.items():
            setattr(attention, name, projection)
    setattr(model, ARCHITECTURE_ATTR, new_arch)
