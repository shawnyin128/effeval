"""Grouped pairwise Givens rotations (ParoQuant, github.com/z-lab/paroquant@8e5ac77).

Each rotation step pairs up the channels of every group of `group_size` input channels and rotates every pair
(a, b) by its own angle: a' = cos*a + sin*b, b' = cos*b - sin*a (kernels/paroquant_kernels/interface.py,
`RotateTensorFunc.backward`). A rotation sequence applies R steps in order.

Shape symbols: R rotations, H input channels, G = H / group_size groups, B rows.
Layout of the pair data: `pairs` [R, H] holds, per group, the in-group channel indices
(i0, j0, i1, j1, ...); pair k of group g rotates channels g*group_size + i_k and g*group_size + j_k by
`angles[r, g * group_size / 2 + k]`.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.promote_types(dtype, torch.float32)


def _pair_index(idx: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat channel indices (first, second) of every pair of one rotation step; idx is [H]."""
    groups = idx.numel() // group_size
    offsets = torch.arange(groups, device=idx.device, dtype=torch.int64).unsqueeze(1) * group_size
    grouped = idx.long().view(groups, group_size)
    first = grouped[:, 0:group_size:2] + offsets  # [G, group_size / 2]
    second = grouped[:, 1:group_size:2] + offsets
    return first.reshape(-1), second.reshape(-1)


def _rotate_step(x: torch.Tensor, idx: torch.Tensor, theta: torch.Tensor, group_size: int) -> torch.Tensor:
    """One rotation step on x [B, H] with idx [H] and theta [H / 2]."""
    first, second = _pair_index(idx, group_size)
    a = x[:, first]  # [B, H / 2]
    b = x[:, second]
    cos = theta.cos()
    sin = theta.sin()
    return x.index_copy(1, first, cos * a + sin * b).index_copy(1, second, cos * b - sin * a)


class _PairwiseRotate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pairs, angles, group_size):
        dtype = x.dtype
        t = x.reshape(-1, x.shape[-1]).to(_compute_dtype(dtype))
        theta = angles.to(t.dtype)
        for step in range(pairs.shape[0]):
            t = _rotate_step(t, pairs[step], theta[step], group_size)
        ctx.group_size = group_size
        ctx.save_for_backward(pairs, angles, t)
        return t.reshape(x.shape).to(dtype)

    @staticmethod
    def backward(ctx, grad_out):
        pairs, angles, y = ctx.saved_tensors
        group_size = ctx.group_size
        compute = y.dtype
        theta = angles.to(compute)
        g = grad_out.reshape(-1, y.shape[-1]).to(compute)
        t = y
        grad_theta = torch.zeros_like(theta)
        for step in range(pairs.shape[0] - 1, -1, -1):
            t_prev = _rotate_step(t, pairs[step], -theta[step], group_size)
            g_prev = _rotate_step(g, pairs[step], -theta[step], group_size)
            if ctx.needs_input_grad[2]:
                first, second = _pair_index(pairs[step], group_size)
                shape = (t_prev.shape[0], -1, group_size // 2)
                a = t_prev[:, first].view(shape)  # [B, G, group_size / 2]
                b = t_prev[:, second].view(shape)
                ga = g[:, first].view(shape)
                gb = g[:, second].view(shape)
                grad_s = (ga * b - gb * a).sum(dim=0)  # [G, group_size / 2]
                grad_c = (ga * a + gb * b).sum(dim=0)
                theta_groups = theta[step].view(-1, group_size // 2)
                grad_theta[step] = (grad_s * theta_groups.cos() - grad_c * theta_groups.sin()).reshape(-1)
            t, g = t_prev, g_prev
        grad_x = g.reshape(grad_out.shape).to(grad_out.dtype) if ctx.needs_input_grad[0] else None
        grad_angles = grad_theta.to(angles.dtype) if ctx.needs_input_grad[2] else None
        return grad_x, None, grad_angles, None


def pairwise_rotate(
    x: torch.Tensor, pairs: torch.Tensor, angles: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Apply the R rotation steps in order to the last dimension of `x` [..., H]; returns the same shape.

    `pairs` is [R, H] (integer in-group channel indices), `angles` [R, H / 2]. Arithmetic runs in float32 (or
    float64 for float64 input) and the result is cast back to `x.dtype`. Differentiable in `x` and `angles`.
    """
    if pairs.shape != (angles.shape[0], x.shape[-1]) or angles.shape[1] * 2 != x.shape[-1]:
        raise ValueError(
            f"pairs {tuple(pairs.shape)} and angles {tuple(angles.shape)} do not fit x {tuple(x.shape)}"
        )
    if x.shape[-1] % group_size != 0 or group_size % 2 != 0:
        raise ValueError(f"group_size {group_size} must be even and divide {x.shape[-1]}")
    return _PairwiseRotate.apply(x, pairs, angles, group_size)


def pairwise_rotate_inverse(
    x: torch.Tensor, pairs: torch.Tensor, angles: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Inverse of `pairwise_rotate`: the steps in reverse order with negated angles."""
    return pairwise_rotate(x, torch.flip(pairs, dims=[0]), -torch.flip(angles, dims=[0]), group_size)


def _independent_channel_pairs(
    pairs: list[tuple[int, int]], dim: int, num_rotations: int, num_pairs_each: int
) -> list[list[tuple[int, int]]]:
    """Greedy split of a pair list into `num_rotations` lists of channel-disjoint pairs (<= num_pairs_each).

    A pair used by one rotation is not used again by a later one.
    """
    available = np.ones((dim, dim), dtype=bool)
    np.fill_diagonal(available, False)
    rotations_pairs = []
    for _ in range(num_rotations):
        independent_pairs: list[tuple[int, int]] = []
        available_in_rotation = available.copy()
        for i, j in pairs:
            if len(independent_pairs) == num_pairs_each:
                break
            if not available_in_rotation[i, j]:
                continue
            independent_pairs.append((i, j))
            available_in_rotation[i, :] = False
            available_in_rotation[j, :] = False
            available_in_rotation[:, i] = False
            available_in_rotation[:, j] = False
            available[i, j] = False
            available[j, i] = False
        rotations_pairs.append(independent_pairs)
    return rotations_pairs


def random_rotation_pairs(
    num_groups: int, group_size: int, num_rotations: int, num_pairs_factor: float, seed: int
) -> list[list[tuple[int, int]]]:
    """Per rotation, the (global channel index) pairs of all groups: random channel-disjoint pairs per group.

    Each group shuffles its group_size*(group_size-1)/2 candidate pairs with one shared `random.Random(seed)`
    (groups in order), then every rotation takes the first `int(group_size * num_pairs_factor)` pairs that
    share no channel with an earlier pick of the same rotation and were not picked by an earlier rotation.
    """
    rand = random.Random(seed)
    sorted_pairs: list[list[tuple[int, int]]] = []
    for _ in range(num_groups):
        candidates = [(i, j) for i in range(group_size) for j in range(i + 1, group_size)]
        rand.shuffle(candidates)
        sorted_pairs.append(candidates)

    pairs_k_groups: list[list[tuple[int, int]]] = [[] for _ in range(num_rotations)]
    num_pairs_per_group = int(group_size * num_pairs_factor)
    for group in range(num_groups):
        offset = group * group_size
        per_rotation = _independent_channel_pairs(
            sorted_pairs[group], group_size, num_rotations, num_pairs_per_group
        )
        for rotation in range(num_rotations):
            pairs_k_groups[rotation].extend((c1 + offset, c2 + offset) for c1, c2 in per_rotation[rotation])
    return pairs_k_groups


def align_pairs(pairs: list[tuple[int, int]], group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Kernel layout of one rotation's pairs: (in-group indices [H] int32, dummy-pair mask [H / 2] bool).

    `pairs` lists global channel pairs, group by group in ascending order. A group with fewer than
    group_size / 2 pairs is completed with dummy pairs over its unused channels (mask True); a pair that
    reuses a channel of its group raises ValueError.
    """
    pair_groups = []
    mask_groups = []
    group = 0
    pointer = 0
    while pointer < len(pairs):
        used = [False] * group_size
        count = 0
        group_pairs = []
        group_mask = []
        while count < group_size // 2:
            if (
                pointer < len(pairs)
                and pairs[pointer][0] - group * group_size < group_size
                and pairs[pointer][1] - group * group_size < group_size
            ):
                first, second = pairs[pointer]
                if used[first % group_size] or used[second % group_size]:
                    raise ValueError("a pair reuses a channel of its group")
                used[first % group_size] = used[second % group_size] = True
                group_pairs.append((first, second))
                group_mask.append(False)
                pointer += 1
            else:
                free = [i for i in range(group_size) if not used[i]]
                if len(free) < 2:
                    raise ValueError("cannot complete the group with a dummy pair")
                used[free[0]] = used[free[1]] = True
                group_pairs.append((free[0], free[1]))
                group_mask.append(True)
            count += 1
        group += 1
        pair_groups.append(group_pairs)
        mask_groups.append(group_mask)
    flat = torch.tensor(
        [c for group_pairs in pair_groups for pair in group_pairs for c in pair], dtype=torch.int32
    )
    mask = torch.tensor([m for group_mask in mask_groups for m in group_mask], dtype=torch.bool)
    return flat % group_size, mask
