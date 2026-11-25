"""Per-unit k-means codebooks for weight matrices.

Shape symbols: G codebook units, S values per unit, K centroids per unit, O/D rows/columns of a weight.
A unit is one row of the weight (`group_size` None) or `group_size` consecutive values of a row. Rows whose
length is not a multiple of group_size are zero-padded; `mask` marks the real values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from effeval.core.fragments import CodebookFormat


def to_units(weight: torch.Tensor, group_size: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a [O, D] weight into units [G, S] and the boolean mask of real values [G, S].

    A group_size of None or at least D makes every row one unit.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a [out, in] weight, got shape {tuple(weight.shape)}")
    columns = weight.shape[1]
    if group_size is None or group_size >= columns:
        return weight, torch.ones_like(weight, dtype=torch.bool)
    pad = -columns % group_size
    mask = torch.ones_like(weight, dtype=torch.bool)
    if pad:
        weight = F.pad(weight, (0, pad))
        mask = F.pad(mask, (0, pad), value=False)
    return weight.reshape(-1, group_size), mask.reshape(-1, group_size)


def from_units(units: torch.Tensor, group_size: int | None, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `to_units`: [G, S] back to the [O, D] weight shape, dropping padding."""
    rows, columns = shape
    if group_size is None or group_size >= columns:
        return units
    return units.reshape(rows, -1)[:, :columns]


def _nearest(units: torch.Tensor, mask: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Index of the closest centroid per value [G, S] (the first one on ties); padding gets index 0."""
    distance = (units.unsqueeze(2) - centroids.unsqueeze(1)).pow(2)
    return distance.masked_fill(~mask.unsqueeze(2), float("inf")).argmin(dim=2)


def _member_sum(values: torch.Tensor, mask: torch.Tensor, assignments: torch.Tensor, k: int):
    """Per-unit sum and count of the real values of each cluster: two [G, K] tensors."""
    g = values.shape[0]
    index = torch.arange(g, device=values.device).unsqueeze(1) * k + assignments.long()
    weight = mask.to(values.dtype)
    sums = torch.zeros(g * k, dtype=values.dtype, device=values.device)
    counts = torch.zeros_like(sums)
    sums.scatter_add_(0, index.reshape(-1), (values * weight).reshape(-1))
    counts.scatter_add_(0, index.reshape(-1), weight.reshape(-1))
    return sums.view(g, k), counts.view(g, k)


def _kmeans_pp(units: torch.Tensor, mask: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """k-means++ seeding per unit: [G, K] values drawn with probability proportional to squared distance."""
    g = units.shape[0]
    rows = torch.arange(g, device=units.device)
    generator = torch.Generator(device=units.device)
    generator.manual_seed(seed)
    centroids = torch.empty((g, k), device=units.device, dtype=units.dtype)

    first = mask.to(units.dtype)
    first = first / (first.sum(dim=1, keepdim=True) + 1e-8)
    centroids[:, 0] = units[rows, torch.multinomial(first, 1, generator=generator).squeeze(1)]
    nearest = (units - centroids[:, :1]).pow(2).masked_fill(~mask, 0.0)
    for i in range(1, k):
        probability = nearest / (nearest.sum(dim=1, keepdim=True) + 1e-8)
        centroids[:, i] = units[rows, torch.multinomial(probability, 1, generator=generator).squeeze(1)]
        distance = (units - centroids[:, i : i + 1]).pow(2).masked_fill(~mask, float("inf"))
        nearest = torch.minimum(nearest, distance)
    return centroids


def kmeans(
    units: torch.Tensor, mask: torch.Tensor, k: int, iters: int = 300, tol: float = 1e-6, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit K centroids per unit: k-means++ seeding (a generator seeded with `seed`, created per call), then
    Lloyd iterations until no centroid moves by `tol` or `iters` is reached.

    Returns (centroids [G, K] in units' dtype, assignments [G, S] uint8). An emptied cluster gets centroid
    0. K = 1 is the mean of the real values. Assignments are the nearest centroid of the returned centroids.
    Raises RuntimeError when a unit has fewer distinct real values than K (k-means++ has nothing to draw).
    """
    if not 1 <= k <= 256:
        raise ValueError(f"centroids must be in [1, 256], got {k}")
    if k == 1:
        counts = mask.sum(dim=1, keepdim=True).to(units.dtype)
        centroids = (units * mask).sum(dim=1, keepdim=True) / counts
    else:
        centroids = _kmeans_pp(units, mask, k, seed)
        for _ in range(iters):
            sums, counts = _member_sum(units, mask, _nearest(units, mask, centroids), k)
            updated = sums / counts.masked_fill(counts == 0, 1.0)
            moved = (centroids - updated).abs().max()
            centroids = updated
            if moved < tol:
                break
    return centroids, _nearest(units, mask, centroids).to(torch.uint8)


def gather(centroids: torch.Tensor, assignments: torch.Tensor) -> torch.Tensor:
    """Values [G, S]: the centroid each element is assigned to."""
    return centroids.gather(1, assignments.long())


def scatter_mean_grad(gradient: torch.Tensor, mask: torch.Tensor, assignments: torch.Tensor, k: int):
    """Mean of the real elements' gradient [G, S] over each cluster: [G, K], 0 for an empty cluster."""
    sums, counts = _member_sum(gradient, mask, assignments, k)
    return sums / counts.masked_fill(counts == 0, 1.0)


def input_diag(
    inputs: torch.Tensor, target_inputs: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column statistics [D] of linear inputs: A = E[x^2] clamped below at eps, B = E[x * x_target]."""
    columns = inputs.shape[-1]
    x = inputs.reshape(-1, columns).float()
    target = target_inputs.reshape(-1, columns).float()
    return (x * x).mean(dim=0).clamp_min(eps), (x * target).mean(dim=0)


def recluster_diag(
    units: torch.Tensor,
    mask: torch.Tensor,
    centroids: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    group_size: int | None,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Re-assign every element of the original weight units [G, S] to the centroid minimizing
    `(b_j * w - a_j * c)^2`, with `a` and `b` the per-column vectors [D] of `input_diag`. Returns uint8.

    With a == b this is the nearest centroid. Padding is never assigned a meaningful index.
    """
    rows = shape[0]
    a_units, _ = to_units(a.to(units.dtype).unsqueeze(0).expand(rows, -1).contiguous(), group_size)
    b_units, _ = to_units(b.to(units.dtype).unsqueeze(0).expand(rows, -1).contiguous(), group_size)
    score = (units * b_units).unsqueeze(-1) - a_units.unsqueeze(-1) * centroids.unsqueeze(1)
    score = score.pow(2).masked_fill(~mask.unsqueeze(-1), float("inf"))
    return score.argmin(dim=2).to(torch.uint8)


def fake_quantize(weight: torch.Tensor, fmt: CodebookFormat) -> torch.Tensor:
    """Replace each element of a [O, D] weight by its nearest k-means centroid; same dtype and shape."""
    group_size = fmt.group_size if fmt.granularity == "per_group" else None
    work = weight.float() if fmt.kmeans.compute_dtype == "float32" else weight
    units, mask = to_units(work, group_size)
    centroids, assignments = kmeans(
        units, mask, fmt.centroids, fmt.kmeans.iters, fmt.kmeans.tol, fmt.kmeans.seed
    )
    return from_units(gather(centroids, assignments), group_size, tuple(weight.shape)).to(weight.dtype)
