"""DuQuant rotation source: per-linear-input block rotations and zigzag permutation searched on activations.

For each linear input DuQuant (https://github.com/Hsu1023/DuQuant, commit d56cfc6) applies, per row of
`block_size` channels, R1, then a zigzag channel permutation P, then a second block rotation R2; the weight
gets the same input-side map x -> x @ T, T = diag(R1) P diag(R2), folded offline (W @ T) while the activation
transform runs online. Each R is a random orthogonal base matrix with channel 0 exchanged by a greedy search
for the activation's largest outlier; the search tensor is the mean of the calibration sequences propagated
layer by layer. Linears reading the same tensor (q/k/v, gate/up) share one (R, P), as in the official code.
Randomness comes from one CPU generator seeded with the source's seed and consumed in forward-execution order.

Shape symbols: B block size, C input channels (a multiple of B), L calibration tokens.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import torch
from torch import nn

from effeval.core.graph import LayerGraph
from effeval.core.preprocess import SOURCE_VERSIONS, RotationEntry, position_dim
from effeval.methods.preprocess.engine import (
    ArtifactRef,
    Context,
    EngineError,
    StageResult,
    attached_online,
    check_position,
    source,
)
from effeval.ops import rotation as rot
from effeval.ops.artifacts import artifact_key
from effeval.ops.calibration import LayerRunner
from effeval.ops.online import OnlineOp


@functools.cache
def base_rotation(n: int) -> torch.Tensor:
    """The float32 [n, n] base orthogonal matrix of DuQuant's get_rot.py (a Givens product, 0 <-> n-1)."""
    rotation = torch.eye(n)
    for i in range(n - 1):
        rot_i = torch.eye(n)
        cos = 1 / np.sqrt(i + 2)
        sin = np.sqrt(i + 1) / np.sqrt(i + 2)
        rot_i[i, i] = rot_i[i + 1, i + 1] = cos
        rot_i[i, i + 1] = -sin
        rot_i[i + 1, i] = sin
        rotation = torch.matmul(rot_i, rotation)
    index = torch.arange(n)
    index[0], index[n - 1] = n - 1, 0
    return rotation[index][:, index]


def random_rotation(n: int, generator: torch.Generator) -> torch.Tensor:
    """DuQuant's utils.get_rot: the base matrix times a random orthogonal [n-1, n-1] block (Gaussian QR)."""
    q, _ = torch.linalg.qr(torch.randn(n - 1, n - 1, generator=generator))
    q = torch.cat([torch.zeros(n - 1, 1), q], dim=1)
    q = torch.cat([torch.zeros(1, n), q], dim=0)
    q[0, 0] = 1
    return torch.matmul(base_rotation(n), q)


def _swap_channel_zero(matrix: torch.Tensor, j: int) -> torch.Tensor:
    index = torch.arange(matrix.shape[0])
    index[0], index[j] = j, 0
    return matrix[index][:, index]


def search_rotation(
    x: torch.Tensor, block_size: int, max_steps: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """One greedy outlier-flattening block rotation: (x @ R applied per block as [L, C], R [B, B]).

    Each step swaps the position of the current largest-magnitude entry into channel 0 of the random
    orthogonal base (`exchange_id` is the entry's row in the [-1, B] view taken modulo B, as in the
    official code) and applies it; the prefix of steps with the lowest max - min is kept.
    """
    channels = x.shape[-1]
    work = x.detach().clone().reshape(-1, block_size)
    base = random_rotation(block_size, generator).to(work.device)
    exchange_ids: list[int] = []
    peaks: list[float] = []
    for _ in range(max_steps):
        r, c = divmod(work.argmax().item(), work.shape[-1])
        r2, c2 = divmod(work.argmin().item(), work.shape[-1])
        peaks.append((work[r, c] - work[r2, c2]).item())
        exchange_ids.append(r if work[r, c].abs() > work[r2, c2].abs() else r2)
        work = torch.matmul(work, _swap_channel_zero(base, exchange_ids[-1] % block_size).to(work))
    r, c = divmod(work.argmax().item(), work.shape[-1])
    r2, c2 = divmod(work.argmin().item(), work.shape[-1])
    peaks.append((work[r, c] - work[r2, c2]).item())
    keep = torch.argmin(torch.tensor(peaks)).item()
    matrix = torch.eye(block_size).to(x.device)
    matrix = matrix.to(x.dtype)
    for exchange_id in exchange_ids[:keep]:
        matrix = torch.matmul(matrix, _swap_channel_zero(base, exchange_id % block_size).to(matrix))
    rotated = torch.matmul(x.detach().reshape(-1, block_size), matrix).reshape(-1, channels)
    return rotated, matrix


def zigzag_permutation(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Channel order [C] that deals channels, sorted by max |x| over tokens, down and up across the blocks.

    x [L, C]; blocks are filled 0, 1, .., n-1, n-1, .., 1, 0, 0, 1, ..; inside a block channels are sorted by
    descending max |x|. The permuted tensor is `x[:, perm]`.
    """
    channels = x.shape[-1]
    column_max = x.detach().abs().max(dim=0).values.tolist()
    ranked = sorted(enumerate(column_max), key=lambda pair: pair[1], reverse=True)
    buckets: list[list[tuple[int, float]]] = [[] for _ in range(channels // block_size)]
    current, going_up = 0, True
    for item in ranked:
        buckets[current].append(item)
        if going_up:
            current += 1
            if current == len(buckets):
                current -= 1
                going_up = False
        else:
            current -= 1
            if current == -1:
                current += 1
                going_up = True
    perm = torch.zeros(channels, dtype=torch.long)
    for i, bucket in enumerate(buckets):
        bucket.sort(key=lambda pair: pair[1], reverse=True)
        perm[i * block_size : (i + 1) * block_size] = torch.tensor([channel for channel, _ in bucket])
    return perm


def search_transform(
    x: torch.Tensor, block_size: int, max_steps: int, permutation_times: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """DuQuant's (R list, permutation list) for activations x [..., C]: a rotation, then `permutation_times`
    times (zigzag permutation, rotation). Returns rotations [times + 1, B, B] and permutations [times, C].
    """
    if x.shape[-1] % block_size:
        raise EngineError(f"{x.shape[-1]} channels are not a multiple of the DuQuant block size {block_size}")
    rotations: list[torch.Tensor] = []
    permutations: list[torch.Tensor] = []
    current = x
    for _ in range(permutation_times):
        current, matrix = search_rotation(current, block_size, max_steps, generator)
        rotations.append(matrix)
        permutation = zigzag_permutation(current, block_size)
        permutations.append(permutation)
        current = current[:, permutation.to(current.device)]
    current, matrix = search_rotation(current, block_size, max_steps, generator)
    rotations.append(matrix)
    channels = x.shape[-1]
    return torch.stack(rotations), (
        torch.stack(permutations) if permutations else torch.zeros(0, channels, dtype=torch.long)
    )


def apply_transform(
    x: torch.Tensor, rotations: torch.Tensor, permutations: torch.Tensor, block_size: int
) -> torch.Tensor:
    """x @ T for T = diag(R[0]) P[0] diag(R[1]) ... diag(R[-1]) over the last dimension of x [..., C]."""
    shape = x.shape
    for i in range(len(permutations)):
        x = torch.matmul(x.reshape(-1, block_size), rotations[i].to(x)).reshape(shape)
        x = x.index_select(-1, permutations[i].to(x.device))
    return torch.matmul(x.reshape(-1, block_size), rotations[-1].to(x)).reshape(shape)


class BlockRotatePermute(nn.Module):
    """Online op x -> x @ T (`apply_transform`), computed in `compute_dtype` and returned in x's dtype."""

    def __init__(
        self,
        rotations: torch.Tensor,
        permutations: torch.Tensor,
        block_size: int,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.block_size = block_size
        self.compute_dtype = compute_dtype
        self.register_buffer("rotations", rotations.detach().clone().to(compute_dtype))
        self.register_buffer("permutations", permutations.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotations = self.rotations.to(x.device)
        permutations = self.permutations.to(x.device)
        return apply_transform(x.to(self.compute_dtype), rotations, permutations, self.block_size).to(x.dtype)

    def matrix(self, channels: int) -> torch.Tensor:
        """The dense float64 [C, C] matrix T of this transform."""
        eye = torch.eye(channels, dtype=torch.float64, device=self.rotations.device)
        return apply_transform(eye, self.rotations.double(), self.permutations, self.block_size)


def _observe(seen: dict[str, torch.Tensor], position: str, _module: nn.Module, args: tuple) -> None:
    """Forward-pre-hook body: keep the first float32 input a linear receives, by position."""
    seen.setdefault(position, args[0].float().clone())


def _linear_of(position: str) -> str:
    return position[len("linear.") : -len(".input")]


def _plan(ctx: Context, graph: LayerGraph, entries: list[RotationEntry]) -> dict[int, dict[str, list[str]]]:
    """Per decoder layer, the linear-input positions to transform grouped by the tensor they read.

    Returns layer -> {producer node id: [position, ..]} with positions in entry order. Raises EngineError for
    entries that are not online linear inputs, widths not divisible by the block size, or differing sources.
    """
    net = ctx.net
    source_json = entries[0].source.model_dump_json()
    block = entries[0].source.block_size
    if block < 2 or block & (block - 1):
        raise EngineError(f"DuQuant block_size must be a power of two >= 2, got {block}")
    plan: dict[int, dict[str, list[str]]] = {}
    for entry in entries:
        check_position(ctx, "rotation", entry)
        if entry.source.model_dump_json() != source_json:
            raise EngineError("all duquant entries of one step must use the same source settings")
        if entry.mode != "online" or not (
            entry.position.startswith("linear.") and entry.position.endswith(".input")
        ):
            raise EngineError(f"duquant applies online at linear inputs, not {entry.mode} {entry.position!r}")
        width = position_dim(graph, net.shape, entry.position)
        if width % block:
            raise EngineError(f"{entry.position} has {width} channels, not a multiple of block_size {block}")
        [(producer, _site)] = graph.producers(net.linear_node(_linear_of(entry.position)).id)
        for layer in net.layer_indices(entry.layers):
            positions = plan.setdefault(layer, {}).setdefault(producer.id, [])
            if entry.position in positions:
                raise EngineError(f"position {entry.position!r} appears twice for layer {layer}")
            positions.append(entry.position)
    return plan


def _search_all(
    ctx: Context, plan: dict[int, dict[str, list[str]]], src: Any, pending: list[OnlineOp]
) -> dict[str, torch.Tensor]:
    """Searched (rotations, permutations) per layer and input group, keyed '<layer>/<first position>/R|P'."""
    net = ctx.net
    windows = ctx.windows()
    generator = torch.Generator().manual_seed(src.seed)
    found: dict[str, torch.Tensor] = {}
    with attached_online(net.model, pending), LayerRunner(net.model, windows, propagate="original") as runner:
        first = torch.stack([h.float() for h in runner.inputs]).mean(dim=0)
        dtype = runner.inputs[0].dtype
        runner.inputs = [first.to(dtype)]
        for layer_index, layer in runner:
            groups = plan.get(layer_index)
            if not groups:
                continue
            seen: dict[str, torch.Tensor] = {}
            handles = []
            for positions in groups.values():
                for position in positions:
                    module = net.module(layer_index, net.linear_node(_linear_of(position)))
                    handles.append(
                        module.register_forward_pre_hook(functools.partial(_observe, seen, position))
                    )
            try:
                runner.forward(layer, runner.inputs[0])
            finally:
                for handle in handles:
                    handle.remove()
            order = list(seen)
            for positions in sorted(groups.values(), key=lambda ps: min(order.index(p) for p in ps)):
                rotations, permutations = search_transform(
                    seen[positions[0]], src.block_size, src.max_steps, src.permutation_times, generator
                )
                found[f"{layer_index}/{positions[0]}/R"] = rotations.cpu()
                found[f"{layer_index}/{positions[0]}/P"] = permutations.cpu()
    return found


@source("duquant", "rotation")
def apply_duquant(
    model: nn.Module, graph: LayerGraph, entries: list[RotationEntry], ctx: Context
) -> StageResult:
    """Search and apply DuQuant's block rotation / zigzag permutation / block rotation at linear inputs.

    The search runs on the mean calibration sequence propagated through the layers as they are (offline edits
    of earlier steps, `ctx.pending_online` ops attached), without quantizers. Raises EngineError for a step
    without calibration, a non-online or non-linear-input position, a width not divisible by the block size,
    or entries with different source settings.
    """
    if not entries:
        return StageResult()
    net = ctx.net
    src = entries[0].source
    plan = _plan(ctx, graph, entries)
    key = artifact_key(
        model=ctx.model_id,
        calibration=ctx.calibration,
        statistic=f"duquant:v{SOURCE_VERSIONS['rotation']['duquant']}:{src.model_dump_json()}:"
        + ",".join(f"{layer}:{sorted(groups.items())}" for layer, groups in sorted(plan.items())),
        upstream=ctx.upstream,
    )
    found = ctx.artifacts.get_or_compute(key, lambda: _search_all(ctx, plan, src, list(ctx.pending_online)))

    result = StageResult(artifacts={"duquant": ArtifactRef(key, "DuQuant rotations and permutations")})
    for layer_index in sorted(plan):
        for positions in plan[layer_index].values():
            lead = positions[0]
            transform = BlockRotatePermute(
                found[f"{layer_index}/{lead}/R"], found[f"{layer_index}/{lead}/P"], src.block_size
            )
            matrix = transform.matrix(position_dim(graph, net.shape, lead))
            for position in positions:
                linear = net.module(layer_index, net.linear_node(_linear_of(position)))
                linear.weight.data.copy_(rot.fold_input(linear.weight, matrix))
                result.online.append(OnlineOp(position, transform, [layer_index]))
    n_groups = sum(len(groups) for groups in plan.values())
    result.edits.append(
        f"duquant: block {src.block_size}, {src.max_steps} steps, {src.permutation_times} permutation(s), "
        f"seed {src.seed}: {n_groups} input groups in {len(plan)} layers, weights folded, transforms online"
    )
    return result
