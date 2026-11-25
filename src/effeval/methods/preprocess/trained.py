"""Trained sources: a learned residual rotation (CodeQuant's AOS) and learned norm scaling (CodeQuant2).

Both minimise the activation quantization error `ops.orthogonal.act_quant_mse` of residual-stream norm outputs
on `TextBatches` calibration, with forward hooks on the norm modules of the unmodified model (the gains are
already folded into the consumers for the rotation). The rotation is fitted before it is folded; the trained
matrix and the loss history are cached as an artifact, so a cached run folds without training.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils import parametrize

from effeval.core.graph import LayerGraph
from effeval.core.preprocess import SOURCE_VERSIONS, RotationEntry, ScalingEntry
from effeval.methods.preprocess.engine import (
    ArtifactRef,
    Context,
    EngineError,
    StageResult,
    attached_online,
    check_position,
    source,
)
from effeval.methods.preprocess.rotation import fold_residual, prepare_residual
from effeval.methods.preprocess.scaling import _fold_offline, _group
from effeval.ops.artifacts import artifact_key
from effeval.ops.calibration import text_batches
from effeval.ops.orthogonal import act_quant_mse, cayley_rotation, trainable


def micro_batches(batch, size: int | None) -> list[tuple[dict[str, torch.Tensor], float]]:
    """Split a batch (mapping of tensors with the rows on dim 0) into chunks of `size` rows.

    Returns (chunk, weight) pairs, weight = chunk rows / all rows (a single chunk with weight 1 for `size`
    None). The loss of every row is a mean over the rows' elements, so the weighted sum of the chunk losses
    equals the full-batch loss and so do the accumulated gradients, up to float rounding.
    """
    rows = len(next(iter(batch.values())))
    if size is None or size >= rows:
        return [(dict(batch), 1.0)]
    chunks = [(i, min(i + size, rows)) for i in range(0, rows, size)]
    return [({k: v[a:b] for k, v in batch.items()}, (b - a) / rows) for a, b in chunks]


def train_rotation(
    model: nn.Module,
    norms: Sequence[nn.Module],
    batches: Sequence,
    rotation: nn.Module,
    bits: int,
    lr: float,
    micro_batch_size: int | None = None,
) -> list[float]:
    """Train `rotation` (a `cayley_rotation`) with Adam, one step per batch; returns the loss per step.

    Each step runs `model` on the batch without gradients and sums `act_quant_mse(output @ R, bits)` over
    the outputs of `norms`, taken in float32 and multiplied with R moved to the output's device. Padding
    tokens count. With `micro_batch_size` the batch is split in chunks (see `micro_batches`) whose weighted
    losses are backpropagated one by one, with the Cayley matrix built once per chunk instead of once per
    norm. The model's training flag is restored.
    """
    optimizer = torch.optim.Adam([trainable(rotation)], lr=lr)
    losses: list[torch.Tensor] = []
    home = trainable(rotation).device

    def hook(_module, _args, output):
        with torch.enable_grad():
            rotated = output.detach().float() @ rotation.weight.to(output.device)
            losses.append(act_quant_mse(rotated, bits).to(home))

    handles = [norm.register_forward_hook(hook) for norm in norms]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    history: list[float] = []
    try:
        for batch in batches:
            optimizer.zero_grad()
            step_loss = 0.0
            for chunk, weight in micro_batches(batch, micro_batch_size):
                losses.clear()
                with parametrize.cached() if micro_batch_size is not None else contextlib.nullcontext():
                    with torch.no_grad():
                        model(**{k: v.to(device) for k, v in chunk.items()}, use_cache=False)
                    total = torch.stack(losses).sum() * weight
                    total.backward()
                step_loss += total.item()
            optimizer.step()
            history.append(step_loss)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return history


def train_scaling(
    model: nn.Module,
    norms: dict[str, nn.Module],
    batches: Sequence,
    init: dict[str, torch.Tensor],
    bits: int,
    lr: float,
    micro_batch_size: int | None = None,
) -> tuple[dict[str, torch.Tensor], list[float]]:
    """Train one scaling vector per norm in `norms` (name -> module), each with its own Adam.

    The loss of vector s on its norm is `act_quant_mse(output / s, bits)` with the output taken detached in
    float32; the model forward does not depend on s. `init[name]` is the starting vector. With
    `micro_batch_size` the gradients of the weighted chunk losses are accumulated (see `micro_batches`).
    Returns the final vectors (float32, CPU) and the sum of the per-norm losses per step.
    """
    params = {
        name: nn.Parameter(init[name].detach().clone().float().to(norm.weight.device))
        for name, norm in norms.items()
    }
    optimizers = {name: torch.optim.Adam([param], lr=lr) for name, param in params.items()}
    losses: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook(_module, _args, output):
            with torch.enable_grad():
                scaled = output.detach().float() / params[name].to(output.device)
                losses[name] = act_quant_mse(scaled, bits)

        return hook

    handles = [norm.register_forward_hook(make_hook(name)) for name, norm in norms.items()]
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    history: list[float] = []
    try:
        for batch in batches:
            for optimizer in optimizers.values():
                optimizer.zero_grad()
            step_loss = 0.0
            for chunk, weight in micro_batches(batch, micro_batch_size):
                losses.clear()
                with torch.no_grad():
                    model(**{k: v.to(device) for k, v in chunk.items()}, use_cache=False)
                for loss in losses.values():
                    (loss * weight).backward()
                    step_loss += loss.item() * weight
            for optimizer in optimizers.values():
                optimizer.step()
            history.append(step_loss)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return {name: param.detach().cpu() for name, param in params.items()}, history


def residual_norms(net) -> list[nn.Module]:
    """The hidden-axis norm modules reading the residual stream, layer by layer (input norm first)."""
    return [
        net.module(layer, norm_node)
        for layer in range(len(net.layers))
        for norm_node, _consumers in net.stream_consumers()
    ]


def _flat_batches(ctx: Context, calibration) -> list:
    return [batch for epoch in text_batches(calibration, ctx.tokenizer) for batch in epoch]


@source("trained", "rotation")
def apply_trained_rotation(
    model: nn.Module, graph: LayerGraph, entries: list[RotationEntry], ctx: Context
) -> StageResult:
    """Fold the norm gains, train the residual rotation on the gain-free model, and fold it.

    The loss is taken over every hidden-axis norm that reads the residual stream, of the model with the gains
    folded and earlier steps' online ops attached. Raises EngineError unless the step has exactly one entry.
    """
    if len(entries) != 1:
        raise EngineError("a trained rotation step takes exactly one residual entry")
    [entry] = entries
    check_position(ctx, "rotation", entry)
    net, src = ctx.net, entry.source
    result = StageResult()
    prepare_residual(net, entry, result)

    key = artifact_key(
        model=ctx.model_id,
        calibration=None,
        statistic=f"trained_rotation:v{SOURCE_VERSIONS['rotation']['trained']}:{src.model_dump_json()}:"
        + json.dumps(entry.options, sort_keys=True),
        upstream=ctx.upstream,
    )

    def compute() -> dict[str, torch.Tensor]:
        rotation = cayley_rotation(net.shape.hidden, src.init_seed, net.embedding.weight.device)
        with attached_online(model, list(ctx.pending_online)):
            history = train_rotation(
                model,
                residual_norms(net),
                _flat_batches(ctx, src.calibration),
                rotation,
                src.act.bits,
                src.lr,
                src.micro_batch_size,
            )
        return {"R": rotation.weight.detach().cpu(), "losses": torch.tensor(history, dtype=torch.float64)}

    found = ctx.artifacts.get_or_compute(key, compute)
    fold_residual(net, entry, found["R"].double(), result)
    losses = found["losses"]
    result.artifacts["trained_rotation"] = ArtifactRef(key, "Trained residual rotation and its loss history")
    result.edits.append(
        f"trained rotation: {losses.numel()} Adam steps (lr {src.lr}, init seed {src.init_seed}), "
        f"act_quant_mse {losses[0].item():.6g} -> {losses[-1].item():.6g}"
    )
    return result


@source("trained", "scaling")
def apply_trained_scaling(
    model: nn.Module, graph: LayerGraph, entries: list[ScalingEntry], ctx: Context
) -> StageResult:
    """Train one scaling vector per norm output of the entries' positions and layers, and fold each into its
    norm gain and the consumers' columns.

    All vectors train in one pass over the calibration batches, on the model as it is (earlier steps' offline
    edits applied, their online ops attached). Raises EngineError for entries with different source settings,
    a duplicate position, or an online mode.
    """
    net = ctx.net
    src = entries[0].source
    seen: set[str] = set()
    for entry in entries:
        check_position(ctx, "scaling", entry)
        if entry.source != src:
            raise EngineError("the entries of one trained scaling step must share their source settings")
        if entry.position in seen:
            raise EngineError(f"position {entry.position!r} appears twice in one step")
        seen.add(entry.position)
    targets = [
        (_group(net, entry), layer, f"{entry.position}@{layer}")
        for entry in entries
        for layer in net.layer_indices(entry.layers)
    ]
    key = artifact_key(
        model=ctx.model_id,
        calibration=None,
        statistic=f"trained_scaling:v{SOURCE_VERSIONS['scaling']['trained']}:{src.model_dump_json()}:"
        + ",".join(name for _group_, _layer, name in targets),
        upstream=ctx.upstream,
    )

    def compute() -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(src.init_seed)
        init = {
            name: 1 + 0.01 * torch.randn(net.shape.hidden, generator=generator) for _g, _l, name in targets
        }
        norms = {name: net.module(layer, group.prev) for group, layer, name in targets}
        with attached_online(model, list(ctx.pending_online)):
            vectors, history = train_scaling(
                model,
                norms,
                _flat_batches(ctx, src.calibration),
                init,
                src.act.bits,
                src.lr,
                src.micro_batch_size,
            )
        return {**vectors, "losses": torch.tensor(history, dtype=torch.float64)}

    found = ctx.artifacts.get_or_compute(key, compute)
    for group, layer, name in targets:
        _fold_offline(net, layer, group, found[name])
    losses = found["losses"]
    result = StageResult(artifacts={"trained_scaling": ArtifactRef(key, "Trained norm scaling vectors")})
    result.edits.append(
        f"trained scaling: {len(targets)} vectors, {losses.numel()} Adam steps (lr {src.lr}, "
        f"init seed {src.init_seed}), summed act_quant_mse {losses[0].item():.6g} -> {losses[-1].item():.6g}"
    )
    return result
