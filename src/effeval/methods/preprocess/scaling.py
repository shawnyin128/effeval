"""Scaling sources: SmoothQuant and AWQ per-channel scales folded into norm gains, weights or online ops.

A scale s on a channel divides the activation and multiplies the consuming weights' input columns, so the
full-precision function is unchanged. Offline, the division is folded into a norm gain (`norm.<module>.out`
positions) or into the producing linear's rows (`linear.mlp.down_proj.input`, producer up_proj). Online
(SmoothQuant only, `linear.<canon>.input` positions) it is a runtime `ChannelDivide` on the linear's input.
SmoothQuant: s = max|x|^alpha / max|W|^(1 - alpha) from activation maxima. AWQ: s = mean|x|^ratio normalised
by sqrt(max * min), with the ratio searched per group on the output error of the quantized block
(`search_scales`), then an optional per-group weight clip search (`search_clip`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from effeval.core.fragments import IntFormat
from effeval.core.graph import GraphNode, LayerGraph
from effeval.core.preprocess import AwqScaling, Entry, ScalingEntry, SmoothQuantScaling
from effeval.methods.preprocess.engine import (
    ArtifactRef,
    Context,
    EngineError,
    Net,
    StageResult,
    attached_online,
    check_position,
    source,
)
from effeval.ops import sites
from effeval.ops.artifacts import artifact_key
from effeval.ops.calibration import LayerRunner, act_stats
from effeval.ops.online import OnlineOp
from effeval.ops.quant.quant import fake_quantize
from effeval.ops.scaling import (
    ChannelDivide,
    fold_into_norm_gain,
    scale_input_columns,
    scale_output_rows,
)

SMOOTH_MIN = 1e-5
AWQ_SCALE_MIN = 1e-4


@dataclass(frozen=True)
class Group:
    """One scale: `prev` (a norm, or the linear whose rows are divided) feeds `consumers`
    (columns multiplied).

    `inspect` is the module path, inside a decoder layer, of the smallest module containing every consumer;
    AWQ compares its output when searching.
    """

    position: str
    prev: GraphNode
    consumers: list[GraphNode]
    inspect: str


def _common_module(paths: list[str]) -> str:
    prefix: list[str] = []
    for parts in zip(*(path.split(".") for path in paths), strict=False):
        if len(set(parts)) != 1:
            break
        prefix.append(parts[0])
    return ".".join(prefix)


def _group(net: Net, entry: Entry) -> Group:
    """The scale group of a scaling entry. Raises EngineError for positions this module does not implement."""
    graph = net.graph
    position = entry.position
    if position.startswith("norm.") and position.endswith(".out"):
        module = position[len("norm.") : -len(".out")]
        norm = next((n for n in graph.of_kind("norm") if n.module == module), None)
        if norm is None:
            raise EngineError(f"unknown position {position!r}")
        consumers = [c for c, _ in graph.consumers(norm.id) if c.kind == "linear"]
        if not consumers:
            raise EngineError(f"{position}: the norm feeds no linear")
        inspect = _common_module([c.module for c in consumers])
        if not inspect:
            raise EngineError(f"{position}: the consumers share no module to compare outputs on")
        return Group(position, norm, consumers, inspect)
    if position.startswith("linear.") and position.endswith(".input"):
        consumer = net.linear_node(position[len("linear.") : -len(".input")])
        [(producer, _site)] = graph.producers(consumer.id)
        if producer.kind == "mul":
            up = next((p for p, _ in graph.producers(producer.id) if p.kind == "linear"), None)
            if up is not None:
                return Group(position, up, [consumer], consumer.module)
    raise EngineError(f"scaling at {position!r} is not implemented")


FIRST_CONSUMER = ("q_proj", "up_proj")
"""Linear names (last module path segment) preferred as a group's first consumer, in this order: the linear
whose weight DuQuant's `first_consumer` reduction uses for the attention input and the MLP input."""


def first_consumer(group: Group) -> GraphNode:
    """The consumer whose weight `first_consumer` uses: q_proj or up_proj if present, else the first one."""
    by_name = {node.module.rsplit(".", 1)[-1]: node for node in group.consumers}
    return next((by_name[name] for name in FIRST_CONSUMER if name in by_name), group.consumers[0])


def _check_entries(ctx: Context, entries: list[Any]) -> None:
    seen: set[str] = set()
    for entry in entries:
        check_position(ctx, "scaling", entry)
        online_ok = entry.source.type == "smoothquant" and entry.position.startswith("linear.")
        if entry.mode != "offline" and not online_ok:
            raise EngineError(f"online scaling at {entry.position!r} is not implemented")
        if entry.position in seen:
            raise EngineError(f"position {entry.position!r} appears twice in one step")
        seen.add(entry.position)


@source("smoothquant", "scaling")
@source("awq", "scaling")
def apply_scaling(
    model: nn.Module, graph: LayerGraph, entries: list[ScalingEntry], ctx: Context
) -> StageResult:
    """Apply the step's SmoothQuant or AWQ entries (one source type per step) to `model` in place.

    Raises EngineError for an unsupported position or mode, a missing calibration, mixed source types, or
    (AWQ) a missing or non-integer weight quantizer in `ctx.downstream`.
    """
    _check_entries(ctx, entries)
    if len({e.source.type for e in entries}) != 1:
        raise EngineError("one scaling step takes one source type; put SmoothQuant and AWQ in separate steps")
    if ctx.calibration is None:
        raise EngineError(f"{entries[0].source.type} needs a calibration fragment on its step")
    result = StageResult()
    if entries[0].source.type == "smoothquant":
        _smoothquant(ctx, entries, result)
    else:
        _awq(ctx, entries, result)
    return result


# ---------------------------------------------------------------- SmoothQuant


def smoothquant_scales(
    act_max: torch.Tensor, weights: list[torch.Tensor], alpha: float, weight_reduce: str
) -> torch.Tensor:
    """SmoothQuant scale max|x|^alpha / max|W|^(1 - alpha) per input channel, float32 [C].

    `weights` are the consumers' [out, in] weights; the weight maximum is taken over all of them
    (`group_max`) or over the first (`first_consumer`). Scale and weight maximum are clamped to >= 1e-5.
    """
    used = weights if weight_reduce == "group_max" else weights[:1]
    w_max = torch.stack([w.abs().amax(dim=0).float() for w in used]).amax(dim=0).clamp(min=SMOOTH_MIN)
    return (act_max.to(w_max.device).float().pow(alpha) / w_max.pow(1 - alpha)).clamp(min=SMOOTH_MIN)


def _fold_offline(net: Net, layer: int, group: Group, scales: torch.Tensor) -> None:
    """Divide the group's input by `scales` in `layer`: into the norm gain or the producer's rows, and the
    consumers' columns are multiplied."""
    if group.prev.kind == "norm":
        gain = net.module(layer, group.prev).weight
        gain.data.copy_(fold_into_norm_gain(gain, scales))
    else:
        prev = net.module(layer, group.prev)
        prev.weight.data.copy_(scale_output_rows(prev.weight, scales))
        if prev.bias is not None:
            prev.bias.data.copy_(scale_output_rows(prev.bias[:, None], scales)[:, 0])
    for node in group.consumers:
        linear = net.module(layer, node)
        linear.weight.data.copy_(scale_input_columns(linear.weight, scales))


def _smoothquant(ctx: Context, entries: list[ScalingEntry], result: StageResult) -> None:
    net = ctx.net
    groups = []
    for entry in entries:
        if entry.mode == "online":
            consumer = net.linear_node(entry.position[len("linear.") : -len(".input")])
            groups.append(Group(entry.position, consumer, [consumer], consumer.module))
        else:
            groups.append(_group(net, entry))
    probes = sorted({f"linear.{first_consumer(g).canon}.input" for g in groups})
    key = artifact_key(
        model=ctx.model_id,
        calibration=ctx.calibration,
        statistic="smoothquant:absmax:" + ",".join(probes),
        upstream=ctx.upstream,
    )

    def compute_stats() -> dict[str, torch.Tensor]:
        with attached_online(net.model, list(ctx.pending_online)):
            observed = act_stats(net.model, ctx.windows(), probes, "absmax")
        return {f"{site}@{layer}": value for (site, layer), value in observed.items()}

    stats = ctx.artifacts.get_or_compute(key, compute_stats)
    result.artifacts["smoothquant.act_max"] = ArtifactRef(key, "per-channel max |x| at the smoothed inputs")
    for entry, group in zip(entries, groups, strict=True):
        src: SmoothQuantScaling = entry.source
        layers = net.layer_indices(entry.layers)
        first = first_consumer(group)
        probe = f"linear.{first.canon}.input"
        others = [node for node in group.consumers if node is not first]
        for layer in layers:
            weights = [net.module(layer, node).weight for node in [first, *others]]
            scales = smoothquant_scales(stats[f"{probe}@{layer}"], weights, src.alpha, src.weight_reduce)
            if entry.mode == "online":
                linear = net.module(layer, first)
                linear.weight.data.copy_(scale_input_columns(linear.weight, scales))
                result.online.append(OnlineOp(entry.position, ChannelDivide(scales), [layer]))
            else:
                _fold_offline(net, layer, group, scales)
        target = "an online division on the input" if entry.mode == "online" else "the norm gain or producer"
        result.edits.append(
            f"{entry.position}: smoothquant alpha {src.alpha} ({src.weight_reduce}) folded into {target} "
            f"and {len(group.consumers)} consumer linears of {len(layers)} layers"
        )


# ---------------------------------------------------------------- AWQ


def _cast(value: Any, dtype: torch.dtype) -> Any:
    """`value` with every floating tensor (also inside tuples, lists, dicts) converted to `dtype`."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, (tuple, list)):
        return type(value)(_cast(v, dtype) for v in value)
    if isinstance(value, dict):
        return {k: _cast(v, dtype) for k, v in value.items()}
    return value


def _block_output(block: nn.Module, x: torch.Tensor, kwargs: dict[str, Any]) -> torch.Tensor:
    out = block(x, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def search_scales(
    block: nn.Module,
    linears: list[nn.Linear],
    quantizers: list,
    x: torch.Tensor,
    kwargs: dict[str, Any],
    n_grid: int,
) -> tuple[torch.Tensor, float]:
    """AWQ scale search (llm-awq auto_scale.py `_search_module_scale`): returns (scales [C], best ratio).

    `x` [n, L, C] is the input of `linears` (members of `block`), `quantizers[i]` fake-quantizes the weight of
    `linears[i]`. For each ratio in {i / n_grid}: s = mean|x|^ratio / sqrt(max * min), the linears' weights
    become
    quantize(W * s) / s, and the mean squared error of `block`'s output against the unquantized output is
    measured; the first strictly smallest error wins. The weights are restored afterwards.
    """
    org_out = _block_output(block, x, kwargs)
    x_max = x.abs().reshape(-1, x.shape[-1]).mean(0)
    original = [fc.weight.data.clone() for fc in linears]
    best_error, best_ratio, best_scales = float("inf"), -1.0, None
    for step in range(n_grid):
        ratio = step * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=AWQ_SCALE_MIN).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc, quantize in zip(linears, quantizers, strict=True):
            fc.weight.mul_(scales.view(1, -1))
            fc.weight.data = quantize(fc.weight.data) / scales.view(1, -1)
        loss = (org_out - _block_output(block, x, kwargs)).float().pow(2).mean().item()
        if loss < best_error:
            best_error, best_ratio, best_scales = loss, ratio, scales
        for fc, weight in zip(linears, original, strict=True):
            fc.weight.data = weight.clone()
    if best_scales is None:
        raise EngineError("awq scale search found no finite loss")
    return best_scales.detach(), best_ratio


def weight_group_size(fmt: Any, in_features: int) -> int:
    """Values per scale of an integer weight format: its group size, or the whole row when per_channel."""
    if not isinstance(fmt, IntFormat) or fmt.granularity not in ("per_group", "per_channel"):
        raise EngineError(
            "awq needs integer weight quantizers with per_group or per_channel granularity downstream, "
            f"got {fmt!r}"
        )
    return fmt.group_size if fmt.granularity == "per_group" else in_features


@torch.no_grad()
def search_clip(
    weight: torch.Tensor,
    x: torch.Tensor,
    fmt: IntFormat,
    n_grid: int,
    max_shrink: float,
    n_tokens: int,
) -> torch.Tensor:
    """AWQ clip search (llm-awq auto_clip.py `auto_clip_layer`): per output row and group, the best max |w|.

    `weight` [out, in] and `x` [..., in] (the linear's input) are float tensors; each group of
    `weight_group_size` weights is clamped to +-max_val for max_val = org_max * (1 - i / n_grid),
    i < int(max_shrink * n_grid), quantized with `fmt`, and the candidate with the smallest mean squared error
    of
    the group's output over about `n_tokens` input rows wins (the first on ties). Returns [out, groups, 1].
    """
    group_size = weight_group_size(fmt, weight.shape[1])
    unit_fmt = fmt.model_copy(update={"granularity": "per_group", "group_size": group_size})
    x = x.reshape(-1, x.shape[-1])
    x = x.reshape(1, x.shape[0], -1, group_size)
    step = x.shape[1] // n_tokens
    if step == 0:
        raise EngineError(f"awq clip search needs at least {n_tokens} calibration tokens, got {x.shape[1]}")
    x = x[:, 0::step]
    w_all = weight.reshape(weight.shape[0], 1, -1, group_size)
    batch = 256 if w_all.shape[0] % 256 == 0 else 64
    if w_all.shape[0] % batch != 0:
        batch = w_all.shape[0]
    best_all = []
    for start in range(0, w_all.shape[0], batch):
        w = w_all[start : start + batch]
        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # [co, 1, groups, 1]
        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        org_out = (x * w).sum(dim=-1)  # [co, tokens, groups]
        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            cur_w = torch.clamp(w, -max_val, max_val)
            q_w = fake_quantize(cur_w.reshape(-1, group_size), unit_fmt, "weight").reshape(cur_w.shape)
            cur_out = (x * q_w).sum(dim=-1)
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            better = err < min_errs
            min_errs[better] = err[better]
            best_max_val[better] = max_val[better]
        best_all.append(best_max_val)
    return torch.cat(best_all, dim=0).squeeze(1)


def apply_clip(weight: torch.Tensor, max_val: torch.Tensor) -> None:
    """Clamp `weight` [out, in] in place to +-max_val [out, groups, 1] per group."""
    shape = weight.shape
    grouped = weight.data.reshape(*max_val.shape[:2], -1)
    limit = max_val.to(weight.device).to(weight.dtype)
    weight.data.copy_(torch.clamp(grouped, -limit, limit).reshape(shape))


def _capture_inputs(
    layer: nn.Module, hidden: list[torch.Tensor], forward, paths: list[str]
) -> dict[str, torch.Tensor]:
    """Float32 [n, L, C] inputs of the linears at `paths` over all `hidden` windows of one layer."""
    captured: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
    handles = [
        layer.get_submodule(path).register_forward_pre_hook(
            lambda _m, args, path=path: captured[path].append(args[0].detach().float())
        )
        for path in paths
    ]
    try:
        for h in hidden:
            forward(layer, h)
    finally:
        for handle in handles:
            handle.remove()
    return {path: torch.cat(parts, dim=0) for path, parts in captured.items()}


def _weight_formats(ctx: Context) -> dict[tuple[str, int | None], Any]:
    """(site, layer) -> the downstream format of every weight site the quantizer rules assign."""
    assignment, _ = sites.resolve(ctx.net.model, ctx.downstream)
    return {key: fmt for key, (_rule, _loc, fmt) in assignment.items() if key[0].endswith(".weight")}


def _identity(weight: torch.Tensor) -> torch.Tensor:
    return weight


def _awq_layer(
    net: Net,
    layer_index: int,
    layer: nn.Module,
    pairs: list[tuple[ScalingEntry, Group]],
    formats: dict[tuple[str, int | None], Any],
    feats: dict[str, torch.Tensor],
    kwargs: dict[str, Any],
    out: dict[str, torch.Tensor],
) -> None:
    """Search scales and clip limits of one unedited layer; writes `scale.*`, `ratio.*`, `clip.*` to `out`.

    The clip search sees the weights as they are after every fold (column scales, up_proj row scales).
    """
    layout = net.layout
    scaled: dict[str, torch.Tensor] = {}
    row_scaled: dict[str, torch.Tensor] = {}
    src: AwqScaling = pairs[0][0].source
    for entry, group in pairs:
        if layer_index not in net.layer_indices(entry.layers):
            continue
        block = copy.deepcopy(layer.get_submodule(group.inspect)).float()
        linears, quantizers = [], []
        for node in group.consumers:
            relative = node.module[len(group.inspect) :].lstrip(".")
            linears.append(block.get_submodule(relative))
            fmt = formats.get((f"linear.{node.canon}.weight", layer_index))
            quantizers.append(
                _identity if fmt is None else (lambda w, fmt=fmt: fake_quantize(w, fmt, "weight"))
            )
        block_kwargs = _cast(kwargs, torch.float32) if group.inspect == layout.attention_path else {}
        block_kwargs.pop("use_cache", None)
        x = feats[group.consumers[0].module]
        scales, ratio = search_scales(block, linears, quantizers, x, block_kwargs, src.n_grid)
        out[f"scale.{layer_index}.{group.position}"] = scales.cpu()
        out[f"ratio.{layer_index}.{group.position}"] = torch.tensor(ratio)
        for node in group.consumers:
            scaled[node.module] = scales
        if group.prev.kind == "linear":
            row_scaled[group.prev.module] = scales
    if src.clip_search is None:
        return
    search = src.clip_search
    for path, canon in layout.linears.items():
        if path.rsplit(".", 1)[-1] in search.skip:
            continue
        fmt = formats.get((f"linear.{canon}.weight", layer_index))
        if fmt is None:
            continue
        weight = layer.get_submodule(path).weight.detach().float()
        x = feats[path]
        if path in scaled:
            weight = weight * scaled[path].view(1, -1)
            x = x / scaled[path].view(1, -1)
        if path in row_scaled:
            weight = weight / row_scaled[path].view(-1, 1)
        out[f"clip.{layer_index}.{canon}"] = search_clip(
            weight, x, fmt, search.n_grid, search.max_shrink, search.n_tokens
        ).cpu()


def _apply_awq_layer(
    net: Net, layer_index: int, pairs: list[tuple[ScalingEntry, Group]], tensors: dict[str, torch.Tensor]
) -> None:
    layer = net.layers[layer_index]
    for _entry, group in pairs:
        name = f"scale.{layer_index}.{group.position}"
        if name not in tensors:
            continue
        scales = tensors[name]
        _fold_offline(net, layer_index, group, scales)
    for path, canon in net.layout.linears.items():
        name = f"clip.{layer_index}.{canon}"
        if name in tensors:
            apply_clip(layer.get_submodule(path).weight, tensors[name])


def _awq(ctx: Context, entries: list[ScalingEntry], result: StageResult) -> None:
    net = ctx.net
    if ctx.downstream is None:
        raise EngineError(
            "awq searches against the quantizer's weight rules; this job has no quantizer rules"
        )
    if len({e.source.model_dump_json() for e in entries}) != 1:
        raise EngineError("the awq entries of one step must share the same source parameters")
    pairs = [(entry, _group(net, entry)) for entry in entries]
    formats = _weight_formats(ctx)
    layers = sorted({i for entry, _ in pairs for i in net.layer_indices(entry.layers)})
    windows = ctx.windows()
    if len({w.shape[1] for w in windows}) != 1:
        raise EngineError("awq needs calibration windows of equal length (sampling: concat_blocks)")
    weight_rules = [
        r.model_dump(mode="json") for r in ctx.downstream if any(s.endswith(".weight") for s in r.sites)
    ]
    key = artifact_key(
        model=ctx.model_id,
        calibration=ctx.calibration,
        statistic="awq:" + pairs[0][0].source.model_dump_json() + ":" + ",".join(e.position for e in entries),
        upstream=ctx.upstream,
        downstream=weight_rules,
    )

    def compute() -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        feed_paths = list(net.layout.linears)
        with attached_online(net.model, list(ctx.pending_online)), LayerRunner(
            net.model, windows, propagate="original"
        ) as runner:
            for index, layer in runner:
                if index not in layers:
                    continue
                feats = _capture_inputs(layer, runner.inputs, runner.forward, feed_paths)
                _awq_layer(net, index, layer, pairs, formats, feats, runner.kwargs, out)
        return out

    tensors = ctx.artifacts.get_or_compute(key, compute)
    result.artifacts["awq.scales"] = ArtifactRef(key, "searched scales, ratios and clip limits per layer")
    for index in layers:
        _apply_awq_layer(net, index, pairs, tensors)
    for entry, group in pairs:
        ratios = [tensors[f"ratio.{i}.{group.position}"].item() for i in layers]
        result.edits.append(
            f"{entry.position}: awq scales folded in {len(layers)} layers (mean searched ratio "
            f"{sum(ratios) / len(ratios):.3f})"
        )
    clipped = sum(name.startswith("clip.") for name in tensors)
    if clipped:
        result.edits.append(f"awq: clipped {clipped} weight matrices")
