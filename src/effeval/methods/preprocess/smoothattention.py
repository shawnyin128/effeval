"""SmoothAttention source: per-channel scales of the K cache (and, as a non-paper variant, the V cache).

Statistic: a[g, d] = max |x| of the attention operand (`attn.k`, or `attn.v` for `attn.v_head`) over all
calibration tokens, per KV head g and channel d; `granularity: shared` takes the maximum over g. The scale is
lambda = clamp(a, 1e-5) ** alpha.
- `attn.qk_pre_rope` (QServe): K is divided and Q multiplied by lambda before RoPE, folded offline into the
  q/k projections or, when RoPE's producers are head_dim norms (Qwen3), into the norm gains. RoPE rotates
  channel i with channel i + D/2, so lambda is tied over each pair (`pair_tie`) to commute with it.
- `attn.qk_rope` (online, no paper): K / lambda and Q * lambda as runtime ops on the post-RoPE operands.
- `attn.v_head` (offline, no paper): v_proj rows / s and o_proj columns * s, s tied over each KV group.
"""

from __future__ import annotations

import torch
from torch import nn

from effeval.core.graph import GraphNode, LayerGraph, attention_output_position
from effeval.core.preprocess import ScalingEntry, SmoothAttentionScaling, source_position_problem
from effeval.methods.preprocess.engine import Context, EngineError, Net, StageResult, check_position, source
from effeval.ops.calibration import attention_operand_absmax
from effeval.ops.norms import HeadwiseRMSNorm, install_headwise_norm
from effeval.ops.online import OnlineOp
from effeval.ops.scaling import (
    HeadChannelScale,
    fold_into_norm_gain,
    scale_input_columns,
    scale_output_rows,
)

LAMBDA_MIN = 1e-5
ORDER = ("attn.qk_pre_rope", "attn.v_head", "attn.qk_rope")


def pair_tie(a: torch.Tensor) -> torch.Tensor:
    """Max over each RoPE channel pair (i, i + D/2) of `a [..., D]`, written back to both channels."""
    half = a.shape[-1] // 2
    tied = torch.maximum(a[..., :half], a[..., half:])
    return torch.cat([tied, tied], dim=-1)


def expand_to_query_heads(t: torch.Tensor, heads: int) -> torch.Tensor:
    """`t [Hkv, D]` repeated so that each KV head's row serves its `heads // Hkv` consecutive query heads."""
    return t.repeat_interleave(heads // t.shape[0], dim=0)


def _per_kv_head(t: torch.Tensor, kv_heads: int) -> torch.Tensor:
    """`t` as [Hkv, D]: a shared [D] scale is repeated for every KV head."""
    return t if t.ndim == 2 else t[None, :].expand(kv_heads, -1)


def _qk_producers(graph: LayerGraph) -> dict[str, tuple[GraphNode, GraphNode | None]]:
    """For "q" and "k": the projection linear and the head_dim norm between it and RoPE (None without one)."""
    [rope] = graph.of_kind("rope")
    out: dict[str, tuple[GraphNode, GraphNode | None]] = {}
    for producer, _site in graph.producers(rope.id):
        norm = producer if producer.kind == "norm" else None
        linear = graph.producers(producer.id)[0][0] if norm is not None else producer
        out["q" if linear.canon == "attn.q_proj" else "k"] = (linear, norm)
    return out


def _value_and_output_nodes(graph: LayerGraph) -> tuple[GraphNode, GraphNode]:
    """The linear producing V (edge site `attn.v`) and the linear fed by the attention output."""
    edge = next(((s, d) for s, d, site in graph.edges if site == "attn.v"), None)
    o = None if edge is None else next((c for c, _ in graph.consumers(edge[1]) if c.kind == "linear"), None)
    if edge is None or o is None:
        raise EngineError("the architecture graph has no value -> output projection path")
    return graph.node(edge[0]), o


def _multiply_rows(linear: nn.Linear, factor: torch.Tensor) -> None:
    """Multiply output row i of the weight (and bias) by factor[i], in float64."""
    factor = factor.to(linear.weight.device).double()
    linear.weight.data.copy_((linear.weight.double() * factor[:, None]).to(linear.weight.dtype))
    if linear.bias is not None:
        linear.bias.data.copy_((linear.bias.double() * factor).to(linear.bias.dtype))


def _divide_rows(linear: nn.Linear, factor: torch.Tensor) -> None:
    linear.weight.data.copy_(scale_output_rows(linear.weight, factor))
    if linear.bias is not None:
        linear.bias.data.copy_(scale_output_rows(linear.bias[:, None], factor)[:, 0])


def _headwise_norm(net: Net, layer: int, node: GraphNode, heads: int) -> HeadwiseRMSNorm:
    parent_path, _, name = node.module.rpartition(".")
    attention = net.layers[layer].get_submodule(parent_path)
    existing = getattr(attention, name)
    return (
        existing if isinstance(existing, HeadwiseRMSNorm) else install_headwise_norm(attention, name, heads)
    )


def fold_qk_pre_rope(net: Net, layer: int, lam: torch.Tensor) -> None:
    """Divide K by `lam` and multiply Q by `lam[g(h)]` before RoPE in decoder layer `layer`.

    `lam` is [Hkv, D] or [D] (shared by all KV heads). Producers that are linears get their rows scaled;
    head_dim norms get their gain scaled, a [D] `lam` keeping the shared gain and a [Hkv, D] one replacing
    the norm by a HeadwiseRMSNorm. `lam` must be equal on the two channels of every RoPE pair for the model
    function to be kept.
    """
    shape = net.shape
    producers = _qk_producers(net.graph)
    for side, heads in (("k", shape.kv_heads), ("q", shape.heads)):
        linear_node, norm_node = producers[side]
        per_head = _per_kv_head(lam, shape.kv_heads)
        if side == "q":
            per_head = expand_to_query_heads(per_head, shape.heads)
        divide = side == "k"
        if norm_node is None:
            linear = net.module(layer, linear_node)
            (_divide_rows if divide else _multiply_rows)(linear, per_head.reshape(-1))
            continue
        if lam.ndim == 1:
            gain = net.module(layer, norm_node).weight
            factor = lam
        else:
            gain = _headwise_norm(net, layer, norm_node, heads).weight
            factor = per_head
        factor = factor.to(gain.device)
        if divide:
            gain.data.copy_(fold_into_norm_gain(gain, factor))
        else:
            gain.data.copy_((gain.double() * factor.double()).to(gain.dtype))


def fold_v_head(net: Net, layer: int, s: torch.Tensor) -> None:
    """Divide V by `s` in v_proj's rows and multiply o_proj's columns by `s[g(h)]` in decoder layer `layer`.

    `s` is [Hkv, D] or [D] (shared by all KV heads).
    """
    shape = net.shape
    v_node, o_node = _value_and_output_nodes(net.graph)
    per_kv = _per_kv_head(s, shape.kv_heads)
    _divide_rows(net.module(layer, v_node), per_kv.reshape(-1))
    o_proj = net.module(layer, o_node)
    columns = expand_to_query_heads(per_kv, shape.heads).reshape(-1)
    o_proj.weight.data.copy_(scale_input_columns(o_proj.weight, columns))


def _scales(
    stats: dict[int, torch.Tensor], src: SmoothAttentionScaling, tie: bool
) -> dict[int, torch.Tensor]:
    """Per layer lambda = clamp(a, 1e-5) ** alpha from the [Hkv, D] maxima, with the granularity applied."""
    out = {}
    for layer, a in stats.items():
        if tie:
            a = pair_tie(a)
        if src.granularity == "shared":
            a = a.amax(dim=0)
        out[layer] = a.clamp(min=LAMBDA_MIN).pow(src.alpha)
    return out


def _range(scales: dict[int, torch.Tensor]) -> str:
    low = min(s.min().item() for s in scales.values())
    high = max(s.max().item() for s in scales.values())
    return f"scale range [{low:.4g}, {high:.4g}]"


def _pre_rope(ctx: Context, entry: ScalingEntry, result: StageResult) -> None:
    net, src = ctx.net, entry.source
    layers = net.layer_indices(entry.layers)
    stats = attention_operand_absmax(net.model, ctx.windows(), "attn.k", layers, [])
    scales = _scales(stats, src, tie=True)
    producers = _qk_producers(net.graph)
    target = "q/k_norm gains" if producers["k"][1] is not None else "q_proj/k_proj rows"
    for layer in layers:
        fold_qk_pre_rope(net, layer, scales[layer])
    result.edits.append(
        f"attn.qk_pre_rope: smoothattention alpha {src.alpha} ({src.granularity}) folded into {target} "
        f"of {len(layers)} layers, {_range(scales)}"
    )


def _v_head(ctx: Context, entry: ScalingEntry, result: StageResult) -> None:
    net, src = ctx.net, entry.source
    output = attention_output_position(net.graph)
    for op in ctx.pending_online:
        if op.site in ("attn.v", output):
            raise EngineError(
                f"scaling at attn.v_head after an online op at {op.site}: put the scaling step first"
            )
    layers = net.layer_indices(entry.layers)
    stats = attention_operand_absmax(net.model, ctx.windows(), "attn.v", layers, [])
    scales = _scales(stats, src, tie=False)
    for layer in layers:
        fold_v_head(net, layer, scales[layer])
    result.edits.append(
        f"attn.v_head: smoothattention alpha {src.alpha} ({src.granularity}) folded into v_proj rows and "
        f"o_proj columns of {len(layers)} layers, {_range(scales)}"
    )


def _qk_rope(ctx: Context, entry: ScalingEntry, result: StageResult) -> None:
    net, src = ctx.net, entry.source
    layers = net.layer_indices(entry.layers)
    stats = attention_operand_absmax(net.model, ctx.windows(), "attn.k", layers, list(ctx.pending_online))
    scales = _scales(stats, src, tie=False)
    for layer in layers:
        lam = scales[layer]
        q_lam = lam if lam.ndim == 1 else expand_to_query_heads(lam, net.shape.heads)
        result.online.append(OnlineOp("attn.k", HeadChannelScale(1.0 / lam), [layer]))
        result.online.append(OnlineOp("attn.q", HeadChannelScale(q_lam), [layer]))
    result.edits.append(
        f"attn.qk_rope: smoothattention alpha {src.alpha} ({src.granularity}) online K / scale and "
        f"Q * scale on {len(layers)} layers, {_range(scales)}"
    )


@source("smoothattention", "scaling")
def apply_smoothattention(
    model: nn.Module, graph: LayerGraph, entries: list[ScalingEntry], ctx: Context
) -> StageResult:
    """Apply the step's SmoothAttention entries: `attn.qk_pre_rope`, then `attn.v_head`, then `attn.qk_rope`.

    The offline folds go first so that the online statistic sees them; each entry has its own calibration
    pass. Raises EngineError for a position that is unavailable or not served by this source, a duplicate
    position, a missing calibration fragment, or a V scaling after an online op on V or on o_proj's input.
    """
    by_position: dict[str, ScalingEntry] = {}
    for entry in entries:
        check_position(ctx, "scaling", entry)
        problem = source_position_problem("scaling", entry.source.type, entry.position)
        if problem is not None:
            raise EngineError(problem)
        if entry.position in by_position:
            raise EngineError(f"position {entry.position!r} appears twice in one step")
        by_position[entry.position] = entry
    if ctx.calibration is None:
        raise EngineError("smoothattention needs a calibration fragment")
    result = StageResult()
    handlers = {"attn.qk_pre_rope": _pre_rope, "attn.v_head": _v_head, "attn.qk_rope": _qk_rope}
    for position in ORDER:
        if position in by_position:
            handlers[position](ctx, by_position[position], result)
    return result
