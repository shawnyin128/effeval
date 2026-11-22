"""Rotation sources (QuaRot-style): residual-stream rotation, value/output head rotation, online rotations.

All edits keep the full-precision function of the model unchanged. Conventions: activations are row vectors,
a rotation R replaces x by x @ R, consumer weights become W @ R, producer weights R.T @ W (`ops.rotation`).
Positions handled: `residual` (offline), `attn.v_head` (offline: v_proj rows and o_proj columns;
online: runtime op on the V operand and o_proj columns), `linear.<canon>.input` (online, weight half
folded offline), `attn.qk_rope` (online on Q and K after RoPE). At `attn.v_head` and `attn.qk_rope` the
option `per_kv_head` draws one matrix per KV head (seed + head index) instead of one shared matrix.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from effeval.core.graph import GraphNode, LayerGraph, hadamard_size_ok
from effeval.core.preprocess import (
    Entry,
    RotationEntry,
    plain_hadamard_ok,
    position_dim,
    rotation_option_problem,
    rotation_size_problem,
)
from effeval.methods.preprocess.engine import (
    Context,
    EngineError,
    Net,
    StageResult,
    check_position,
    source,
)
from effeval.ops import rotation as rot
from effeval.ops.online import HeadMix, HeadwiseRightMultiply, OnlineOp, RightMultiply
from effeval.ops.scaling import scale_input_columns


def rotation_matrix(src: Any, n: int, position: str) -> torch.Tensor:
    """The float64 [n, n] orthogonal matrix a hadamard / random_hadamard / random_orthogonal source defines.

    Raises EngineError when a Hadamard-family source has no valid matrix of that size at `position`.
    """
    problem = rotation_size_problem(src, position, n)
    if problem is not None:
        raise EngineError(problem)
    if src.type == "hadamard":
        return rot.hadamard(n)
    if src.type == "random_hadamard":
        if plain_hadamard_ok(position, n):
            return rot.random_hadamard(n, src.seed)
        return rot.block_hadamard(n)
    if src.type == "random_orthogonal":
        return rot.random_orthogonal(n, src.seed)
    raise EngineError(f"{src.type!r} is not a matrix source")


def per_kv_head_matrices(src: Any, net: Net, position: str) -> torch.Tensor:
    """Float64 [kv_heads, head_dim, head_dim]: matrix g is drawn with seed `src.seed + g` for KV head g."""
    head_dim = net.shape.head_dim
    return torch.stack(
        [
            rotation_matrix(src.model_copy(update={"seed": src.seed + g}), head_dim, position)
            for g in range(net.shape.kv_heads)
        ]
    )


def expand_to_query_heads(matrices: torch.Tensor, heads: int) -> torch.Tensor:
    """[kv_heads, D, D] -> [heads, D, D]: query head h gets the matrix of KV head h // (heads // kv_heads)."""
    return matrices.repeat_interleave(heads // matrices.shape[0], dim=0)


def _set(param: nn.Parameter, value: torch.Tensor) -> None:
    param.data.copy_(value)


def _value_and_output_nodes(graph: LayerGraph) -> tuple[GraphNode, GraphNode]:
    """The linear producing V (edge site `attn.v`) and the linear fed by the attention output."""
    v = next((graph.node(s) for s, _d, site in graph.edges if site == "attn.v"), None)
    pv = next((graph.node(d) for _s, d, site in graph.edges if site == "attn.v"), None)
    o = None if pv is None else next((c for c, _ in graph.consumers(pv.id) if c.kind == "linear"), None)
    if v is None or o is None:
        raise EngineError("the architecture graph has no value -> output projection path")
    return v, o


def _rotate_bias(linear: nn.Linear, matrix: torch.Tensor, head_dim: int | None = None) -> None:
    """Rotate a producer's bias with its output: b @ R, per head_dim block when `head_dim` is given."""
    if linear.bias is None:
        return
    bias = linear.bias.double()
    matrix = matrix.to(bias.device).double()
    if head_dim is not None:
        bias = bias.reshape(-1, head_dim)
    _set(linear.bias, (bias @ matrix).reshape(-1).to(linear.bias.dtype))


def _rotate_bias_blocks(linear: nn.Linear, matrices: torch.Tensor) -> None:
    """Rotate a producer's bias per head_dim block i with matrices[i]: b_i @ R_i."""
    if linear.bias is None:
        return
    blocks = linear.bias.double().reshape(matrices.shape[0], 1, -1)
    rotated = blocks @ matrices.to(blocks.device).double()
    _set(linear.bias, rotated.reshape(-1).to(linear.bias.dtype))


def _rotate_residual(net: Net, entry: RotationEntry, result: StageResult) -> None:
    shape = net.shape
    options = entry.options
    q = rotation_matrix(entry.source, shape.hidden, "residual")
    embedding, lm_head = net.embedding, net.lm_head

    if lm_head.weight.data_ptr() == embedding.weight.data_ptr():
        if options.get("untie_embeddings") is not True:
            raise EngineError(
                "the model ties lm_head to the embedding; rotating the residual stream needs "
                "options.untie_embeddings: true"
            )
        lm_head.weight = nn.Parameter(embedding.weight.detach().clone())
        net.model.config.tie_word_embeddings = False
        net.model.config.get_text_config().tie_word_embeddings = False
        result.edits.append("residual: untied lm_head from the embedding")
    if options.get("center_embeddings", False):
        weight = embedding.weight.double()
        _set(embedding.weight, (weight - weight.mean(dim=-1, keepdim=True)).to(embedding.weight.dtype))
        result.edits.append("residual: centred embedding rows (QuaRot behaviour; not function-preserving)")

    layer_norms = net.stream_consumers()
    for norm_node, consumers in layer_norms:
        if not norm_node.gain:
            continue
        for layer in range(len(net.layers)):
            gain = net.module(layer, norm_node).weight
            for linear_node in consumers:
                linear = net.module(layer, linear_node)
                _set(linear.weight, scale_input_columns(linear.weight, gain))
            gain.data.fill_(1)
    final_gain = net.final_norm.weight
    _set(lm_head.weight, scale_input_columns(lm_head.weight, final_gain))
    final_gain.data.fill_(1)
    result.edits.append(
        f"residual: folded norm gains into {len(layer_norms)} input groups per layer and lm_head"
    )

    _set(embedding.weight, rot.fold_input(embedding.weight, q))
    _set(lm_head.weight, rot.fold_input(lm_head.weight, q))
    for layer in range(len(net.layers)):
        for _norm_node, consumers in layer_norms:
            for linear_node in consumers:
                linear = net.module(layer, linear_node)
                _set(linear.weight, rot.fold_input(linear.weight, q))
        for linear_node in net.stream_producers():
            linear = net.module(layer, linear_node)
            _set(linear.weight, rot.fold_output(linear.weight, q))
            _rotate_bias(linear, q)
    result.edits.append(
        f"residual: {entry.source.type} {shape.hidden}x{shape.hidden} folded into embedding, "
        f"{len(net.layers)} layers and lm_head"
    )


def _o_proj_online_op(
    heads: int,
    head_dim: int,
    full: torch.Tensor,
    value_matrix: torch.Tensor | None,
    plain_kron: bool,
    query_blocks: torch.Tensor | None = None,
) -> torch.nn.Module:
    """Online op for o_proj's input: `full` as a whole, or only the part the V side has already applied.

    `query_blocks` [heads, D, D] (per_kv_head) replaces the shared `value_matrix` on the block diagonal.
    """
    if value_matrix is None and query_blocks is None:
        return RightMultiply(full)
    if plain_kron:
        return HeadMix(rot.hadamard(heads), head_dim)
    if query_blocks is not None:
        per_head = torch.block_diag(*query_blocks.double())
    else:
        per_head = torch.kron(torch.eye(heads, dtype=torch.float64), value_matrix)
    return RightMultiply(per_head.T @ full)


def _rotate_attention_values(
    net: Net,
    v_entry: RotationEntry | None,
    o_entry: RotationEntry | None,
    result: StageResult,
) -> None:
    """`attn.v_head` (v_proj rows and o_proj columns per head) with `linear.attn.o_proj.input` online."""
    shape = net.shape
    heads, head_dim = shape.heads, shape.head_dim
    v_node, o_node = _value_and_output_nodes(net.graph)
    v_layers = set(net.layer_indices(v_entry.layers)) if v_entry else set()
    o_layers = set(net.layer_indices(o_entry.layers)) if o_entry else set()
    per_kv_head = v_entry is not None and v_entry.options.get("per_kv_head", False)
    value_matrix = kv_matrices = query_blocks = None
    if v_entry is not None:
        if per_kv_head:
            kv_matrices = per_kv_head_matrices(v_entry.source, net, "attn.v_head")
            query_blocks = expand_to_query_heads(kv_matrices, heads)
        else:
            value_matrix = rotation_matrix(v_entry.source, head_dim, "attn.v_head")
    online_v = v_entry is not None and v_entry.mode == "online"
    full = None if o_entry is None else rotation_matrix(o_entry.source, heads * head_dim, o_entry.position)

    if not online_v:
        for layer in sorted(v_layers):
            v_proj = net.module(layer, v_node)
            if per_kv_head:
                _set(v_proj.weight, rot.fold_output_per_head_blocks(v_proj.weight, kv_matrices))
                _rotate_bias_blocks(v_proj, kv_matrices)
            else:
                _set(v_proj.weight, rot.fold_output_per_head(v_proj.weight, value_matrix, head_dim))
                _rotate_bias(v_proj, value_matrix, head_dim)
    for layer in sorted(v_layers | o_layers):
        o_proj = net.module(layer, o_node)
        if layer in o_layers:
            _set(o_proj.weight, rot.fold_input(o_proj.weight, full))
        elif per_kv_head:
            _set(o_proj.weight, rot.fold_input_per_head_blocks(o_proj.weight, query_blocks))
        else:
            _set(o_proj.weight, rot.fold_input_per_head(o_proj.weight, value_matrix, head_dim))
    if v_entry:
        kind = "per-KV-head " if per_kv_head else ""
        if online_v:
            v_op = HeadwiseRightMultiply(kv_matrices) if per_kv_head else RightMultiply(value_matrix)
            result.online.append(OnlineOp("attn.v", v_op, v_entry.layers))
            where = "online on the V operand and folded into o_proj columns"
        else:
            where = "folded into v_proj rows and o_proj columns"
        result.edits.append(
            f"attn.v_head: {v_entry.source.type} {kind}{head_dim}x{head_dim} {where} "
            f"of {len(v_layers)} layers"
        )
    if o_entry is None:
        return
    both = sorted(v_layers & o_layers)
    only_o = sorted(o_layers - v_layers)
    plain_kron = (
        v_entry is not None
        and v_entry.source.type == "hadamard"
        and o_entry.source.type == "hadamard"
        and head_dim & (head_dim - 1) == 0
        and hadamard_size_ok(heads)
    )
    if both:
        op = _o_proj_online_op(heads, head_dim, full, value_matrix, plain_kron, query_blocks)
        result.online.append(OnlineOp(o_entry.position, op, both))
    if only_o:
        result.online.append(OnlineOp(o_entry.position, RightMultiply(full), only_o))
    result.edits.append(
        f"{o_entry.position}: {o_entry.source.type} {heads * head_dim}-wide matrix folded into o_proj; "
        f"online {'H_heads (x) I' if plain_kron else 'matrix'} on {len(both)} layers with attn.v_head, "
        f"full matrix on {len(only_o)} layers"
    )


def _rotate_linear_input(net: Net, entry: RotationEntry, graph: LayerGraph, result: StageResult) -> None:
    canon = entry.position[len("linear.") : -len(".input")]
    node = net.linear_node(canon)
    n = position_dim(graph, net.shape, entry.position)
    matrix = rotation_matrix(entry.source, n, entry.position)
    layers = net.layer_indices(entry.layers)
    for layer in layers:
        linear = net.module(layer, node)
        _set(linear.weight, rot.fold_input(linear.weight, matrix))
    result.online.append(OnlineOp(entry.position, RightMultiply(matrix), entry.layers))
    result.edits.append(
        f"{entry.position}: {entry.source.type} {n}x{n} online, weight half folded in {len(layers)} layers"
    )


def _rotate_qk(net: Net, entry: RotationEntry, result: StageResult) -> None:
    if entry.options.get("per_kv_head", False):
        kv_matrices = per_kv_head_matrices(entry.source, net, "attn.qk_rope")
        q_matrices = expand_to_query_heads(kv_matrices, net.shape.heads)
        result.online.append(OnlineOp("attn.q", HeadwiseRightMultiply(q_matrices), entry.layers))
        result.online.append(OnlineOp("attn.k", HeadwiseRightMultiply(kv_matrices), entry.layers))
        kind = "per-KV-head "
    else:
        op = RightMultiply(rotation_matrix(entry.source, net.shape.head_dim, "attn.qk_rope"))
        result.online.append(OnlineOp("attn.q", op, entry.layers))
        result.online.append(OnlineOp("attn.k", op, entry.layers))
        kind = ""
    result.edits.append(
        f"attn.qk_rope: {entry.source.type} {kind}{net.shape.head_dim}x{net.shape.head_dim} online on Q and K"
    )


@source("hadamard", "rotation")
@source("random_hadamard", "rotation")
@source("random_orthogonal", "rotation")
def apply_rotation(model: nn.Module, graph: LayerGraph, entries: list[Entry], ctx: Context) -> StageResult:
    """Apply the step's matrix-source rotation entries; the residual rotation goes first (it folds gains).

    Raises EngineError for an unsupported position, a duplicate position, or a size with no matrix.
    """
    net = ctx.net
    by_position: dict[str, RotationEntry] = {}
    for entry in entries:
        check_position(ctx, "rotation", entry)
        problem = rotation_option_problem(entry.position, entry.source, entry.options)
        if problem is not None:
            raise EngineError(problem)
        if entry.position in by_position:
            raise EngineError(f"position {entry.position!r} appears twice in one step")
        by_position[entry.position] = entry
    result = StageResult()
    if "residual" in by_position:
        _rotate_residual(net, by_position.pop("residual"), result)
    v_entry = by_position.pop("attn.v_head", None)
    _v_node, o_node = _value_and_output_nodes(graph)
    o_entry = by_position.pop(f"linear.{o_node.canon}.input", None)
    if v_entry or o_entry:
        _rotate_attention_values(net, v_entry, o_entry, result)
    qk_entry = by_position.pop("attn.qk_rope", None)
    if qk_entry:
        _rotate_qk(net, qk_entry, result)
    for position, entry in by_position.items():
        if not position.startswith("linear."):
            raise EngineError(f"rotation at {position!r} is not implemented")
        _rotate_linear_input(net, entry, graph, result)
    return result
