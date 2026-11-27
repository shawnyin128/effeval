"""Layer graph of a dense decoder architecture and the preprocessing positions it offers.

The graph describes one decoder layer plus the global modules around the layers (embedding, final norm,
lm_head). It is data (extracted by `ops.introspect`, shipped in `core/data/architectures/`), so this module is
torch-free. `positions` derives, from the graph and a model's shape, every place a rotation or scaling can be
applied and whether each kind x mode combination is valid there.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from effeval.core.fragments import LayerSelector
from effeval.core.types import StrictModel

NodeKind = Literal["residual", "norm", "linear", "rope", "matmul", "softmax", "add", "act", "mul", "route"]
OK = "ok"

# Hadamard matrices of size K * 2^m exist for these K: QuaRot's (hadamard_utils.get_hadK) and Paley I.
_HADAMARD_BASES = (1, 12, 20, 28, 36, 40, 52, 60, 68, 108, 140, 152, 156, 172, 200)


class ModelShape(StrictModel):
    """Dimensions of one checkpoint, read from its config.json."""

    hidden: int
    intermediate: int
    num_layers: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int
    tie_word_embeddings: bool
    num_experts: int | None = Field(default=None, description="MoE only: routed experts per MoE block.")
    experts_per_token: int | None = Field(default=None, description="MoE only: experts a token is sent to.")
    moe_intermediate: int | None = Field(default=None, description="MoE only: width of one routed expert.")
    shared_intermediate: int | None = Field(
        default=None, description="MoE only: width of the shared expert MLP, if the model has one."
    )

    @model_serializer(mode="wrap")
    def _omit_unset_moe_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        for key in ("num_experts", "experts_per_token", "moe_intermediate", "shared_intermediate"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class GraphNode(StrictModel):
    id: str
    kind: NodeKind
    module: str | None = Field(
        default=None, description="Module path inside a decoder layer, if the node is one."
    )
    canon: str | None = Field(
        default=None, description="Canonical linear name (`attn.q_proj`), linears only."
    )
    block: Literal["attn", "mlp"] | None = None
    gain: bool | None = Field(default=None, description="Norm only: has an elementwise gain.")
    axis: Literal["hidden", "head_dim"] | None = Field(
        default=None, description="Norm only: normalised axis."
    )
    fn: str | None = Field(default=None, description="Activation only: function name.")
    out_site: str | None = Field(default=None, description="Quant site of this node's output, if it has one.")
    layers: LayerSelector | None = Field(
        default=None, description="Decoder layers that have this node; None means every layer."
    )
    experts: bool | None = Field(
        default=None,
        description="True when `module` has a `*` where the routed expert index goes (one per expert).",
    )

    @model_serializer(mode="wrap")
    def _omit_unset_moe_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        for key in ("layers", "experts"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class GraphGlobals(StrictModel):
    embedding: str
    final_norm: str
    lm_head: str
    norm_kind: Literal["rms", "layer"]


class LayerGraph(StrictModel):
    """One decoder layer as nodes (modules and operators) and activation edges `(from, to, site)`."""

    version: int = 1
    globals: GraphGlobals | None = None
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[tuple[str, str, str | None]] = Field(default_factory=list)
    unsupported: list[str] = Field(
        default_factory=list, description="Reasons no graph is offered, e.g. ['moe']; nodes are then empty."
    )

    @model_validator(mode="after")
    def _edges_name_nodes(self) -> LayerGraph:
        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate node ids in {ids}")
        for src, dst, _site in self.edges:
            if src not in ids or dst not in ids:
                raise ValueError(f"edge {src}->{dst} names an unknown node")
        if not self.unsupported and self.globals is None:
            raise ValueError("a supported graph needs globals")
        return self

    def node(self, node_id: str) -> GraphNode:
        return next(n for n in self.nodes if n.id == node_id)

    def producers(self, node_id: str) -> list[tuple[GraphNode, str | None]]:
        return [(self.node(s), site) for s, d, site in self.edges if d == node_id]

    def consumers(self, node_id: str) -> list[tuple[GraphNode, str | None]]:
        return [(self.node(d), site) for s, d, site in self.edges if s == node_id]

    def of_kind(self, kind: NodeKind) -> list[GraphNode]:
        return [n for n in self.nodes if n.kind == kind]


def merge_overrides(graph: LayerGraph, overrides: dict[str, Any]) -> LayerGraph:
    """Apply hand fixes to an extracted graph.

    `overrides` may hold `nodes` ({node id: {field: value}}), `edges_remove` and `edges_add` (lists of
    `[from, to, site]`). Raises ValueError on an unknown node id or an edge to remove that does not exist.
    """
    unknown = set(overrides) - {"nodes", "edges_remove", "edges_add"}
    if unknown:
        raise ValueError(f"unknown graph override keys {sorted(unknown)}")
    doc = graph.model_dump()
    by_id = {n["id"]: n for n in doc["nodes"]}
    for node_id, fields in overrides.get("nodes", {}).items():
        if node_id not in by_id:
            raise ValueError(f"graph override names unknown node {node_id!r}")
        by_id[node_id].update(fields)
    edges = [tuple(e) for e in doc["edges"]]
    for edge in overrides.get("edges_remove", []):
        if tuple(edge) not in edges:
            raise ValueError(f"graph override removes missing edge {edge}")
        edges.remove(tuple(edge))
    edges += [tuple(e) for e in overrides.get("edges_add", [])]
    doc["edges"] = edges
    return LayerGraph.model_validate(doc)


def hadamard_size_ok(n: int) -> bool:
    """Whether a Hadamard matrix of size `n` exists here (n = K * 2^m, K a known base)."""
    if n < 1:
        return False
    return any(n % k == 0 and (n // k) & (n // k - 1) == 0 for k in _HADAMARD_BASES)


Availability = dict[str, dict[str, str]]
"""kind -> mode -> `OK` or the reason the combination is not valid."""


def positions(graph: LayerGraph, shape: ModelShape) -> dict[str, Availability]:
    """Every preprocessing position of `graph` for a model of `shape`, with the validity of each kind x mode.

    Returns `{position id: {"rotation": {"offline": r, "online": r}, "scaling": {...}}}` where `r` is the
    string "ok" or the reason (a sentence fragment) the combination cannot be used. Position ids:
    `residual`, `norm.<module>.out`, `linear.<canon>.input`, `attn.v_head`, `attn.qk_pre_rope` (only with a
    rope node), `attn.qk_rope`. Insertion order is deterministic (graph order). A graph with `unsupported`
    entries has no positions.
    """
    if graph.unsupported:
        return {}
    out: dict[str, Availability] = {}
    gqa = shape.kv_heads != shape.heads
    q_norm = any(n.kind == "norm" and n.axis == "head_dim" for n in graph.nodes)

    out["residual"] = _residual(graph)

    for norm in graph.of_kind("norm"):
        if norm.axis != "hidden":
            continue
        gain_reason = "the norm has no gain to fold the scale into"
        out[f"norm.{norm.module}.out"] = _cell(
            rot_off="the producer is a norm with an elementwise gain; rotate the residual instead",
            rot_on="each linear gets its own matrix: use the per-linear inputs",
            sc_off=OK if norm.gain else gain_reason,
            sc_on=OK,
        )

    for lin in graph.of_kind("linear"):
        [(producer, _site)] = graph.producers(lin.id)
        out[f"linear.{lin.canon}.input"] = _linear_input(graph, lin, producer, gqa)

    v_lin, o_lin = _value_and_output(graph)
    if v_lin is not None and o_lin is not None:
        out["attn.v_head"] = _cell(
            rot_off=OK,
            rot_on=OK,
            sc_off=OK,
            sc_on="fully foldable offline, nothing needs to run online",
        )

    if graph.of_kind("rope"):
        out["attn.qk_pre_rope"] = _qk_pre_rope(graph, q_norm)
        between = "q_norm/k_norm gains and RoPE sit" if q_norm else "RoPE sits"
        out["attn.qk_rope"] = _cell(
            rot_off=f"{between} between q/k_proj and QK^T",
            rot_on=OK,
            sc_off="RoPE pairs channels, so an offline scale must sit before RoPE: use attn.qk_pre_rope",
            sc_on=OK,
        )
    return out


def _qk_pre_rope(graph: LayerGraph, q_norm: bool) -> Availability:
    """Q/K before RoPE: only scaling folds, into the q/k linears or into head_dim norm gains."""
    [rope] = graph.of_kind("rope")
    blocked = next(
        (
            p
            for p, _site in graph.producers(rope.id)
            if not (p.kind == "linear" or (p.kind == "norm" and p.axis == "head_dim" and p.gain))
        ),
        None,
    )
    return _cell(
        rot_off="only a rotation inside each RoPE channel pair commutes with RoPE, and it cannot move an "
        "outlier out of its pair; not offered"
        + ("; q_norm/k_norm gains also block the fold" if q_norm else ""),
        rot_on="an online rotation before RoPE would have to be undone before RoPE; rotate after RoPE "
        "(attn.qk_rope)",
        sc_off=OK if blocked is None else f"{blocked.id} has no weight or gain to fold the scale into",
        sc_on="fully foldable offline, nothing needs to run online",
    )


def _cell(*, rot_off: str, rot_on: str, sc_off: str, sc_on: str) -> Availability:
    return {
        "rotation": {"offline": rot_off, "online": rot_on},
        "scaling": {"offline": sc_off, "online": sc_on},
    }


def _residual(graph: LayerGraph) -> Availability:
    assert graph.globals is not None
    stream = {n.id for n in graph.nodes if n.kind in ("residual", "add")}
    reason = OK
    if graph.globals.norm_kind != "rms":
        reason = "a LayerNorm subtracts the mean, which a rotation does not commute with"
    for src, dst, _site in graph.edges:
        if src in stream and dst not in stream and graph.node(dst).kind != "norm":
            reason = f"{dst} reads the residual stream and is not a norm"
        if dst in stream and src not in stream and graph.node(src).kind not in ("linear", "route"):
            reason = f"{src} writes the residual stream and is not a linear"
    return _cell(
        rot_off=reason,
        rot_on="one matrix is shared by the embedding, every layer and lm_head; nothing runs at inference",
        sc_off="the norm over the whole vector is not invariant to per-channel scaling",
        sc_on="the norm over the whole vector is not invariant to per-channel scaling",
    )


def _linear_input(graph: LayerGraph, lin: GraphNode, producer: GraphNode, gqa: bool) -> Availability:
    if producer.kind == "norm":
        siblings = len(graph.consumers(producer.id))
        rot_off = "same as the norm output: rotate the residual instead"
        sc_off = (
            OK
            if siblings == 1
            else f"the input is shared with sibling linears; use norm.{producer.module}.out"
        )
    elif producer.kind == "mul":
        act = next((p for p, _ in graph.producers(producer.id) if p.kind == "act"), None)
        gate = f"{act.fn}(gate)" if act is not None and act.fn else "the gate"
        rot_off = f"{gate} * up is a nonlinear product"
        has_branch = any(p.kind == "linear" for p, _ in graph.producers(producer.id))
        sc_off = OK if has_branch else "the product has no linear branch to fold the scale into"
    elif producer.kind == "linear":
        reason = (
            f"the input is the latent of {producer.canon}, whose width the decomposition chooses per layer; "
            "set the latent rotation in the method config"
        )
        return _cell(rot_off=reason, rot_on=reason, sc_off=reason, sc_on=reason)
    elif producer.kind == "matmul":
        rot_off = (
            "an across-head rotation does not commute: each head has its own softmax(QK)V; "
            "the per-head part is attn.v_head"
        )
        sc_off = "on GQA one value channel feeds several query heads' o_proj columns" if gqa else OK
    else:
        rot_off = "the input comes from the residual stream; rotate the residual instead"
        sc_off = "the producer has no weight to fold the scale into"
    return _cell(rot_off=rot_off, rot_on=OK, sc_off=sc_off, sc_on=OK)


def _value_and_output(graph: LayerGraph) -> tuple[GraphNode | None, GraphNode | None]:
    """The linear producing V (edge with site `attn.v`) and the linear consuming the attention output."""
    v = next((graph.node(s) for s, _d, site in graph.edges if site == "attn.v"), None)
    pv = next((graph.node(d) for _s, d, site in graph.edges if site == "attn.v"), None)
    o = None
    if pv is not None:
        o = next((c for c, _ in graph.consumers(pv.id) if c.kind == "linear"), None)
    return v, o


def attention_output_position(graph: LayerGraph) -> str | None:
    """The `linear.<canon>.input` position of the linear consuming the attention output, or None."""
    _v, o = _value_and_output(graph)
    return None if o is None else f"linear.{o.canon}.input"


def input_width(graph: LayerGraph, shape: ModelShape, lin: GraphNode) -> int:
    """Width of the channel a linear node reads: `hidden` after a norm, `heads * head_dim` after attention,
    and after a gating product the width of the MLP that product belongs to (routed expert, shared expert,
    dense).

    Raises ValueError when the graph needs a MoE dimension `shape` does not have, or when the producer is a
    linear (a latent of decomposition-chosen width).
    """
    [(producer, _site)] = graph.producers(lin.id)
    if producer.kind == "linear":
        raise ValueError(f"{lin.canon} reads the latent of {producer.canon}, whose width is chosen per layer")
    if producer.kind == "matmul":
        return shape.heads * shape.head_dim
    if producer.kind != "mul":
        return shape.hidden
    width = (
        shape.moe_intermediate
        if producer.experts
        else shape.shared_intermediate
        if producer.id == "shared_mul"
        else shape.intermediate
    )
    if width is None:
        raise ValueError(f"{producer.id} needs a MoE width the model shape does not have")
    return width
