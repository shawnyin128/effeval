"""Architecture layouts of low-rank decompositions and their rank budget; torch-free.

A decomposition replaces dense linears by factors. The rewritten layout is an `ArchitectureDef` derived from
the stored one, so rotation, site resolution and GPTQ keep reading it through `ops.sites.architecture`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from effeval.core.catalog import ArchitectureDef
from effeval.core.graph import GraphNode, LayerGraph, ModelShape

ARCHITECTURE_ATTR = "_effeval_architecture"
QKV_SHARED = "qkv_shared"
LSVD = "lsvd"
QKV_CANONS = ("attn.q_proj", "attn.k_proj", "attn.v_proj")
DOWN_CANON = "attn.qkv_down"
DOWN_NAME = "qkv_down"
UP_NAME = "up"
QKV_DOWN_CANONS = ("attn.q_down", "attn.k_down", "attn.v_down")
FFN_CANONS = ("mlp.gate_proj", "mlp.up_proj")
SVD_DOWN_NAME, SVD_UP_NAME, DENSE_NAME = "svd_down", "svd_up", "dense"


def _qkv_nodes(arch: ArchitectureDef) -> tuple[LayerGraph, dict[str, GraphNode], str]:
    """Effective graph, the q/k/v linear nodes by canon, and the id of the one norm they read.

    Raises ValueError unless the effective graph is supported and has exactly one q, k and v linear reading
    one norm at `<attention_path>.<name>`.
    """
    graph = arch.effective_graph()
    if graph is None or graph.unsupported:
        raise ValueError(f"{arch.name} has no supported layer graph to factor q/k/v in")
    if arch.attention_path is None:
        raise ValueError(f"{arch.name} has no attention module path")
    nodes: dict[str, GraphNode] = {}
    for canon in QKV_CANONS:
        found = [n for n in graph.of_kind("linear") if n.canon == canon]
        if len(found) != 1:
            raise ValueError(
                f"{arch.name} has {len(found)} linear nodes for {canon}; factoring needs exactly one"
            )
        nodes[canon] = found[0]
    producers = {canon: graph.producers(node.id) for canon, node in nodes.items()}
    if any(len(p) != 1 for p in producers.values()):
        raise ValueError(f"{arch.name}: q/k/v must each have a single producer")
    norm_ids = {p[0][0].id for p in producers.values()}
    [norm_id] = norm_ids if len(norm_ids) == 1 else [None]
    if norm_id is None or graph.node(norm_id).kind != "norm":
        raise ValueError(f"{arch.name}: q/k/v must read the output of one shared norm")
    for canon, node in nodes.items():
        expected = f"{arch.attention_path}.{canon.split('.')[1]}"
        if node.module != expected:
            raise ValueError(f"{arch.name}: {canon} is at {node.module!r}, expected {expected!r}")
    return graph, nodes, norm_id


def factor_qkv_architecture(arch: ArchitectureDef) -> ArchitectureDef:
    """Layout of an attention whose q/k/v are `up(qkv_down(x))` with a shared down factor.

    The `qkv_down` linear sits between the norm and the three projections; each projection's module path gains
    `.up` and keeps its canonical name. Returns a copy with `graph_overrides` folded in. Raises ValueError
    unless the effective graph is supported and has exactly one q, k and v linear reading one norm.
    """
    graph, nodes, norm_id = _qkv_nodes(arch)
    if any(n.id == DOWN_NAME for n in graph.nodes):
        raise ValueError(f"{arch.name} already has a node {DOWN_NAME!r}")

    down_module = f"{arch.attention_path}.{DOWN_NAME}"
    down_site = f"linear.{DOWN_CANON}.input"
    q_id = nodes[QKV_CANONS[0]].id
    qkv_ids = {n.id for n in nodes.values()}
    doc = graph.model_dump()
    new_nodes: list[dict] = []
    for node in doc["nodes"]:
        if node["id"] == q_id:
            new_nodes.append(
                {
                    "id": DOWN_NAME,
                    "kind": "linear",
                    "module": down_module,
                    "canon": DOWN_CANON,
                    "block": "attn",
                }
            )
        if node["id"] in qkv_ids:
            node["module"] = f"{node['module']}.{UP_NAME}"
        new_nodes.append(node)
    new_edges: list[tuple[str, str, str | None]] = []
    down_edge_added = False
    for src, dst, site in doc["edges"]:
        if src == norm_id and dst in qkv_ids:
            if not down_edge_added:
                new_edges.append((norm_id, DOWN_NAME, down_site))
                down_edge_added = True
            new_edges.append((DOWN_NAME, dst, site))
        else:
            new_edges.append((src, dst, site))
    doc["nodes"], doc["edges"] = new_nodes, new_edges

    linears: dict[str, str] = {}
    for path, canon in arch.linears.items():
        if canon == QKV_CANONS[0]:
            linears[down_module] = DOWN_CANON
        linears[f"{path}.{UP_NAME}" if canon in QKV_CANONS else path] = canon
    return arch.model_copy(
        update={"linears": linears, "graph": LayerGraph.model_validate(doc), "graph_overrides": {}}
    )


def ffn_branch_canons(canon: str) -> tuple[str, str]:
    """Canons of the SVD branch's down and up linears of a gate/up canon.

    `mlp.gate_proj` -> (`mlp.gate_svd_down`, `mlp.gate_svd_up`). Raises ValueError for another canon.
    """
    if canon not in FFN_CANONS:
        raise ValueError(f"{canon!r} is not one of {FFN_CANONS}")
    stem = canon.removesuffix("_proj")
    return f"{stem}_{SVD_DOWN_NAME}", f"{stem}_{SVD_UP_NAME}"


def factor_qkv_per_head_architecture(arch: ArchitectureDef) -> ArchitectureDef:
    """Layout of an attention whose q, k and v each read their own down factor `<p>_down`.

    `<p>_down` sits between the norm and projection p; p's module path gains `.up` and keeps its canonical
    name. Raises ValueError under the same preconditions as `factor_qkv_architecture`.
    """
    graph, nodes, norm_id = _qkv_nodes(arch)
    by_id = {node.id: canon for canon, node in nodes.items()}
    down_canons = dict(zip(QKV_CANONS, QKV_DOWN_CANONS, strict=True))
    down_ids = {canon: down_canons[canon].split(".")[1] for canon in QKV_CANONS}
    if clash := {n.id for n in graph.nodes} & set(down_ids.values()):
        raise ValueError(f"{arch.name} already has nodes {sorted(clash)}")

    doc = graph.model_dump()
    new_nodes: list[dict] = []
    for node in doc["nodes"]:
        canon = by_id.get(node["id"])
        if canon is not None:
            new_nodes.append(
                {
                    "id": down_ids[canon],
                    "kind": "linear",
                    "module": f"{arch.attention_path}.{down_ids[canon]}",
                    "canon": down_canons[canon],
                    "block": "attn",
                }
            )
            node["module"] = f"{node['module']}.{UP_NAME}"
        new_nodes.append(node)
    new_edges: list[tuple[str, str, str | None]] = []
    for src, dst, site in doc["edges"]:
        canon = by_id.get(dst)
        if src == norm_id and canon is not None:
            new_edges.append((norm_id, down_ids[canon], f"linear.{down_canons[canon]}.input"))
            new_edges.append((down_ids[canon], dst, site))
        else:
            new_edges.append((src, dst, site))
    doc["nodes"], doc["edges"] = new_nodes, new_edges

    linears: dict[str, str] = {}
    for path, canon in arch.linears.items():
        if canon in QKV_CANONS:
            linears[f"{arch.attention_path}.{down_ids[canon]}"] = down_canons[canon]
            linears[f"{path}.{UP_NAME}"] = canon
        else:
            linears[path] = canon
    return arch.model_copy(
        update={"linears": linears, "graph": LayerGraph.model_validate(doc), "graph_overrides": {}}
    )


def hybrid_ffn_architecture(arch: ArchitectureDef, hybrid_layers: list[int] | None) -> ArchitectureDef:
    """Layout of an MLP whose gate and up each have a dense branch and, in `hybrid_layers`, an SVD branch.

    The gate/up module becomes `<path>.dense` (canon unchanged); the nodes `<c>_svd_down` and `<c>_svd_up`
    (modules `<path>.svd_down` / `<path>.svd_up`) exist in `hybrid_layers` only, None meaning every layer.
    Raises ValueError unless gate and up are single linears reading one norm and feeding `act` and `mul`.
    """
    graph = arch.effective_graph()
    if graph is None or graph.unsupported:
        raise ValueError(f"{arch.name} has no supported layer graph to split the FFN in")
    nodes: dict[str, GraphNode] = {}
    norm_ids: set[str] = set()
    sinks: dict[str, str] = {}
    for canon, sink_kind in zip(FFN_CANONS, ("act", "mul"), strict=True):
        found = [n for n in graph.of_kind("linear") if n.canon == canon]
        if len(found) != 1:
            raise ValueError(
                f"{arch.name} has {len(found)} linear nodes for {canon}; splitting needs exactly one"
            )
        node = found[0]
        if node.module != canon:
            raise ValueError(f"{arch.name}: {canon} is at {node.module!r}, expected {canon!r}")
        producers = graph.producers(node.id)
        if len(producers) != 1 or producers[0][0].kind != "norm":
            raise ValueError(f"{arch.name}: {canon} must read the output of one norm")
        consumers = graph.consumers(node.id)
        if len(consumers) != 1 or consumers[0][0].kind != sink_kind:
            raise ValueError(f"{arch.name}: {canon} must feed exactly one {sink_kind!r} node")
        nodes[canon] = node
        norm_ids.add(producers[0][0].id)
        sinks[canon] = consumers[0][0].id
    if len(norm_ids) != 1:
        raise ValueError(f"{arch.name}: gate and up must read the same norm")
    [norm_id] = norm_ids
    branch_ids = {canon: tuple(c.split(".")[1] for c in ffn_branch_canons(canon)) for canon in FFN_CANONS}
    if graph_ids := {n.id for n in graph.nodes} & {i for pair in branch_ids.values() for i in pair}:
        raise ValueError(f"{arch.name} already has nodes {sorted(graph_ids)}")

    by_id = {node.id: canon for canon, node in nodes.items()}
    layers = None if hybrid_layers is None else list(hybrid_layers)
    doc = graph.model_dump()
    new_nodes: list[dict] = []
    for node in doc["nodes"]:
        canon = by_id.get(node["id"])
        if canon is not None:
            node["module"] = f"{node['module']}.{DENSE_NAME}"
        new_nodes.append(node)
        if canon is not None:
            for node_id, name, branch in zip(
                branch_ids[canon], (SVD_DOWN_NAME, SVD_UP_NAME), ffn_branch_canons(canon), strict=True
            ):
                new_nodes.append(
                    {
                        "id": node_id,
                        "kind": "linear",
                        "module": f"{canon}.{name}",
                        "canon": branch,
                        "block": "mlp",
                        "layers": layers,
                    }
                )
    new_edges: list[tuple[str, str, str | None]] = []
    for src, dst, site in doc["edges"]:
        new_edges.append((src, dst, site))
        canon = by_id.get(dst)
        if src == norm_id and canon is not None:
            down_id, up_id = branch_ids[canon]
            down_canon, up_canon = ffn_branch_canons(canon)
            new_edges.append((norm_id, down_id, f"linear.{down_canon}.input"))
            new_edges.append((down_id, up_id, f"linear.{up_canon}.input"))
            new_edges.append((up_id, sinks[canon], None))
    doc["nodes"], doc["edges"] = new_nodes, new_edges

    linears: dict[str, str] = {}
    for path, canon in arch.linears.items():
        if canon in FFN_CANONS:
            down_canon, up_canon = ffn_branch_canons(canon)
            linears[f"{path}.{DENSE_NAME}"] = canon
            linears[f"{path}.{SVD_DOWN_NAME}"] = down_canon
            linears[f"{path}.{SVD_UP_NAME}"] = up_canon
        else:
            linears[path] = canon
    return arch.model_copy(
        update={"linears": linears, "graph": LayerGraph.model_validate(doc), "graph_overrides": {}}
    )


def lsvd_architecture(arch: ArchitectureDef, config: Mapping[str, Any]) -> ArchitectureDef:
    """Per-head q/k/v layout, plus the hybrid FFN layout for every layer when `config["ffn"]["enabled"]`."""
    layout = factor_qkv_per_head_architecture(arch)
    if config.get("ffn", {}).get("enabled", False):
        layout = hybrid_ffn_architecture(layout, None)
    return layout


ARCHITECTURE_TRANSFORMS: dict[str, Callable[[ArchitectureDef, Mapping[str, Any]], ArchitectureDef]] = {
    QKV_SHARED: lambda arch, _config: factor_qkv_architecture(arch),
    LSVD: lsvd_architecture,
}


def transform_architecture(
    arch: ArchitectureDef, name: str, config: Mapping[str, Any] | None = None
) -> ArchitectureDef:
    """Apply the named entry of `ARCHITECTURE_TRANSFORMS` with the method config (None = empty).

    Raises ValueError for an unknown name or an unsupported layout.
    """
    if name not in ARCHITECTURE_TRANSFORMS:
        raise ValueError(f"unknown architecture transform {name!r}; known: {sorted(ARCHITECTURE_TRANSFORMS)}")
    return ARCHITECTURE_TRANSFORMS[name](arch, config or {})


def per_head_budget(num_layers: int, hidden: int, head_dim: int, heads: int, rank_ratio: float) -> int:
    """Singular values kept per projection over all layers: `int(rank_ratio * num_layers * r_full) * heads`.

    `r_full = head_dim * hidden / (hidden + head_dim)` is the rank at which a head's factors hold as many
    parameters as the dense head.
    """
    r_full = head_dim * hidden / (hidden + head_dim)
    return int(rank_ratio * (num_layers * r_full)) * heads


def hybrid_ffn_sizes(d_ffn: int, hidden: int, svd_ratio: float, rank_ratio: float) -> tuple[int, int]:
    """`(n_svd, rank)`: channels on the SVD branch and its rank.

    `n_svd = min(max(1, int(d_ffn * svd_ratio)), d_ffn - 1)`; `rank = int(rank_ratio * n_svd * hidden /
    (n_svd + hidden))` clamped to `[1, min(n_svd, hidden)]`.
    """
    n_svd = min(max(1, int(d_ffn * svd_ratio)), d_ffn - 1)
    rank = int(rank_ratio * n_svd * hidden / (n_svd + hidden))
    return n_svd, min(max(1, rank), min(n_svd, hidden))


@dataclass(frozen=True)
class QkvBudget:
    """Rank budget of the shared-down q/k/v factorisation.

    `kept` singular values over all layers, `mean_rank` per layer, `param_ratio` the factored over the dense
    q/k/v parameters, `latent_ratio` the mean rank over the K/V width.
    """

    kept: int
    mean_rank: float
    param_ratio: float
    latent_ratio: float


def qkv_budget(shape: ModelShape, rank_ratio: float) -> QkvBudget:
    """Budget of `rank_ratio` as QSVD defines it: `int(rank_ratio / 2 * num_layers * hidden)` values kept."""
    kept = int(rank_ratio / 2 * (shape.num_layers * shape.hidden))
    mean_rank = kept / shape.num_layers
    out_total = (shape.heads + 2 * shape.kv_heads) * shape.head_dim
    return QkvBudget(
        kept=kept,
        mean_rank=mean_rank,
        param_ratio=mean_rank * (shape.hidden + out_total) / (shape.hidden * out_total),
        latent_ratio=mean_rank / (2 * shape.kv_heads * shape.head_dim),
    )
