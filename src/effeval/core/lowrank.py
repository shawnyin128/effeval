"""Architecture layouts of low-rank decompositions and their rank budget; torch-free.

A decomposition replaces dense linears by factors. The rewritten layout is an `ArchitectureDef` derived from
the stored one, so rotation, site resolution and GPTQ keep reading it through `ops.sites.architecture`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from effeval.core.catalog import ArchitectureDef
from effeval.core.graph import GraphNode, LayerGraph, ModelShape

ARCHITECTURE_ATTR = "_effeval_architecture"
QKV_SHARED = "qkv_shared"
QKV_CANONS = ("attn.q_proj", "attn.k_proj", "attn.v_proj")
DOWN_CANON = "attn.qkv_down"
DOWN_NAME = "qkv_down"
UP_NAME = "up"


def factor_qkv_architecture(arch: ArchitectureDef) -> ArchitectureDef:
    """Layout of an attention whose q/k/v are `up(qkv_down(x))` with a shared down factor.

    The `qkv_down` linear sits between the norm and the three projections; each projection's module path gains
    `.up` and keeps its canonical name. Returns a copy with `graph_overrides` folded in. Raises ValueError
    unless the effective graph is supported and has exactly one q, k and v linear reading one norm.
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


ARCHITECTURE_TRANSFORMS: dict[str, Callable[[ArchitectureDef], ArchitectureDef]] = {
    QKV_SHARED: factor_qkv_architecture
}


def transform_architecture(arch: ArchitectureDef, name: str) -> ArchitectureDef:
    """Apply the named entry of `ARCHITECTURE_TRANSFORMS`; ValueError for an unknown name or layout."""
    if name not in ARCHITECTURE_TRANSFORMS:
        raise ValueError(f"unknown architecture transform {name!r}; known: {sorted(ARCHITECTURE_TRANSFORMS)}")
    return ARCHITECTURE_TRANSFORMS[name](arch)


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
