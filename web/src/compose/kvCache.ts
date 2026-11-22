// Frontend-only graph rewrite that draws the K and V caches as nodes between the projections and the attention matmuls.
import type { GraphNode, LayerGraph } from "./ArchDiagram";

const CACHES = [
  { site: "attn.k", id: "k_cache" },
  { site: "attn.v", id: "v_cache" },
] as const;

/**
 * Returns a copy of `graph` where each edge carrying the site `attn.k` / `attn.v` runs through a `cache` node
 * (`k_cache` / `v_cache`, `out_site` = that site): `src -> cache` keeps the site, `cache -> dst` carries none.
 * A graph without such an edge, or already holding the cache node, is returned unchanged.
 */
export function withKvCache(graph: LayerGraph): LayerGraph {
  let nodes = graph.nodes;
  let edges = graph.edges;
  for (const { site, id } of CACHES) {
    if (nodes.some((n) => n.id === id)) continue;
    const at = edges.findIndex(([, , s]) => s === site);
    if (at < 0) continue;
    const [src, dst] = edges[at];
    const cache: GraphNode = { id, kind: "cache", out_site: site };
    const srcAt = nodes.findIndex((n) => n.id === src);
    nodes = [...nodes.slice(0, srcAt + 1), cache, ...nodes.slice(srcAt + 1)];
    edges = [...edges.slice(0, at), [src, id, site], [id, dst, null], ...edges.slice(at + 1)];
  }
  return nodes === graph.nodes ? graph : { ...graph, nodes, edges };
}
