// Interactive diagram of one decoder layer, drawn from an architecture graph (core/graph.py LayerGraph).
// Two selection modes: preprocessing positions (clickable points, validity from positions()) and
// quantization targets (weight boxes, activation edges, KV cache).
import { useId, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from "react";
import { useT } from "../i18n";
import { Input, Segmented } from "../ui/controls";
import { withKvCache } from "./kvCache";

export type NodeKind = "residual" | "norm" | "linear" | "rope" | "matmul" | "softmax" | "add" | "act" | "mul" | "cache" | "route";

export interface GraphNode {
  id: string;
  kind: NodeKind;
  module?: string | null;
  canon?: string | null;
  block?: "attn" | "mlp" | null;
  gain?: boolean | null;
  axis?: "hidden" | "head_dim" | null;
  fn?: string | null;
  out_site?: string | null;
  /** MoE: decoder layers that have this node (None = all). */
  layers?: unknown;
  /** MoE: `module` has a `*` for the routed expert index; the node stands for every expert. */
  experts?: boolean | null;
}

export interface GraphGlobals {
  embedding: string;
  final_norm: string;
  lm_head: string;
  norm_kind: "rms" | "layer";
}

export interface LayerGraph {
  version: number;
  globals: GraphGlobals | null;
  nodes: GraphNode[];
  edges: [string, string, string | null][];
  unsupported: string[];
}

export type PreprocessKind = "rotation" | "scaling";
export type ApplyMode = "offline" | "online";
/** `"ok"` or the reason the combination is not valid. */
export type Availability = Record<PreprocessKind, Record<ApplyMode, string>>;
export type PositionTable = Record<string, Availability>;

/** Shape of the JSON fixtures (and of the describe data WP7b will wire in). */
export interface ArchGraphPayload {
  architecture: string;
  layers: number;
  graph: LayerGraph;
  positions: PositionTable;
}

/** What is applied at a positions-mode point; `via` marks a rotation that comes from the shared residual matrix. */
export interface PointMark {
  kind: PreprocessKind;
  mode?: ApplyMode;
  via?: boolean;
}

/** Positions mode: how one point is drawn and whether it can be clicked. */
export interface PointState {
  clickable: boolean;
  /** Why the point cannot be used (shown on hover when not clickable). */
  reason?: string;
  /** Extra tooltip line for a clickable point. */
  note?: string;
  marks: PointMark[];
  /** Alternates 0/1 per change so the highlight animation restarts; set while the point was just changed. */
  flash?: 0 | 1;
  /** The point's marks are about to be removed. */
  leaving?: boolean;
}

export type LayerSelection = { kind: "all" } | { kind: "range"; from: number; to: number };

export interface ArchDiagramProps {
  graph: LayerGraph;
  /** "positions": preprocessing points. "targets": quantization targets (weights, activations, KV cache). */
  mode: "positions" | "targets";
  /** Targets mode: selected site names. */
  value?: string[];
  /** Targets mode: called with the next selection. */
  onChange?: (next: string[]) => void;
  /** Positions mode: availability per position id, from `positions()`; decides which points exist. */
  positions?: PositionTable;
  /** Positions mode: drawing state per position id. */
  points?: Record<string, PointState>;
  /** Positions mode: called with the id of a clicked clickable point. */
  onPoint?: (id: string) => void;
  /** Positions mode: content shown in a popover anchored at the last clicked point; `onDismiss` closes it. */
  popover?: { content: ReactNode; onDismiss: () => void };
  /** Targets mode: site name -> reason it cannot be selected. */
  disabled?: Record<string, string>;
  numLayers?: number;
  tied?: boolean;
  /** MoE models: routed experts per block and how many each token uses; labels the expert block. */
  moe?: { experts: number; topK?: number };
}

interface Item {
  id: string;
  label: string;
  x: number;
  y: number;
  shape: "dot" | "weight" | "chip";
  w?: number;
  h?: number;
  tag?: string;
  /** Draw the tag under the dot instead of above it. */
  tagBelow?: boolean;
  reason?: string;
  note?: string;
  marks?: PointMark[];
  /** Positions mode: where the pills of the dot are drawn. */
  pillSide?: PillSide;
  flash?: 0 | 1;
  leaving?: boolean;
}

interface Placed {
  node: GraphNode;
  row: number;
  x: number;
  y: number;
  w: number;
  h: number;
}

const GAP_X = 52;
const SLOT = 46;
const H = 30;
const CIRCLE = 26;
const PAD = 16;
const RAIL_GAP = 44;
const ROW_GAP = 8;
/** How far a cache node is moved off the line of an edge that bypasses it (the Q edge around the K cache). */
const CACHE_NUDGE = 30;
/** Distance from the layer output to the lm_head chip: box margin, final norm chip, gaps. */
const HEAD_OFFSET = 24 + 62 + 80;

/** Offset of each of the two shadow copies drawn behind an expert node, and the margin of the frame around them. */
const STACK = 3;
const FRAME_PAD = 5;

const NORM_LABEL = { rms: "RMS", layer: "LN" } as const;
const ACT_LABEL: Record<string, string> = { silu: "SiLU", gelu: "GELU", relu: "ReLU", swish: "Swish" };

const isRouter = (n: GraphNode) => n.kind === "linear" && !!n.canon && n.canon.endsWith(".router");

/** Quantization site of a linear node's tensor: the router has its own `moe.router.*` family, others `linear.<canon>.*`. */
function linearSite(n: GraphNode, tensor: "weight" | "input"): string {
  return isRouter(n) ? `moe.router.${tensor}` : `linear.${n.canon}.${tensor}`;
}

function weightSub(canon: string): string {
  const name = canon.split(".").pop() ?? canon;
  const base = name.replace(/_proj$/, "");
  return base.length === 1 ? base.toUpperCase() : base;
}

function nodeSize(n: GraphNode): { w: number; h: number } {
  if (n.kind === "linear") return { w: 56, h: H };
  if (n.kind === "matmul" || n.kind === "mul" || n.kind === "add" || n.kind === "route") return { w: CIRCLE, h: CIRCLE };
  if (n.kind === "softmax") return { w: 62, h: 24 };
  if (n.kind === "cache") return { w: 62, h: 30 };
  if (n.kind === "residual") return { w: 28, h: 24 };
  return { w: 46, h: 24 };
}

function isBypass(graph: LayerGraph, src: string, dst: string): boolean {
  const s = graph.nodes.find((n) => n.id === src);
  const d = graph.nodes.find((n) => n.id === dst);
  return d?.kind === "add" && (s?.kind === "residual" || s?.kind === "add");
}

/**
 * Longest-path columns. The stream is cut after the first residual add: nodes before it form row 0 and the
 * rest row 1, so a layer is two rows (attention, MLP) of equal width: the spacing of the narrower row is
 * stretched. Within a column, slots follow the mean row of the producers.
 */
export function layout(graph: LayerGraph): { placed: Placed[]; width: number; rows: { top: number; bottom: number }[] } {
  const ids = graph.nodes.map((n) => n.id);
  const flow = graph.edges.filter(([s, d]) => !isBypass(graph, s, d));
  const depth = new Map<string, number>(ids.map((i) => [i, 0]));
  for (let pass = 0; pass < ids.length; pass++) {
    let changed = false;
    for (const [s, d] of graph.edges) {
      if ((depth.get(d) ?? 0) < (depth.get(s) ?? 0) + 1) {
        depth.set(d, (depth.get(s) ?? 0) + 1);
        changed = true;
      }
    }
    if (!changed) break;
  }
  const addDepths = graph.nodes.filter((n) => n.kind === "add").map((n) => depth.get(n.id)!);
  const maxDepth = Math.max(0, ...depth.values());
  const cut = addDepths.length > 0 && Math.min(...addDepths) < maxDepth ? Math.min(...addDepths) : maxDepth;
  const rowOf = (id: string) => (depth.get(id)! > cut ? 1 : 0);
  const colOf = (id: string) => depth.get(id)! - (rowOf(id) === 1 ? cut + 1 : 0);
  const sizes = new Map(graph.nodes.map((n) => [n.id, nodeSize(n)]));
  const rowCount = rowOf(ids[ids.length - 1]) === 1 || graph.nodes.some((n) => rowOf(n.id) === 1) ? 2 : 1;

  const colX = new Map<string, number>();
  const colWidths = Array.from({ length: rowCount }, (_, r) => {
    const members = graph.nodes.filter((n) => rowOf(n.id) === r);
    const cols = Math.max(...members.map((n) => colOf(n.id))) + 1;
    return Array.from({ length: cols }, (_, c) => Math.max(0, ...members.filter((n) => colOf(n.id) === c).map((n) => sizes.get(n.id)!.w)));
  });
  const natural = colWidths.map((ws) => ws.reduce((a, w) => a + w, 0) + (ws.length - 1) * GAP_X);
  const widths = natural.map(() => Math.max(...natural));
  colWidths.forEach((ws, r) => {
    const gap = ws.length > 1 ? (widths[r] - ws.reduce((a, w) => a + w, 0)) / (ws.length - 1) : GAP_X;
    let cursor = 0;
    ws.forEach((w, c) => {
      colX.set(`${r}:${c}`, cursor + w / 2);
      cursor += w + gap;
    });
  });

  const y = new Map<string, number>();
  /** Whether a same-row edge jumps over column `c` close to height `at`, which would run hidden behind a node placed there. */
  const passesBehind = (id: string, c: number, r: number, at: number) => flow.some(([s, d]) => s !== id && d !== id
    && rowOf(s) === r && rowOf(d) === r && colOf(s) < c && colOf(d) > c && Math.abs((y.get(s) ?? 0) - at) < SLOT);
  const order = [...graph.nodes].sort((a, b) => depth.get(a.id)! - depth.get(b.id)!);
  for (let r = 0; r < rowCount; r++) {
    const cols = Math.max(...graph.nodes.filter((n) => rowOf(n.id) === r).map((n) => colOf(n.id))) + 1;
    for (let c = 0; c < cols; c++) {
      const members = order.filter((n) => rowOf(n.id) === r && colOf(n.id) === c);
      const ideal = members.map((n, i) => {
        const preds = [...new Set(flow.filter(([, d]) => d === n.id).map(([s]) => s))].filter((p) => rowOf(p) === r);
        const ys = preds.map((p) => y.get(p) ?? 0);
        const mean = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
        return { n, order: i, y: n.kind === "cache" && passesBehind(n.id, c, r, mean) ? mean + CACHE_NUDGE : mean };
      });
      ideal.sort((a, b) => a.y - b.y || a.order - b.order);
      const slots: number[] = [];
      ideal.forEach((e, i) => slots.push(i === 0 ? e.y : Math.max(e.y, slots[i - 1] + SLOT)));
      const shift = ideal.reduce((a, e, i) => a + (e.y - slots[i]), 0) / ideal.length;
      ideal.forEach((e, i) => {
        const snap = ideal.length === 1 && ["add", "norm", "residual"].includes(e.n.kind);
        y.set(e.n.id, snap ? 0 : slots[i] + shift);
      });
    }
  }

  const local = graph.nodes.map((n) => {
    const { w, h } = sizes.get(n.id)!;
    const row = rowOf(n.id);
    return { node: n, row, x: colX.get(`${row}:${colOf(n.id)}`)!, y: y.get(n.id)!, w, h };
  });
  const span = (r: number) => {
    const ps = local.filter((p) => p.row === r);
    return { top: Math.min(...ps.map((p) => p.y - p.h / 2)), bottom: Math.max(...ps.map((p) => p.y + p.h / 2)) };
  };
  const rows = [span(0)];
  let placed = local;
  if (rowCount === 2) {
    const s1 = span(1);
    const shift = rows[0].bottom + RAIL_GAP + ROW_GAP + 30 - s1.top;
    placed = local.map((p) => (p.row === 1 ? { ...p, y: p.y + shift } : p));
    rows.push({ top: s1.top + shift, bottom: s1.bottom + shift });
  }
  return { placed, width: Math.max(...widths), rows };
}

/** Outline of a cylinder (the cache symbol) centred on the origin: body plus the front arc of the top cap. */
function cylinderPath(w: number, h: number): string {
  const rx = w / 2;
  const ry = 4;
  const top = -h / 2 + ry;
  return `M${-rx},${top} A${rx},${ry} 0 0 1 ${rx},${top} V${h / 2 - ry} A${rx},${ry} 0 0 1 ${-rx},${h / 2 - ry} Z M${-rx},${top} A${rx},${ry} 0 0 0 ${rx},${top}`;
}

function curve(x1: number, y1: number, x2: number, y2: number): string {
  const mx = (x1 + x2) / 2;
  return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
}

/** Point at fraction `f` of the cubic produced by `curve` (control points share the x midpoint). */
function curvePoint(x1: number, y1: number, x2: number, y2: number, f: number): { x: number; y: number } {
  const mx = (x1 + x2) / 2;
  const u = 1 - f;
  const at = (a: number, b: number, c: number, d: number) =>
    u * u * u * a + 3 * u * u * f * b + 3 * u * f * f * c + f * f * f * d;
  return { x: at(x1, mx, mx, x2), y: at(y1, y1, y2, y2) };
}

export interface Geometry {
  placed: Placed[];
  byId: Map<string, Placed>;
  edges: { src: Placed; dst: Placed; site: string | null; offset: number; bypass: boolean }[];
  width: number;
  /** y of the residual bypass line of each row. */
  rails: number[];
  /** y where a stream crossing from row 0 to row 1 runs horizontally. */
  crossY: number;
  top: number;
  bottom: number;
  /** The node nothing reads from: the layer output. */
  sink: Placed;
}

function useGeometry(graph: LayerGraph): Geometry {
  return useMemo(() => computeGeometry(graph), [graph]);
}

export function computeGeometry(graph: LayerGraph): Geometry {
  const { placed, width, rows } = layout(graph);
  const byId = new Map(placed.map((p) => [p.node.id, p]));
  const pairCount = new Map<string, number>();
  const pairSeen = new Map<string, number>();
  for (const [s, d] of graph.edges) pairCount.set(`${s}>${d}`, (pairCount.get(`${s}>${d}`) ?? 0) + 1);
  const edges = graph.edges.map(([s, d, site]) => {
    const key = `${s}>${d}`;
    const i = pairSeen.get(key) ?? 0;
    pairSeen.set(key, i + 1);
    const n = pairCount.get(key)!;
    return { src: byId.get(s)!, dst: byId.get(d)!, site, offset: (i - (n - 1) / 2) * 18, bypass: isBypass(graph, s, d) };
  });
  const sink = [...placed].reverse().find((p) => !edges.some((e) => e.src === p)) ?? placed[placed.length - 1];
  return {
    placed,
    byId,
    edges,
    width,
    rails: rows.map((r) => r.top - 30),
    crossY: rows[0].bottom + RAIL_GAP / 2,
    top: rows[0].top,
    bottom: rows[rows.length - 1].bottom,
    sink,
  };
}

function linearInputId(dst: Placed, posMode: boolean): string | null {
  if (dst.node.kind !== "linear" || !dst.node.canon) return null;
  return posMode ? `linear.${dst.node.canon}.input` : linearSite(dst.node, "input");
}

function describeId(id: string): string {
  return id.replace(/^(linear|moe)\./, "").replace(/\./g, " ").replace(/_/g, " ");
}

/** The selectable points of the diagram for the current mode. */
export function buildItems(
  graph: LayerGraph,
  geo: Geometry,
  props: Pick<ArchDiagramProps, "mode" | "positions" | "points" | "disabled">,
  t: (s: string, v?: Record<string, string | number>) => string,
): Item[] {
  const items = new Map<string, Item>();
  const add = (item: Item) => {
    if (!items.has(item.id)) items.set(item.id, item);
  };
  const posMode = props.mode === "positions";
  const table = props.positions ?? {};

  const reasonFor = (id: string): string | undefined => {
    if (!posMode) return props.disabled?.[id];
    const state = props.points?.[id];
    if (!state) return t("Not available for this model");
    return state.clickable ? undefined : state.reason;
  };
  const push = (id: string, label: string, x: number, y: number, extra: Partial<Item> = {}) => {
    if (posMode && !(id in table)) return;
    const state = posMode ? props.points?.[id] : undefined;
    add({ id, label, x, y, shape: "dot", reason: reasonFor(id), note: state?.note, marks: state?.marks, flash: state?.flash, leaving: state?.leaving, ...extra });
  };

  const first = geo.placed.find((p) => p.node.kind === "residual");
  if (posMode && first) {
    const target = geo.edges.find((e) => e.bypass && e.src === first);
    const x2 = target ? target.dst.x : first.x + 60;
    push("residual", t("Residual stream"), (first.x + x2) / 2, geo.rails[0], { shape: "chip", w: 96, h: 18, tag: t("residual") });
  }

  for (const p of geo.placed) {
    if (p.node.kind === "norm" && p.node.axis === "hidden" && p.node.module && posMode) {
      push(`norm.${p.node.module}.out`, `${p.node.module} ${t("output")}`, p.x + p.w / 2 + 14, p.y);
    }
    if (p.node.kind === "linear" && p.node.canon && !posMode) {
      const site = linearSite(p.node, "weight");
      add({
        id: site,
        label: `${weightSub(p.node.canon)} ${t("weight")}`,
        x: p.x,
        y: p.y,
        shape: "weight",
        w: p.w,
        h: p.h,
        reason: reasonFor(site),
      });
    }
    if (!posMode && p.node.kind === "cache" && p.node.out_site) {
      const kv = p.node.id === "k_cache" ? t("K cache") : t("V cache");
      add({ id: p.node.out_site, label: kv, x: p.x, y: p.y, shape: "chip", w: p.w, h: p.h, tag: kv, reason: reasonFor(p.node.out_site) });
    } else if (!posMode && p.node.out_site) {
      const outs = geo.edges.filter((e) => e.src === p);
      if (!outs.some((e) => e.site === p.node.out_site)) {
        push(p.node.out_site, describeId(p.node.out_site), p.x + p.w / 2 + 14, p.y);
      }
    }
  }

  const hasVHead = "attn.v_head" in table;
  const hasQkRope = "attn.qk_rope" in table;
  let qEdge: { x1: number; y1: number; x2: number; y2: number } | null = null;
  let kMid: { x: number; y: number } | null = null;
  const hasPreRope = "attn.qk_pre_rope" in table;
  const qk = new Map<string, { x: number; y: number }[]>();
  const addTo = (id: string, mid: { x: number; y: number }) => qk.set(id, [...(qk.get(id) ?? []), mid]);
  for (const e of geo.edges) {
    if (e.bypass) continue;
    const x1 = e.src.x + e.src.w / 2;
    const y1 = e.src.y + e.offset;
    const x2 = e.dst.x - e.dst.w / 2;
    const y2 = e.dst.y + e.offset;
    const mid = curvePoint(x1, y1, x2, y2, 0.5);
    if (e.site === "attn.q") qEdge = { x1, y1, x2, y2 };
    if (e.site === "attn.k") kMid = mid;
    const li = linearInputId(e.dst, posMode);
    if (li) push(li, describeId(li), x2 - 14, y2);
    if (posMode) {
      if (hasVHead && e.site === "attn.v") push("attn.v_head", t("v_proj output per head"), mid.x, mid.y, { tag: t("v head") });
      if (hasPreRope && e.dst.node.kind === "rope") addTo("attn.qk_pre_rope", mid);
    } else if (e.site && !li && e.site.startsWith("attn.") && e.dst.node.kind !== "cache") {
      push(e.site, describeId(e.site), mid.x, mid.y);
    }
  }
  if (posMode && hasQkRope && qEdge && kMid) {
    // The K edge ends at the K cache, the Q edge runs on to the matmul: take the Q edge point level with the K edge midpoint.
    const f = Math.min(0.95, Math.max(0.05, (kMid.x - qEdge.x1) / (qEdge.x2 - qEdge.x1)));
    addTo("attn.qk_rope", kMid);
    addTo("attn.qk_rope", curvePoint(qEdge.x1, qEdge.y1, qEdge.x2, qEdge.y2, f));
  }
  for (const [id, pts] of qk) {
    const mx = pts.reduce((a, p) => a + p.x, 0) / pts.length;
    const my = pts.reduce((a, p) => a + p.y, 0) / pts.length;
    if (id === "attn.qk_pre_rope") push(id, t("Q and K before RoPE"), mx, my, { tag: t("pre-RoPE") });
    else push(id, t("Q and K after RoPE"), mx, my, { tag: t("QK rope") });
  }

  if (!posMode && graph.globals) {
    const last = geo.sink;
    add({
      id: "lm_head.weight",
      label: `${graph.globals.lm_head} ${t("weight")}`,
      x: geo.sink.x + geo.sink.w / 2 + HEAD_OFFSET,
      y: last.y,
      shape: "chip",
      w: 64,
      h: 24,
      tag: "lm_head",
      reason: reasonFor("lm_head.weight"),
    });
  }
  const list = [...items.values()];
  if (posMode) placePills(list, geo);
  return list;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

const PILL_W = 14;
const PILL_H = 12;
const PILL_GAP = 2;
const PILL_NEAR = 11;
const TAG_CHAR_W = 5.6;

/** Pills of a dot are stacked above or below it, or laid side by side above or below it. */
export type PillSide = "above" | "below" | "above-row" | "below-row";
const PILL_SIDES: PillSide[] = ["above", "below", "above-row", "below-row"];

/** Centre of the `i`-th of `n` pills relative to the dot. */
function pillCenter(side: PillSide, i: number, n: number): { dx: number; dy: number } {
  const sign = side.startsWith("below") ? 1 : -1;
  if (side.endsWith("row")) return { dx: (i - (n - 1) / 2) * (PILL_W + PILL_GAP), dy: sign * (PILL_NEAR + PILL_H / 2) };
  return { dx: 0, dy: sign * (PILL_NEAR + i * (PILL_H + PILL_GAP) + PILL_H / 2) };
}

/** Boxes (diagram coordinates) of up to `n` pills of a dot item. */
export function pillBoxes(item: Pick<Item, "x" | "y">, side: PillSide, n = 2): Box[] {
  return Array.from({ length: n }, (_, i) => {
    const { dx, dy } = pillCenter(side, i, n);
    return { x1: item.x + dx - PILL_W / 2, y1: item.y + dy - PILL_H / 2, x2: item.x + dx + PILL_W / 2, y2: item.y + dy + PILL_H / 2 };
  });
}

/** Box of the tag text of a dot item (estimated from its length). */
export function tagBox(item: Item): Box | null {
  if (!item.tag || item.shape !== "dot") return null;
  const base = item.y + (item.tagBelow ? 20 : -13);
  const half = (item.tag.length * TAG_CHAR_W) / 2;
  return { x1: item.x - half, y1: base - 9, x2: item.x + half, y2: base + 2 };
}

const hits = (a: Box, b: Box) => a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2;

function itemBox(i: Item): Box {
  const hw = i.shape === "dot" ? 7 : i.w! / 2;
  const hh = i.shape === "dot" ? 7 : i.h! / 2;
  return { x1: i.x - hw, y1: i.y - hh, x2: i.x + hw, y2: i.y + hh };
}

/** Chooses per dot where its pills go so they avoid nodes, other dots, tags and other pills (the first side that is free, else the least crowded). */
function placePills(items: Item[], geo: Geometry): void {
  const nodes: Box[] = geo.placed.map((p) => ({ x1: p.x - p.w / 2, y1: p.y - p.h / 2, x2: p.x + p.w / 2, y2: p.y + p.h / 2 }));
  const fixed = [...nodes, ...items.map(itemBox), ...items.map(tagBox).filter((b): b is Box => !!b)];
  const taken: Box[] = [];
  for (const item of items) {
    if (item.shape !== "dot") continue;
    const own = itemBox(item);
    const cost = (side: PillSide) => pillBoxes(item, side)
      .reduce((n, b) => n + fixed.filter((f) => f !== own && hits(b, f)).length + taken.filter((f) => hits(b, f)).length, 0);
    item.pillSide = PILL_SIDES.map((side) => ({ side, cost: cost(side) })).sort((a, b) => a.cost - b.cost)[0].side;
    taken.push(...pillBoxes(item, item.pillSide));
  }
}

/** The pills a point shows: one per kind, solid-edged when applied here, dashed when only shared. */
function pillsOf(marks: PointMark[]): { kind: PreprocessKind; via: boolean; online: boolean }[] {
  return (["rotation", "scaling"] as const).flatMap((kind) => {
    const own = marks.find((m) => m.kind === kind && !m.via);
    const via = marks.find((m) => m.kind === kind && m.via);
    const m = own ?? via;
    return m ? [{ kind, via: !own, online: own?.mode === "online" }] : [];
  });
}

function NodeLabel({ n, globals }: { n: GraphNode; globals: GraphGlobals | null }) {
  const t = useT();
  if (n.kind === "linear" && n.canon) {
    return (
      <text className="arch-text" textAnchor="middle" dominantBaseline="central">
        W<tspan dy="4" fontSize="9">{weightSub(n.canon)}</tspan>
      </text>
    );
  }
  const text =
    n.kind === "route" ? "Σ"
    : n.kind === "matmul" || n.kind === "mul" ? "⊗"
    : n.kind === "add" ? "⊕"
    : n.kind === "softmax" ? "Softmax"
    : n.kind === "rope" ? "RoPE"
    : n.kind === "cache" ? (n.id === "k_cache" ? t("K cache") : t("V cache"))
    : n.kind === "act" ? ACT_LABEL[n.fn ?? ""] ?? n.fn ?? "act"
    : n.kind === "norm" ? (NORM_LABEL[globals?.norm_kind ?? "rms"] + (n.axis === "head_dim" ? "·d" : ""))
    : "X";
  const big = n.kind === "matmul" || n.kind === "mul" || n.kind === "add" || n.kind === "route";
  return (
    <text className={big ? "arch-text arch-op" : "arch-text"} textAnchor="middle" dominantBaseline="central">
      {text}
    </text>
  );
}

function nodeTitle(n: GraphNode, t: (s: string) => string): string {
  if (n.kind === "cache") return n.id === "k_cache" ? t("K cache") : t("V cache");
  const kind = n.kind === "matmul" ? "matmul" : n.kind;
  return n.module ? `${kind}: ${n.module}` : kind;
}

export function ArchDiagram(props: ArchDiagramProps) {
  const { graph, mode, value = [], onChange, onPoint, popover, numLayers, tied, moe } = props;
  const t = useT();
  const markerId = useId();
  const shown = useMemo(() => withKvCache(graph), [graph]);
  const geo = useGeometry(shown);
  const outerRef = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string; reason?: string } | null>(null);
  const [anchor, setAnchor] = useState<{ x: number; y: number } | null>(null);

  const items = useMemo(
    () => buildItems(shown, geo, props, t),
    [shown, geo, mode, props.positions, props.points, props.disabled, t],
  );

  if (graph.unsupported.length > 0 || graph.nodes.length === 0) {
    return (
      <div className="arch arch-empty muted">
        {t("No diagram for this architecture")}
        {graph.unsupported.length > 0 ? ` (${graph.unsupported.join(", ")})` : ""}
      </div>
    );
  }

  const selected = new Set(mode === "positions" ? Object.entries(props.points ?? {}).filter(([, p]) => p.marks.length > 0).map(([id]) => id) : value);
  const place = (el: Element) => {
    const outer = outerRef.current;
    if (!outer) return null;
    const r = el.getBoundingClientRect();
    const o = outer.getBoundingClientRect();
    return { x: r.left - o.left + r.width / 2, y: r.top - o.top };
  };
  const toggle = (item: Item, el: Element) => {
    if (item.reason) return;
    if (mode === "positions") {
      setAnchor(place(el));
      onPoint?.(item.id);
    } else onChange?.(selected.has(item.id) ? value.filter((v) => v !== item.id) : [...value, item.id]);
  };
  const showTip = (el: Element, item: Item) => {
    const at = place(el);
    if (at) setTip({ ...at, text: item.label, reason: item.reason ?? item.note });
  };

  const g = graph.globals;
  const first = geo.placed.find((p) => p.node.kind === "residual") ?? geo.placed[0];
  const last = geo.sink;
  const chipW = 78;
  const left = g ? chipW + 36 : 0;
  const boxTop = geo.rails[0] - 26;
  const boxBottom = geo.bottom + 38;
  const r0 = geo.width + 28;
  const r1 = last.x + last.w / 2 + 24;
  const headX = last.x + last.w / 2 + HEAD_OFFSET;
  const normX = headX - 80;
  const totalW = PAD + left + Math.max(r0, g ? headX + 32 : r1) + PAD;
  const boxPath = `M-28,${boxTop} H${r0} V${boxBottom} H-28 Z`;
  const tieY = boxBottom + 22;
  const totalH = tieY - boxTop + 18 + 2 * PAD;
  const ox = PAD + left;
  const oy = -boxTop + PAD;
  const embX = -(first.w / 2) - 36 - chipW / 2;
  const expertNodes = geo.placed.filter((p) => p.node.experts);
  const expertFrame = expertNodes.length === 0 ? null : {
    x1: Math.min(...expertNodes.map((p) => p.x - p.w / 2)) - FRAME_PAD,
    y1: Math.min(...expertNodes.map((p) => p.y - p.h / 2)) - FRAME_PAD - STACK * 2,
    x2: Math.max(...expertNodes.map((p) => p.x + p.w / 2)) + FRAME_PAD + STACK * 2,
    y2: Math.max(...expertNodes.map((p) => p.y + p.h / 2)) + FRAME_PAD,
  };

  const itemEl = (item: Item) => {
    const on = selected.has(item.id);
    const disabled = !!item.reason;
    const pills = pillsOf(item.marks ?? []);
    const flash = item.flash === undefined ? "" : ` flash-${item.flash}`;
    const aria = `${item.label}${disabled ? `, ${t("unavailable")}: ${item.reason}` : ""}`;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        toggle(item, e.currentTarget);
      }
    };
    return (
      <g
        key={item.id}
        className={`arch-item arch-${item.shape}${on ? " on" : ""}${flash}${item.leaving ? " leaving" : ""}${disabled ? " off" : ""}`}
        transform={`translate(${item.x},${item.y})`}
        role={mode === "positions" ? "button" : "checkbox"}
        aria-pressed={mode === "positions" ? on : undefined}
        aria-checked={mode === "positions" ? undefined : on}
        aria-disabled={disabled}
        aria-label={aria}
        tabIndex={0}
        data-id={item.id}
        onClick={(e) => toggle(item, e.currentTarget)}
        onKeyDown={onKey}
        onMouseEnter={(e) => showTip(e.currentTarget, item)}
        onMouseLeave={() => setTip(null)}
        onFocus={(e) => showTip(e.currentTarget, item)}
        onBlur={() => setTip(null)}
      >
        {item.shape === "dot" ? (
          <>
            <circle className="arch-hit" r={11} />
            {pills.length === 2 ? (
              <>
                <circle className="arch-shape arch-split-ring" r={7} />
                <path className={`arch-half k-rotation${pills[0].via ? " via" : ""}`} d="M0,-7 A7,7 0 0 0 0,7 Z" />
                <path className={`arch-half k-scaling${pills[1].via ? " via" : ""}`} d="M0,-7 A7,7 0 0 1 0,7 Z" />
              </>
            ) : (
              <circle className={`arch-shape${pills.length ? ` ${pills[0].kind === "rotation" ? "mk-r" : "mk-s"}${pills[0].via ? " via" : ""}` : ""}`} r={6} />
            )}
          </>
        ) : (
          <>
            <rect className="arch-hit" x={-(item.w! / 2) - 2} y={-(item.h! / 2) - 2} width={item.w! + 4} height={item.h! + 4} rx={8} />
            <rect className="arch-shape" x={-(item.w! / 2)} y={-(item.h! / 2)} width={item.w!} height={item.h!} rx={item.shape === "chip" ? item.h! / 2 : 6} />
          </>
        )}
        {item.shape === "weight" && (
          <text className="arch-text" textAnchor="middle" dominantBaseline="central">
            W<tspan dy="4" fontSize="9">{item.label.split(" ")[0]}</tspan>
          </text>
        )}
        {item.shape === "chip" && (
          <text className="arch-text arch-chip-text" textAnchor="middle" dominantBaseline="central">{item.tag ?? item.label}</text>
        )}
        {item.shape === "dot" && item.tag && (
          <text className="arch-tag" y={item.tagBelow ? 20 : -13} textAnchor="middle">{item.tag}</text>
        )}
        {item.shape === "dot" && pills.map((pl, i) => {
          const { dx, dy } = pillCenter(item.pillSide ?? "above", i, pills.length);
          const cls = `${pl.kind === "rotation" ? "k-rotation" : "k-scaling"}${pl.via ? " via" : pl.online ? " solid" : ""}`;
          return (
            <g key={pl.kind} className={`arch-pill ${cls}`} transform={`translate(${dx},${dy})`}>
              <rect x={-PILL_W / 2} y={-PILL_H / 2} width={PILL_W} height={PILL_H} rx={PILL_H / 2} />
              <text textAnchor="middle" dominantBaseline="central">{pl.kind === "rotation" ? "R" : "S"}</text>
            </g>
          );
        })}
      </g>
    );
  };

  return (
    <div className="arch" ref={outerRef} data-mode={mode}>
      <div className="arch-scroll">
        <svg
          className="arch-svg"
          viewBox={`0 0 ${totalW} ${totalH}`}
          style={{ minWidth: Math.min(totalW, 720), aspectRatio: `${totalW} / ${totalH}` }}
          role="group"
          aria-label={mode === "positions" ? t("Architecture diagram: choose where to apply the method") : t("Architecture diagram: choose what to quantize")}
        >
          <defs>
            <marker id={markerId} viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M0,1 L7,4 L0,7 z" className="arch-arrow" />
            </marker>
          </defs>
          <g transform={`translate(${ox},${oy})`}>
            <path className="arch-layer" d={boxPath} />
            {typeof numLayers === "number" && (
              <text className="arch-layer-label" x={-(first.w / 2) - 6} y={boxBottom - 8}>{t("Decoder layer × {n}", { n: numLayers })}</text>
            )}

            {expertFrame && (
              <g className="arch-experts">
                <rect x={expertFrame.x1} y={expertFrame.y1} width={expertFrame.x2 - expertFrame.x1} height={expertFrame.y2 - expertFrame.y1} rx={8}
                  fill="none" stroke="currentColor" strokeOpacity={0.4} strokeDasharray="5 4" />
                <text x={expertFrame.x1 + 6} y={expertFrame.y2 + 11} fontSize={10} fill="currentColor" fillOpacity={0.7}>
                  {!moe ? t("routed experts") : moe.topK === undefined ? t("{n} experts", { n: moe.experts }) : t("{n} experts, top-{k}", { n: moe.experts, k: moe.topK })}
                </text>
              </g>
            )}

            {geo.edges.map((e, i) => {
              const x1 = e.src.x + e.src.w / 2;
              const x2 = e.dst.x - e.dst.w / 2;
              if (e.bypass) {
                const rail = geo.rails[e.dst.row];
                const lead = e.src.row === e.dst.row ? "" : ` V${geo.crossY} H${-14}`;
                const d = `M${x1},${e.src.y} H${x1 + 14}${lead} V${rail} H${e.dst.x} V${e.dst.y - e.dst.h / 2 - 1}`;
                return <path key={i} className={`arch-edge arch-rail${selected.has("residual") ? " on" : ""}`} d={d} markerEnd={`url(#${markerId})`} />;
              }
              if (e.src.row !== e.dst.row) {
                const d = `M${x1},${e.src.y} H${x1 + 14} V${geo.crossY} H${-14} V${e.dst.y} H${x2}`;
                return <path key={i} className="arch-edge" d={d} markerEnd={`url(#${markerId})`} />;
              }
              const gate = isRouter(e.src.node) && e.dst.node.kind === "route";
              return <path key={i} className="arch-edge" strokeDasharray={gate ? "4 3" : undefined} d={curve(x1, e.src.y + e.offset, x2, e.dst.y + e.offset)} markerEnd={`url(#${markerId})`} />;
            })}

            {g && (
              <>
                <path className="arch-edge" d={`M${embX + chipW / 2},0 H${-first.w / 2 - 1}`} markerEnd={`url(#${markerId})`} />
                <path className="arch-edge" d={`M${last.x + last.w / 2},${last.y} H${normX - 26 - 1}`} markerEnd={`url(#${markerId})`} />
                <path className="arch-edge" d={`M${normX + 26},${last.y} H${headX - 32 - 1}`} markerEnd={`url(#${markerId})`} />
                <g className="arch-global" transform={`translate(${embX},0)`}>
                  <rect x={-chipW / 2} y={-12} width={chipW} height={24} rx={12} />
                  <text textAnchor="middle" dominantBaseline="central">{t("Embedding")}</text>
                </g>
                <g className="arch-global" transform={`translate(${normX},${last.y})`}>
                  <rect x={-26} y={-12} width={52} height={24} rx={12} />
                  <text textAnchor="middle" dominantBaseline="central">{NORM_LABEL[g.norm_kind]}</text>
                </g>
                {tied && (
                  <g>
                    <path className="arch-tie" d={`M${embX},12 V${tieY} H${headX} V${last.y + 12}`} />
                    <g className="arch-global arch-tied" transform={`translate(${(embX + headX) / 2},${tieY})`}>
                      <rect x={-34} y={-10} width={68} height={20} rx={10} />
                      <text textAnchor="middle" dominantBaseline="central">{t("tied weights")}</text>
                    </g>
                  </g>
                )}
              </>
            )}

            {geo.placed.map((p) => {
              const n = p.node;
              const round = n.kind === "matmul" || n.kind === "mul" || n.kind === "add" || n.kind === "route";
              return (
                <g key={n.id} className={`arch-node arch-k-${n.kind}${n.kind === "route" ? " arch-k-mul" : ""}`} transform={`translate(${p.x},${p.y})`}
                  onMouseEnter={(e) => mode === "positions" || n.kind !== "linear" ? showTip(e.currentTarget, { id: n.id, label: nodeTitle(n, t), x: 0, y: 0, shape: "dot" }) : undefined}
                  onMouseLeave={() => setTip(null)}>
                  {n.experts && [STACK * 2, STACK].map((d) => (
                    <g key={d} transform={`translate(${d},${-d})`}>
                      {round ? <circle r={p.w / 2} /> : <rect x={-p.w / 2} y={-p.h / 2} width={p.w} height={p.h} rx={n.kind === "linear" ? 6 : 12} />}
                    </g>
                  ))}
                  {round ? <circle r={p.w / 2} /> : n.kind === "cache" ? <path d={cylinderPath(p.w, p.h)} /> : <rect x={-p.w / 2} y={-p.h / 2} width={p.w} height={p.h} rx={n.kind === "linear" ? 6 : 12} />}
                  {mode === "positions" || (n.kind !== "linear" && n.kind !== "cache") ? <NodeLabel n={n} globals={g} /> : null}
                </g>
              );
            })}

            {g && (
              <g className="arch-global" transform={`translate(${headX},${last.y})`}>
                <rect x={-32} y={-12} width={64} height={24} rx={12} />
                <text textAnchor="middle" dominantBaseline="central">lm_head</text>
              </g>
            )}

            {items.map(itemEl)}
          </g>
        </svg>
      </div>
      {popover && anchor && (
        <div className="arch-popover" style={{ left: anchor.x, top: anchor.y }}
          onKeyDown={(e) => { if (e.key === "Escape") popover.onDismiss(); }}>
          {popover.content}
        </div>
      )}
      {tip && (
        <div className="arch-tip" role="tooltip" style={{ left: tip.x, top: tip.y }}>
          <span>{tip.text}</span>
          {tip.reason && <span className="arch-tip-reason">{tip.reason}</span>}
        </div>
      )}
      <div className="arch-legend faint">
        {mode === "positions" ? (
          <>
            <span className="arch-key arch-key-r" />{t("Rotation")}
            <span className="arch-key arch-key-s" />{t("Scaling")}
            <span className="arch-key arch-key-split" />{t("both")}
            <span className="arch-key arch-key-via" />{t("shared (via residual or norm)")}
            <span className="arch-key" />{t("available")}
            <span className="arch-key arch-key-off" />{t("unavailable (hover for the reason)")}
            <span className="arch-key-pill solid">R</span>{t("online")}
            <span className="arch-key-pill">S</span>{t("offline")}
          </>
        ) : (
          <>
            <span className="arch-key arch-key-on" />{t("selected")}
            <span className="arch-key" />{t("available")}
            <span className="arch-key arch-key-off" />{t("unavailable (hover for the reason)")}
          </>
        )}
      </div>
    </div>
  );
}

/** Layer chip row: all layers, or an inclusive range of layer indices. */
export function LayerSelector({ value, onChange, numLayers }: {
  value: LayerSelection;
  onChange: (next: LayerSelection) => void;
  /** Model depth; when unknown the range is unbounded above. */
  numLayers?: number;
}) {
  const t = useT();
  const last = numLayers === undefined ? Number.MAX_SAFE_INTEGER : Math.max(0, numLayers - 1);
  const clamp = (n: number) => Math.min(last, Math.max(0, Number.isFinite(n) ? Math.trunc(n) : 0));
  const range = value.kind === "range" ? value : { kind: "range" as const, from: 0, to: numLayers === undefined ? 0 : last };
  return (
    <div className="arch-layers">
      <Segmented
        value={value.kind}
        onChange={(v) => onChange(v === "all" ? { kind: "all" } : range)}
        options={[
          { value: "all", label: t("All layers") },
          { value: "range", label: t("Layer range") },
        ]}
      />
      {value.kind === "range" && (
        <span className="arch-range">
          <Input
            type="number"
            min={0}
            max={numLayers === undefined ? undefined : last}
            aria-label={t("First layer")}
            value={value.from}
            onChange={(e) => onChange({ kind: "range", from: clamp(Number(e.target.value)), to: Math.max(value.to, clamp(Number(e.target.value))) })}
          />
          <span className="faint">{t("to")}</span>
          <Input
            type="number"
            min={0}
            max={numLayers === undefined ? undefined : last}
            aria-label={t("Last layer")}
            value={value.to}
            onChange={(e) => onChange({ kind: "range", from: Math.min(value.from, clamp(Number(e.target.value))), to: clamp(Number(e.target.value)) })}
          />
          {numLayers !== undefined && <span className="faint">{t("of {n}", { n: numLayers })}</span>}
        </span>
      )}
    </div>
  );
}
