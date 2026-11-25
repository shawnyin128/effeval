// Model preprocessing panel: where preprocessing (rotation / scaling) acts (one architecture diagram),
// how it is applied (offline / online) and where its matrices or scales come from (source).
import { Check, ChevronDown, ChevronRight, RotateCcw, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { Describe, JsonSchema, PreprocessEntry, PreprocessSource, PreprocessSpec, PreprocessStep } from "../api";
import { methodTitle } from "../api";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { Button, Field, Input, Segmented, Select, Switch } from "../ui/controls";
import { Badge, Help } from "../ui/display";
import { ArchDiagram, LayerSelector, type ApplyMode, type LayerGraph, type LayerSelection, type PointMark, type PointState, type PositionTable, type PreprocessKind } from "./ArchDiagram";
import { ScalarField, problemAt, type Problems } from "./MethodConfig";

type Rec = Record<string, any>;
type Translate = ReturnType<typeof useT>;

const SOURCE_LABEL: Record<string, string> = {
  hadamard: "Hadamard",
  random_hadamard: "Random Hadamard",
  random_orthogonal: "Random orthogonal",
  duquant: "DuQuant",
  smoothquant: "SmoothQuant",
  awq: "AWQ",
  trained: "Trained",
};
const DATASETS = ["c4", "wikitext2", "pile_val"];

// ---------------------------------------------------------------------------------------------------------
// Value helpers
// ---------------------------------------------------------------------------------------------------------

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Rec).sort(([a], [b]) => (a < b ? -1 : 1)).map(([k, v]) => `${JSON.stringify(k)}:${stable(v)}`).join(",")}}`;
  }
  return JSON.stringify(value ?? null);
}

function resolveRef(schema: JsonSchema, root: JsonSchema): JsonSchema {
  return schema.$ref ? root.$defs?.[schema.$ref.split("/").pop() as string] ?? schema : schema;
}

/** Defaults of every field of an object schema that declares one (`default_factory` fields declare none). */
function schemaDefaults(schema: JsonSchema): Rec {
  const out: Rec = {};
  for (const [key, raw] of Object.entries(schema.properties ?? {})) {
    if (key === "type") out.type = raw.const ?? raw.default;
    else if ("default" in raw) out[key] = raw.default;
  }
  return out;
}

function nullableRef(raw: JsonSchema, root: JsonSchema): JsonSchema | null {
  const target = raw.anyOf?.find((s) => s.$ref);
  return target && raw.anyOf?.some((s) => s.type === "null") ? resolveRef(target, root) : null;
}

/** A source with every field the schema defaults filled, so two spellings of the same source compare equal. */
function normalizeSource(source: Rec, sources: Describe["preprocess_sources"], kind: string): Rec {
  const spec = sources?.[kind]?.[source.type];
  if (!spec) return source;
  const root = spec.schema;
  const out: Rec = { ...schemaDefaults(root), ...source };
  for (const [key, raw] of Object.entries(root.properties ?? {})) {
    const nested = nullableRef(raw, root);
    if (nested && out[key] && typeof out[key] === "object") out[key] = { ...schemaDefaults(nested), ...out[key] };
  }
  return out;
}

function normalizeSteps(steps: PreprocessStep[], sources: Describe["preprocess_sources"]): unknown {
  return steps.map((step) => ({
    kind: step.kind,
    calibration: step.calibration ?? null,
    entries: step.entries.map((e) => ({
      position: e.position,
      layers: e.layers ?? "all",
      mode: e.mode,
      source: normalizeSource(e.source, sources, step.kind),
      options: e.options ?? {},
    })),
  }));
}

export function sameSteps(a: PreprocessStep[], b: PreprocessStep[], sources: Describe["preprocess_sources"]): boolean {
  return stable(normalizeSteps(a, sources)) === stable(normalizeSteps(b, sources));
}

/** The steps a preprocess value runs: its own, else its preset's. */
export function effectiveSteps(value: PreprocessSpec | undefined, describe: Describe): PreprocessStep[] {
  return value?.steps ?? (value?.preset ? describe.preprocess_presets?.[value.preset]?.steps : undefined) ?? [];
}

/** Whether the steps differ from the named preset's. */
export function isModified(value: PreprocessSpec | undefined, describe: Describe): boolean {
  const preset = value?.preset ? describe.preprocess_presets?.[value.preset] : undefined;
  return !!preset && !!value?.steps && !sameSteps(value.steps, preset.steps, describe.preprocess_sources);
}

/** Run label "<preprocess title>[*] + <method title>[ (<method preset>)]" (the server builds the same for stored records). */
export function preprocessLabel(
  value: { preset?: string | null; steps?: PreprocessStep[] } | null | undefined, describe: Describe, method: string,
  t: Translate, methodPreset?: string | null,
): string {
  const title = methodTitle(describe, method) + (value && methodPreset && methodPreset !== "default" ? ` (${methodPreset})` : "");
  if (!value) return title;
  const preset = value.preset ? describe.preprocess_presets?.[value.preset] : undefined;
  if (!preset) return `${t("Custom preprocessing")} + ${title}`;
  return `${preset.title}${isModified(value as PreprocessSpec, describe) ? "*" : ""} + ${title}`;
}

function sourceSummary(source: Rec): string {
  const rest = Object.entries(source).filter(([k, v]) => k !== "type" && v !== null && v !== undefined && typeof v !== "object");
  const label = SOURCE_LABEL[source.type] ?? source.type;
  return rest.length ? `${label} (${rest.map(([k, v]) => `${k} ${v}`).join(", ")})` : label;
}

/** One compact line per entry, e.g. "rotation · residual · offline · Random Hadamard (seed 0)". */
export function preprocessLines(steps: PreprocessStep[]): string[] {
  return steps.flatMap((step) => step.entries.map((e) => [step.kind, e.position, e.mode, sourceSummary(e.source)].join(" · ")));
}

function toSelection(layers: unknown): LayerSelection | null {
  if (layers === undefined || layers === "all") return { kind: "all" };
  const m = typeof layers === "string" ? /^(\d+)-(\d+)$/.exec(layers) : null;
  return m ? { kind: "range", from: Number(m[1]), to: Number(m[2]) } : null;
}

const fromSelection = (s: LayerSelection) => (s.kind === "all" ? "all" : `${s.from}-${s.to}`);

// ---------------------------------------------------------------------------------------------------------
// Points, cards and edits (pure)
// ---------------------------------------------------------------------------------------------------------

const RESIDUAL_HINT = "rotate the residual instead";
const NORM_HINT = /use (norm\.\S+\.out)/;
const KINDS: PreprocessKind[] = ["rotation", "scaling"];
const SOURCE_DEFAULT: Record<PreprocessKind, string> = { rotation: "random_hadamard", scaling: "smoothquant" };

/** Whether `position` matches a glob whose only wildcard is `*`. */
export function globMatch(pattern: string, position: string): boolean {
  const re = pattern.split("*").map((p) => p.replace(/[.+?^${}()|[\]\\]/g, "\\$&")).join(".*");
  return new RegExp(`^${re}$`).test(position);
}

/** Whether a source may be used at `position` (a source without a position list is allowed anywhere). */
export function sourceAllowed(spec: PreprocessSource | undefined, position: string): boolean {
  return !spec?.positions || spec.positions.some((p) => globMatch(p, position));
}

/** The source a new entry of `kind` at `position` starts with: the kind's default when allowed there, else the first allowed implemented source. */
export function defaultSourceType(kind: PreprocessKind, position: string, sources: Describe["preprocess_sources"]): string {
  const specs = sources?.[kind] ?? {};
  if (sourceAllowed(specs[SOURCE_DEFAULT[kind]], position)) return SOURCE_DEFAULT[kind];
  return Object.entries(specs).find(([, s]) => s.implemented && sourceAllowed(s, position))?.[0] ?? SOURCE_DEFAULT[kind];
}

const KV_ROTATION_POSITIONS = ["attn.qk_rope", "attn.v_head"];
const RANDOM_ROTATIONS = ["random_hadamard", "random_orthogonal"];

/** Whether the "Per KV head" option applies: a random rotation at a KV-cache rotation position. */
export function hasPerKvHead(kind: PreprocessKind, position: string, sourceType: string): boolean {
  return kind === "rotation" && KV_ROTATION_POSITIONS.includes(position) && RANDOM_ROTATIONS.includes(sourceType);
}

const rankOf = (table: PositionTable, position: string) => {
  const i = Object.keys(table).indexOf(position);
  return i < 0 ? Infinity : i;
};

/**
 * The position whose offline entry already covers `kind` at `position`, or null. A rotation at a norm output or a
 * linear input behind a norm is the residual rotation; a scaling at one of several linears sharing a norm output
 * is the scaling folded into that norm.
 */
export function shareTarget(table: PositionTable, position: string, kind: PreprocessKind): string | null {
  const avail = table[position]?.[kind];
  if (!avail || position === "residual") return null;
  if (kind === "rotation") return avail.offline.includes(RESIDUAL_HINT) && table.residual?.rotation.offline === "ok" ? "residual" : null;
  const target = NORM_HINT.exec(avail.offline)?.[1];
  return target && table[target]?.scaling.offline === "ok" ? target : null;
}

export interface AddPlan {
  position: string;
  mode: ApplyMode;
}

/** The entry that choosing `kind` at `position` writes: offline at the position, else offline at its share target, else online. */
export function addPlan(table: PositionTable, position: string, kind: PreprocessKind): AddPlan | null {
  const avail = table[position]?.[kind];
  if (!avail) return null;
  if (avail.offline === "ok") return { position, mode: "offline" };
  const target = shareTarget(table, position, kind);
  if (target) return { position: target, mode: "offline" };
  return avail.online === "ok" ? { position, mode: "online" } : null;
}

export interface PointInfo {
  /** Entries at exactly this position. */
  own: PointMark[];
  /** Kinds covered here by an offline entry at the share target. */
  via: PreprocessKind[];
  /** Kinds that can be chosen here. */
  possible: PreprocessKind[];
  /** Kinds in effect here, own or shared. */
  applied: PreprocessKind[];
  clickable: boolean;
}

export function pointInfo(steps: PreprocessStep[], table: PositionTable, position: string): PointInfo {
  const own: PointMark[] = steps.flatMap((s) => s.entries.filter((e) => e.position === position).map((e) => ({ kind: s.kind, mode: e.mode })));
  const via = KINDS.filter((k) => {
    const target = shareTarget(table, position, k);
    return !!target && steps.some((s) => s.kind === k && s.entries.some((e) => e.position === target && e.mode === "offline"));
  });
  const possible = KINDS.filter((k) => addPlan(table, position, k));
  const applied = KINDS.filter((k) => own.some((m) => m.kind === k) || via.includes(k));
  return { own, via, possible, applied, clickable: possible.length > 0 || own.length > 0 || via.length > 0 };
}

/** One position with the entries (one per kind) applied there. */
export interface Card {
  position: string;
  lines: { kind: PreprocessKind; step: number; entry: number }[];
}

/** Cards in the position order of the diagram; lines inside a card follow the step order. */
export function buildCards(steps: PreprocessStep[], table: PositionTable): Card[] {
  const cards = new Map<string, Card>();
  steps.forEach((s, step) => s.entries.forEach((e, entry) => {
    const card = cards.get(e.position) ?? { position: e.position, lines: [] };
    card.lines.push({ kind: s.kind, step, entry });
    cards.set(e.position, card);
  }));
  return [...cards.values()].sort((a, b) => rankOf(table, a.position) - rankOf(table, b.position));
}

export const rowKey = (kind: PreprocessKind, position: string) => `${kind}:${position}`;

/** Rows and points that differ between two step lists (mode and source type count; other parameters do not). */
export function changedKeys(prev: PreprocessStep[], next: PreprocessStep[], table: PositionTable): { rows: string[]; added: string[]; removed: string[]; points: string[] } {
  const sig = (steps: PreprocessStep[]) => new Map(steps.flatMap((s) => s.entries.map((e) => [rowKey(s.kind, e.position), `${e.mode}|${e.source.type}`] as const)));
  const a = sig(prev);
  const b = sig(next);
  const added = [...b.keys()].filter((k) => !a.has(k));
  const rows = [...b].filter(([k, v]) => a.get(k) !== v).map(([k]) => k);
  const removed = [...a.keys()].filter((k) => !b.has(k));
  const touched = new Set([...rows, ...removed].map((k) => k.split(":").slice(1).join(":")));
  const touchedKinds = new Map([...rows, ...removed].map((k) => [k.split(":").slice(1).join(":"), k.split(":")[0]]));
  const points = Object.keys(table).filter((p) => touched.has(p)
    || KINDS.some((k) => { const target = shareTarget(table, p, k); return !!target && touched.has(target) && touchedKinds.get(target) === k; }));
  return { rows, added, removed, points };
}

function withCalibration(step: PreprocessStep, sources: Describe["preprocess_sources"]): PreprocessStep {
  const needs = step.entries.some((e) => sources?.[step.kind]?.[e.source.type]?.needs_data);
  if (needs && !step.calibration) return { ...step, calibration: newCalibration() };
  if (!needs && step.calibration) return { ...step, calibration: null };
  return step;
}

/** Adds an entry of `kind` at `plan`, creating the step when needed (a new rotation step goes first, a new scaling step last). */
export function addEntry(
  steps: PreprocessStep[], table: PositionTable, sources: Describe["preprocess_sources"], tied: boolean, plan: AddPlan, kind: PreprocessKind,
): PreprocessStep[] {
  const idx = steps.findIndex((s) => s.kind === kind);
  const step: PreprocessStep = idx >= 0 ? steps[idx] : { kind, entries: [] };
  if (step.entries.some((e) => e.position === plan.position)) return steps;
  const donor = step.entries.map((e) => e.source).find((s) => sourceAllowed(sources?.[kind]?.[s.type], plan.position));
  const options: Rec = plan.position === "residual" && kind === "rotation" ? { untie_embeddings: tied, center_embeddings: false } : {};
  const entry: PreprocessEntry = {
    position: plan.position, layers: "all", mode: plan.mode, source: donor ? { ...donor } : newSource(kind, defaultSourceType(kind, plan.position, sources), sources), options,
  };
  const at = step.entries.findIndex((e) => rankOf(table, e.position) > rankOf(table, plan.position));
  const entries = at < 0 ? [...step.entries, entry] : [...step.entries.slice(0, at), entry, ...step.entries.slice(at)];
  const next = withCalibration({ ...step, entries }, sources);
  if (idx >= 0) return steps.map((s, i) => (i === idx ? next : s));
  return kind === "rotation" ? [next, ...steps] : [...steps, next];
}

/** Whether an offline scaling folded into a norm gain runs before the residual rotation, which would cancel it. */
export function scalingCancelledByRotation(steps: PreprocessStep[]): boolean {
  const norm = steps.findIndex((s) => s.kind === "scaling" && s.entries.some((e) => e.mode === "offline" && e.position.startsWith("norm.")));
  const rotation = steps.findIndex((s) => s.kind === "rotation" && s.entries.some((e) => e.position === "residual"));
  return norm >= 0 && rotation >= 0 && norm < rotation;
}

/** Removes the entry of `kind` at `position`; a step left without entries is dropped. */
export function removeEntry(steps: PreprocessStep[], sources: Describe["preprocess_sources"], position: string, kind: PreprocessKind): PreprocessStep[] {
  return steps
    .map((s) => (s.kind === kind ? withCalibration({ ...s, entries: s.entries.filter((e) => e.position !== position) }, sources) : s))
    .filter((s) => s.entries.length > 0);
}

/** Chooses `kind` at `position` when it is not in effect there, else removes it (a shared kind removes the entry at its share target). */
export function toggleKind(
  steps: PreprocessStep[], table: PositionTable, sources: Describe["preprocess_sources"], tied: boolean, position: string, kind: PreprocessKind,
): PreprocessStep[] {
  const info = pointInfo(steps, table, position);
  if (info.own.some((m) => m.kind === kind)) return removeEntry(steps, sources, position, kind);
  if (info.via.includes(kind)) return removeEntry(steps, sources, shareTarget(table, position, kind)!, kind);
  const plan = addPlan(table, position, kind);
  return plan ? addEntry(steps, table, sources, tied, plan, kind) : steps;
}

/** Drawing state of every point of the diagram. */
export function pointStates(steps: PreprocessStep[], table: PositionTable, t: Translate): Record<string, PointState> {
  const kindLabel = (k: PreprocessKind) => (k === "rotation" ? t("Rotation") : t("Scaling"));
  const shared = (k: PreprocessKind) => (k === "rotation"
    ? t("Shares the residual rotation matrix (offline): attention and FFN inputs use one matrix")
    : t("Shares the scaling folded into the norm gain (offline)"));
  return Object.fromEntries(Object.keys(table).map((id) => {
    const info = pointInfo(steps, table, id);
    const marks: PointMark[] = [...info.own, ...info.via.filter((k) => !info.own.some((m) => m.kind === k)).map((kind) => ({ kind, via: true }))];
    const reason = KINDS.flatMap((k) => (["offline", "online"] as const).map((m) => `${kindLabel(k)} ${t(m)}: ${table[id][k][m]}`)).join(". ");
    return [id, {
      clickable: info.clickable, marks, reason: info.clickable ? undefined : reason,
      note: info.via.length ? info.via.map(shared).join(" ") : undefined,
    }];
  }));
}

/** Human title of a position built from the graph, e.g. "Q/K/V input (after input_layernorm)". */
export function positionTitle(graph: LayerGraph, id: string, t: Translate): string {
  if (id === "residual") return t("Residual stream");
  if (id === "attn.v_head") return t("V head");
  if (id === "attn.qk_rope") return t("Q/K (after RoPE)");
  if (id === "attn.qk_pre_rope") return t("Q/K (before RoPE)");
  const short = (canon: string) => canon.split(".").pop() as string;
  const letter = (canon: string) => {
    const base = short(canon).replace(/_proj$/, "");
    return base.length === 1 ? base.toUpperCase() : base;
  };
  const nodeOf = (nid: string) => graph.nodes.find((n) => n.id === nid);
  const norm = /^norm\.(.+)\.out$/.exec(id);
  if (norm) {
    const node = graph.nodes.find((n) => n.kind === "norm" && n.module === norm[1]);
    const canons = node ? [...new Set(graph.edges.filter(([s]) => s === node.id).map(([, d]) => nodeOf(d)).filter((n) => n?.kind === "linear" && n.canon).map((n) => n!.canon as string))] : [];
    return canons.length ? t("{targets} input (after {norm})", { targets: canons.map(letter).join("/"), norm: norm[1] }) : `${norm[1]} ${t("output")}`;
  }
  const lin = /^linear\.(.+)\.input$/.exec(id);
  if (lin) {
    const node = graph.nodes.find((n) => n.kind === "linear" && n.canon === lin[1]);
    const producer = node ? graph.edges.filter(([, d]) => d === node.id).map(([s]) => nodeOf(s)).find((n) => n?.kind === "norm") : undefined;
    return producer?.module
      ? t("{name} input (after {norm})", { name: short(lin[1]), norm: producer.module })
      : t("{name} input", { name: short(lin[1]) });
  }
  return id;
}

// ---------------------------------------------------------------------------------------------------------
// Source and calibration forms
// ---------------------------------------------------------------------------------------------------------

function newSource(kind: string, type: string, sources: Describe["preprocess_sources"]): PreprocessEntry["source"] {
  const spec = sources?.[kind]?.[type];
  return spec ? { ...schemaDefaults(spec.schema), type } : { type };
}

function newCalibration(): Rec {
  return { dataset: "wikitext2", n_samples: 128, seqlen: 2048, seed: 0, sampling: "windows" };
}

function SourceFields({ source, spec, onChange, errorAt }: {
  source: Rec; spec: PreprocessSource; onChange: (next: Rec) => void; errorAt: (field: string) => string | undefined;
}) {
  const t = useT();
  const root = spec.schema;
  const defaults = schemaDefaults(root);
  const set = (key: string, value: unknown) => onChange({ ...source, [key]: value === undefined ? defaults[key] : value });
  const fields = Object.entries(root.properties ?? {}).filter(([key]) => key !== "type" && key !== "calibration");
  if (!spec.implemented) return <p className="faint small">{t("Not implemented yet.")}</p>;
  if (fields.length === 0) return <p className="faint small">{t("No parameters.")}</p>;
  return (
    <div className="grid-3">
      {fields.map(([key, raw]) => {
        const label = t(key.charAt(0).toUpperCase() + key.slice(1).replaceAll("_", " "));
        const nested = nullableRef(raw, root);
        if (nested) {
          const value = source[key] as Rec | null | undefined;
          return (
            <div key={key} className="pre-nested">
              <Field label={<>{label}<Help text={raw.description} /></>}>
                <Switch checked={!!value} onChange={(on) => set(key, on ? schemaDefaults(nested) : null)} label={label} />
              </Field>
              {value && (
                <div className="grid-2">
                  {Object.entries(nested.properties ?? {}).map(([sub, subRaw]) => (
                    <ScalarField key={sub} name={sub} schema={resolveRef(subRaw, root)} value={value[sub]} invalid={false}
                      onChange={(v) => set(key, { ...value, [sub]: v === undefined ? schemaDefaults(nested)[sub] : v })} />
                  ))}
                </div>
              )}
            </div>
          );
        }
        const constChoice = raw.anyOf?.find((s) => s.const !== undefined);
        if (constChoice && raw.anyOf?.some((s) => s.type === "null")) {
          return (
            <Field key={key} label={<>{label}<Help text={raw.description} /></>} error={errorAt(key)}>
              <Select value={source[key] === null || source[key] === undefined ? "" : String(source[key])} placeholder={t("none")}
                onChange={(v) => set(key, v === "" ? null : v)} options={[{ value: String(constChoice.const), label: String(constChoice.const) }]} />
            </Field>
          );
        }
        return (
          <ScalarField key={key} name={key} schema={raw} value={source[key]} invalid={!!errorAt(key)} error={errorAt(key)}
            onChange={(v) => set(key, v)} />
        );
      })}
    </div>
  );
}

const PILE_SAMPLING = ["documents", "concat_blocks", "random_crop"];

function CalibrationFields({ value, onChange }: { value: Rec; onChange: (next: Rec) => void }) {
  const t = useT();
  const set = (patch: Rec) => {
    const next = { ...value, ...patch };
    if ("dataset" in patch) next.sampling = patch.dataset === "pile_val" ? (PILE_SAMPLING.includes(String(value.sampling)) ? value.sampling : "documents") : "windows";
    onChange(next);
  };
  const num = (key: string, min: number) => (
    <Input type="number" min={min} value={String(value[key] ?? "")} onChange={(e) => set({ [key]: Number(e.target.value) })} />
  );
  return (
    <div className="grid-3">
      <Field label={t("Dataset")}><Select value={String(value.dataset)} onChange={(v) => set({ dataset: v })} options={DATASETS.map((d) => ({ value: d, label: d }))} /></Field>
      <Field label={<>{t("Sampling")}<Help text={t("windows: random token windows; documents: one document per forward pass; concat_blocks: documents joined and cut into blocks; random_crop: shuffled documents, each cropped at a random offset, then joined and cut into blocks (QServe)")} /></>}>
        <Select value={String(value.sampling)} onChange={(v) => set({ sampling: v })} disabled={value.dataset !== "pile_val"}
          options={(value.dataset === "pile_val" ? PILE_SAMPLING : ["windows"]).map((s) => ({ value: s, label: s }))} />
      </Field>
      <Field label={t("Samples")}>{num("n_samples", 1)}</Field>
      <Field label={t("Sequence length")}>{num("seqlen", 8)}</Field>
      <Field label={t("Seed")}>{num("seed", 0)}</Field>
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------
// One line of a position card (kind + mode + source)
// ---------------------------------------------------------------------------------------------------------

function EntryLine({ rowId, kind, peers, entry, table, sources, tied, numLayers, open, onToggle, onChange, onRemove, onApplyToAll, problems, base, note, effect }: {
  rowId: string;
  kind: PreprocessKind;
  /** Number of entries of this kind. */
  peers: number;
  entry: PreprocessEntry;
  table: PositionTable;
  sources: NonNullable<Describe["preprocess_sources"]>;
  tied: boolean;
  numLayers?: number;
  open: boolean;
  onToggle: () => void;
  onChange: (next: PreprocessEntry) => void;
  onRemove: () => void;
  onApplyToAll: () => void;
  problems: Problems;
  base: string;
  note?: string;
  /** Highlight class while the line was just changed or is leaving. */
  effect: string;
}) {
  const t = useT();
  const avail = table[entry.position]?.[kind];
  const reason = (mode: ApplyMode) => (avail ? (avail[mode] === "ok" ? undefined : avail[mode]) : t("Not available for this model"));
  const typeSpecs = sources[kind] ?? {};
  const spec = typeSpecs[entry.source.type];
  const errors = [...new Set([...problems].filter(([path]) => path === normalize(base) || path.startsWith(`${normalize(base)}.`)).map(([, m]) => m))];
  const selection = toSelection(entry.layers);
  const global = entry.position === "residual";
  const noPaper = !!spec?.no_paper?.some((p) => globMatch(p, entry.position));
  const perKvHead = hasPerKvHead(kind, entry.position, entry.source.type);
  return (
    <div className={`pre-line k-${kind} ${errors.length ? "invalid" : ""} ${effect}`} data-row={rowId}>
      <div className="pre-line-head">
        <button type="button" className="btn btn-ghost btn-sm pre-toggle" aria-expanded={open} aria-label={t("Edit source")} onClick={onToggle}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
        <span className="pre-kind-badge">{kind === "rotation" ? t("Rotation") : t("Scaling")}</span>
        <Segmented value={entry.mode} onChange={(m) => onChange({ ...entry, mode: m as ApplyMode })}
          options={(["offline", "online"] as const).map((m) => ({ value: m, label: t(m), disabled: !!reason(m) && entry.mode !== m, title: reason(m) }))} />
        <span className="pre-source mono small truncate" title={sourceSummary(entry.source)}>{sourceSummary(entry.source)}</span>
        <Button variant="ghost" size="sm" icon={<Trash2 size={13} />} aria-label={t("Remove position")} title={t("Remove position")} onClick={onRemove} />
      </div>
      {note && <div className="pre-extra faint tiny">{note}</div>}
      {noPaper && <div className="pre-extra faint tiny">{t("No published paper for this source at this position.")}</div>}
      {errors.map((m) => <div key={m} className="field-error">{m}</div>)}
      {open && (
        <div className="pre-entry-body stack gap-3">
          <Field label={<>{t("Source")}<Help text={t("Where the matrix or scale comes from.")} /></>}>
            <Segmented value={entry.source.type}
              onChange={(type) => {
                const { per_kv_head: _drop, ...rest } = entry.options;
                onChange({ ...entry, source: newSource(kind, type, sources), options: hasPerKvHead(kind, entry.position, type) ? entry.options : rest });
              }}
              options={Object.entries(typeSpecs).filter(([type, s]) => type === entry.source.type || sourceAllowed(s, entry.position)).map(([type, s]) => ({
                value: type,
                label: <>{SOURCE_LABEL[type] ?? type}{!s.implemented && <span className="faint tiny"> · {t("coming soon")}</span>}</>,
                disabled: !s.implemented && entry.source.type !== type,
                title: s.implemented ? undefined : t("Coming soon"),
              }))} />
          </Field>
          {spec && (
            <SourceFields source={entry.source} spec={spec} onChange={(source) => onChange({ ...entry, source: source as PreprocessEntry["source"] })}
              errorAt={(field) => problemAt(problems, `${base}.source.${field}`)} />
          )}
          <div className="grid-2">
            <Field label={<>{t("Layers")}<Help text={t("Decoder layers this position is applied to.")} /></>} error={problemAt(problems, `${base}.layers`)}>
              {global ? <span className="faint small">{t("All layers (one matrix for the whole model)")}</span>
                : selection ? <LayerSelector value={selection} numLayers={numLayers} onChange={(s) => onChange({ ...entry, layers: fromSelection(s) })} />
                  : <Input className="mono" value={String(entry.layers)} onChange={(e) => onChange({ ...entry, layers: e.target.value || "all" })} />}
            </Field>
            {((global && kind === "rotation") || perKvHead) && (
              <Field label={t("Options")}>
                <div className="stack gap-2">
                  {global && kind === "rotation" && (
                    <>
                      <label className="row gap-2 small">
                        <Switch checked={entry.options.untie_embeddings === true} label={t("Untie embeddings")}
                          onChange={(v) => onChange({ ...entry, options: { ...entry.options, untie_embeddings: v } })} />
                        {t("Untie embeddings")}{tied && <Badge tone="orange">{t("required: tied lm_head")}</Badge>}
                      </label>
                      <label className="row gap-2 small">
                        <Switch checked={entry.options.center_embeddings === true} label={t("Center embeddings")}
                          onChange={(v) => onChange({ ...entry, options: { ...entry.options, center_embeddings: v } })} />
                        {t("Center embeddings")}
                      </label>
                    </>
                  )}
                  {perKvHead && (
                    <label className="row gap-2 small">
                      <Switch checked={entry.options.per_kv_head === true} label={t("Per KV head")}
                        onChange={(v) => onChange({ ...entry, options: { ...entry.options, per_kv_head: v } })} />
                      {t("Per KV head")}
                    </label>
                  )}
                </div>
              </Field>
            )}
          </div>
          {peers > 1 && (
            <div><Button size="sm" variant="link" onClick={onApplyToAll}>{t("Use this source for every position of this kind")}</Button></div>
          )}
        </div>
      )}
    </div>
  );
}

const normalize = (path: string) => path.replace(/\[(\d+)\]/g, ".$1");

// ---------------------------------------------------------------------------------------------------------
// The panel
// ---------------------------------------------------------------------------------------------------------

const FADE_MS = 220;
const FLASH_MS = 1100;

export function Preprocess({ value, onChange, modelCode, problems, method, onPickQuantizer }: {
  value: PreprocessSpec | undefined;
  onChange: (next: PreprocessSpec | undefined) => void;
  modelCode: string;
  problems: Problems;
  /** The spec's current method name and preset; with `onPickQuantizer` it enables the paired-quantizer buttons. */
  method?: { name?: string; preset?: string | null };
  onPickQuantizer?: (method: string, preset: string) => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const presets = describe.preprocess_presets ?? {};
  const sources = describe.preprocess_sources ?? {};
  const model = describe.models[modelCode];
  const graph = model ? describe.architectures[model.architecture]?.graph ?? null : null;
  const table = model?.positions ?? null;
  const steps = effectiveSteps(value, describe);
  const preset = value?.preset ? presets[value.preset] : undefined;
  const modified = isModified(value, describe);
  const tied = !!model?.tie_word_embeddings;
  const ready = !!model && !!graph && !!table;
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [chooser, setChooser] = useState<string | null>(null);
  const [focus, setFocus] = useState<{ key: string } | null>(null);
  const [flash, setFlash] = useState<{ n: number; rows: Set<string>; points: Set<string> } | null>(null);
  const [leaving, setLeaving] = useState<{ rows: Set<string>; points: Set<string> } | null>(null);
  const flashCount = useRef(0);
  const timers = useRef<number[]>([]);
  const rowsRef = useRef<HTMLDivElement>(null);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);
  useEffect(() => {
    if (!flash) return;
    const id = window.setTimeout(() => setFlash(null), FLASH_MS);
    return () => clearTimeout(id);
  }, [flash]);
  useEffect(() => {
    if (focus) rowsRef.current?.querySelector(`[data-row="${focus.key}"]`)?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [focus]);

  const highlight = (rows: string[], points: string[]) => {
    flashCount.current += 1;
    setFlash({ n: flashCount.current, rows: new Set(rows), points: new Set(points) });
  };
  const setValue = (next: PreprocessSpec | undefined) => {
    if (table) {
      const diff = changedKeys(steps, effectiveSteps(next, describe), table);
      if (diff.rows.length || diff.points.length) highlight(diff.rows, diff.points);
      if (diff.added.length) setFocus({ key: diff.added[0] });
    }
    onChange(next);
  };
  const specOf = (next: PreprocessStep[]): PreprocessSpec | undefined => {
    if (next.length === 0) return undefined;
    if (preset && sameSteps(next, preset.steps, describe.preprocess_sources)) return { preset: preset.name };
    return { ...(value?.preset ? { preset: value.preset } : {}), steps: next };
  };
  const commit = (next: PreprocessStep[]) => setValue(specOf(next));
  const commitFading = (next: PreprocessStep[]) => {
    const diff = table ? changedKeys(steps, next, table) : null;
    if (!diff || diff.removed.length === 0) return commit(next);
    setLeaving({ rows: new Set(diff.removed), points: new Set(diff.points) });
    timers.current.push(window.setTimeout(() => { setLeaving(null); commit(next); }, FADE_MS));
  };
  const reveal = (keys: string[]) => {
    if (keys.length === 0) return;
    setOpen((o) => new Set([...o, ...keys]));
    setFocus({ key: keys[0] });
  };
  const editStep = (i: number, edit: (s: PreprocessStep) => PreprocessStep) => commit(steps.map((s, k) => (k === i ? withCalibration(edit(s), sources) : s)));

  const infos = ready ? Object.fromEntries(Object.keys(table!).map((id) => [id, pointInfo(steps, table!, id)])) : {};
  const cards = ready ? buildCards(steps, table!) : [];
  const kinds = new Set(steps.map((s) => s.kind));
  const wrongOrder = scalingCancelledByRotation(steps);
  const positions = Object.keys(table ?? {});
  const sharedRotation = ready && positions.some((p) => shareTarget(table!, p, "rotation") === "residual");
  const parity = flash ? flash.n % 2 : 0;
  const effectOf = (key: string) => (leaving?.rows.has(key) ? "leaving" : flash?.rows.has(key) ? `flash-${parity}` : "");

  const choose = (position: string, kind: PreprocessKind) => {
    const next = toggleKind(steps, table!, sources, tied, position, kind);
    if (infos[position].applied.includes(kind)) return commitFading(next);
    commit(next);
    const plan = addPlan(table!, position, kind);
    if (plan) reveal([rowKey(kind, plan.position)]);
  };

  const onPoint = (id: string) => {
    const info = infos[id];
    if (!info) return;
    reveal(info.own.map((m) => rowKey(m.kind, id)));
    if (info.possible.length > 1 || info.via.length > 0) setChooser(id);
    else {
      setChooser(null);
      if (info.possible.length === 1 && info.applied.length === 0) choose(id, info.possible[0]);
    }
  };

  const kindName = (k: PreprocessKind) => (k === "rotation" ? t("Rotation") : t("Scaling"));
  const chooserContent = chooser && infos[chooser] && (
    <>
      <div className="arch-popover-head">
        <span>{positionTitle(graph!, chooser, t)}</span>
        <Button size="sm" variant="ghost" icon={<X size={13} />} aria-label={t("Close")} title={t("Close")} onClick={() => setChooser(null)} />
      </div>
      {infos[chooser].possible.map((k) => (
        <button key={k} type="button" className={`pre-kind-btn k-${k} ${infos[chooser].applied.includes(k) ? "on" : ""}`} aria-pressed={infos[chooser].applied.includes(k)}
          onClick={() => choose(chooser, k)}>
          {infos[chooser].applied.includes(k) && <Check size={13} />}{kindName(k)}
        </button>
      ))}
      {infos[chooser].via.length > 0 && <div className="arch-popover-note">{t("A shared entry covers this point: removing it removes that entry.")}</div>}
    </>
  );

  const states = ready ? pointStates(steps, table!, t) : {};
  for (const [id, st] of Object.entries(states)) {
    if (flash?.points.has(id)) st.flash = parity as 0 | 1;
    if (leaving?.points.has(id)) st.leaving = true;
  }

  return (
    <div className="stack gap-3 preprocess">
      {preset && modified && (
        <div className="row">
          <Badge tone="orange">{t("Modified from {name}", { name: preset.title })}</Badge>
          <Button variant="link" icon={<RotateCcw size={12} />} onClick={() => setValue({ preset: preset.name })}>{t("Restore preset")}</Button>
        </div>
      )}

      {Object.keys(presets).length > 0 && (
        <Field label={t("Start from a preset")}>
          <div className="row gap-2 pre-presets" role="group" aria-label={t("Start from a preset")}>
            {Object.values(presets).map((p) => (
              <button key={p.name} type="button" className={`pre-preset ${value?.preset === p.name ? "on" : ""}`} aria-pressed={value?.preset === p.name}
                title={p.description} aria-label={p.title} onClick={() => setValue({ preset: p.name })}>
                {p.title}{value?.preset === p.name && modified ? "*" : ""}
              </button>
            ))}
          </div>
        </Field>
      )}

      {onPickQuantizer && preset && Object.keys(preset.quantizers).length > 0 && (
        <Field label={t("Pair with a quantizer")}>
          <div className="row gap-2 pre-presets" role="group" aria-label={t("Pair with a quantizer")}>
            {Object.entries(preset.quantizers).flatMap(([name, names]) => names.map((p) => (
              <button key={`${name}:${p}`} type="button" className={`pre-preset ${method?.name === name && method?.preset === p ? "on" : ""}`}
                aria-pressed={method?.name === name && method?.preset === p} onClick={() => onPickQuantizer(name, p)}>
                {methodTitle(describe, name)} · {p}
              </button>
            )))}
          </div>
        </Field>
      )}

      {!model && <p className="faint small">{t("Pick a model first: the positions come from its architecture.")}</p>}
      {model && !ready && <p className="faint small">{t("This model has no inspected architecture graph, so preprocessing positions cannot be chosen.")}</p>}
      {preset && <p className="faint small">{preset.description}</p>}

      {ready && (
        <>
          <Field label={<>{t("Where to apply")}<Help text={t("Click a point to apply Rotation or Scaling there. Blue is Rotation, green is Scaling; the pills show offline (outlined) or online (filled). Greyed points cannot be used for this model; hover for the reason.")} /></>}>
            <ArchDiagram graph={graph!} mode="positions" positions={table!} points={states} onPoint={onPoint}
              popover={chooserContent ? { content: chooserContent, onDismiss: () => setChooser(null) } : undefined}
              numLayers={model?.shape?.num_layers} tied={tied} />
          </Field>

          {steps.length > 0 && (
            <div className="row gap-3">
              {kinds.size === 2 && (
                <span className="row gap-2">
                  <span className="faint small">{t("Order")}<Help text={t("With both, the order matters: the residual rotation must come before a scaling folded into a norm.")} /></span>
                  <Segmented value={steps[0].kind} onChange={(first) => {
                    if (first === steps[0].kind) return;
                    highlight(cards.flatMap((c) => c.lines.map((l) => rowKey(l.kind, c.position))), []);
                    onChange(specOf([...steps].reverse()));
                  }} options={[
                    { value: "rotation", label: t("Rotation first") }, { value: "scaling", label: t("Scaling first") },
                  ]} />
                  {wrongOrder && (
                    <span className="field-error row gap-2">
                      {t("The residual rotation folds the norm gains into the linears and cancels a scaling placed before it.")}
                      <Button size="sm" variant="link" onClick={() => onChange(specOf([...steps].reverse()))}>{t("Fix order")}</Button>
                    </span>
                  )}
                </span>
              )}
              <span className="spacer" />
              <Button size="sm" variant="ghost" icon={<Trash2 size={13} />} onClick={() => commitFading([])}>{t("Clear all")}</Button>
            </div>
          )}
          {!wrongOrder && problemAt(problems, "preprocess.steps") && <div className="field-error">{problemAt(problems, "preprocess.steps")}</div>}
          {steps.map((_, i) => problemAt(problems, `preprocess.steps[${i}]`) && <div key={i} className="field-error">{problemAt(problems, `preprocess.steps[${i}]`)}</div>)}

          {cards.length > 0 && (
            <div className="pre-cards" ref={rowsRef}>
              {cards.map((card) => {
                const info = infos[card.position];
                const missing = info.possible.filter((k) => !info.applied.includes(k));
                const scalingShared = positions.some((p) => shareTarget(table!, p, "scaling") === card.position);
                return (
                  <section key={card.position} className="pre-card">
                    <div className="pre-card-head">
                      <span className="strong">{positionTitle(graph!, card.position, t)}</span>
                      <span className="mono faint tiny">{card.position}</span>
                    </div>
                    {card.lines.map((l) => {
                      const step = steps[l.step];
                      const entry = step.entries[l.entry];
                      const key = rowKey(l.kind, card.position);
                      const note = l.kind === "rotation" && card.position === "residual" && sharedRotation
                        ? t("Also rotates the attention and FFN norm outputs of every layer with the same matrix.")
                        : l.kind === "scaling" && scalingShared && entry.mode === "offline"
                          ? t("Also covers every linear that reads this norm output (folded into the norm gain).")
                          : undefined;
                      return (
                        <EntryLine key={key} rowId={key} kind={l.kind} peers={step.entries.length} entry={entry} table={table!} sources={sources} tied={tied}
                          numLayers={model?.shape?.num_layers} open={open.has(key)} base={`preprocess.steps[${l.step}].entries[${l.entry}]`} problems={problems}
                          note={note} effect={effectOf(key)}
                          onToggle={() => setOpen((o) => { const n = new Set(o); if (n.has(key)) n.delete(key); else n.add(key); return n; })}
                          onChange={(next) => editStep(l.step, (s) => ({ ...s, entries: s.entries.map((e, j) => (j === l.entry ? next : e)) }))}
                          onRemove={() => commitFading(removeEntry(steps, sources, card.position, l.kind))}
                          onApplyToAll={() => editStep(l.step, (s) => ({ ...s, entries: s.entries.map((e) => (sourceAllowed(sources[l.kind]?.[entry.source.type], e.position) ? { ...e, source: { ...entry.source } } : e)) }))} />
                      );
                    })}
                    {missing.map((k) => (
                      <button key={k} type="button" className={`pre-add k-${k}`} onClick={() => choose(card.position, k)}>
                        {k === "scaling" ? t("+ Scaling here") : t("+ Rotation here")}
                      </button>
                    ))}
                  </section>
                );
              })}
            </div>
          )}

          {steps.map((step, i) => step.calibration && step.entries.some((e) => sources[step.kind]?.[e.source.type]?.needs_data) && (
            <Field key={step.kind} label={<>{step.kind === "rotation" ? t("Rotation calibration data") : t("Scaling calibration data")}<Help text={t("The sources used here are fitted on sample text.")} /></>}
              error={problemAt(problems, `preprocess.steps[${i}].calibration`)}>
              <CalibrationFields value={step.calibration} onChange={(calibration) => editStep(i, (s) => ({ ...s, calibration }))} />
            </Field>
          ))}
        </>
      )}
    </div>
  );
}
