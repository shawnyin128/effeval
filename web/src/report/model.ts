// Pure logic behind the Report view: experiment blocks, config rows, shared/varying fields, metric pivot.
import type { Describe, Row } from "../api";
import { categoryOf, metricDef, secondaryColumns } from "./metrics";
import { formatLimit, limitKey, specLimit, summarizeLimits, type Limit, type LimitSummary } from "./limits";

export type Spec = Record<string, any>;

export interface ConfigField {
  id: string;
  label: string;
  path: string;
  virtual: boolean;
  /** The label is a UI string that goes through the translator. */
  translate?: boolean;
}

export interface MetricColumn {
  id: string;
  task: string;
  metric: string;
  group: string;
  order: number;
  category: string;
  higherIsBetter: boolean | null;
  detail: boolean;
  /** Protocol settings that differ between the records shown in this column, e.g. "mode: loglikelihood / generation". */
  mixed: string[];
}

export interface AverageColumn {
  label: string;
  metric: string;
  columns: MetricColumn[];
}

export interface ConfigRow {
  key: string;
  isBaseline: boolean;
  method: string;
  config: Spec;
  flat: Map<string, string>;
  cells: Map<string, { value: string; row: Row; runs: number; all: Row[] }>;
  /** Sample limit of each task evaluated in this row. */
  limits: Map<string, Limit>;
  limit: LimitSummary;
  /** What a baseline row must share with this row to be compared with it: the single limit, or the per-task map. */
  limitKey: string;
  tasks: { task: string; mode: string; evaluator: string }[];
}

export interface Block {
  id: string;
  model: string;
  field: string;
  rows: ConfigRow[];
  fields: ConfigField[];
  varying: Set<string>;
  shared: Spec;
  metrics: MetricColumn[];
  metricGroups: { label: string; columns: MetricColumn[] }[];
}

const PROTOCOL: [string, (spec: Spec) => unknown][] = [
  ["task version", (s) => s.task?.version],
  ["mode", (s) => s.task?.mode],
  ["few-shot", (s) => s.task?.num_fewshot],
  ["evaluator", (s) => s.evaluator && `${s.evaluator.name} ${s.evaluator.version}`],
];

const CONFIG_ROOTS = ["model", "method", "runtime", "decoding", "generation", "determinism"];
const HIDDEN_PATHS = new Set(["model.name", "model.checkpoint", "model.revision", "model.architecture", "method.name", "method.version",
  "decoding.mode", "decoding.draft.model.checkpoint", "decoding.draft.model.revision", "decoding.draft.model.architecture", "decoding.draft.method.version"]);

/** Config columns of decoding and the provider: their label (instead of the last path segment) by path. */
const DECODING_LABELS: Record<string, string> = {
  "runtime.name": "Provider",
  "decoding.draft.model.name": "Draft model",
  "decoding.draft.model.dtype": "Draft precision",
  "decoding.draft.method.name": "Draft method",
  "decoding.num_speculative_tokens": "Draft tokens per step (k)",
  "decoding.verification": "Verification",
};
const PROVIDER_LABELS: Record<string, string> = { hf: "transformers", vllm: "vLLM", sglang: "SGLang" };
export const QUANT_FIELDS: ConfigField[] = [
  { id: "quant.weight", label: "Weight", path: "method.config.rules", virtual: true },
  { id: "quant.activation", label: "Activation", path: "method.config.rules", virtual: true },
  { id: "quant.attention", label: "Attention", path: "method.config.rules", virtual: true },
];

export const SAMPLES_FIELD: ConfigField = { id: "samples", label: "Samples", path: "task.limit", virtual: true };

/** Text of a row's Samples column: "All", a count, a percentage, or "Varies by task" when its tasks have different limits. */
export function samplesText(row: ConfigRow): string {
  if (row.limit.mixed) return "Varies by task";
  const limit = row.limit.limits[0] ?? null;
  return limit === null ? "All" : formatLimit(limit);
}

/** Per-task limits of a row whose tasks differ, one "task: limit" per line; "" for a uniform row. */
export function samplesDetail(row: ConfigRow): string {
  return row.limit.mixed ? [...row.limits].map(([task, l]) => `${task}: ${l === null ? "All" : formatLimit(l)}`).join("\n") : "";
}

/** The part of a frozen job spec that defines a configuration row (everything except the task). */
export function configOf(spec: Spec): Spec {
  return Object.fromEntries(CONFIG_ROOTS.filter((k) => k in spec).map((k) => [k, spec[k]]));
}

function stable(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stable).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.keys(value as Spec).sort().map((k) => `${JSON.stringify(k)}:${stable((value as Spec)[k])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/** Flatten nested objects to dotted paths; arrays stay whole (JSON) so rule lists compare as one value.
 *  Empty objects contribute nothing, so `config: {}` and an absent config compare equal. */
export function flatten(obj: Spec, prefix = "", out = new Map<string, string>()): Map<string, string> {
  for (const [k, v] of Object.entries(obj)) {
    const path = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === "object" && !Array.isArray(v)) flatten(v, path, out);
    else out.set(path, stable(v));
  }
  return out;
}

export function unflatten(entries: [string, string][]): Spec {
  const out: Spec = {};
  for (const [path, json] of entries) {
    const parts = path.split(".");
    let node = out;
    parts.slice(0, -1).forEach((p) => (node = node[p] ??= {}));
    node[parts[parts.length - 1]] = JSON.parse(json);
  }
  return out;
}

function formatQuant(q: Spec): string {
  const gran = String(q.granularity ?? "").replace("per_", "").replace("channel", "ch").replace("token", "tok").replace("tensor", "tensor");
  const group = (q.granularity === "per_group" ? ` g${q.group_size}${q.group_axis === "token" ? " along tokens" : ""}` : "")
    + (q.zero_point === "float" ? " float zero" : "") + (q.residual ? ` +${q.residual} recent exact` : "");
  if (q.type === "int") return `INT${q.bits} ${q.symmetric ? "sym" : "asym"} ${gran}${group}`;
  if (q.type === "fp") return `FP ${String(q.encoding).toUpperCase()} ${gran}${group}`;
  if (q.type === "mx") return `MX ${String(q.element).toUpperCase()} b${q.block_size} ${String(q.scale).toUpperCase()}`;
  return JSON.stringify(q);
}

/** Readable Weight / Activation / Attention summaries of quantization rules. */
export function quantSummary(rules: unknown): Record<string, string> {
  const out: Record<string, string[]> = { "quant.weight": [], "quant.activation": [], "quant.attention": [] };
  // Attention settings keep the operands they apply to (Q, K, V), so "K/V: INT4 …" reads as KV-cache quantization.
  const attention = new Map<string, Set<string>>();
  if (!Array.isArray(rules)) return {};
  for (const rule of rules as Spec[]) {
    const layers = rule.layers === undefined || rule.layers === "all" ? "" : ` @L${rule.layers}`;
    const text = `${formatQuant(rule.quant ?? {})}${layers}`;
    for (const site of (rule.sites ?? []) as string[]) {
      if (site.startsWith("attn.") || site.startsWith("kv_cache")) {
        const operands = site.startsWith("kv_cache") ? ["K", "V"] : [site.slice("attn.".length).toUpperCase()];
        if (!attention.has(text)) attention.set(text, new Set());
        operands.forEach((o) => attention.get(text)!.add(o));
        continue;
      }
      const kind = site.endsWith(".weight") ? "quant.weight" : "quant.activation";
      if (!out[kind].includes(text)) out[kind].push(text);
    }
  }
  const order = ["Q", "K", "V"];
  out["quant.attention"] = [...attention].map(([text, operands]) =>
    `${[...operands].sort((a, b) => order.indexOf(a) - order.indexOf(b)).join("/")}: ${text}`);
  return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v.join("; ")]));
}

export function fieldValue(row: ConfigRow, field: ConfigField): string {
  if (field.id === SAMPLES_FIELD.id) return samplesText(row);
  if (field.virtual) return quantSummary(row.config.method?.config?.rules)[field.id] || "—";
  const raw = row.flat.get(field.path);
  if (raw === undefined) return "—";
  const value = JSON.parse(raw);
  if (field.path === "runtime.name") return PROVIDER_LABELS[value] ?? String(value);
  return value === null ? "—" : typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** Row identity of each record: its configuration, plus its sample limit when the configuration was evaluated
 *  under more than one limit for some task. A configuration whose every task has a single limit stays one row
 *  (so a run that gave tasks different limits is not torn apart); otherwise every limit gets its own row. */
function rowKeys(records: Row[]): Map<Row, { config: string; key: string }> {
  const limitsByConfig = new Map<string, Map<string, Set<Limit>>>();
  const configKeys = new Map<Row, string>();
  for (const row of records) {
    const spec = row.record.spec as Spec;
    const config = stable(configOf(spec));
    configKeys.set(row, config);
    const tasks = limitsByConfig.get(config) ?? new Map<string, Set<Limit>>();
    limitsByConfig.set(config, tasks);
    tasks.set(spec.task?.name, (tasks.get(spec.task?.name) ?? new Set<Limit>()).add(specLimit(spec)));
  }
  const keys = new Map<Row, { config: string; key: string }>();
  for (const row of records) {
    const config = configKeys.get(row)!;
    const split = [...limitsByConfig.get(config)!.values()].some((limits) => limits.size > 1);
    keys.set(row, { config, key: split ? `${config}|limit:${limitKey(specLimit(row.record.spec as Spec))}` : config });
  }
  return keys;
}

/** Sort position of a limit: the full set first, then larger limits before smaller. */
function limitOrder(limit: Limit): number {
  return limit === null ? Infinity : limit;
}

export function buildBlocks(rows: Row[], describe: Describe): Block[] {
  const byBlock = new Map<string, Row[]>();
  for (const row of rows) {
    const spec = row.record.spec as Spec;
    const key = `${row.model}|${spec.field}`;
    byBlock.set(key, [...(byBlock.get(key) ?? []), row]);
  }
  const blocks: Block[] = [];
  for (const [id, records] of byBlock) {
    const [model, field] = id.split("|");
    const configRows = new Map<string, ConfigRow>();
    const configRank = new Map<string, number>();
    const metricIds = new Map<string, MetricColumn>();
    const ordered = [...records].sort((a, b) => b.created_at.localeCompare(a.created_at));
    const keys = rowKeys(ordered);
    for (const row of ordered) {
      const spec = row.record.spec as Spec;
      const config = configOf(spec);
      const { config: configKey, key } = keys.get(row)!;
      if (!configRank.has(configKey)) configRank.set(configKey, configRank.size);
      let entry = configRows.get(key);
      if (!entry) {
        entry = { key, isBaseline: spec.method?.name === "base" && !spec.decoding, method: spec.method?.name, config, flat: flatten(config),
          cells: new Map(), limits: new Map(), limit: summarizeLimits([]), limitKey: "", tasks: [] };
        configRows.set(key, entry);
      }
      entry.limits.set(spec.task?.name, specLimit(spec));
      if (!entry.tasks.some((t) => t.task === spec.task?.name)) {
        entry.tasks.push({ task: spec.task?.name, mode: spec.task?.mode, evaluator: spec.evaluator?.name });
      }
      for (const [metric, value] of Object.entries(row.metrics)) {
        const mid = `${spec.task?.name}:${metric}`;
        if (!metricIds.has(mid)) {
          const category = categoryOf(describe, spec.task?.name);
          const def = metricDef(describe, metric, spec.task?.name);
          metricIds.set(mid, { id: mid, task: spec.task?.name, metric, group: category.label, order: category.order,
            category: category.name, higherIsBetter: def.higher_is_better, detail: def.placement === "detail", mixed: [] });
        }
        const existing = entry.cells.get(mid);
        if (existing) {
          existing.runs += 1;
          existing.all.push(row);
        } else entry.cells.set(mid, { value, row, runs: 1, all: [row] });
      }
    }
    for (const entry of configRows.values()) {
      entry.limit = summarizeLimits([...entry.limits.values()]);
      entry.limitKey = entry.limit.mixed ? limitKey([...entry.limits].sort(([a], [b]) => a.localeCompare(b))) : limitKey(entry.limit.limits[0] ?? null);
    }
    for (const column of metricIds.values()) {
      const specs = [...configRows.values()].flatMap((r) => r.cells.get(column.id)?.all ?? []).map((r) => r.record.spec as Spec);
      column.mixed = PROTOCOL.flatMap(([label, read]) => {
        const values = [...new Set(specs.map((s) => String(read(s) ?? "—")))];
        return values.length > 1 ? [`${label}: ${values.join(" / ")}`] : [];
      });
    }
    const first = (r: ConfigRow) => limitOrder(r.limit.limits[0] ?? null);
    const list = [...configRows.values()].sort((a, b) => Number(b.isBaseline) - Number(a.isBaseline) || a.method.localeCompare(b.method)
      || configRank.get(stable(a.config))! - configRank.get(stable(b.config))!
      || Math.sign(first(b) - first(a)) || a.limitKey.localeCompare(b.limitKey));
    const allPaths = new Set(list.flatMap((r) => [...r.flat.keys()]));
    const varying = new Set<string>();
    const sharedEntries: [string, string][] = [];
    for (const path of allPaths) {
      const values = new Set(list.map((r) => r.flat.get(path) ?? "null"));
      if (values.size > 1) varying.add(path);
      else sharedEntries.push([path, [...values][0]]);
    }
    const hasRules = list.some((r) => Array.isArray(r.config.method?.config?.rules));
    const quantFields = hasRules ? QUANT_FIELDS.filter((f) => list.some((r) => fieldValue(r, f) !== "—")) : [];
    for (const f of quantFields) if (new Set(list.map((r) => fieldValue(r, f))).size > 1) varying.add(f.id);
    if (new Set(list.map(samplesText)).size > 1) varying.add(SAMPLES_FIELD.id);
    const fields: ConfigField[] = [
      ...quantFields,
      SAMPLES_FIELD,
      ...[...allPaths].filter((p) => !HIDDEN_PATHS.has(p) && !(hasRules && p === "method.config.rules")).sort()
        .map((p) => ({ id: p, label: DECODING_LABELS[p] ?? p.split(".").slice(-1)[0], path: p, virtual: false, translate: p in DECODING_LABELS })),
    ];
    const rank = (m: MetricColumn) => {
      const at = describe.tasks[m.task]?.metrics.findIndex((x) => x.name === m.metric) ?? -1;
      return at < 0 ? 99 : at;
    };
    const metrics = [...metricIds.values()].sort((a, b) =>
      a.order - b.order || a.task.localeCompare(b.task) || rank(a) - rank(b) || a.metric.localeCompare(b.metric));
    const groups: { label: string; columns: MetricColumn[] }[] = [];
    for (const m of metrics) {
      const last = groups[groups.length - 1];
      if (last && last.label === m.group) last.columns.push(m);
      else groups.push({ label: m.group, columns: [m] });
    }
    blocks.push({ id, model, field, rows: list, fields, varying, shared: unflatten(sharedEntries), metrics, metricGroups: groups });
  }
  return blocks.sort((a, b) => a.model.localeCompare(b.model) || a.field.localeCompare(b.field));
}

/** The registry's average columns that cover at least two of the given metric columns. */
export function averageColumns(describe: Describe, metrics: MetricColumn[]): AverageColumn[] {
  const secondary = new Set(secondaryColumns(describe, metrics));
  return describe.averages
    .map((a) => ({ label: a.label, metric: a.metric, columns: metrics.filter((m) => !secondary.has(m.id) && a.categories.includes(m.category)) }))
    .filter((a) => a.columns.length >= 2);
}

/** Unweighted mean of a row's values over an average's columns; null if any is missing. */
export function averageOf(average: AverageColumn, row: ConfigRow): number | null {
  const values = average.columns.map((c) => row.cells.get(c.id)?.value);
  if (values.some((v) => v === undefined)) return null;
  return values.reduce((sum, v) => sum + Number(v), 0) / values.length;
}

export function defaultFields(block: Block): string[] {
  const limited = block.rows.some((r) => r.limit.limited);
  return block.fields.filter((f) => (block.varying.has(f.id) || (limited && f.id === SAMPLES_FIELD.id)) && f.id !== "method.preset").slice(0, 4).map((f) => f.id);
}

