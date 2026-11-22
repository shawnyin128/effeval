// Metric rendering driven by the server's metric registry (core/data/metrics.yaml); no metric is named here.
import type { Cell, Describe, MetricDef, TaskDef } from "../api";

const FALLBACK: MetricDef = {
  name: "", label: "", kind: "quality", description: "", unit: "",
  format: { style: "fixed", decimals: 4 }, higher_is_better: null, placement: "column",
};

/**
 * The registry entry for `name`. Metrics the registry does not declare get a neutral 4-decimal definition whose
 * direction comes from the task's own declaration; standard errors are details, everything else a column.
 */
export function metricDef(describe: Describe, name: string, task?: TaskDef | string): MetricDef {
  const declared = describe.metrics[name];
  if (declared) return declared;
  const definition = typeof task === "string" ? describe.tasks[task] : task;
  const own = definition?.metrics.find((m) => m.name === name);
  return {
    ...FALLBACK, name, label: name,
    higher_is_better: own?.higher_is_better ?? null,
    placement: own?.stderr_of ? "detail" : "column",
  };
}

/** Metric names in display priority; a name matches by its base before any ",<filter>" suffix. */
const METRIC_PRIORITY = ["acc", "accuracy", "exact_match", "pass@1", "ppl", "word_perplexity"];

/**
 * The metric to show for a task by default: the first of METRIC_PRIORITY among `names` (default: the task's
 * own metrics), else the first name that is not a standard error or a count, else the first name.
 */
export function primaryMetric(describe: Describe, task: string, names?: string[]): string | undefined {
  const own = describe.tasks[task]?.metrics ?? [];
  const available = names ?? own.map((m) => m.name);
  const spread = (name: string) => own.find((m) => m.name === name)?.stderr_of != null || /_stderr(,|$)|\.n$/.test(name);
  const candidates = available.filter((n) => !spread(n));
  for (const wanted of METRIC_PRIORITY) {
    const hit = candidates.find((n) => n.split(",")[0] === wanted);
    if (hit !== undefined) return hit;
  }
  return candidates[0] ?? available[0];
}

/** Ids of the columns that are not their task's primary metric; `names` are the metrics each task reports. */
export function secondaryColumns(describe: Describe, columns: { id: string; task: string; metric: string }[]): string[] {
  const reported = new Map<string, string[]>();
  for (const c of columns) reported.set(c.task, [...(reported.get(c.task) ?? []), c.metric]);
  return columns.filter((c) => c.metric !== primaryMetric(describe, c.task, reported.get(c.task))).map((c) => c.id);
}

/** Column header text: the metric's label plus its unit. */
export function metricHeader(describe: Describe, name: string, task?: TaskDef | string): string {
  const def = metricDef(describe, name, task);
  return def.unit ? `${def.label} (${def.unit})` : def.label;
}

/** Display text of a full-precision metric value according to its registered format. */
export function formatMetric(describe: Describe, name: string, value: string | number | null | undefined, task?: TaskDef | string): string {
  if (value === null || value === undefined || value === "") return "";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  const { style, decimals } = metricDef(describe, name, task).format;
  if (style === "percent") return `${(n * 100).toFixed(decimals)}%`;
  if (style === "scientific") return n.toExponential(decimals);
  if (style === "integer") return Math.round(n).toLocaleString();
  if (style === "duration") return n >= 1000 ? `${(n / 1000).toFixed(decimals)} s` : `${n.toFixed(decimals)} ms`;
  if (n !== 0 && Math.abs(n) < 10 ** -decimals) return n.toExponential(2);
  return n.toFixed(decimals);
}

/**
 * A table cell's value for `name`: the formatted mean, and for a repeated run "mean ± std (n=3)" where std is the
 * sample std; "n=2/3" while some repeats are still missing.
 */
export function formatCellValue(describe: Describe, name: string, cell: Cell, task?: TaskDef | string): string {
  const value = cell.metrics?.[name];
  if (value === undefined) return "";
  const text = formatMetric(describe, name, value, task);
  const spread = cell.spread?.[name];
  if (!spread || spread.std === null) return text;
  const planned = cell.repeats?.length ?? spread.n;
  return `${text} ± ${formatMetric(describe, name, spread.std, task)} (n=${spread.n}${planned > spread.n ? `/${planned}` : ""})`;
}

/** The band a task's columns sit in: its category's label and order, or the raw category when undeclared. */
export function categoryOf(describe: Describe, task: string): { label: string; order: number; name: string } {
  const name = describe.tasks[task]?.category ?? "";
  const def = describe.categories[name];
  return def ? { label: def.label, order: def.order, name } : { label: name || "Other", order: 999, name };
}
