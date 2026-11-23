import { useEffect, useMemo, useState } from "react";
import type { Cell, Describe, TableView } from "../api";
import { useApp } from "../components/common";
import { providerLabel, rolesOf, usesDecoding } from "../compose/Decoding";
import { useT } from "../i18n";
import { formatCellValue, metricDef, metricHeader } from "../report/metrics";
import { Select } from "../ui/controls";

export type GridRow = TableView["blocks"][number]["rows"][number];
type Translate = (key: string, vars?: Record<string, string | number>) => string;
type Empty = { reason: "reference" | "not_applicable" | "not_reported" | "pending"; title: string };

/** Whether a table shows the decoding columns: its field has a draft role or some row has a draft. */
export function hasDecodingColumns(describe: Describe, table: TableView): boolean {
  return usesDecoding(rolesOf(describe, table.definition.field)) || table.blocks.some((b) => b.rows.some((r) => !!r.draft));
}

/** Whether a task column is a speed task. */
export function isSpeedTask(describe: Describe, task: string): boolean {
  return describe.tasks[task]?.evaluator === "speed";
}

const taskMetricNames = (describe: Describe, task: string): string[] =>
  (describe.tasks[task]?.metrics ?? []).filter((m) => m.stderr_of == null).map((m) => m.name);

/**
 * Metric names offered for speed columns: the speed tasks' own metrics in task order, then the comparatives
 * (base names before ".") found in any cell, with "speedup" first when the table has comparisons or speculative rows.
 */
export function speedMetricOptions(describe: Describe, table: TableView): string[] {
  const speedTasks = table.tasks.filter((task) => isSpeedTask(describe, task));
  if (speedTasks.length === 0) return [];
  const own = [...new Set(speedTasks.flatMap((task) => taskMetricNames(describe, task)))];
  const rows = table.blocks.flatMap((b) => b.rows);
  const comparatives = new Set<string>();
  for (const row of rows) for (const task of speedTasks) {
    Object.keys(row.cells[task]?.comparisons ?? {}).forEach((k) => comparatives.add(k.split(".")[0]));
  }
  if (rows.some((r) => !!r.draft)) comparatives.add("speedup");
  const extra = [...comparatives].filter((n) => !own.includes(n)).sort((a, b) => Number(b === "speedup") - Number(a === "speedup"));
  return [...extra.filter((n) => n === "speedup"), ...own, ...extra.filter((n) => n !== "speedup")];
}

/** "speedup" when offered and some row has a draft, else the first speed task's first metric. */
export function defaultSpeedMetric(describe: Describe, table: TableView): string {
  const options = speedMetricOptions(describe, table);
  const speculative = table.blocks.some((b) => b.rows.some((r) => !!r.draft));
  if (speculative && options.includes("speedup")) return "speedup";
  const first = table.tasks.find((task) => isSpeedTask(describe, task));
  return (first && taskMetricNames(describe, first)[0]) ?? options[0] ?? "";
}

/** The cell as formatCellValue reads it for a comparative: comparisons stand in for metrics. */
export const comparativeView = (cell: Cell): Cell => ({ ...cell, metrics: cell.comparisons, spread: cell.comparison_spread });

/** A speed cell's text for `name`, from metrics or comparisons (mean ± std for repeats), or the reason it is empty. */
export function speedCell(describe: Describe, row: GridRow, cell: Cell, task: string, name: string, t: Translate): { text: string } | Empty {
  const own = taskMetricNames(describe, task).includes(name);
  const text = own ? formatCellValue(describe, name, cell, task) : formatCellValue(describe, name, comparativeView(cell), task);
  if (text !== "") return { text };
  const metric = metricDef(describe, name, task).label;
  const provider = providerLabel(describe, row.runtime);
  if (row.baseline && !own) {
    return { reason: "reference", title: t("This row is the autoregressive reference the other rows are compared with.") };
  }
  if (!row.decoding) return { reason: "not_applicable", title: t("Autoregressive decoding has no {metric}.", { metric }) };
  if (row.runtime && row.runtime !== "hf") return { reason: "not_reported", title: t("{provider} does not report {metric}.", { provider, metric }) };
  return {
    reason: "pending",
    title: own ? t("No value for this metric.")
      : t("Speedup and output match appear once the autoregressive reference has a result and both runs' per-sample files are on the server."),
  };
}

/** The text shown for a speed cell and the hover lines saying what a value was computed against. */
export function speedCellView(describe: Describe, row: GridRow, cell: Cell, task: string, name: string, t: Translate):
  { text: string; title?: string; empty: boolean } {
  const shown = speedCell(describe, row, cell, task, name, t);
  if ("reason" in shown) return { text: shown.reason === "reference" ? t("ref") : shown.reason === "pending" ? "—" : t("n/a"), title: shown.title, empty: true };
  const comparative = !taskMetricNames(describe, task).includes(name);
  const title = comparative
    ? t("Computed against the autoregressive reference of this model on {provider}.", { provider: providerLabel(describe, row.runtime) })
    : undefined;
  return { text: shown.text, title, empty: false };
}

/** Hover lines for the comparatives of a cell (those without "." in their name), with the reference they refer to. */
export function comparisonLines(describe: Describe, row: GridRow, cell: Cell, task: string, t: Translate): string[] {
  const names = Object.keys(cell.comparisons ?? {}).filter((k) => !k.includes("."));
  if (names.length === 0) return [];
  const view = comparativeView(cell);
  return [
    names.map((k) => `${metricDef(describe, k, task).label} ${formatCellValue(describe, k, view, task)}`).join(" · "),
    t("Computed against the autoregressive reference of this model on {provider}.", { provider: providerLabel(describe, row.runtime) }),
  ];
}

/** Header cells of the decoding columns. */
export function DecodingHeads() {
  const t = useT();
  return <>{["Decoding", "Draft model", "k", "Provider"].map((h) => <th key={h} className="xt-config">{t(h)}</th>)}</>;
}

/** Body cells of one row: the decoding mode, draft code, k and provider; the mode's title is the verification. */
export function DecodingCells({ row }: { row: GridRow }) {
  const t = useT();
  const { describe } = useApp();
  return (
    <>
      <td className="xt-config" title={row.decoding?.verification}>{row.decoding ? t("Speculative") : t("Autoregressive")}</td>
      <td className="xt-config mono" title={row.draft ?? undefined}>{row.draft ?? "—"}</td>
      <td className="xt-config">{row.decoding?.k ?? "—"}</td>
      <td className="xt-config">{providerLabel(describe, row.runtime)}</td>
    </>
  );
}

/** The speed metric chosen for a table (kept in localStorage under `storageKey`), its options and a setter. */
export function useSpeedMetric(describe: Describe, table: TableView | null, storageKey: string): { name: string; options: string[]; set: (name: string) => void } {
  const options = useMemo(() => (table ? speedMetricOptions(describe, table) : []), [describe, table]);
  const [stored, setStored] = useState<string | null>(() => localStorage.getItem(storageKey));
  useEffect(() => setStored(localStorage.getItem(storageKey)), [storageKey]);
  const name = stored !== null && options.includes(stored) ? stored : table ? defaultSpeedMetric(describe, table) : "";
  const set = (next: string) => {
    localStorage.setItem(storageKey, next);
    setStored(next);
  };
  return { name, options, set };
}

/** Select of the speed columns' metric. */
export function SpeedMetricSelect({ describe, name, options, onChange }: {
  describe: Describe;
  name: string;
  options: string[];
  onChange: (name: string) => void;
}) {
  const t = useT();
  if (options.length === 0) return null;
  return (
    <span className="row" style={{ gap: 6 }}>
      <span className="faint tiny">{t("Speed metric")}</span>
      <span style={{ width: 170 }}>
        <Select value={name} onChange={onChange} ariaLabel={t("Speed metric")}
          options={options.map((n) => ({ value: n, label: metricHeader(describe, n) }))} />
      </span>
    </span>
  );
}
