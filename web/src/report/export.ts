// Paper-style table export of Report blocks: best value bold, second best underlined, our methods highlighted.
import { rowMethodTitle, type Describe, type TableView } from "../api";
import { formatCellValue, formatMetric, metricDef, primaryMetric } from "./metrics";
import { limitFootnote, rowLimitFootnote, summarizeLimits, type Limit, type LimitSummary } from "./limits";
import { SAMPLES_FIELD, averageColumns, averageOf, fieldValue, samplesText, type Block, type ConfigField, type ConfigRow } from "./model";

export type Mark = "best" | "second" | null;

export interface ExportColumn {
  label: string;
  band: string;
  higherIsBetter: boolean | null;
  /** Sample limits of the results in this column; a subset gets "†" in the label. */
  limit?: LimitSummary;
}

export interface ExportRow {
  method: string;
  ours: boolean;
  values: (number | null)[];
  cells: string[];
  marks: Mark[];
  /** Rows are only ranked against rows with the same limit key (sample count). */
  limitKey: string;
}

export interface ExportSection {
  model: string;
  /** One baseline row per sample count, each with the text for the group column. */
  baselines: { row: ExportRow; group: string }[];
  groups: { label: string; rows: ExportRow[] }[];
}

export interface ExportTable {
  groupHeader: string;
  columns: ExportColumn[];
  sections: ExportSection[];
  /** Footnote explaining "†" marks; null when every result covers the full sample set. */
  footnote: string | null;
}

const DAGGER = " †";

function withFootnote(table: Omit<ExportTable, "footnote">): ExportTable {
  const columns = table.columns.map((c) => (c.limit?.limited ? { ...c, label: c.label + DAGGER } : c));
  return { ...table, columns, footnote: limitFootnote(table.columns) };
}

const PRECISION: Record<string, string> = { bfloat16: "BF16", float16: "FP16", float32: "FP32" };

/** A configuration row's display name: the stored results' label ("QuaRot + GPTQ"), else the method's title. */
export function configRowTitle(describe: Describe, row: ConfigRow): string {
  const first = row.cells.values().next().value;
  return rowMethodTitle(describe, { method: row.method, label: first?.row.label });
}

function precisionLabel(dtype: string): string {
  return PRECISION[dtype] ?? dtype.toUpperCase();
}

function rank(rows: ExportRow[], columns: ExportColumn[]): void {
  if (rows.length < 2) return;
  columns.forEach((column, c) => {
    if (column.higherIsBetter === null) return;
    const sign = column.higherIsBetter ? -1 : 1;
    const distinct = [...new Set(rows.map((r) => r.values[c]).filter((v): v is number => v !== null))].sort((a, b) => sign * (a - b));
    rows.forEach((r) => {
      const v = r.values[c];
      r.marks[c] = v === null ? null : v === distinct[0] ? "best" : v === distinct[1] ? "second" : null;
    });
  });
}

/** Build a paper table from blocks that share the given visible metric columns and grouping fields. */
export function exportTable(describe: Describe, blocks: { block: Block; fields: ConfigField[] }[]): ExportTable {
  const first = blocks[0];
  const withSamples = ({ block, fields }: { block: Block; fields: ConfigField[] }) =>
    fields.some((f) => f.id === SAMPLES_FIELD.id) || !block.rows.some((r) => r.limit.limited) ? fields : [...fields, SAMPLES_FIELD];
  const metricKeys = [...new Map(blocks.flatMap(({ block }) => block.metrics.map((m) => [`${m.task}:${m.metric}`, m] as const))).values()];
  const averages = first ? averageColumns(describe, metricKeys) : [];
  const perTask = new Map<string, number>();
  metricKeys.forEach((m) => perTask.set(m.task, (perTask.get(m.task) ?? 0) + 1));
  const columns: ExportColumn[] = [
    ...metricKeys.map((m) => ({
      label: perTask.get(m.task)! > 1 ? `${m.task} ${metricDef(describe, m.metric, m.task).label}` : m.task,
      band: m.group,
      higherIsBetter: m.higherIsBetter,
    })),
    ...averages.map((a) => ({
      label: a.label, band: `avg:${a.label}`, higherIsBetter: metricDef(describe, a.metric).higher_is_better,
    })),
  ];
  const toRow = (block: Block, row: ConfigRow): ExportRow => {
    const values: (number | null)[] = [];
    const cells: string[] = [];
    metricKeys.forEach((m) => {
      const cell = row.cells.get(`${m.task}:${m.metric}`);
      const raw = cell?.value;
      const limit = row.limits.get(m.task) ?? null;
      values.push(raw === undefined ? null : Number(raw));
      cells.push(raw === undefined ? "–" : formatMetric(describe, m.metric, raw) + (cell && row.limit.mixed && limit !== null ? "†" : ""));
    });
    averages.forEach((a) => {
      const scoped = { ...a, columns: a.columns.filter((c) => block.metrics.some((m) => m.id === c.id)) };
      const v = scoped.columns.length >= 2 ? averageOf(scoped, row) : null;
      const subset = row.limit.mixed && scoped.columns.some((c) => (row.limits.get(c.task) ?? null) !== null);
      values.push(v);
      cells.push(v === null ? "–" : formatMetric(describe, a.metric, v) + (subset ? "†" : ""));
    });
    const ours = !(describe.methods[row.method]?.builtin ?? true);
    const label = row.isBaseline ? precisionLabel(String(row.config.model?.dtype ?? row.method)) : configRowTitle(describe, row);
    const method = row.limit.limited && !row.limit.mixed ? label + DAGGER : label;
    return { method, ours, values, cells, marks: columns.map(() => null), limitKey: row.limitKey };
  };
  const uniformSubsets: { name: string; limit: number }[] = [];
  const perTaskSubsets: { name: string; limits: [string, Limit][] }[] = [];
  const noteSubset = (section: string, name: string, row: ConfigRow) => {
    const full = [blocks.length > 1 ? section : "", name].filter(Boolean).join(" ");
    if (row.limit.mixed) perTaskSubsets.push({ name: full, limits: [...row.limits] });
    else if (row.limit.limited) uniformSubsets.push({ name: full, limit: row.limit.limits[0] as number });
  };
  const sections = blocks.map((part) => {
    const { block } = part;
    const fields = withSamples(part);
    const groups = new Map<string, ExportRow[]>();
    const baselines: ExportSection["baselines"] = [];
    for (const row of block.rows) {
      const exported = toRow(block, row);
      if (row.isBaseline) {
        baselines.push({ row: exported, group: row.limit.limited ? samplesText(row) : "–" });
        noteSubset(block.model, exported.method.replace(DAGGER, ""), row);
        continue;
      }
      const label = fields.map((f) => fieldValue(row, f)).join(" · ");
      groups.set(label, [...(groups.get(label) ?? []), exported]);
      noteSubset(block.model, [label, configRowTitle(describe, row)].filter(Boolean).join(" "), row);
    }
    groups.forEach((rows) => {
      const byLimit = new Map<string, ExportRow[]>();
      rows.forEach((r) => byLimit.set(r.limitKey, [...(byLimit.get(r.limitKey) ?? []), r]));
      byLimit.forEach((same) => rank(same, columns));
    });
    return { model: block.model, baselines, groups: [...groups].map(([label, rows]) => ({ label, rows })) };
  });
  return { groupHeader: (first ? withSamples(first) : []).map((f) => f.label).join(" · "), columns, sections, footnote: rowLimitFootnote(uniformSubsets, perTaskSubsets) };
}

function arrow(column: ExportColumn, up: string, down: string): string {
  return column.higherIsBetter === null ? "" : column.higherIsBetter ? up : down;
}

const LATEX_SPECIAL: Record<string, string> = { "\\": "\\textbackslash{}", "&": "\\&", "%": "\\%", "$": "\\$", "#": "\\#", _: "\\_", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}" };

function tex(text: string): string {
  return text.replace(/[\\&%$#_{}~^]/g, (c) => LATEX_SPECIAL[c]).replace(/–/g, "--").replace(/†/g, "$^\\dagger$").replace(/±/g, "$\\pm$");
}

function bands(columns: ExportColumn[]): number[] {
  const sizes: number[] = [];
  columns.forEach((c, i) => {
    if (i > 0 && columns[i - 1].band === c.band) sizes[sizes.length - 1] += 1;
    else sizes.push(1);
  });
  return sizes;
}

/** LaTeX in the lab's paper style: table*, resizebox, booktabs rules, multirow models and groups, our rows tinted.
 *  Needs \usepackage{booktabs,multirow,graphicx} and \usepackage[table]{xcolor}. */
export function toLatex(table: ExportTable): string {
  const hasGroups = table.groupHeader !== "";
  const lead = hasGroups ? 3 : 2;
  const width = lead + table.columns.length;
  const sizes = bands(table.columns);
  const spec = `@{}${"c".repeat(lead)}|${sizes.map((n) => "c".repeat(n)).join("|")}@{}`;
  const ruled = new Set<number>([lead - 1]);
  sizes.reduce((at, n) => (ruled.add(at + n - 1), at + n), lead);
  ruled.delete(width - 1);
  const head = (text: string, i: number) => (ruled.has(i) ? `\\multicolumn{1}{c}{${text}}` : text);
  const headers = ["\\textbf{Models}", ...(hasGroups ? [`\\textbf{${tex(table.groupHeader)}}`] : []), "\\textbf{Methods}",
    ...table.columns.map((c) => `\\textbf{${tex(c.label)}}${arrow(c, " $\\uparrow$", " $\\downarrow$")}`)];
  const mark = (text: string, m: Mark) => (m === "best" ? `\\textbf{${text}}` : m === "second" ? `\\underline{${text}}` : text);
  const tint = "\\cellcolor{violet!8}";
  const line = (groupCell: string | null, row: ExportRow) => {
    const cells = [
      row.ours ? `\\textbf{${tex(row.method)}}` : tex(row.method),
      ...row.cells.map((v, i) => mark(tex(v), row.marks[i])),
    ].map((c) => (row.ours ? tint + c : c));
    const group = groupCell === null ? [] : [row.ours && groupCell === "" ? "\\cellcolor{white!10}" : groupCell];
    return `& ${[...group, ...cells].join(" & ")} \\\\`;
  };
  const out = [
    "% \\usepackage{booktabs,multirow,graphicx} \\usepackage[table]{xcolor}",
    "\\begin{table*}[t]\\caption{}",
    "\\label{tab:}",
    "\\centering",
    "\\resizebox{\\textwidth}{!}{",
    `\\begin{tabular}{${spec}}`,
    "\\toprule",
    `${headers.map(head).join(" & ")} \\\\`,
    "\\midrule",
  ];
  table.sections.forEach((section, s) => {
    const total = section.baselines.length + section.groups.reduce((n, g) => n + g.rows.length, 0);
    out.push(`\\multirow{${total}}{*}{${tex(section.model)}}`);
    if (section.baselines.length) {
      section.baselines.forEach((b) => out.push(line(hasGroups ? tex(b.group) : null, b.row)));
      if (section.groups.length) out.push(`\\cmidrule(lr){2-${width}}`);
    }
    section.groups.forEach((group, g) => {
      group.rows.forEach((row, r) => {
        const label = r > 0 ? "" : group.rows.length > 1 ? `\\multirow{${group.rows.length}}{*}{${tex(group.label)}}` : tex(group.label);
        out.push(line(hasGroups ? label : null, row));
      });
      if (g < section.groups.length - 1) out.push(`\\cmidrule(lr){2-${width}}`);
    });
    out.push(s < table.sections.length - 1 ? "\\midrule" : "\\bottomrule");
  });
  out.push("\\end{tabular}}");
  if (table.footnote) out.push(`\\par\\vspace{2pt}{\\footnotesize ${tex(table.footnote)}}`);
  out.push("\\end{table*}");
  return out.join("\n");
}

/** Flat header and rows with marks, for Markdown and HTML. */
export function flatRows(table: ExportTable): { header: string[]; rows: { cells: string[]; marks: Mark[]; ours: boolean }[]; footnote: string | null } {
  const hasGroups = table.groupHeader !== "";
  const header = ["Models", ...(hasGroups ? [table.groupHeader] : []), "Methods", ...table.columns.map((c) => `${c.label}${arrow(c, " ↑", " ↓")}`)];
  const lead = hasGroups ? 3 : 2;
  const rows: { cells: string[]; marks: Mark[]; ours: boolean }[] = [];
  for (const section of table.sections) {
    const push = (group: string, row: ExportRow) =>
      rows.push({ cells: [section.model, ...(hasGroups ? [group] : []), row.method, ...row.cells], marks: [...Array(lead).fill(null), ...row.marks], ours: row.ours });
    section.baselines.forEach((b) => push(b.group, b.row));
    section.groups.forEach((g) => g.rows.forEach((r) => push(g.label, r)));
  }
  return { header, rows, footnote: table.footnote };
}

/** Paper table of an experiment table (live or frozen): one section per model, each task's primary metric. */
export function exportFromGrid(describe: Describe, grid: TableView): ExportTable {
  const frozen = new Map((grid.columns ?? []).map((c) => [c.task, c]));
  const primary = (task: string) => frozen.get(task)?.metric ?? primaryMetric(describe, task);
  const categoryOf = (task: string) => frozen.get(task)?.category ?? describe.tasks[task]?.category ?? "";
  const band = (task: string) => {
    const column = frozen.get(task);
    return column ? { label: column.band, order: column.order } : describe.categories[categoryOf(task)];
  };
  const tasks = [...grid.tasks].sort((a, b) => (band(a)?.order ?? 99) - (band(b)?.order ?? 99));
  const averages = describe.averages
    .map((a) => ({ ...a, tasks: tasks.filter((t) => a.categories.includes(categoryOf(t))) }))
    .filter((a) => a.tasks.length >= 2);
  const taskLimits = new Map(tasks.map((t) => [t, summarizeLimits(
    grid.blocks.flatMap((b) => b.rows.map((r) => r.cells[t]?.limit)).filter((l): l is number | null => l !== undefined).map((l) => (l === -1 ? null : l)))]));
  const columns: ExportColumn[] = [
    ...tasks.map((t) => ({
      label: t, band: band(t)?.label ?? "", higherIsBetter: metricDef(describe, primary(t) ?? "", t).higher_is_better,
      limit: taskLimits.get(t),
    })),
    ...averages.map((a) => ({
      label: a.label, band: `avg:${a.label}`, higherIsBetter: metricDef(describe, a.metric).higher_is_better,
      limit: summarizeLimits(a.tasks.flatMap((t) => taskLimits.get(t)?.limits ?? [])),
    })),
  ];
  const sections = grid.blocks.map((block) => {
    const toRow = (row: TableView["blocks"][number]["rows"][number]): ExportRow => {
      const value = (t: string) => {
        const cell = row.cells[t];
        const raw = cell?.status === "done" ? cell.metrics?.[primary(t) ?? ""] : undefined;
        return raw === undefined ? null : Number(raw);
      };
      const values = tasks.map(value);
      const cells = tasks.map((t, i) => {
        if (values[i] === null) return "–";
        const limit = row.cells[t]?.limit;
        return formatCellValue(describe, primary(t) ?? "", row.cells[t], t) + (columns[i].limit?.mixed && limit != null && limit !== -1 ? "†" : "");
      });
      averages.forEach((a) => {
        const parts = a.tasks.map(value);
        const v = parts.some((x) => x === null) ? null : (parts as number[]).reduce((x, y) => x + y, 0) / parts.length;
        values.push(v);
        cells.push(v === null ? "–" : formatMetric(describe, a.metric, v));
      });
      const ours = !(describe.methods[row.method.name]?.builtin ?? true);
      return { method: row.label, ours, values, cells, marks: columns.map(() => null), limitKey: "" };
    };
    const baselineRow = block.rows.find((r) => r.baseline);
    const rows = block.rows.filter((r) => !r.baseline).map(toRow);
    rank(rows, columns);
    const dtype = String((grid.definition.shared.model as { dtype?: string } | undefined)?.dtype ?? describe.models[block.model]?.dtype ?? "");
    const baselines = baselineRow ? [{ row: { ...toRow(baselineRow), method: dtype ? precisionLabel(dtype) : baselineRow.label }, group: "–" }] : [];
    return { model: block.model, baselines, groups: rows.length ? [{ label: "", rows }] : [] };
  });
  return withFootnote({ groupHeader: "", columns, sections });
}
