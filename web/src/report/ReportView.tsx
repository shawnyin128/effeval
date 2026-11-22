import * as yaml from "js-yaml";
import { Columns3, Copy, FileText, FolderMinus, Search, Trash2, TriangleAlert, X } from "lucide-react";
import { Popover } from "radix-ui";
import { useEffect, useMemo, useState } from "react";
import { api, type Row } from "../api";
import { ProblemMark, Uploader, useApp } from "../components/common";
import { PublishSelected, RecordPanel } from "../components/RecordPanel";
import { useT } from "../i18n";
import { Button, Checkbox, Switch } from "../ui/controls";
import { Badge } from "../ui/display";
import { Dialog, useToast } from "../ui/overlay";
import { ConfigView } from "./ConfigView";
import type { Describe } from "../api";
import { copyTable, copyText } from "../ui/clipboard";
import { configRowTitle, exportTable, flatRows, toLatex } from "./export";
import { CopyMenu } from "../ui/CopyMenu";
import { formatMetric, metricHeader, secondaryColumns } from "./metrics";
import { CellLimit, limitText, specLimit } from "./limits";
import { SAMPLES_FIELD, averageColumns, averageOf, buildBlocks, defaultFields, fieldValue, samplesDetail, type Block, type ConfigField, type ConfigRow, type Spec } from "./model";

const SUMMARY_PATHS: [string, (v: unknown, t: (key: string) => string) => string][] = [
  ["model.dtype", (v) => String(v)],
  ["method.preset", (v, t) => (v ? `${t("preset")} ${v}` : "")],
  ["determinism.seed", (v, t) => `${t("seed")} ${v}`],
  ["generation.batch_size", (v, t) => `${t("batch")} ${v}`],
];

function sharedSummary(shared: Spec, t: (key: string) => string): string {
  return SUMMARY_PATHS.map(([path, fmt]) => {
    const value = path.split(".").reduce<any>((node, k) => node?.[k], shared);
    return value === undefined ? "" : fmt(value, t);
  }).filter(Boolean).join(" · ");
}

/** A config column's value as shown; the Samples column's "All" / "Varies by task" are translated. */
function shownValue(row: ConfigRow, field: ConfigField, t: (key: string) => string): string {
  const value = fieldValue(row, field);
  return field.id === SAMPLES_FIELD.id && Number.isNaN(Number(value.replace("%", ""))) ? t(value) : value;
}

function presetSources(describe: ReturnType<typeof useApp>["describe"], config: Spec): Record<string, string> {
  const method = config.method?.name;
  const name = config.method?.preset;
  const preset = method && name ? describe.presets[method]?.[name] : undefined;
  return Object.fromEntries(Object.entries(preset?.values ?? {}).map(([path, v]) => [path, v.source]));
}

function storedFields(block: Block): string[] {
  return JSON.parse(localStorage.getItem(`effeval.report.cols.${block.id}`) ?? "null") ?? defaultFields(block);
}

function storedHiddenMetrics(describe: Describe, block: Block): string[] {
  return JSON.parse(localStorage.getItem(`effeval.report.hidden-metrics.v2.${block.id}`) ?? "null") ?? secondaryColumns(describe, block.metrics);
}

/** A block as currently shown: hidden metric columns removed, plus the config columns chosen for it. */
function shown(block: Block, selected: string[], hidden: string[]): { block: Block; fields: ConfigField[] } {
  const metrics = block.metrics.filter((m) => !hidden.includes(m.id));
  const metricGroups = block.metricGroups.map((g) => ({ ...g, columns: g.columns.filter((m) => !hidden.includes(m.id)) })).filter((g) => g.columns.length);
  return { block: { ...block, metrics, metricGroups }, fields: block.fields.filter((f) => selected.includes(f.id)) };
}

async function copyPaper(describe: Describe, parts: { block: Block; fields: ConfigField[] }[], format: "markdown" | "latex") {
  const table = exportTable(describe, parts);
  if (format === "latex") return copyText(toLatex(table));
  const { header, rows, footnote } = flatRows(table);
  return copyTable(header, rows.map((r) => ({ cells: r.cells, marks: r.marks, highlight: r.ours })), footnote);
}

export function ReportView({ rows, project, showOwners, onChanged }: { rows: Row[]; project: string | null; showOwners: boolean; onChanged?: () => void }) {
  const { describe } = useApp();
  const t = useT();
  const toast = useToast();
  const [query, setQuery] = useState("");
  const [colored, setColored] = useState(() => localStorage.getItem("effeval.report.colored") !== "0");
  const [recordId, setRecordId] = useState<string | null>(null);
  const record = rows.find((r) => r.record_id === recordId) ?? null;
  useEffect(() => {
    if (!recordId) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setRecordId(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [recordId]);
  const blocks = useMemo(() => buildBlocks(rows, describe), [rows, describe]);
  const visible = blocks.filter((b) => !query || `${b.model} ${b.field}`.includes(query.trim().toLowerCase()));

  return (
    <div className="docked">
    <div className="report">
      <div className="report-toolbar">
        <div className="search">
          <Search size={14} />
          <input className="input" placeholder={t("Filter blocks")} value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
        <div className="report-jump">
          {visible.map((b) => (
            <button key={b.id} className="jump" onClick={() => document.getElementById(`block-${b.id}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}>
              <span className="mono">{b.model}</span><span className="faint">· {b.field}</span><span className="count">{b.rows.length}</span>
            </button>
          ))}
        </div>
        <span className="spacer" />
        {visible.length > 1 && (
          <CopyMenu label={t("Copy all")}
            onMarkdown={async () => { await copyPaper(describe, visible.map((b) => shown(b, storedFields(b), storedHiddenMetrics(describe, b))), "markdown"); toast(t("Copied as Markdown")); }}
            onLatex={async () => { await copyPaper(describe, visible.map((b) => shown(b, storedFields(b), storedHiddenMetrics(describe, b))), "latex"); toast(t("Copied LaTeX")); }} />
        )}
        <label className="row small clickable nowrap-row">
          <Switch checked={colored} onChange={(v) => { setColored(v); localStorage.setItem("effeval.report.colored", v ? "1" : "0"); }} label={t("Compare to baseline")} />
          {t("Compare to baseline")}
        </label>
      </div>
      {visible.map((b) => <ReportBlock key={b.id} block={b} project={project} colored={colored} showOwners={showOwners} onOpenRecord={(r) => setRecordId(r.record_id)} onChanged={onChanged} />)}
    </div>
      {record && (
        <div className="dock">
          <RecordPanel row={record} project={project} onClose={() => setRecordId(null)} onChanged={onChanged} />
        </div>
      )}
    </div>
  );
}

function ReportBlock({ block: full, project, colored, showOwners, onOpenRecord, onChanged }: { block: Block; project: string | null; colored: boolean; showOwners: boolean; onOpenRecord: (row: Row) => void; onChanged?: () => void }) {
  const { describe } = useApp();
  const t = useT();
  const toast = useToast();
  const storageKey = `effeval.report.cols.${full.id}`;
  const [selected, setSelected] = useState<string[]>(() => storedFields(full));
  const metricsKey = `effeval.report.hidden-metrics.v2.${full.id}`;
  const [hiddenMetrics, setHiddenMetrics] = useState<string[]>(() => storedHiddenMetrics(describe, full));
  const block = useMemo(() => shown(full, [], hiddenMetrics).block, [full, hiddenMetrics]);
  const chooseMetric = (id: string, on: boolean) => {
    const next = on ? hiddenMetrics.filter((m) => m !== id) : [...hiddenMetrics, id];
    setHiddenMetrics(next);
    localStorage.setItem(metricsKey, JSON.stringify(next));
  };
  const [configRow, setConfigRow] = useState<ConfigRow | null>(null);
  const [runs, setRuns] = useState<{ metric: string; rows: Row[] } | null>(null);
  const cellTitle = (cell: { value: string; row: Row; runs: number }) => {
    const lines = [cell.value + (cell.row.owner ? ` · ${cell.row.owner}` : "")];
    if (cell.runs > 1) lines.push(t("{n} runs, newest shown", { n: cell.runs }));
    return lines.join("\n");
  };
  const [sharedOpen, setSharedOpen] = useState(false);
  const fields = block.fields.filter((f) => selected.includes(f.id));
  const averages = averageColumns(describe, full.metrics);
  const columnTitle = (m: Block["metrics"][number]) => `${m.task} · ${metricHeader(describe, m.metric)}`;
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [removal, setRemoval] = useState<{ what: string; rows: Row[]; baseline: boolean } | null>(null);
  const recordsOf = (rows: ConfigRow[], task?: string) => {
    const unique = new Map<string, Row>();
    rows.forEach((r) => r.cells.forEach((c) => c.all.forEach((x) => (!task || (x.record.spec as Spec).task?.name === task) && unique.set(x.record_id, x)))); 
    return [...unique.values()];
  };
  const rowLabel = (row: ConfigRow) => [row.isBaseline ? t("baseline") : configRowTitle(describe, row), ...(row.isBaseline ? [] : fields.filter((f) => f.id !== SAMPLES_FIELD.id).map((f) => fieldValue(row, f)).filter((v) => v !== "—")),
    ...(row.limit.limited ? [`${t("Samples")} ${shownValue(row, SAMPLES_FIELD, t)}`] : [])].join(" · ");
  const askRemoveRows = (rows: ConfigRow[]) => setRemoval({ what: rows.map(rowLabel).join(", "), rows: recordsOf(rows), baseline: rows.some((r) => r.isBaseline) });
  const askRemoveColumn = (task: string) => setRemoval({ what: task, rows: recordsOf(full.rows, task), baseline: false });
  const hasBaselineProblem = block.rows.some((r) => [...r.cells.values()].some((c) => c.row.problems.includes("reference_trashed")));
  const hasOtherProblems = block.rows.some((r) => [...r.cells.values()].some((c) => c.row.problems.some((p) => p !== "reference_trashed")));
  const sameLimit = (a: Row, b: Row) => specLimit(a.record.spec as Spec) === specLimit(b.record.spec as Spec);
  const baselineFor = (row: ConfigRow) => (row.isBaseline ? undefined : block.rows.find((b) => b.isBaseline && b.limitKey === row.limitKey));
  const uncompared = block.rows.filter((r) => !r.isBaseline && !baselineFor(r)).length;

  const choose = (id: string, on: boolean) => {
    const next = on ? [...selected, id] : selected.filter((s) => s !== id);
    setSelected(next);
    localStorage.setItem(storageKey, JSON.stringify(next));
  };
  const tone = (row: ConfigRow, metricId: string, higher: boolean | null): string => {
    const baseline = baselineFor(row);
    if (!colored || !baseline || higher === null) return "";
    const a = row.cells.get(metricId)?.value;
    const b = baseline.cells.get(metricId)?.value;
    if (a === undefined || b === undefined || Number(a) === Number(b)) return "";
    if (!sameLimit(row.cells.get(metricId)!.row, baseline.cells.get(metricId)!.row)) return "";
    return (Number(a) > Number(b)) === higher ? "better" : "worse";
  };
  const changedVsBaseline = (row: ConfigRow) => {
    const baseline = baselineFor(row);
    return new Set(baseline ? [...row.flat.keys()].filter((p) => row.flat.get(p) !== baseline.flat.get(p)) : []);
  };

  const copy = async (format: "markdown" | "latex") => {
    await copyPaper(describe, [{ block, fields }], format);
    toast(format === "latex" ? t("Copied LaTeX") : t("Copied as Markdown"));
  };

  return (
    <section className="report-block card" id={`block-${block.id}`}>
      <header className="report-title">
        <span className="mono strong">{block.model}</span>
        <Badge>{block.field}</Badge>
        <span className="faint small report-summary" title={sharedSummary(block.shared, t)}>{sharedSummary(block.shared, t)}</span>
        {colored && uncompared > 0 && (
          <span className="note-tag" title={block.rows.some((r) => r.isBaseline)
            ? t("Rows without a baseline row of the same sample count are not coloured.") : t("Rows are not coloured because this block has no baseline row.")}>{t("No baseline row")}</span>
        )}
        <span className="spacer" />
        {picked.size > 0 && (
          <>
            <span className="small"><b>{t("{n} rows selected", { n: picked.size })}</b></span>
            <PublishSelected rows={recordsOf(block.rows.filter((r) => picked.has(r.key)))} onDone={() => { setPicked(new Set()); onChanged?.(); }} />
            <Button size="sm" variant="danger" icon={<Trash2 size={13} />} onClick={() => askRemoveRows(block.rows.filter((r) => picked.has(r.key)))}>{t("Remove…")}</Button>
            <button className="icon-button" aria-label={t("Clear")} onClick={() => setPicked(new Set())}><X size={13} /></button>
          </>
        )}
        <Popover.Root>
          <Popover.Trigger asChild><Button size="sm" variant="ghost" icon={<Columns3 size={13} />}>{t("Columns")}</Button></Popover.Trigger>
          <Popover.Portal>
            <Popover.Content className="popover-content" align="end" sideOffset={4} style={{ width: 340, maxHeight: 420, overflow: "auto", padding: 6 }}>
              {block.fields.some((f) => block.varying.has(f.id)) && <div className="menu-label">{t("Varies in this block")}</div>}
              {block.fields.filter((f) => block.varying.has(f.id)).map((f) => (
                <label key={f.id} className="pick">
                  <Checkbox checked={selected.includes(f.id)} onChange={(v) => choose(f.id, v)} />
                  <span>{f.translate ? t(f.label) : f.label}</span><span className="faint tiny mono path">{f.virtual && f.id.startsWith("quant.") ? t("from rules") : f.path}</span>
                </label>
              ))}
              {block.fields.some((f) => block.varying.has(f.id)) && block.fields.some((f) => !block.varying.has(f.id)) && <div className="menu-sep" />}
              {block.fields.some((f) => !block.varying.has(f.id)) && <div className="menu-label">{t("Same for every row")}</div>}
              {block.fields.filter((f) => !block.varying.has(f.id)).map((f) => (
                <label key={f.id} className="pick muted">
                  <Checkbox checked={selected.includes(f.id)} onChange={(v) => choose(f.id, v)} />
                  <span>{f.translate ? t(f.label) : f.label}</span><span className="faint tiny mono path">{f.virtual && f.id.startsWith("quant.") ? t("from rules") : f.path}</span>
                </label>
              ))}
              <div className="menu-sep" />
              <div className="menu-label">{t("Metrics")}</div>
              {full.metrics.map((m) => (
                <label key={m.id} className="pick">
                  <Checkbox checked={!hiddenMetrics.includes(m.id)} onChange={(v) => chooseMetric(m.id, v)} />
                  <span className="mono">{m.task}</span><span className="faint tiny path">{metricHeader(describe, m.metric)}</span>
                </label>
              ))}
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
        <Button size="sm" variant="ghost" icon={<FileText size={13} />} onClick={() => setSharedOpen(true)}>{t("Shared config")}</Button>
        <CopyMenu size="sm" onMarkdown={() => copy("markdown")} onLatex={() => copy("latex")} />
      </header>
      {hasBaselineProblem && <div className="report-note warning"><TriangleAlert size={13} />{t("The baseline these results were compared against is in the recycle bin, so comparisons to it are unavailable.")}</div>}
      <div className="report-scroll">
        <table className="report-table">
          <thead>
            <tr>
              <th rowSpan={2} className="z-config pick-col">
                <Checkbox checked={picked.size === 0 ? false : picked.size === block.rows.length ? true : "indeterminate"} label={t("Select all rows")}
                  onChange={(v) => setPicked(v ? new Set(block.rows.map((r) => r.key)) : new Set())} />
              </th>
              <th rowSpan={2} className="z-config">{t("Method")}</th>
              {fields.map((f) => <th key={f.id} rowSpan={2} className="z-config" title={f.path}>{f.id === SAMPLES_FIELD.id || f.translate ? t(f.label) : f.label}</th>)}
              <th rowSpan={2} className="z-config">{t("Extra")}</th>
              {showOwners && <th rowSpan={2} className="z-config">{t("Uploaded by")}</th>}
              {block.metricGroups.map((g) => (
                <th key={g.label} colSpan={g.columns.length} className="z-result band">{t(g.label)}</th>
              ))}
              {averages.length > 0 && <th colSpan={averages.length} className="z-result band">{t("Average")}</th>}
            </tr>
            <tr>
              {block.metrics.map((m) => (
                <th key={m.id} className="z-result" title={[`${m.task} · ${metricHeader(describe, m.metric)}`, ...m.mixed].join("\n")}>
                  <span className="th-title">
                    {m.mixed.length > 0 && <TriangleAlert size={12} className="mixed-icon" />}{columnTitle(m)}
                    <button className="col-remove" aria-label={t("Remove this task's results")} title={t("Remove this task's results")} onClick={() => askRemoveColumn(m.task)}><Trash2 size={11} /></button>
                  </span>
                </th>
              ))}
              {averages.map((a) => <th key={a.label} className="z-result derived" title={a.columns.map((c) => c.task).join(" + ")}>{t(a.label)}</th>)}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row) => {
              const extra = block.fields.filter((f) => block.varying.has(f.id) && !selected.includes(f.id)).length;
              return (
                <tr key={row.key} className={row.isBaseline ? "baseline" : ""}>
                  <td className="z-config pick-col"><Checkbox checked={picked.has(row.key)} label={t("Select row")} onChange={(v) => setPicked((s) => { const next = new Set(s); if (v) next.add(row.key); else next.delete(row.key); return next; })} /></td>
                  <td className="z-config mono">{configRowTitle(describe, row)}{row.isBaseline && <span className="base-tag">{t("baseline")}</span>}</td>
                  {fields.map((f) => <td key={f.id} className="z-config" title={f.id === SAMPLES_FIELD.id && samplesDetail(row) ? samplesDetail(row) : fieldValue(row, f)}>{shownValue(row, f, t)}</td>)}
                  <td className="z-config">
                    <button className="extra" onClick={() => setConfigRow(row)}>
                      <FileText size={12} />{extra > 0 ? t("{n} more", { n: extra }) : t("view")}
                    </button>
                  </td>
                  {showOwners && (
                    <td className="z-config">
                      <span className="owner-stack">
                        {[...new Map([...row.cells.values()].map((c) => [c.row.owner, c.row])).values()].map((r) => <Uploader key={r.owner} row={r} compact />)}
                      </span>
                    </td>
                  )}
                  {block.metrics.map((m) => {
                    const cell = row.cells.get(m.id);
                    return (
                      <td key={m.id} className={`z-result num ${cell ? "link" : "blank"} ${cell && cell.row.state !== "active" ? "stale" : tone(row, m.id, m.higherIsBetter)} ${cell && ownProblems(cell.row).length ? "invalid" : ""}`}
                        title={cell ? cellTitle(cell) : undefined}
                        onClick={() => cell && onOpenRecord(cell.row)}>
                        {cell && <ProblemMark problems={ownProblems(cell.row)} />}
                        {cell ? formatMetric(describe, m.metric, cell.value) : "—"}
                        {cell && row.limit.mixed && <CellLimit limit={specLimit(cell.row.record.spec as Spec)} t={t} />}
                        {cell && cell.runs > 1 && (
                          <button className="runs-tag" onClick={(e) => { e.stopPropagation(); setRuns({ metric: m.metric, rows: cell.all }); }}>×{cell.runs}</button>
                        )}
                      </td>
                    );
                  })}
                  {averages.map((a) => {
                    const value = averageOf(a, row);
                    return <td key={a.label} className="z-result num derived" title={value === null ? undefined : String(value)}>{value === null ? "—" : formatMetric(describe, a.metric, value)}</td>;
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {hasOtherProblems && <div className="report-note"><span className="problem-mark">!</span>{t("A result marked ! is invalid (in the recycle bin, redo pending or retracted); hover it for the reason.")}</div>}
      <RemoveDialog removal={removal} project={project}
        onClose={() => setRemoval(null)} onDone={() => { setRemoval(null); setPicked(new Set()); onChanged?.(); }} />
      <Dialog open={!!configRow} onOpenChange={(o) => !o && setConfigRow(null)} wide
        title={configRow ? configRowTitle(describe, configRow) : undefined}
        footer={configRow && <Button icon={<Copy size={13} />} onClick={async () => { await navigator.clipboard.writeText(yaml.dump(configRow.config)); toast(t("Copied YAML")); }}>{t("Copy YAML")}</Button>}>
        {configRow && (
          <div className="stack">
            <ConfigView config={configRow.config} sources={presetSources(describe, configRow.config)} changed={changedVsBaseline(configRow)} />
            <div className="row gap-2">
              {configRow.tasks.map((task) => <Badge key={task.task} mono>{task.task} · {task.mode} · {task.evaluator}</Badge>)}
            </div>
          </div>
        )}
      </Dialog>
      <Dialog open={!!runs} onOpenChange={(o) => !o && setRuns(null)} title={t("Repeated runs")}>
        {runs && <RunsList describe={describe} metric={runs.metric} rows={runs.rows} onOpen={(row) => { setRuns(null); onOpenRecord(row); }} />}
      </Dialog>
      <Dialog open={sharedOpen} onOpenChange={setSharedOpen} wide title={t("Shared config")}
        footer={<Button icon={<Copy size={13} />} onClick={async () => { await navigator.clipboard.writeText(yaml.dump(block.shared)); toast(t("Copied YAML")); }}>{t("Copy YAML")}</Button>}>
        <ConfigView config={block.shared} sources={presetSources(describe, block.rows.find((r) => !r.isBaseline)?.config ?? {})} />
      </Dialog>
    </section>
  );
}

/** Problems worth marking on a single result; a trashed baseline is reported once for the whole block instead. */
function ownProblems(row: Row) {
  return row.problems.filter((p) => p !== "reference_trashed");
}

/** Confirm removing whole rows or a column of results: from the project (kept in the library) or to the recycle bin. */
function RemoveDialog({ removal, project, onClose, onDone }: {
  removal: { what: string; rows: Row[]; baseline: boolean } | null;
  project: string | null;
  onClose: () => void;
  onDone: () => void;
}) {
  const { me } = useApp();
  const t = useT();
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  if (!removal) return null;
  const mine = removal.rows.filter((r) => r.owner === me.username || me.role === "admin");
  const run = async (fn: () => Promise<unknown>, done: string) => {
    setBusy(true);
    try {
      await fn();
      toast(done);
      onDone();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };
  const n = removal.rows.length;
  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} title={t("Remove {n} results", { n })}
      footer={<>
        <Button onClick={onClose}>{t("Cancel")}</Button>
        {project && <Button disabled={busy} icon={<FolderMinus size={13} />}
          onClick={() => run(() => Promise.all(removal.rows.map((r) => api.unlink(project, r.record_id))), t("Removed {n} results from the project", { n }))}>{t("Remove from project")}</Button>}
        <Button variant="danger" disabled={busy || mine.length === 0} icon={<Trash2 size={13} />}
          onClick={() => run(() => Promise.all(mine.map((r) => api.trash(r.record_id))), t("Moved {n} results to the recycle bin", { n: mine.length }))}>{t("Move to recycle bin")}</Button>
      </>}>
      <div className="stack-sm">
        <div><b>{removal.what}</b> · {t("{n} results", { n })}</div>
        {project && <div className="small muted">{t("Remove from project: the results stay in your library and in other projects; add them back with Add results.")}</div>}
        <div className="small muted">{t("Move to recycle bin: the results disappear everywhere, including other projects, until restored from the recycle bin.")}</div>
        {mine.length < n && <div className="small" style={{ color: "var(--orange)" }}>{t("Only {n} of them are yours and can go to the recycle bin.", { n: mine.length })}</div>}
        {removal.baseline && <div className="small" style={{ color: "var(--orange)" }}>{t("Other rows are compared against this baseline; their comparisons will be unavailable.")}</div>}
      </div>
    </Dialog>
  );
}

function RunsList({ describe, metric, rows, onOpen }: { describe: Describe; metric: string; rows: Row[]; onOpen: (row: Row) => void }) {
  const t = useT();
  const values = rows.map((r) => Number(r.metrics[metric]));
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const sd = values.length > 1 ? Math.sqrt(values.reduce((a, v) => a + (v - mean) ** 2, 0) / (values.length - 1)) : 0;
  return (
    <div className="stack-sm">
      <table className="list">
        <thead><tr><th>{metricHeader(describe, metric)}</th><th>{t("Created")}</th><th>{t("Uploaded by")}</th><th>{t("Evaluator")}</th><th>{t("Samples")}</th><th>{t("Mode")}</th></tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.record_id} className="clickable" onClick={() => onOpen(r)}>
              <td className="mono" title={r.metrics[metric]}>{formatMetric(describe, metric, r.metrics[metric])}</td>
              <td className="small muted">{r.created_at.slice(0, 16).replace("T", " ")}</td>
              <td className="small">{r.owner}</td>
              <td className="small">{r.evaluator}</td>
              <td className="small">{specLimit(r.record.spec as Spec) === null ? t("all samples") : limitText(specLimit(r.record.spec as Spec) as number, t)}</td>
              <td className="small">{r.mode}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="small muted">{t("Mean")} {formatMetric(describe, metric, mean)} · {t("SD")} {formatMetric(describe, metric, sd)} · n={values.length}</div>
    </div>
  );
}
