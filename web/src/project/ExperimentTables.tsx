import { Flag, FlagOff, FolderPlus, Globe, Pencil, Snowflake, Plus, RotateCw, Table2, Terminal, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, methodTitle, type Cell, type CellStatus, type Rendered, type Row, type Snapshot, type TableView } from "../api";
import { copyPaperGrid } from "./SnapshotView";
import { CopyMenu } from "../ui/CopyMenu";
import { initials, useApp } from "../components/common";
import { RecordPanel } from "../components/RecordPanel";
import { useT } from "../i18n";
import { useLive } from "../live";
import { CommandDialog, warnDuplicates } from "../pages/Compose";
import type { Describe } from "../api";
import { CellLimit, LimitTag, summarizeLimits } from "../report/limits";
import { formatCellValue, metricHeader } from "../report/metrics";
import { Button } from "../ui/controls";
import { Badge, Empty } from "../ui/display";
import { useToast } from "../ui/overlay";
import { DecodingCells, DecodingHeads, SpeedMetricSelect, comparisonLines, hasDecodingColumns, isSpeedTask, speedCellView, useSpeedMetric } from "./decodingColumns";
import { TableEditor } from "./TableEditor";
import { providerLabel } from "../compose/Decoding";

const STATUS_LABEL: Record<CellStatus, string> = {
  done: "Done",
  reusable: "Reusable",
  running: "Running",
  failed: "Failed",
  pending: "Pending",
  invalid: "Cannot run",
};
const ORDER: CellStatus[] = ["done", "reusable", "running", "failed", "pending", "invalid"];

function primaryMetric(describe: Describe, task: string, cell: Cell): string {
  const name = describe.tasks[task]?.metrics[0]?.name ?? Object.keys(cell.metrics ?? {})[0];
  return name ? formatCellValue(describe, name, cell, task) : "";
}

/** The runs behind a cell: its repeats, or the cell itself for a single run. */
const runsOf = (cell: Cell): Cell[] => cell.repeats ?? [cell];

function cellTitle(cell: Cell, t: (k: string, v?: Record<string, string | number>) => string, comparisons: string[] = []): string {
  const lines = [t(STATUS_LABEL[cell.status])];
  if (cell.metrics) lines.push(Object.entries(cell.metrics).map(([k, v]) => `${k} ${v}`).join(" · "));
  lines.push(...comparisons);
  if (cell.repeats) lines.push(t("{n} of {m} runs done", { n: cell.repeats.filter((r) => r.status === "done").length, m: cell.repeats.length }));
  if (cell.owner && cell.status !== "done") lines.push(`${t("Uploaded by")} ${cell.owner}`);
  if (cell.activity) lines.push(`${cell.activity.owner} · ${t(cell.activity.status)} · ${cell.activity.at.slice(0, 16).replace("T", " ")}${cell.activity.message ? ` · ${cell.activity.message}` : ""}`);
  if (cell.claim) lines.push(`${t("Claimed by")} ${cell.claim.by}`);
  return lines.join("\n");
}

/** A project's experiment tables: every planned cell with its live status, claims and run commands. */
export function ExperimentTables({ project, onResultsChanged }: { project: string; onResultsChanged: () => void }) {
  const t = useT();
  const toast = useToast();
  const [tables, setTables] = useState<TableView[] | null>(null);
  const [editing, setEditing] = useState<TableView | "new" | null>(null);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const load = useCallback(async () => {
    try {
      const [loaded, snaps] = await Promise.all([api.tables(project), api.snapshots(project)]);
      setTables(loaded);
      setSnapshots(snaps);
    } catch (e) {
      toast(t("Could not load results"), { tone: "error", description: (e as Error).message });
    }
  }, [project, toast, t]);
  useEffect(() => {
    load();
  }, [load]);
  useLive(["projects", "records", "runs"], load);
  const changed = () => {
    load();
    onResultsChanged();
  };
  const [record, setRecord] = useState<Row | null>(null);
  const openRecord = async (recordId: string) => {
    try {
      setRecord(await api.record(recordId));
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  useEffect(() => {
    if (!record) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setRecord(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [record]);

  return (
    <div className="docked">
    <div className="stack grow">
      {tables !== null && tables.length === 0 ? (
        <div className="card">
          <Empty icon={<Table2 size={20} />} title={t("No experiment tables yet")}
            action={<Button variant="primary" icon={<Plus size={14} />} onClick={() => setEditing("new")}>{t("New table")}</Button>} />
        </div>
      ) : (
        <>
          {(tables ?? []).map((table) => (
            <TableBlock key={table.table_id} project={project} table={table} snapshots={snapshots.filter((s) => s.table_id === table.table_id)}
              onEdit={() => setEditing(table)} onChanged={changed} onOpenRecord={openRecord} />
          ))}
          {tables && <div><Button icon={<Plus size={14} />} onClick={() => setEditing("new")}>{t("New table")}</Button></div>}
        </>
      )}
      {editing !== null && (
        <TableEditor project={project} table={editing === "new" ? null : editing} onClose={() => setEditing(null)} onSaved={() => { setEditing(null); load(); }} />
      )}
    </div>
      {record && (
        <div className="dock">
          <RecordPanel row={record} project={project} onClose={() => setRecord(null)}
            onChanged={async () => { changed(); setRecord(await api.record(record.record_id).catch(() => null)); }} />
        </div>
      )}
    </div>
  );
}

function TableBlock({ project, table, snapshots, onEdit, onChanged, onOpenRecord }: {
  project: string;
  table: TableView;
  snapshots: Snapshot[];
  onEdit: () => void;
  onChanged: () => void;
  onOpenRecord: (recordId: string) => void;
}) {
  const { me, describe } = useApp();
  const t = useT();
  const toast = useToast();
  const [selected, setSelected] = useState<Map<string, Cell>>(new Map());
  const [rendered, setRendered] = useState<Rendered | null>(null);
  const [deleting, setDeleting] = useState(false);

  const live = useMemo(() => {
    const byKey = new Map<string, Cell>();
    table.blocks.forEach((b) => b.rows.forEach((r) => Object.values(r.cells).forEach((c) => c.status !== "invalid" && byKey.set(c.run_key, c))));
    return byKey;
  }, [table]);
  const chosen = [...selected.keys()].map((k) => live.get(k)).filter((c): c is Cell => !!c);
  const toggle = (cell: Cell) => setSelected((s) => {
    const next = new Map(s);
    if (next.has(cell.run_key)) next.delete(cell.run_key);
    else next.set(cell.run_key, cell);
    return next;
  });
  const act = async (fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) toast(done);
      setSelected(new Map());
      onChanged();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const command = async (cells: Cell[], force: boolean) => {
    try {
      const result = await api.tableCommand(project, table.table_id, cells.flatMap(runsOf).map((c) => c.run_key), force);
      warnDuplicates(toast, t, result.issues);
      setRendered(result);
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const open = (cell: Cell) => cell.record_id && onOpenRecord(cell.record_id);
  const limits = useMemo(() => Object.fromEntries(table.tasks.map((task) => [task, summarizeLimits(
    table.blocks.flatMap((b) => b.rows.map((r) => r.cells[task]?.limit)).filter((l): l is number | null => l !== undefined)
      .map((l) => (l === -1 ? null : l)))])), [table]);

  const speed = useSpeedMetric(describe, table, `effeval.xt.speedMetric.${table.table_id}`);
  const decoding = hasDecodingColumns(describe, table);
  const copyGrid = async (format: "markdown" | "latex") => {
    await copyPaperGrid(describe, table, format, (task) => (isSpeedTask(describe, task) ? speed.name : undefined));
    toast(format === "latex" ? t("Copied LaTeX") : t("Copied as Markdown"));
  };
  const freeze = async () => {
    try {
      const snap = await api.freezeTable(project, table.table_id);
      toast(t("Version frozen"), { description: snap.snapshot_id });
      onChanged();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };

  const runnable = chosen.filter((c) => runsOf(c).some((r) => r.status === "pending" || r.status === "failed"));
  const reusable = chosen.filter((c) => c.status === "reusable");
  const finished = chosen.filter((c) => c.status === "done" || c.status === "reusable");
  const claimable = chosen.filter((c) => !c.claim && c.status !== "done");
  const mine = chosen.filter((c) => c.claim && (c.claim.by === me.username || me.role === "admin"));
  const publishable = chosen.flatMap(runsOf).filter((c) => c.status === "done" && c.visibility === "private" && (c.owner === me.username || me.role === "admin"));

  return (
    <section className="card xt">
      <header className="xt-head">
        <span className="xt-title">{table.definition.name}</span>
        <Badge>{table.definition.field}</Badge>
        <span className="xt-counts">
          {ORDER.filter((s) => table.counts[s]).map((s) => (
            <span key={s} className={`xt-count ${s}`}><span className={`xt-dot ${s}`} />{t(STATUS_LABEL[s])} {table.counts[s]}</span>
          ))}
        </span>
        <span className="spacer" />
        <SpeedMetricSelect describe={describe} name={speed.name} options={speed.options} onChange={speed.set} />
        <CopyMenu size="sm" onMarkdown={() => copyGrid("markdown")} onLatex={() => copyGrid("latex")} />
        <Button size="sm" variant="ghost" icon={<Snowflake size={13} />} onClick={freeze}>{t("Freeze version")}</Button>
        <Button size="sm" variant="ghost" icon={<Pencil size={13} />} onClick={onEdit}>{t("Edit")}</Button>
        {deleting
          ? <><Button size="sm" onClick={() => setDeleting(false)}>{t("Cancel")}</Button>
            <Button size="sm" variant="danger" onClick={() => act(() => api.deleteTable(project, table.table_id), t("Deleted"))}>{t("Delete table")}</Button></>
          : <Button size="sm" variant="ghost" icon={<Trash2 size={13} />} aria-label={t("Delete table")} onClick={() => setDeleting(true)} />}
      </header>
      {snapshots.length > 0 && (
        <div className="xt-snapshots">
          <Snowflake size={12} />
          {snapshots.map((snap) => (
            <a key={snap.snapshot_id} className="chip" href={`#/project/${project}/snapshot/${snap.snapshot_id}`} title={snap.snapshot_id}>
              {snap.title ?? `${snap.created_at.slice(0, 16).replace("T", " ")} · ${snap.created_by}`}
            </a>
          ))}
        </div>
      )}
      {table.blocks.map((block) => (
        <div key={block.model} className="xt-block">
          <div className="xt-model mono">{block.model}</div>
          <div className="xt-scroll">
            <table className="xt-grid">
              <thead>
                <tr>
                  <th className="xt-rowhead">{t("Method")}</th>
                  {decoding && <DecodingHeads />}
                  {table.tasks.map((task) => (
                    <th key={task}>{task}<LimitTag summary={limits[task]} t={t} />
                      {isSpeedTask(describe, task) && <div className="faint tiny">{metricHeader(describe, speed.name, task)}</div>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {block.rows.map((row) => {
                  const errors = row.issues.filter((i) => i.severity === "error");
                  return (
                    <tr key={row.key} className={row.baseline ? "baseline" : ""}>
                      <th className="xt-rowhead" title={errors.map((e) => e.message).join("\n") || undefined}>
                        <span className={row.baseline ? "" : "mono"}>{decoding && row.baseline ? t("Autoregressive ({provider})", { provider: providerLabel(describe, row.runtime) }) : row.label}</span>
                        {row.baseline && <span className="base-tag">{t("recommended baseline")}</span>}
                        {!row.baseline && row.method.preset !== undefined && row.method.preset !== null && <span className="faint tiny mono"> {row.title || methodTitle(describe, row.method.name)}·{row.method.preset}</span>}
                        {errors.length > 0 && <Badge tone="red">{t("{n} conflicts", { n: errors.length })}</Badge>}
                      </th>
                      {decoding && <DecodingCells row={row} />}
                      {table.tasks.map((task) => {
                        const cell = row.cells[task];
                        if (!cell) return <td key={task} className="xt-td"><span className="xt-cell blank static">—</span></td>;
                        if (cell.status === "invalid") {
                          const reasons = (cell.issues ?? []).map((i) => i.hint ? `${i.message} (${i.hint})` : i.message).join("\n");
                          return <td key={task} className="xt-td"><span className="xt-cell invalid static" title={reasons}><span className="xt-dot invalid" /><span className="xt-label">{t("Cannot run")}</span></span></td>;
                        }
                        const isSelected = selected.has(cell.run_key);
                        const speedView = isSpeedTask(describe, task) && (cell.status === "done" || cell.status === "reusable")
                          ? speedCellView(describe, row, cell, task, speed.name, t) : null;
                        const shown = speedView ? speedView.text : primaryMetric(describe, task, cell);
                        return (
                          <td key={task} className="xt-td">
                            <button className={`xt-cell ${cell.status} ${isSelected ? "selected" : ""}`} title={cellTitle(cell, t, speedView ? comparisonLines(describe, row, cell, task, t) : [])}
                              onClick={() => (cell.status === "done" ? open(cell) : toggle(cell))}>
                              <span className={`xt-dot ${cell.status}`} />
                              {cell.status === "done" && <span className={`xt-value${speedView?.empty ? " faint" : ""}`} title={speedView?.title}>{shown}</span>}
                              {limits[task].mixed && <CellLimit limit={cell.limit === -1 ? null : cell.limit} t={t} />}
                              {cell.status === "reusable" && <><span className="xt-value faint" title={speedView?.title}>{shown}</span><span className="xt-tag">{t("Reusable")}</span></>}
                              {cell.status === "running" && <><span className="xt-label">{t("Running")}</span>{cell.activity?.owner && <span className="avatar xs">{initials(cell.activity.owner)}</span>}</>}
                              {cell.status === "failed" && <span className="xt-label">{t("Failed")}</span>}
                              {cell.status === "pending" && <span className="xt-label">{t("Pending")}</span>}
                              {cell.claim && cell.status !== "running" && <span className="xt-claim" title={`${t("Claimed by")} ${cell.claim.by}`}><Flag size={10} />{initials(cell.claim.by)}</span>}
                            </button>
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      {chosen.length > 0 && (
        <div className="xt-actions">
          <span className="small"><b>{t("{n} cells selected", { n: chosen.length })}</b></span>
          <button className="icon-button" aria-label={t("Clear")} onClick={() => setSelected(new Map())}><X size={13} /></button>
          <span className="spacer" />
          {claimable.length > 0 && <Button size="sm" icon={<Flag size={13} />} onClick={() => act(() => api.claim(project, claimable.flatMap(runsOf).map((c) => c.run_key)), t("Claimed"))}>{t("Claim")}</Button>}
          {mine.length > 0 && <Button size="sm" icon={<FlagOff size={13} />} onClick={() => act(() => Promise.all(mine.flatMap(runsOf).map((c) => api.release(project, c.run_key))), t("Released"))}>{t("Release")}</Button>}
          {reusable.length > 0 && <Button size="sm" icon={<FolderPlus size={13} />} onClick={() => act(() => api.link(project, reusable.flatMap(runsOf).filter((c) => c.status === "reusable").map((c) => c.record_id!)), t("Added {n}", { n: reusable.length }))}>{t("Add to project")}</Button>}
          {publishable.length > 0 && <Button size="sm" icon={<Globe size={13} />} onClick={() => act(() => Promise.all(publishable.map((c) => api.publish(c.record_id!))), t("Published"))}>{t("Publish")}</Button>}
          {finished.length > 0 && <Button size="sm" icon={<RotateCw size={13} />} onClick={() => command(chosen, true)}>{t("Re-run anyway")}</Button>}
          {runnable.length > 0 && <Button size="sm" variant="primary" icon={<Terminal size={13} />} onClick={() => command(chosen, false)}>{t("Copy command")}</Button>}
        </div>
      )}
      <CommandDialog rendered={rendered} onClose={() => setRendered(null)} />
    </section>
  );
}
