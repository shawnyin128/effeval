import { FolderOpen, Pencil, Snowflake, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type Describe, type Snapshot, type TableView } from "../api";
import { formatTime, useApp } from "../components/common";
import { useT } from "../i18n";
import { exportFromGrid, flatRows, toLatex } from "../report/export";
import { CellLimit, LimitTag, summarizeLimits } from "../report/limits";
import { formatCellValue, metricHeader } from "../report/metrics";
import { copyTable, copyText } from "../ui/clipboard";
import { CopyMenu } from "../ui/CopyMenu";
import { Button, Input } from "../ui/controls";
import { Badge, PageHeader } from "../ui/display";
import { useToast } from "../ui/overlay";
import { DecodingCells, DecodingHeads, SpeedMetricSelect, comparisonLines, hasDecodingColumns, isSpeedTask, speedCellView, useSpeedMetric } from "./decodingColumns";
import { providerLabel } from "../compose/Decoding";

/** Copy an experiment table (live or frozen) as a paper table. */
export async function copyPaperGrid(describe: Describe, grid: TableView, format: "markdown" | "latex", metricFor?: (task: string) => string | undefined): Promise<void> {
  const table = exportFromGrid(describe, grid, metricFor);
  if (format === "latex") return copyText(toLatex(table));
  const { header, rows, footnote } = flatRows(table);
  return copyTable(header, rows.map((r) => ({ cells: r.cells, marks: r.marks, highlight: r.ours })), footnote);
}

/** Read-only page of a frozen experiment table; numbers and display precision are those stored at freeze time. */
export function SnapshotView({ project, snapshotId }: { project: string; snapshotId: string }) {
  const { describe, me } = useApp();
  const t = useT();
  const toast = useToast();
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [editing, setEditing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  useEffect(() => {
    api.snapshot(project, snapshotId).then(setSnap, (e) => toast(t("Action failed"), { tone: "error", description: (e as Error).message }));
  }, [project, snapshotId]);
  const speed = useSpeedMetric(describe, snap?.body ?? null, `effeval.snapshot.speedMetric.${snapshotId}`);
  if (!snap?.body) return <div className="page" />;
  const body = snap.body;
  const frozen: Describe = { ...describe, metrics: { ...describe.metrics, ...body.metrics }, averages: body.averages, categories: body.categories };
  const decoding = hasDecodingColumns(frozen, body);
  const copy = async (format: "markdown" | "latex") => {
    await copyPaperGrid(frozen, body, format, (task) => (isSpeedTask(frozen, task) ? speed.name : undefined));
    toast(format === "latex" ? t("Copied LaTeX") : t("Copied as Markdown"));
  };
  const limits = Object.fromEntries(body.tasks.map((task) => [task, summarizeLimits(
    body.blocks.flatMap((b) => b.rows.map((r) => r.cells[task]?.limit)).filter((l): l is number | null => l !== undefined)
      .map((l) => (l === -1 ? null : l)))]));
  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const rename = (title: string) => act(async () => {
    setEditing(false);
    const next = title.trim() && title.trim() !== body.definition.name ? title.trim() : "";
    await api.renameSnapshot(project, snapshotId, next);
    setSnap({ ...snap, title: next || null });
  });
  const remove = () => act(async () => {
    await api.deleteSnapshot(project, snapshotId);
    toast(t("Deleted"));
    window.location.hash = `#/project/${project}`;
  });
  const canDelete = snap.created_by === me.username || me.role === "admin";
  const title = editing
    ? <Input autoFocus defaultValue={snap.title ?? body.definition.name} style={{ fontSize: 20, height: 36, width: 420 }}
        onBlur={(e) => rename(e.target.value)}
        onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key === "Escape") setEditing(false); }} />
    : <span className="row">{snap.title ?? body.definition.name}
        <button className="icon-button" aria-label={t("Rename")} title={t("Rename")} onClick={() => setEditing(true)}><Pencil size={14} /></button></span>;
  return (
    <div className="page">
      <PageHeader title={title} actions={<>
        {canDelete && (confirming
          ? <><Button size="sm" onClick={() => setConfirming(false)}>{t("Cancel")}</Button><Button size="sm" variant="danger" onClick={remove}>{t("Delete version")}</Button></>
          : <Button variant="ghost" icon={<Trash2 size={14} />} aria-label={t("Delete version")} title={t("Delete version")} onClick={() => setConfirming(true)} />)}
        <Button icon={<FolderOpen size={14} />} onClick={() => (window.location.hash = `#/project/${project}`)}>{t("Open project")}</Button>
        <SpeedMetricSelect describe={frozen} name={speed.name} options={speed.options} onChange={speed.set} />
        <CopyMenu onMarkdown={() => copy("markdown")} onLatex={() => copy("latex")} />
      </>} />
      <div className="row small muted mb-4" style={{ marginTop: -8 }}>
        <Badge tone="accent"><Snowflake size={11} /> {t("Frozen version")}</Badge>
        {snap.title && <span>{body.definition.name}</span>}
        <span>{formatTime(snap.created_at)} · {snap.created_by}</span>
      </div>
      {body.blocks.map((block) => (
        <section key={block.model} className="card xt mb-4">
          <div className="xt-block">
            <div className="xt-model mono">{block.model}</div>
            <table className="xt-grid">
              <thead><tr><th className="xt-rowhead">{t("Method")}</th>{decoding && <DecodingHeads />}{body.tasks.map((task) => (
                <th key={task}>{task}<LimitTag summary={limits[task]} t={t} />
                  {isSpeedTask(frozen, task) && <div className="faint tiny">{metricHeader(frozen, speed.name, task)}</div>}
                </th>
              ))}</tr></thead>
              <tbody>
                {block.rows.map((row) => (
                  <tr key={row.key} className={row.baseline ? "baseline" : ""}>
                    <th className="xt-rowhead">{decoding && row.baseline ? t("Autoregressive ({provider})", { provider: providerLabel(frozen, row.runtime) }) : row.label}</th>
                    {decoding && <DecodingCells row={row} />}
                    {body.tasks.map((task) => {
                      const cell = row.cells[task];
                      if (cell && isSpeedTask(frozen, task) && cell.status === "done") {
                        const view = speedCellView(frozen, row, cell, task, speed.name, t);
                        const hover = [view.title, ...comparisonLines(frozen, row, cell, task, t)].filter(Boolean).join("\n") || undefined;
                        return <td key={task} className="xt-td"><div className="xt-cell done static" title={hover}>{view.empty ? <span className="faint">{view.text}</span> : view.text}{limits[task].mixed && <CellLimit limit={cell.limit === -1 ? null : cell.limit} t={t} />}</div></td>;
                      }
                      const name = body.columns?.find((c) => c.task === task)?.metric ?? frozen.tasks[task]?.metrics[0]?.name ?? "";
                      const raw = cell?.status === "done" ? cell.metrics?.[name] : undefined;
                      return <td key={task} className="xt-td"><div className="xt-cell done static" title={raw}>{raw === undefined ? "–" : formatCellValue(frozen, name, cell, task)}{limits[task].mixed && <CellLimit limit={cell?.limit === -1 ? null : cell?.limit} t={t} />}</div></td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
    </div>
  );
}
