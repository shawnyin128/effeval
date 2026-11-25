import { Columns3, Search, X } from "lucide-react";
import { Popover } from "radix-ui";
import { useCallback, useMemo, useRef, useState } from "react";
import { rowMethodTitle, type Describe, type Row, type Scope } from "../api";
import { limitText, specLimit } from "../report/limits";
import { formatMetric, metricHeader, primaryMetric } from "../report/metrics";
import { copyTable } from "../ui/clipboard";
import { CopyMenu } from "../ui/CopyMenu";
import { Sheet, type SheetColumn, type SheetZone } from "../sheet/Sheet";
import { Button, Checkbox, Segmented } from "../ui/controls";
import { useToast } from "../ui/overlay";
import { useT } from "../i18n";
import { ProblemMark, Uploader, useApp } from "./common";
import { PublishSelected, RecordPanel } from "./RecordPanel";

const GROUPS = ["Config", "Result", "Provenance"] as const;
const POOL_PROVENANCE = new Set(["owner", "created", "projects"]);

/** Columns hidden until chosen: method settings, provenance, and every metric that is not some task's primary one. */
function defaultHidden(columns: SheetColumn<Row>[], scope: string, rows: Row[], describe: Describe): Set<string> {
  const reported = new Map<string, Set<string>>();
  rows.forEach((r) => reported.set(r.task, new Set([...(reported.get(r.task) ?? []), ...Object.keys(r.metrics)])));
  const primary = new Set([...reported].map(([task, names]) => primaryMetric(describe, task, [...names])));
  return new Set(columns.filter((c) => c.key.startsWith("d:") || (c.key.startsWith("m:") && !primary.has(c.key.slice(2)))
    || (c.group === "Provenance" && !(scope === "pool" && POOL_PROVENANCE.has(c.key)))).map((c) => c.key));
}

function buildColumns(rows: Row[], t: (key: string) => string, describe: Describe, picking?: { picked: Set<string>; toggle: (id: string, on: boolean) => void }): SheetColumn<Row>[] {
  const metricKeys = [...new Set(rows.flatMap((r) => Object.keys(r.metrics)))].sort();
  const displayKeys = [...new Set(rows.flatMap((r) => Object.keys(r.display)))].sort();
  return [
    ...(picking ? [{
      key: "pick", header: "", group: "Model", width: 34, value: (r: Row) => (picking.picked.has(r.record_id) ? "1" : "0"),
      render: (r: Row) => (
        <span onMouseDown={(e) => e.stopPropagation()}>
          <Checkbox checked={picking.picked.has(r.record_id)} label={t("Select row")} onChange={(v) => picking.toggle(r.record_id, v)} />
        </span>
      ),
    }] : []),
    { key: "model", header: t("Model"), group: "Model", width: 170, mono: true, value: (r) => r.model },
    { key: "method", header: t("Method"), group: "Config", width: 92, value: (r) => rowMethodTitle(describe, r) },
    { key: "task", header: t("Task"), group: "Config", width: 124, value: (r) => r.task },
    { key: "mode", header: t("Mode"), group: "Config", width: 104, value: (r) => r.mode },
    { key: "samples", header: t("Samples"), group: "Config", width: 120,
      value: (r) => { const l = specLimit(r.record); return l === null ? t("all samples") : limitText(l, t); },
      render: (r) => { const l = specLimit(r.record); return l === null ? <span className="faint">{t("all samples")}</span> : <span className="limit-tag inline">{limitText(l, t)}</span>; } },
    { key: "evaluator", header: t("Evaluator"), group: "Config", width: 92, value: (r) => r.evaluator },
    { key: "dtype", header: t("dtype"), group: "Config", width: 84, value: (r) => r.dtype },
    ...displayKeys.map((k): SheetColumn<Row> => ({
      key: `d:${k}`, header: k, group: "Config", width: 180, mono: true, value: (r) => JSON.stringify(r.display[k]) ?? "",
    })),
    ...metricKeys.map((k): SheetColumn<Row> => ({
      key: `m:${k}`, header: metricHeader(describe, k), group: "Result", width: 140, numeric: true, value: (r) => r.metrics[k],
      render: (r) => <span className="clip" title={r.metrics[k]}>{formatMetric(describe, k, r.metrics[k])}</span>,
    })),
    { key: "seed", header: t("Seed"), group: "Provenance", width: 64, numeric: true, value: (r) => r.seed },
    { key: "gpu", header: t("GPU"), group: "Provenance", width: 110, value: (r) => r.gpu ?? "" },
    { key: "role", header: t("Role"), group: "Provenance", width: 90, value: (r) => r.role },
    { key: "owner", header: t("Uploaded by"), group: "Provenance", width: 120, value: (r) => r.owner ?? "", render: (r) => <Uploader row={r} /> },
    { key: "projects", header: t("Projects"), group: "Provenance", width: 140, mono: true, value: (r) => r.projects.join(", ") },
    { key: "via", header: t("Via"), group: "Provenance", width: 70, value: (r) => r.submitted_via },
    { key: "status", header: t("Status"), group: "Provenance", width: 100, value: (r) => r.annotation?.status ?? "" },
    { key: "tags", header: t("Tags"), group: "Provenance", width: 130, value: (r) => r.annotation?.tags.join(", ") ?? "" },
    { key: "visibility", header: t("Visibility"), group: "Provenance", width: 92, value: (r) => r.visibility },
    { key: "created", header: t("Created"), group: "Provenance", width: 170, value: (r) => r.created_at.replace("T", " ").replace("+00:00", " UTC") },
    { key: "record", header: t("Record"), group: "Provenance", width: 170, mono: true, value: (r) => r.record_id },
  ];
}

const ZONES: Record<string, SheetZone> = {
  Model: { label: "Model", tone: "model" },
  Config: { label: "Config", tone: "config" },
  Result: { label: "Result", tone: "result" },
  Provenance: { label: "Provenance", tone: "provenance" },
};

export function ResultsSheet({ rows, scope, field, onChanged }: {
  rows: Row[];
  scope: Scope | "preview";
  field?: string;
  onChanged?: () => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const toast = useToast();
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState("all");
  const storageKey = `effeval.records.hidden.v2.${scope}.${field ?? ""}`;
  const [stored, setStored] = useState<string[] | null>(() => JSON.parse(localStorage.getItem(storageKey) ?? "null"));
  const [density, setDensity] = useState("compact");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const selected = rows.find((r) => r.record_id === selectedId) ?? null;
  const project = scope.startsWith("project:") ? scope.slice("project:".length) : null;
  const ordered = useRef<Row[]>([]);
  const onOrderChange = useCallback((list: Row[]) => (ordered.current = list), []);

  const tabs = useMemo(() => {
    const counts = new Map<string, number>();
    rows.forEach((r) => counts.set(r.task, (counts.get(r.task) ?? 0) + 1));
    return [{ key: "all", label: t("All"), count: rows.length },
      ...[...counts].sort().map(([task, count]) => ({ key: task, label: task, count }))];
  }, [rows]);

  const visibleRows = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter((r) => (tab === "all" || r.task === tab) &&
      (!q || [r.model, r.method, rowMethodTitle(describe, r), r.task, r.mode, r.evaluator, r.owner, r.record_id, ...(r.annotation?.tags ?? [])]
        .some((v) => v?.toLowerCase().includes(q))));
  }, [rows, tab, query]);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const canPick = scope !== "preview" && scope !== "pool";
  const picking = useMemo(() => canPick ? {
    picked,
    toggle: (id: string, on: boolean) => setPicked((s) => { const next = new Set(s); if (on) next.add(id); else next.delete(id); return next; }),
  } : undefined, [canPick, picked]);
  const allColumns = useMemo(() => buildColumns(visibleRows, t, describe, picking), [visibleRows, t, describe, picking]);
  const hidden = stored ? new Set(stored) : defaultHidden(allColumns, scope, visibleRows, describe);
  const columns = allColumns.filter((c) => !hidden.has(c.key));
  const saveHidden = (next: Set<string> | null) => {
    setStored(next && [...next]);
    if (next) localStorage.setItem(storageKey, JSON.stringify([...next]));
    else localStorage.removeItem(storageKey);
  };
  const toggle = (keys: string[], on: boolean) => {
    const next = new Set(hidden);
    keys.forEach((k) => (on ? next.delete(k) : next.add(k)));
    saveHidden(next);
  };

  const copy = async () => {
    const text = (r: Row, c: SheetColumn<Row>) => {
      const raw = c.value(r);
      if (raw === null || raw === undefined) return "";
      return c.key.startsWith("m:") ? formatMetric(describe, c.key.slice(2), raw) : String(raw);
    };
    const copied = columns.filter((c) => c.key !== "pick");
    await copyTable(copied.map((c) => c.header), ordered.current.map((r) => ({ cells: copied.map((c) => text(r, c)) })));
    toast(t("Copied as Markdown"));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
      <div className="sheet-toolbar">
        <div className="search">
          <Search size={14} />
          <input className="input" placeholder={t("Search")} value={query} onChange={(e) => setQuery(e.target.value)} />
        </div>
        <span className="spacer" />
        {picked.size > 0 && (
          <>
            <span className="small"><b>{t("{n} rows selected", { n: picked.size })}</b></span>
            <Button size="sm" variant="ghost" onClick={() => setPicked(new Set(visibleRows.map((r) => r.record_id)))}>{t("Select all")}</Button>
            <PublishSelected rows={rows.filter((r) => picked.has(r.record_id))} onDone={() => { setPicked(new Set()); onChanged?.(); }} />
            <button className="icon-button" aria-label={t("Clear")} onClick={() => setPicked(new Set())}><X size={13} /></button>
          </>
        )}
        <Popover.Root>
          <Popover.Trigger asChild><Button icon={<Columns3 size={14} />}>{t("Columns")}</Button></Popover.Trigger>
          <Popover.Portal>
            <Popover.Content className="popover-content" align="end" sideOffset={4} style={{ width: 300, maxHeight: 460, overflow: "auto", padding: 6 }}>
              {GROUPS.map((g) => {
                const keys = allColumns.filter((c) => c.group === g).map((c) => c.key);
                if (!keys.length) return null;
                const shown = keys.filter((k) => !hidden.has(k)).length;
                return (
                  <div key={g} className="pick-group">
                    <label className="pick pick-head">
                      <Checkbox checked={shown === keys.length ? true : shown === 0 ? false : "indeterminate"} onChange={(v) => toggle(keys, v)} />
                      <span>{t(g)}</span><span className="faint tiny">{shown}/{keys.length}</span>
                    </label>
                    {allColumns.filter((c) => c.group === g).map((c) => (
                      <label key={c.key} className="pick pick-child">
                        <Checkbox checked={!hidden.has(c.key)} onChange={(v) => toggle([c.key], v)} />
                        <span className={c.key.includes(":") ? "mono" : ""}>{c.header}</span>
                      </label>
                    ))}
                  </div>
                );
              })}
              <div className="menu-sep" />
              <button className="menu-item" style={{ width: "100%" }} onClick={() => saveHidden(null)}>{t("Reset")}</button>
            </Popover.Content>
          </Popover.Portal>
        </Popover.Root>
        <Segmented value={density} onChange={setDensity} options={[{ value: "compact", label: t("Compact") }, { value: "comfortable", label: t("Comfortable") }]} />
        <CopyMenu onMarkdown={copy} />
      </div>
      <div style={{ display: "flex", gap: 16, flex: 1, minHeight: 0 }}>
        <Sheet
          columns={columns}
          frozen={picking ? 2 : 1}
          rows={visibleRows}
          rowKey={(r) => r.record_id}
          rowHeight={density === "compact" ? 26 : 34}
          selectedKey={selectedId}
          onSelect={(r) => setSelectedId(r.record_id)}
          rowClass={(r) => [r.state !== "active" ? "stale" : "", r.problems?.length ? "invalid" : ""].join(" ")}
          rowMark={(r) => r.problems?.length ? <ProblemMark problems={r.problems} /> : null}
          onOrderChange={onOrderChange}
          zones={Object.fromEntries(Object.entries(ZONES).map(([k, z]) => [k, { ...z, label: t(z.label) }]))}
          tabs={tabs}
          activeTab={tab}
          onTab={setTab}
          status={t("{n} rows", { n: visibleRows.length })}
        />
        {selected && scope !== "preview" && (
          <RecordPanel row={selected} project={project} inPool={scope === "pool"} onClose={() => setSelectedId(null)} onChanged={onChanged} />
        )}
      </div>
    </div>
  );
}
