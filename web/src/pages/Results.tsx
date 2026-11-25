import { FlaskConical, FolderClosed, FolderOpen, Plus, RefreshCw, Users } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { api, type Project, type Row, type Scope } from "../api";
import { useApp } from "../components/common";
import { ResultsSheet } from "../components/ResultsSheet";
import { useT } from "../i18n";
import { useLive } from "../live";
import { ReportView } from "../report/ReportView";
import { Button, Segmented } from "../ui/controls";
import { Empty, PageHeader } from "../ui/display";
import { useToast } from "../ui/overlay";
import { ExperimentTables } from "../project/ExperimentTables";
import { AddResultsDialog, ProjectMembers, ProjectSettings } from "./Projects";

const UNFILED = "__unfiled__";

export function Results({ scope }: { scope: Scope }) {
  const { describe, reloadProjects } = useApp();
  const t = useT();
  const toast = useToast();
  const projectName = scope.startsWith("project:") ? scope.slice("project:".length) : null;
  const [project, setProject] = useState<Project | null>(null);
  const [adding, setAdding] = useState(false);
  const sectionKey = `effeval.project.section.${projectName}`;
  const [section, setSection] = useState(() => localStorage.getItem(sectionKey) ?? "tables");
  const showTables = projectName !== null && section === "tables";
  const [rows, setRows] = useState<Row[] | null>(null);
  const [view, setView] = useState(() => localStorage.getItem("effeval.results.view") ?? "report");
  const load = useCallback(async () => {
    try {
      if (projectName) setProject(await api.project(projectName));
      setRows(await api.records(scope));
    } catch (e) {
      toast(t("Could not load results"), { tone: "error", description: (e as Error).message });
    }
  }, [scope, projectName, toast, t]);
  useLive(["records", "projects"], load);
  useEffect(() => {
    setRows(null);
    load();
  }, [load]);
  const byProject = scope === "mine";
  const groupKey = `effeval.results.group.${scope}`;
  const [group, setGroup] = useState(() => localStorage.getItem(groupKey) ?? "");
  const groups = useMemo(() => {
    const counts = new Map<string, number>();
    if (byProject) (rows ?? []).forEach((r) => (r.projects.length ? r.projects : [UNFILED]).forEach((p) => counts.set(p, (counts.get(p) ?? 0) + 1)));
    return [...counts].sort(([a], [b]) => (a === UNFILED ? 1 : b === UNFILED ? -1 : a.localeCompare(b)));
  }, [rows, byProject]);
  const activeGroup = groups.some(([g]) => g === group) ? group : groups[0]?.[0] ?? "";
  const chooseGroup = (g: string) => {
    setGroup(g);
    localStorage.setItem(groupKey, g);
  };
  const groupRows = useMemo(() => !byProject ? rows ?? [] : (rows ?? []).filter((r) =>
    activeGroup === UNFILED ? r.projects.length === 0 : r.projects.includes(activeGroup)), [rows, byProject, activeGroup]);
  const fieldKey = `effeval.results.field.${scope}`;
  const [field, setField] = useState(() => localStorage.getItem(fieldKey) ?? "");
  const fields = useMemo(() => {
    const counts = new Map(Object.keys(describe.fields).map((f) => [f, 0]));
    groupRows.forEach((r) => counts.set(r.field, (counts.get(r.field) ?? 0) + 1));
    return [...counts].filter(([, n]) => !byProject || n > 0).sort(([a], [b]) => a.localeCompare(b));
  }, [groupRows, describe, byProject]);
  const active = fields.some(([f]) => f === field) ? field : (fields.find(([, n]) => n > 0) ?? fields[0])?.[0] ?? "";
  const chooseField = (f: string) => {
    setField(f);
    localStorage.setItem(fieldKey, f);
  };
  const fieldRows = useMemo(() => groupRows.filter((r) => r.field === active), [groupRows, active]);
  const fieldLabel = (f: string) => f.charAt(0).toUpperCase() + f.slice(1);
  const viewToggle = (
    <div className="field-tabs-side">
      <Segmented value={view} onChange={(v) => { setView(v); localStorage.setItem("effeval.results.view", v); }}
        options={[{ value: "report", label: t("Report") }, { value: "records", label: t("Records") }]} />
    </div>
  );

  return (
    <div className={`page ${view === "records" && !showTables ? "fill" : ""}`}>
      <PageHeader
        title={projectName ?? (scope === "mine" ? t("My evaluations") : t("Lab pool"))}
        actions={<>
          {project && <ProjectMembers project={project} onChanged={() => { load(); reloadProjects(); }} />}
          {project && <ProjectSettings project={project}
            onRenamed={async (next) => { window.location.hash = `#/project/${encodeURIComponent(next)}`; await reloadProjects(); }}
            onDeleted={async () => { await reloadProjects(); window.location.hash = "#/mine"; }} />}
          {project && <Button icon={<Plus size={14} />} onClick={() => setAdding(true)}>{t("Add results")}</Button>}
          {projectName && <Segmented value={section} onChange={(v) => { setSection(v); localStorage.setItem(sectionKey, v); }}
            options={[{ value: "tables", label: t("Experiment tables") }, { value: "results", label: t("Results") }]} />}
          <Button variant="ghost" icon={<RefreshCw size={14} />} onClick={load} aria-label={t("Refresh")} />
        </>}
      />
      {showTables ? <ExperimentTables project={projectName} onResultsChanged={load} /> : <>
      {byProject ? (
        <nav className="field-tabs" role="tablist">
          {groups.map(([g, n]) => (
            <button key={g} role="tab" aria-selected={g === activeGroup} className={`field-tab ${g === activeGroup ? "active" : ""}`} onClick={() => chooseGroup(g)}>
              {g === UNFILED ? t("Not in a project") : <><FolderClosed size={14} />{g}</>}<span className="count">{n}</span>
            </button>
          ))}
          <span className="spacer" />
          {fields.length > 1 && <div className="field-tabs-side"><Segmented value={active} onChange={chooseField} options={fields.map(([f]) => ({ value: f, label: fieldLabel(f) }))} /></div>}
          {activeGroup && activeGroup !== UNFILED && <div className="field-tabs-side"><Button size="sm" icon={<FolderOpen size={13} />} onClick={() => (window.location.hash = `#/project/${activeGroup}`)}>{t("Open project")}</Button></div>}
          {viewToggle}
        </nav>
      ) : (
        <nav className="field-tabs" role="tablist">
          {fields.map(([f, n]) => (
            <button key={f} role="tab" aria-selected={f === active} className={`field-tab ${f === active ? "active" : ""}`} onClick={() => chooseField(f)}>
              {fieldLabel(f)}<span className="count">{rows === null ? "" : n}</span>
            </button>
          ))}
          <span className="spacer" />
          {viewToggle}
        </nav>
      )}
      {rows === null ? (
        <div className="card stack-sm" style={{ padding: 16 }}>
          {Array.from({ length: 6 }, (_, i) => <div key={i} className="skeleton" style={{ height: 22 }} />)}
        </div>
      ) : fieldRows.length === 0 ? (
        <div className="card">
          {scope === "mine"
            ? <Empty icon={<FlaskConical size={20} />} title={t("No evaluations yet")} action={<a className="btn btn-primary" href="#/compose">{t("New evaluation")}</a>} />
            : projectName
              ? <Empty icon={<FolderOpen size={20} />} title={t("No results in this project yet")} action={<Button variant="primary" icon={<Plus size={14} />} onClick={() => setAdding(true)}>{t("Add results")}</Button>} />
              : <Empty icon={<Users size={20} />} title={t("Nothing published yet")} />}
        </div>
      ) : view === "report" ? (
        <ReportView key={active} rows={fieldRows} project={projectName} showOwners={scope !== "mine"} inPool={scope === "pool"} onChanged={load} />
      ) : (
        <ResultsSheet key={active} rows={fieldRows} scope={scope} field={active} onChanged={load} />
      )}
      </>}
      {projectName && <AddResultsDialog project={projectName} linked={new Set((rows ?? []).map((r) => r.record_id))}
        open={adding} onOpenChange={setAdding} onAdded={() => { load(); reloadProjects(); }} />}
    </div>
  );
}
