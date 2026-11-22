import { FolderPlus, Plus, Settings2, Trash2, UserPlus, Users, X } from "lucide-react";
import { Popover } from "radix-ui";
import { useEffect, useMemo, useState } from "react";
import { api, rowMethodTitle, type Project, type Row } from "../api";
import { initials, useApp } from "../components/common";
import { useT } from "../i18n";
import { Button, Checkbox, Field, Input, Segmented, Select } from "../ui/controls";
import { Badge } from "../ui/display";
import { Dialog, useToast } from "../ui/overlay";

const DESTINATION_KEY = "effeval.upload.to";

/** The upload destination last chosen on this browser, falling back to private when that project is gone. */
export function useDestination(): [string, (to: string) => void] {
  const { projects } = useApp();
  const [to, setTo] = useState(() => localStorage.getItem(DESTINATION_KEY) ?? "private");
  const valid = !to.startsWith("project:") || projects.some((p) => `project:${p.name}` === to);
  const choose = (next: string) => {
    setTo(next);
    localStorage.setItem(DESTINATION_KEY, next);
  };
  return [valid ? to : "private", choose];
}

/** Destination choices for uploads: private, the lab pool, or one of the user's projects. */
export function DestinationSelect({ value, onChange }: { value: string; onChange: (to: string) => void }) {
  const { projects } = useApp();
  const t = useT();
  return (
    <Select value={value} onChange={onChange} ariaLabel={t("Upload to")} options={[
      { value: "private", label: t("Private") },
      { value: "pool", label: t("Lab pool") },
      ...projects.map((p) => ({ value: `project:${p.name}`, label: `${t("Project")} · ${p.name}` })),
    ]} />
  );
}

export function NewProjectDialog({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { reloadProjects } = useApp();
  const t = useT();
  const toast = useToast();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const create = async () => {
    try {
      await api.createProject(name.trim(), description.trim());
      await reloadProjects();
      onOpenChange(false);
      setName("");
      setDescription("");
      window.location.hash = `#/project/${name.trim()}`;
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} title={t("New project")}
      icon={<div className="icon-tile accent"><FolderPlus size={18} /></div>}
      footer={<><Button onClick={() => onOpenChange(false)}>{t("Cancel")}</Button>
        <Button variant="primary" disabled={!name.trim()} onClick={create}>{t("Create")}</Button></>}>
      <div className="stack">
        <Field label={t("Name")}><Input className="mono" autoFocus value={name} placeholder="paper-a" onChange={(e) => setName(e.target.value.toLowerCase())} /></Field>
        <Field label={t("Description")}><Input value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
      </div>
    </Dialog>
  );
}

/** Member list of a project; the creator (or an admin) can add and remove lab members. */
export function ProjectMembers({ project, onChanged }: { project: Project; onChanged: () => void }) {
  const { me } = useApp();
  const t = useT();
  const toast = useToast();
  const [users, setUsers] = useState<string[]>([]);
  const [pick, setPick] = useState("");
  const manager = me.username === project.created_by || me.role === "admin";
  useEffect(() => {
    if (manager) api.users().then((list) => setUsers(list.filter((u) => !u.disabled).map((u) => u.username))).catch(() => setUsers([]));
  }, [manager]);
  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
      onChanged();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const candidates = users.filter((u) => !project.members.includes(u));
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <Button icon={<Users size={14} />}>
          <span className="avatar-stack">{project.members.slice(0, 4).map((m) => <span key={m} className="avatar xs" title={m}>{initials(m)}</span>)}</span>
          {project.members.length}
        </Button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="popover-content" align="end" sideOffset={4} style={{ width: 280, padding: 6 }}>
          <div className="menu-label">{t("Members")}</div>
          {project.members.map((m) => (
            <div key={m} className="pick" style={{ cursor: "default" }}>
              <span className="avatar xs">{initials(m)}</span><span>{m}</span>
              {m === project.created_by ? <span className="ml-auto"><Badge>{t("creator")}</Badge></span> : manager && (
                <button className="icon-button ml-auto" aria-label={t("Remove")} onClick={() => act(() => api.removeMember(project.name, m))}><X size={13} /></button>
              )}
            </div>
          ))}
          {manager && candidates.length > 0 && (
            <>
              <div className="menu-sep" />
              <div className="row gap-2 nowrap-row" style={{ padding: 6 }}>
                <div className="grow"><Select value={pick} onChange={setPick} placeholder={t("Add member")} options={candidates.map((u) => ({ value: u, label: u }))} /></div>
                <Button size="sm" variant="primary" icon={<UserPlus size={13} />} disabled={!pick} aria-label={t("Add")}
                  onClick={() => act(async () => { await api.addMember(project.name, pick); setPick(""); })} />
              </div>
            </>
          )}
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

/** Rename a project, change its description, or delete it; shown to its creator and admins only. */
export function ProjectSettings({ project, onRenamed, onDeleted }: {
  project: Project;
  onRenamed: (name: string) => void;
  onDeleted: () => void;
}) {
  const { me } = useApp();
  const t = useT();
  const toast = useToast();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description);
  const [confirming, setConfirming] = useState(false);
  const [counts, setCounts] = useState<{ tables: number; snapshots: number } | null>(null);
  if (me.username !== project.created_by && me.role !== "admin") return null;
  const openDialog = () => {
    setName(project.name);
    setDescription(project.description);
    setConfirming(false);
    setOpen(true);
  };
  const askDelete = async () => {
    const [tables, snapshots] = await Promise.all([api.tables(project.name), api.snapshots(project.name)]);
    setCounts({ tables: tables.length, snapshots: snapshots.length });
    setConfirming(true);
  };
  const run = async (fn: () => Promise<void>) => {
    try {
      await fn();
      setOpen(false);
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const changed = name.trim() !== project.name || description.trim() !== project.description;
  return (
    <>
      <Button icon={<Settings2 size={14} />} onClick={openDialog}>{t("Project settings")}</Button>
      <Dialog open={open} onOpenChange={setOpen} title={`${t("Project settings")} · ${project.name}`}
        footer={<><Button onClick={() => setOpen(false)}>{t("Cancel")}</Button>
          <Button variant="primary" disabled={!changed || !name.trim()} onClick={() => run(async () => {
            const updated = await api.updateProject(project.name, { name: name.trim(), description: description.trim() });
            toast(t("Saved"));
            onRenamed(updated.name);
          })}>{t("Save")}</Button></>}>
        <div className="stack">
          <Field label={t("Name")} help={t("Lower-case letters, digits, '.', '_' and '-'.")}>
            <Input className="mono" value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          <Field label={t("Description")}><Input value={description} onChange={(e) => setDescription(e.target.value)} /></Field>
          <div className="divider" />
          {!confirming ? (
            <div className="row">
              <span className="small muted grow">{t("Deleting removes the project's tables, snapshots and member list. The results stay with whoever uploaded them.")}</span>
              <Button variant="danger" icon={<Trash2 size={13} />} onClick={askDelete}>{t("Delete project")}</Button>
            </div>
          ) : (
            <div className="issue error">
              <div className="grow">
                <div className="issue-msg">{t("Delete {name}? This removes {tables} tables, {snapshots} snapshots and {records} result links, and cannot be undone. The results themselves are kept.", { name: project.name, tables: counts?.tables ?? 0, snapshots: counts?.snapshots ?? 0, records: project.records })}</div>
                <div className="row mt-2">
                  <Button size="sm" onClick={() => setConfirming(false)}>{t("Cancel")}</Button>
                  <Button size="sm" variant="danger" onClick={() => run(async () => {
                    await api.deleteProject(project.name);
                    toast(t("Deleted"));
                    onDeleted();
                  })}>{t("Delete project")}</Button>
                </div>
              </div>
            </div>
          )}
        </div>
      </Dialog>
    </>
  );
}

/** Picker that links the user's own results or lab pool results into a project. */
export function AddResultsDialog({ project, linked, open, onOpenChange, onAdded }: {
  project: string;
  linked: Set<string>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAdded: () => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const toast = useToast();
  const [source, setSource] = useState<"pool" | "mine">("pool");
  const [rows, setRows] = useState<Row[]>([]);
  const [query, setQuery] = useState("");
  const [chosen, setChosen] = useState<Set<string>>(new Set());
  useEffect(() => {
    if (open) api.records(source).then(setRows);
  }, [open, source]);
  useEffect(() => {
    if (!open) setChosen(new Set());
  }, [open]);
  const visible = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rows.filter((r) => !linked.has(r.record_id) &&
      (!q || [r.model, r.method, rowMethodTitle(describe, r), r.task, r.owner ?? ""].some((v) => v.toLowerCase().includes(q))));
  }, [rows, linked, query]);
  const toggle = (id: string, on: boolean) => setChosen((c) => { const n = new Set(c); if (on) n.add(id); else n.delete(id); return n; });
  const add = async () => {
    try {
      await api.link(project, [...chosen]);
      toast(t("Added {n}", { n: chosen.size }));
      onAdded();
      onOpenChange(false);
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  return (
    <Dialog open={open} onOpenChange={onOpenChange} wide title={t("Add results")}
      icon={<div className="icon-tile accent"><Plus size={18} /></div>}
      footer={<><Button onClick={() => onOpenChange(false)}>{t("Cancel")}</Button>
        <Button variant="primary" disabled={!chosen.size} onClick={add}>{chosen.size ? t("Add {n}", { n: chosen.size }) : t("Add")}</Button></>}>
      <div className="row mb-3 nowrap-row">
        <Segmented value={source} onChange={(v) => setSource(v as "pool" | "mine")}
          options={[{ value: "pool", label: t("Lab pool") }, { value: "mine", label: t("My evaluations") }]} />
        <Input placeholder={t("Search")} value={query} onChange={(e) => setQuery(e.target.value)} />
      </div>
      <div className="pick-table">
        <table className="list">
          <thead><tr><th className="col-check" /><th>{t("Model")}</th><th>{t("Method")}</th><th>{t("Task")}</th><th>{t("Uploaded by")}</th><th className="right">{t("Metrics")}</th></tr></thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.record_id} onClick={() => toggle(r.record_id, !chosen.has(r.record_id))} className="clickable">
                <td onClick={(e) => e.stopPropagation()}><Checkbox checked={chosen.has(r.record_id)} onChange={(v) => toggle(r.record_id, v)} /></td>
                <td className="mono small">{r.model}</td>
                <td className="small">{rowMethodTitle(describe, r)}</td>
                <td className="small">{r.task}</td>
                <td className="small">{r.owner}</td>
                <td className="mono small right">{Object.entries(r.metrics).map(([k, v]) => `${k} ${Number(v).toFixed(4)}`).join(" · ")}</td>
              </tr>
            ))}
            {visible.length === 0 && <tr><td colSpan={6} className="faint small center" style={{ padding: 24 }}>{query.trim() ? t("No match") : t("Nothing to add")}</td></tr>}
          </tbody>
        </table>
      </div>
    </Dialog>
  );
}
