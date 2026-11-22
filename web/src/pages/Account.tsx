import { ChevronRight, FileCode2, ListChecks, Pencil, RotateCw, Trash2, UserPlus } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, methodTitle, type Command, type Rendered, type Row, type RunItem } from "../api";
import { RecordPanel } from "../components/RecordPanel";
import { CommandDialog } from "./Compose";
import { useDestination } from "./Projects";
import { initials, useApp } from "../components/common";
import { Button, Field, Input, Segmented, Select } from "../ui/controls";
import { Badge, Card, CommandSteps, Empty, PageHeader } from "../ui/display";
import { Dialog, useToast } from "../ui/overlay";
import { useT, useLocale } from "../i18n";
import { useLive } from "../live";
import { applyTheme, savedTheme, type Theme } from "../theme";

export function Runs() {
  const t = useT();
  const toast = useToast();
  const [to] = useDestination();
  const [runs, setRuns] = useState<RunItem[] | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [rendered, setRendered] = useState<Rendered | null>(null);
  const [command, setCommand] = useState<Command | null>(null);
  const [record, setRecord] = useState<Row | null>(null);
  const load = useCallback(() => api.runs().then(setRuns), []);
  useLive(["runs", "records"], load);
  useEffect(() => {
    load();
    const timer = setInterval(load, 10000);
    return () => clearInterval(timer);
  }, [load]);
  const act = async (fn: () => Promise<unknown>) => {
    try {
      await fn();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  return (
    <div className="page">
      <PageHeader title={t("Runs")} />
      <div className="docked">
        <div className="grow">
          {runs && runs.length === 0 && <div className="card"><Empty icon={<ListChecks size={20} />} title={t("No runs yet")} /></div>}
          {runs && runs.length > 0 && (
            <div className="card runs">
              {runs.map((run) => (
                <RunRow key={run.plan_id} run={run} open={openId === run.plan_id} onToggle={() => setOpenId(openId === run.plan_id ? null : run.plan_id)}
                  onRename={(title) => act(async () => { await api.renameRun(run.plan_id, title); load(); })}
                  onDelete={() => act(async () => { await api.deleteRun(run.plan_id); toast(t("Deleted")); load(); })}
                  onResume={() => act(async () => setRendered(await api.resumeRun(run.plan_id, to)))}
                  onCommand={() => act(async () => setCommand(await api.runCommand(run.plan_id, to)))}
                  onOpenRecord={(id) => act(async () => setRecord(await api.record(id)))} />
              ))}
            </div>
          )}
        </div>
        {record && <div className="dock"><RecordPanel row={record} onClose={() => setRecord(null)} onChanged={load} /></div>}
      </div>
      <CommandDialog rendered={rendered} onClose={() => setRendered(null)} />
      <Dialog open={!!command} onOpenChange={(o) => !o && setCommand(null)} wide title={t("Command")}>
        {command && <CommandSteps command={command} />}
      </Dialog>
    </div>
  );
}

function RunRow({ run, open, onToggle, onRename, onDelete, onResume, onCommand, onOpenRecord }: {
  run: RunItem;
  open: boolean;
  onToggle: () => void;
  onRename: (title: string) => void;
  onDelete: () => void;
  onResume: () => void;
  onCommand: () => void;
  onOpenRecord: (recordId: string) => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const [editing, setEditing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const states = run.plan.jobs.map((j) => (run.records[j.job_id] ? "done" : run.heartbeats[j.job_id]?.status ?? "waiting"));
  const done = states.filter((s) => s === "done").length;
  const started = states.some((s) => s !== "waiting");
  const tone = (s: string) => (s === "done" ? "green" : s === "failed" ? "red" : s === "running" ? "accent" : "");
  return (
    <div className={`run-row ${open ? "open" : ""}`}>
      <div className="run-head" onClick={onToggle}>
        <ChevronRight size={14} className="run-chevron" />
        <div className="run-title" onClick={(e) => editing && e.stopPropagation()}>
          {editing ? (
            <Input autoFocus defaultValue={run.title ?? run.summary} onClick={(e) => e.stopPropagation()}
              onBlur={(e) => { setEditing(false); onRename(e.target.value === run.summary ? "" : e.target.value); }}
              onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); if (e.key === "Escape") setEditing(false); }} />
          ) : (
            <>
              <span className="run-name">{run.title ?? run.summary}</span>
              <button className="icon-button" aria-label={t("Rename")} title={t("Rename")} onClick={(e) => { e.stopPropagation(); setEditing(true); }}><Pencil size={12} /></button>
              {run.title && <span className="faint small">{run.summary}</span>}
            </>
          )}
        </div>
        <span className="faint tiny mono">{run.plan_id.slice(0, 13)}</span>
        <span className="small muted w-sm">{run.created_at.slice(0, 16).replace("T", " ")}</span>
        <Badge tone={done === states.length && states.length ? "green" : undefined}>{done}/{states.length}</Badge>
        <span className="run-actions" onClick={(e) => e.stopPropagation()}>
          <Button size="sm" variant="ghost" icon={<FileCode2 size={13} />} aria-label={t("Command")} title={t("Command")} onClick={onCommand} />
          {started && done < states.length && <Button size="sm" variant="ghost" icon={<RotateCw size={13} />} aria-label={t("Resume command")} title={t("Resume command")} onClick={onResume} />}
          {confirming
            ? <Button size="sm" variant="danger" onClick={onDelete}>{t("Delete")}</Button>
            : <Button size="sm" variant="ghost" icon={<Trash2 size={13} />} aria-label={t("Delete")} title={t("Delete")} onClick={() => setConfirming(true)} />}
        </span>
      </div>
      {open && (
        <div className="run-body">
          <table className="list">
            <thead><tr><th className="col-sm" /><th>{t("Model")} · {t("Method")} · {t("Task")}</th><th className="col-md">{t("Status")}</th><th>{t("Message")}</th><th className="col-md" /></tr></thead>
            <tbody>
              {run.plan.jobs.map((job, i) => {
                const recordId = run.records[job.job_id];
                const beat = run.heartbeats[job.job_id];
                return (
                  <tr key={job.job_id}>
                    <td><Badge tone={job.role === "main" ? "accent" : undefined}>{t(job.role)}</Badge></td>
                    <td className="small"><span className="mono">{job.spec.model.name}</span> · {methodTitle(describe, job.spec.method.name)} · {job.spec.task.name} <span className="faint">({job.spec.evaluator.name})</span></td>
                    <td><span className="row small gap-2 nowrap-row"><span className={`dot ${tone(states[i])}`} />{t(states[i])}</span></td>
                    <td className={`small ${states[i] === "failed" ? "error-text" : "muted"}`} title={beat?.message}>{beat?.message}{beat?.at ? <span className="faint"> · {beat.at.slice(5, 16).replace("T", " ")}</span> : null}</td>
                    <td className="actions">
                      {recordId
                        ? <Button size="sm" variant="link" onClick={() => onOpenRecord(recordId)}>{t("View result")}</Button>
                        : <a className="btn btn-link btn-sm" href="#/import">{t("Upload results")}</a>}
                    </td>
                  </tr>
                );
              })}
              {run.plan.reused.map((r) => (
                <tr key={r.run_key}>
                  <td><Badge tone="green">{t("reused")}</Badge></td>
                  <td className="small muted" colSpan={3} title={r.reason}>{t(r.role)} · {r.reason}</td>
                  <td className="actions"><Button size="sm" variant="link" onClick={() => onOpenRecord(r.record_id)}>{t("View result")}</Button></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export function Members() {
  const { me } = useApp();
  const t = useT();
  const toast = useToast();
  const [users, setUsers] = useState<Awaited<ReturnType<typeof api.users>>>([]);
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState({ username: "", password: "", role: "member" });
  const [reset, setReset] = useState<{ user: string; password: string } | null>(null);
  const [editing, setEditing] = useState<{ user: string; username: string; role: string } | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const load = () => api.users().then(setUsers);
  useLive(["users"], load);
  useEffect(() => {
    load();
  }, []);
  const act = async (fn: () => Promise<unknown>, done: string) => {
    try {
      await fn();
      toast(done);
      load();
      return true;
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
      return false;
    }
  };
  return (
    <div className="page">
      <PageHeader title={t("Members")}
        actions={<Button variant="primary" icon={<UserPlus size={14} />} onClick={() => setAdding(true)}>{t("Add member")}</Button>} />
      <Card>
        <table className="list">
          <thead><tr><th>{t("Member")}</th><th className="col-md">{t("Role")}</th><th className="col-md">{t("Status")}</th><th className="col-md">{t("Since")}</th><th className="col-2xl" /></tr></thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.username}>
                <td><span className="row gap-3"><span className="avatar">{initials(u.username)}</span>{u.username}</span></td>
                <td><Badge tone={u.role === "admin" ? "accent" : undefined}>{t(u.role)}</Badge></td>
                <td>{u.disabled ? <Badge tone="red">{t("disabled")}</Badge> : <span className="row small gap-2"><span className="dot green" />{t("active")}</span>}</td>
                <td className="small muted">{u.created_at.slice(0, 10)}</td>
                <td className="actions">
                  <Button variant="link" onClick={() => setEditing({ user: u.username, username: u.username, role: u.role })}>{t("Edit")}</Button>
                  <Button variant="link" onClick={() => setReset({ user: u.username, password: "" })}>{t("Reset password")}</Button>
                  {u.username !== me.username && <Button variant="link" className={u.disabled ? "" : "danger"} onClick={() => act(() => api.setUserEnabled(u.username, !!u.disabled), u.disabled ? t("Enabled") : t("Disabled"))}>{u.disabled ? t("Enable") : t("Disable")}</Button>}
                  {u.username !== me.username && <Button variant="link" className="danger" onClick={() => setRemoving(u.username)}>{t("Delete")}</Button>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <LoginTokens />
      <Dialog open={adding} onOpenChange={setAdding} title={t("Add member")}
        icon={<div className="icon-tile accent"><UserPlus size={18} /></div>}
        footer={<><Button onClick={() => setAdding(false)}>{t("Cancel")}</Button>
          <Button variant="primary" onClick={async () => { if (await act(() => api.addUser(form.username, form.password, form.role), `${t("Added")} ${form.username}`)) { setAdding(false); setForm({ username: "", password: "", role: "member" }); } }}>{t("Add")}</Button></>}>
        <div className="stack">
          <Field label={t("Username")}><Input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} /></Field>
          <Field label={t("Initial password")}><Input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></Field>
          <Field label={t("Role")}><Select value={form.role} onChange={(v) => setForm({ ...form, role: v })} options={[{ value: "member", label: t("member") }, { value: "admin", label: t("admin") }]} /></Field>
        </div>
      </Dialog>
      <Dialog open={!!reset} onOpenChange={(o) => !o && setReset(null)} title={`${t("Reset password")} · ${reset?.user}`}
        footer={<><Button onClick={() => setReset(null)}>{t("Cancel")}</Button>
          <Button variant="primary" onClick={async () => { if (reset && await act(() => api.resetPassword(reset.user, reset.password), t("Password reset"))) setReset(null); }}>{t("Reset")}</Button></>}>
        <Field label={t("New password")}><Input type="password" value={reset?.password ?? ""} onChange={(e) => reset && setReset({ ...reset, password: e.target.value })} /></Field>
      </Dialog>
      <Dialog open={!!editing} onOpenChange={(o) => !o && setEditing(null)} title={`${t("Edit member")} · ${editing?.user}`}
        icon={<div className="icon-tile accent"><Pencil size={18} /></div>}
        footer={<><Button onClick={() => setEditing(null)}>{t("Cancel")}</Button>
          <Button variant="primary" onClick={async () => {
            if (!editing) return;
            const renamedSelf = editing.user === me.username && editing.username !== editing.user;
            if (await act(() => api.updateUser(editing.user, { username: editing.username, role: editing.role }), t("Saved"))) {
              setEditing(null);
              if (renamedSelf) window.location.reload();
            }
          }}>{t("Save")}</Button></>}>
        <div className="stack">
          <Field label={t("Username")}><Input value={editing?.username ?? ""} onChange={(e) => editing && setEditing({ ...editing, username: e.target.value })} /></Field>
          <Field label={t("Role")}>{editing?.user === me.username
            ? <span className="small muted">{t(editing.role)} · {t("You cannot change your own role")}</span>
            : <Select value={editing?.role ?? "member"} onChange={(v) => editing && setEditing({ ...editing, role: v })} options={[{ value: "member", label: t("member") }, { value: "admin", label: t("admin") }]} />}</Field>
        </div>
      </Dialog>
      <Dialog open={!!removing} onOpenChange={(o) => !o && setRemoving(null)} title={`${t("Delete member")} · ${removing}`}
        icon={<div className="icon-tile red"><Trash2 size={18} /></div>}
        description={t("Only members without any data can be deleted; disable the others instead. This cannot be undone.")}
        footer={<><Button onClick={() => setRemoving(null)}>{t("Cancel")}</Button>
          <Button variant="danger" onClick={async () => { if (removing && await act(() => api.deleteUser(removing), t("Deleted"))) setRemoving(null); }}>{t("Delete")}</Button></>} />

    </div>
  );
}

export function Settings({ onLocale }: { onLocale: (locale: "en" | "zh") => Promise<void> }) {
  const t = useT();
  const locale = useLocale();
  const toast = useToast();
  const [password, setPassword] = useState("");
  const [theme, setTheme] = useState(savedTheme);
  return (
    <div className="page">
      <PageHeader title={t("Settings")} />
      <Card title={t("Language")}>
        <Segmented value={locale} onChange={(v: string) => onLocale(v as "en" | "zh")} options={[{ value: "en", label: "English" }, { value: "zh", label: "中文" }]} />
      </Card>
      <Card title={t("Appearance")}>
        <Segmented value={theme} onChange={(v: string) => { applyTheme(v as Theme); setTheme(v as Theme); }}
          options={[{ value: "light", label: t("Light") }, { value: "dark", label: t("Dark") }, { value: "system", label: t("System") }]} />
      </Card>
      <StorageCard />
      <Card title={t("Password")}>
        <div className="row nowrap-row">
          <Input type="password" placeholder={t("New password")} value={password} onChange={(e) => setPassword(e.target.value)} className="max-w-lg" />
          <Button disabled={password.length < 8} onClick={async () => {
            try { await api.changePassword(password); setPassword(""); toast(t("Password changed")); } catch (e) { toast(t("Action failed"), { tone: "error", description: (e as Error).message }); }
          }}>{t("Change")}</Button>
        </div>
      </Card>
    </div>
  );
}

function formatBytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
}

/** Where the server keeps per-sample files: the lab's shared directory (admins) and the user's own. */
function StorageCard() {
  const t = useT();
  const toast = useToast();
  const [usage, setUsage] = useState<Awaited<ReturnType<typeof api.storage>> | null>(null);
  const [paths, setPaths] = useState({ shared: "", personal: "" });
  const load = () => api.storage().then((u) => { setUsage(u); setPaths({ shared: u.shared.path, personal: u.personal.path }); });
  useEffect(() => {
    load();
  }, []);
  if (!usage) return null;
  const save = async (scope: "shared" | "personal") => {
    try {
      const { moved } = await api.setStorage(scope, paths[scope]);
      toast(t("Saved"), { description: t("{n} files moved", { n: moved }) });
      load();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  return (
    <Card title={t("Result storage")}>
      <table className="list">
        <tbody>
          {(["shared", "personal"] as const).map((scope) => (
            <tr key={scope}>
              <td className="small col-md">{scope === "shared" ? t("Lab pool results") : t("My private results")}</td>
              <td><Input className="mono" value={paths[scope]} disabled={!usage[scope].editable} onChange={(e) => setPaths({ ...paths, [scope]: e.target.value })} /></td>
              <td className="small muted col-md">{t("{n} files", { n: usage[scope].files })} · {formatBytes(usage[scope].bytes)}</td>
              <td className="col-sm actions">
                <Button size="sm" disabled={!usage[scope].editable || paths[scope] === usage[scope].path} onClick={() => save(scope)}>{t("Save")}</Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </Card>
  );
}

/** Every member's login tokens (one per machine, issued by `effeval login`), for admins to review and revoke. */
function LoginTokens() {
  const t = useT();
  const toast = useToast();
  const [tokens, setTokens] = useState<Awaited<ReturnType<typeof api.adminTokens>>>([]);
  const load = () => api.adminTokens().then(setTokens);
  useEffect(() => {
    load();
  }, []);
  const day = (iso: string | null) => (iso ? iso.slice(0, 16).replace("T", " ") : "—");
  return (
    <Card title={t("Login tokens")}>
      {tokens.length === 0 ? <p className="faint small">—</p> : (
        <table className="list">
          <thead><tr><th className="col-md">{t("Member")}</th><th>{t("Machine")}</th><th className="col-sm">{t("Rights")}</th><th className="col-md">{t("Last used")}</th><th className="col-md">{t("Expires")}</th><th className="col-sm" /></tr></thead>
          <tbody>
            {tokens.map((tok) => (
              <tr key={tok.token_id}>
                <td><span className="row nowrap-row"><span className="avatar xs">{initials(tok.username)}</span>{tok.username}</span></td>
                <td className="mono small">{tok.label}</td>
                <td>{tok.can_publish ? <Badge tone="accent">{t("can publish")}</Badge> : <Badge>{t("private only")}</Badge>}</td>
                <td className="small muted">{day(tok.last_used)}</td>
                <td className="small muted">{day(tok.expires_at)}</td>
                <td className="actions">
                  <Button variant="link" className="danger" onClick={async () => {
                    try { await api.revokeAdminToken(tok.token_id); toast(t("Revoked")); load(); } catch (e) { toast(t("Action failed"), { tone: "error", description: (e as Error).message }); }
                  }}>{t("Revoke")}</Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
