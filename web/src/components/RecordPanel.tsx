import { ChevronRight, GitCompareArrows, Upload, FolderMinus, FolderPlus, Globe, Lock, RotateCcw, RotateCw, Trash2, Undo2, X } from "lucide-react";
import { Collapsible, DropdownMenu } from "radix-ui";
import { Fragment, useEffect, useState } from "react";
import { api, rowMethodTitle, type Command, type Dependents, type Describe, type Me, type Row } from "../api";
import { DestinationSelect } from "../pages/Projects";
import { Button, Field, Input, Select, Textarea } from "../ui/controls";
import { Badge, CodeBlock, CommandSteps } from "../ui/display";
import { useToast } from "../ui/overlay";
import { limitText, specLimit } from "../report/limits";
import { metricHeader, primaryMetric } from "../report/metrics";
import { FidelityBadges, PaperDiff, formatTime, initials, useApp } from "./common";
import { SampleDiffDialog } from "./SampleDiff";
import { useT } from "../i18n";

function Section({ title, children, defaultOpen = true }: { title: string; children: React.ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <Collapsible.Root open={open} onOpenChange={setOpen} style={{ borderTop: "1px solid var(--line)" }}>
      <Collapsible.Trigger asChild>
        <button className="btn btn-ghost" style={{ width: "100%", justifyContent: "flex-start", borderRadius: 0, height: 40, padding: "0 16px" }}>
          <ChevronRight size={14} style={{ transform: open ? "rotate(90deg)" : undefined, transition: "transform 150ms" }} />
          {title}
        </button>
      </Collapsible.Trigger>
      <Collapsible.Content style={{ padding: "0 16px 16px" }}>{children}</Collapsible.Content>
    </Collapsible.Root>
  );
}

/** Whether `me` may publish the record: it is private or retracted and live, and theirs unless they are an admin. */
export function canPublish(row: Row, me: Me): boolean {
  return row.state !== "trashed" && row.visibility !== "public" && (row.owner === me.username || me.role === "admin");
}

/** Button that publishes every record in `rows` the user may publish; hidden when there is none. */
export function PublishSelected({ rows, onDone }: { rows: Row[]; onDone: () => void }) {
  const { me } = useApp();
  const t = useT();
  const toast = useToast();
  const publishable = rows.filter((r) => canPublish(r, me));
  if (publishable.length === 0) return null;
  const publish = async () => {
    try {
      await Promise.all(publishable.map((r) => api.publish(r.record_id)));
      const skipped = rows.length - publishable.length;
      toast(t("Published {n}", { n: publishable.length }), skipped > 0 ? { description: t("{n} skipped: not yours or already public", { n: skipped }) } : undefined);
      onDone();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  return <Button size="sm" icon={<Globe size={13} />} onClick={publish}>{t("Publish")}</Button>;
}

/** The record's metrics with the task's primary metric first. */
function metricsByPriority(describe: Describe, row: Row): [string, string][] {
  const primary = primaryMetric(describe, row.task, Object.keys(row.metrics));
  return Object.entries(row.metrics).sort(([a], [b]) => Number(b === primary) - Number(a === primary));
}

/** Docked inspector for one record: full-precision values, provenance, annotation and sharing. */
export function RecordPanel({ row, project, inPool = false, onClose, onChanged }: {
  row: Row;
  project?: string | null;
  inPool?: boolean;
  onClose: () => void;
  onChanged?: () => void;
}) {
  const { me, projects, describe } = useApp();
  const t = useT();
  const toast = useToast();
  const record = row.record as Record<string, any>;
  const [tags, setTags] = useState("");
  const [note, setNote] = useState("");
  const [status, setStatus] = useState("unreviewed");
  const [pending, setPending] = useState<null | { kind: "delete"; dependents: Dependents } | { kind: "redo"; to: string; command?: Command }>(null);
  useEffect(() => {
    setPending(null);
  }, [row.record_id]);
  useEffect(() => {
    setTags((row.annotation?.tags ?? []).join(", "));
    setNote(row.annotation?.note ?? "");
    setStatus(row.annotation?.status ?? "unreviewed");
  }, [row]);

  const act = async <T,>(fn: () => Promise<T>, done: string | ((result: T) => string)) => {
    try {
      const result = await fn();
      toast(typeof done === "string" ? done : done(result));
      setPending(null);
      onChanged?.();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const canEdit = row.owner === me.username || me.role === "admin";
  const hasEvidence = String(record.artifacts?.per_sample ?? "").startsWith("server:");
  const [comparing, setComparing] = useState(false);
  const trashed = row.state === "trashed";
  const linkable = projects.filter((p) => !row.projects.includes(p.name));
  const startDelete = async () => {
    try {
      setPending({ kind: "delete", dependents: await api.dependents(row.record_id) });
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  const generateRedo = async (to: string) => {
    try {
      const rendered = await api.redo(row.record_id, to);
      setPending({ kind: "redo", to, command: rendered.commands[0] });
      onChanged?.();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };

  return (
    <aside className="card record-panel no-shrink" style={{ width: 380, display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div className="row nowrap-row gap-3" style={{ padding: "14px 16px", alignItems: "flex-start" }}>
        <div className="grow">
          <div className="mono text-md strong">{row.model}</div>
          <div className="muted small mt-1">{rowMethodTitle(describe, row)} · {row.task} · {row.mode}</div>
        </div>
        <Button variant="ghost" size="sm" icon={<X size={15} />} onClick={onClose} aria-label={t("Close")} />
      </div>
      {row.owner && (
        <div className="uploaded-by">
          <span className="avatar xs">{initials(row.owner)}</span>
          <span><b>{row.owner}</b> · {t("via {via}", { via: row.submitted_via })} · {formatTime(row.created_at)}</span>
          {(record.env?.hostname || row.gpu) && <span className="faint">{[record.env?.hostname, row.gpu].filter(Boolean).join(" · ")}</span>}
        </div>
      )}
      <div style={{ padding: "0 16px 14px" }} className="row">
        <Badge tone={row.visibility === "public" ? "accent" : row.visibility === "retracted" ? "orange" : undefined}>
          {row.visibility === "private" ? <Lock size={11} /> : <Globe size={11} />} {t(row.visibility)}
        </Badge>
        <Badge>{row.role}</Badge>
        <FidelityBadges fidelity={row.fidelity} />
        {specLimit(record.spec) !== null && <Badge tone="orange" title={t("Computed on a subset, not the full sample set")}>{limitText(specLimit(record.spec) as number, t)}</Badge>}
        {row.state === "redo" && <Badge tone="orange">{t("redo pending")}</Badge>}
        {trashed && <Badge tone="red">{t("in trash")}</Badge>}
        {row.projects.map((p) => <Badge key={p} mono>{p}</Badge>)}
      </div>
      {pending?.kind === "delete" && (
        <div className="panel-callout warning">
          <div className="callout-title">{t("Move this result to the trash?")}</div>
          {row.visibility !== "private" && <div className="small">{t("It will also disappear from the lab pool.")}</div>}
          {pending.dependents.comparisons.length > 0 && (
            <div className="small">{t("{n} results compare against it", { n: pending.dependents.comparisons.length })}:
              <ul>{pending.dependents.comparisons.map((d) => <li key={d.record_id} className="mono tiny">{d.model} · {d.owner} · {d.record_id.slice(0, 12)}</li>)}</ul>
            </div>
          )}
          {pending.dependents.projects.length > 0 && (
            <div className="small">{t("Linked in projects")}: <span className="mono">{pending.dependents.projects.join(", ")}</span></div>
          )}
          <div className="row mt-2">
            <span className="spacer" />
            <Button size="sm" onClick={() => setPending(null)}>{t("Cancel")}</Button>
            <Button size="sm" variant="danger" icon={<Trash2 size={13} />} onClick={() => act(() => api.trash(row.record_id), t("Moved to trash"))}>{t("Move to trash")}</Button>
          </div>
        </div>
      )}
      {pending?.kind === "redo" && (
        <div className="panel-callout">
          {pending.command ? (
            <>
              <div className="callout-title">{t("Run this on a compute node")}</div>
              <CommandSteps command={pending.command} />
              <div className="small muted mt-2">
                {t("The new result replaces this one when it arrives; offline bundles go through")} <a href="#/import">{t("Import")}</a>.
              </div>
            </>
          ) : (
            <>
              <div className="callout-title">{t("Redo this result")}</div>
              <Field label={t("Upload to")}><DestinationSelect value={pending.to} onChange={(to) => setPending({ kind: "redo", to })} /></Field>
              <div className="row mt-2">
                <span className="spacer" />
                <Button size="sm" onClick={() => setPending(null)}>{t("Cancel")}</Button>
                <Button size="sm" variant="primary" icon={<RotateCw size={13} />} onClick={() => generateRedo(pending.to)}>{t("Generate command")}</Button>
              </div>
            </>
          )}
        </div>
      )}
      <div style={{ overflow: "auto", flex: 1 }}>
        <Section title={t("Metrics")}>
          <dl className="kv">
            {metricsByPriority(describe, row).map(([k, v]) => (
              <Fragment key={`k${k}`}>
                <dt title={describe.metrics[k]?.description}>{metricHeader(describe, k)}</dt>
                <dd className="mono">{v}</dd>
              </Fragment>
            ))}
          </dl>
          {row.comparisons.reference && <p className="faint tiny mono mt-3">{t("ref")} {row.comparisons.reference}</p>}
          {row.comparisons.reference && hasEvidence && describe.tasks[row.task]?.sample_scores.includes("correct") && (
            <Button size="sm" icon={<GitCompareArrows size={13} />} className="mt-3" onClick={() => setComparing(true)}>{t("Compare samples")}</Button>
          )}
          {comparing && <SampleDiffDialog recordId={row.record_id} open={comparing} onOpenChange={setComparing} />}
        </Section>
        {row.fidelity.preset && (
          <Section title={t("Differences from paper")}>
            <div className="stack-sm">
              <div className="faint tiny mono">{row.fidelity.preset}</div>
              <PaperDiff deviations={row.fidelity.deviations} />
              {row.fidelity.assumed.length > 0 && (
                <div>
                  <div className="field-label mt-2 mb-2">{t("Assumed")}</div>
                  <div className="row gap-1">{row.fidelity.assumed.map((p) => <Badge key={p} tone="orange" mono>{p}</Badge>)}</div>
                </div>
              )}
              {row.fidelity.repro !== "untested" && <div className="small">{t("Reproduction")}: {t(row.fidelity.repro)}</div>}
            </div>
          </Section>
        )}
        {canEdit && !trashed && !hasEvidence && (
          <div className="panel-callout">
            <div className="callout-title">{t("Per-sample file not on the server")}</div>
            <div className="small muted mb-2">{t("Sample comparisons need it.")}</div>
            <label className="btn btn-secondary btn-sm clickable">
              <Upload size={13} />{t("Upload per_sample.jsonl.gz")}
              <input type="file" accept=".gz" hidden onChange={(e) => {
                const file = e.target.files?.[0];
                e.target.value = "";
                if (file) act(() => api.attachEvidence(row.record_id, file), t("Per-sample file attached"));
              }} />
            </label>
          </div>
        )}
        {canEdit && !trashed && (
          <Section title={t("Annotation")}>
            <div className="stack-sm">
              <Field label={t("Tags")}><Input value={tags} placeholder="a, b" onChange={(e) => setTags(e.target.value)} /></Field>
              <Field label={t("Status")}>
                <Select value={status} onChange={setStatus} options={["unreviewed", "verified", "suspect"].map((v) => ({ value: v, label: t(v) }))} />
              </Field>
              <Field label={t("Note")}><Textarea value={note} onChange={(e) => setNote(e.target.value)} className="text-sm" style={{ minHeight: 72, fontFamily: "var(--font-sans)" }} /></Field>
              <div className="row"><span className="spacer" />
                <Button variant="primary" size="sm" onClick={() => act(() => api.annotate(row.record_id, {
                  tags: tags.split(",").map((t) => t.trim()).filter(Boolean), note, status }), t("Saved"))}>{t("Save")}</Button>
              </div>
            </div>
          </Section>
        )}
        <Section title={t("Environment")} defaultOpen={false}>
          <dl className="kv">
            <dt>{t("GPU")}</dt><dd>{record.env?.gpu ?? "—"} ×{record.env?.gpu_count ?? 0}</dd>
            <dt>{t("Host")}</dt><dd>{record.env?.hostname ?? "—"}</dd>
            <dt>{t("CUDA / driver")}</dt><dd>{record.env?.cuda ?? "—"} / {record.env?.driver ?? "—"}</dd>
            <dt>{t("Attention")}</dt><dd className="mono">{record.env?.attn_implementation ?? "—"}</dd>
            {Object.entries(record.env?.packages ?? {}).map(([k, v]) => <Fragment key={`p${k}`}><dt>{k}</dt><dd className="mono">{String(v)}</dd></Fragment>)}
            <dt>{t("Prepare / measure")}</dt><dd>{record.timings?.prepare_s?.toFixed?.(1) ?? "—"} s / {record.timings?.measure_s?.toFixed?.(1) ?? "—"} s</dd>
          </dl>
        </Section>
        <Section title={t("Job spec")} defaultOpen={false}>
          <CodeBlock text={JSON.stringify(record.spec, null, 2)} language="json" />
        </Section>
      </div>
      {trashed && canEdit && (
        <div className="panel-actions">
          <Button variant="primary" icon={<RotateCcw size={14} />} onClick={() => act(() => api.restore(row.record_id), t("Restored"))}>{t("Restore")}</Button>
        </div>
      )}
      {!trashed && (canEdit || project) && (
        <div className="panel-actions">
          {canEdit && row.visibility === "private" && <Button variant="primary" icon={<Globe size={14} />} onClick={() => act(() => api.publish(row.record_id), t("Published"))}>{t("Publish")}</Button>}
          {canEdit && row.visibility === "retracted" && <Button variant="primary" icon={<Globe size={14} />} onClick={() => act(() => api.publish(row.record_id), t("Published"))}>{t("Republish")}</Button>}
          {canEdit && row.visibility !== "private" && <Button icon={<Undo2 size={14} />} onClick={() => act(() => api.retract(row.record_id), ({ unshared }) => unshared === "private" ? t("Set to private") : t("Retracted: others are using it, so it stays in the lab pool with a warning"))}>{t("Remove from the lab pool")}</Button>}
          {linkable.length > 0 && (
            <DropdownMenu.Root>
              <DropdownMenu.Trigger asChild><Button variant="ghost" icon={<FolderPlus size={14} />} aria-label={t("Add to project")} title={t("Add to project")} /></DropdownMenu.Trigger>
              <DropdownMenu.Portal>
                <DropdownMenu.Content className="menu-content" side="top" align="end" sideOffset={4}>
                  {linkable.map((p) => (
                    <DropdownMenu.Item key={p.name} className="menu-item mono" onSelect={() => act(() => api.link(p.name, [row.record_id]), `${t("Added to")} ${p.name}`)}>{p.name}</DropdownMenu.Item>
                  ))}
                </DropdownMenu.Content>
              </DropdownMenu.Portal>
            </DropdownMenu.Root>
          )}
          <span className="spacer" />
          {project && <Button variant="ghost" icon={<FolderMinus size={14} />} aria-label={t("Remove from project")} title={t("Remove from project")} onClick={() => act(() => api.unlink(project, row.record_id), t("Removed from project"))} />}
          {canEdit && row.state === "redo" && <Button variant="ghost" onClick={() => act(() => api.cancelRedo(row.record_id), t("Redo cancelled"))}>{t("Cancel redo")}</Button>}
          {canEdit && row.state === "active" && <Button variant="ghost" icon={<RotateCw size={14} />} aria-label={t("Redo")} title={t("Redo")} onClick={() => setPending({ kind: "redo", to: "private" })} />}
          {canEdit && !inPool && <Button variant="ghost" icon={<Trash2 size={14} />} aria-label={t("Delete")} title={t("Delete")} onClick={startDelete} />}
        </div>
      )}
    </aside>
  );
}
