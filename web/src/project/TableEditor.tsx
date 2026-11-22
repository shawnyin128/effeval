import * as yaml from "js-yaml";
import { Plus, Sparkles, Table2, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { api, methodTitle, type TableDefinition, type TableView, type TaskRef } from "../api";
import { TaskPicker } from "../compose/TaskPicker";
import { FewshotInput, SamplesInput, TaskSettings, extraSettings, hasFewshot, hasLimit } from "../compose/TaskSettings";
import { useApp } from "../components/common";
import { ModelPicker } from "../components/ModelPicker";
import { useT } from "../i18n";
import { Button, Field, Input, Select, Textarea } from "../ui/controls";
import { Dialog, useToast } from "../ui/overlay";

type EditRow = { label: string; method: string; preset: string; config: string };

function toEditRow(row: TableDefinition["rows"][number]): EditRow {
  const config = row.method.config && Object.keys(row.method.config).length ? yaml.dump(row.method.config) : "";
  return { label: row.label, method: row.method.name, preset: row.method.preset ?? "", config };
}

/** Create or edit an experiment table: name, field, models, tasks, one row per method config, shared settings. */
export function TableEditor({ project, table, onClose, onSaved }: {
  project: string;
  table: TableView | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const toast = useToast();
  const initial = table?.definition;
  const [name, setName] = useState(initial?.name ?? "");
  const [field, setField] = useState(initial?.field ?? Object.keys(describe.fields)[0] ?? "");
  const [models, setModels] = useState<string[]>(initial?.models ?? []);
  const [tasks, setTasks] = useState<TaskRef[]>((initial?.tasks ?? []).map((x) => (typeof x === "string" ? { name: x } : x)));
  const setTaskParams = (name: string, params: Record<string, unknown> | undefined) =>
    setTasks((list) => list.map((x) => (x.name === name ? (params ? { ...x, params } : { name: x.name, ...(x.mode ? { mode: x.mode } : {}) }) : x)));
  const [rows, setRows] = useState<EditRow[]>(initial?.rows.map(toEditRow) ?? []);
  const [shared, setShared] = useState(initial && Object.keys(initial.shared).length ? yaml.dump(initial.shared) : "");

  const methods = Object.values(describe.methods).filter((m) => m.field === field && m.name !== "base");
  const suggestions = useMemo(() => methods.filter((m) => m.builtin).flatMap((m) =>
    Object.keys(describe.presets[m.name] ?? {}).map((preset) => ({ method: m.name, preset })))
    .filter((s) => !rows.some((r) => r.method === s.method && r.preset === s.preset)), [methods, rows, describe]);
  const setRow = (index: number, patch: Partial<EditRow>) => setRows((list) => list.map((r, i) => (i === index ? { ...r, ...patch } : r)));

  const save = async () => {
    try {
      const parse = (text: string, what: string) => {
        if (!text.trim()) return {};
        const value = yaml.load(text);
        if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${what}: ${t("The document must be a mapping")}`);
        return value as Record<string, unknown>;
      };
      const definition: TableDefinition = {
        name: name.trim(),
        field,
        models,
        tasks: tasks.map((x) => (x.mode || x.params ? x : x.name)),
        shared: parse(shared, t("Shared run settings")),
        rows: rows.map((r, i) => {
          const config = parse(r.config, `${t("Row")} ${i + 1}`);
          return {
            label: r.label.trim(),
            method: { name: r.method, preset: r.preset || null, ...(Object.keys(config).length ? { config } : {}) },
          };
        }),
      };
      await api.saveTable(project, table?.table_id ?? null, definition);
      onSaved();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()} wide title={table ? t("Edit table") : t("New table")}
      icon={<div className="icon-tile accent"><Table2 size={18} /></div>}
      footer={<><Button onClick={onClose}>{t("Cancel")}</Button>
        <Button variant="primary" disabled={!name.trim() || !models.length || !tasks.length || !rows.length} onClick={save}>{t("Save")}</Button></>}>
      <div className="stack">
        <div className="grid-2">
          <Field label={t("Name")}><Input value={name} placeholder="Table 3 · W4A16" onChange={(e) => setName(e.target.value)} /></Field>
          <Field label={t("Field")}><Select value={field} onChange={setField} options={Object.keys(describe.fields).map((f) => ({ value: f, label: f }))} /></Field>
        </div>
        <Field label={t("Models")}>
          <div className="stack-sm">
            <div className="row gap-2">
              {models.map((m) => (
                <span key={m} className="chip mono">{m}<button className="icon-button" aria-label={t("Remove")} onClick={() => setModels(models.filter((x) => x !== m))}><X size={12} /></button></span>
              ))}
            </div>
            <ModelPicker models={describe.models} value="" onChange={(code) => code && !models.includes(code) && setModels([...models, code])} />
          </div>
        </Field>
        <Field label={t("Tasks")}>
          <div className="stack-sm">
            <TaskPicker allowedModes={describe.fields[field]?.modes ?? []} selected={tasks} onChange={setTasks} />
            {tasks.map((entry) => {
              const task = describe.tasks[entry.name];
              if (!task || (!hasLimit(task) && !hasFewshot(task) && !extraSettings(task).length)) return null;
              return (
                <div key={entry.name} className="task-params">
                  <span className="mono small">{entry.name}</span>
                  {hasFewshot(task) && <Field label={t("Few-shot")}><FewshotInput task={task} params={entry.params} onChange={(p) => setTaskParams(entry.name, p)} /></Field>}
                  {hasLimit(task) && <Field label={t("Samples")}><SamplesInput task={task} params={entry.params} onChange={(p) => setTaskParams(entry.name, p)} /></Field>}
                  {extraSettings(task).length > 0 && <TaskSettings task={task} params={entry.params} invalidPaths={new Set()} pathPrefix="" onChange={(p) => setTaskParams(entry.name, p)} />}
                </div>
              );
            })}
          </div>
        </Field>
        <Field label={t("Rows")}>
          <div className="stack-sm">
            {suggestions.length > 0 && (
              <div className="row gap-2">
                <span className="faint small row gap-1"><Sparkles size={12} />{t("Recommended baselines")}</span>
                {suggestions.map((s) => (
                  <button key={`${s.method}/${s.preset}`} className="chip suggest mono" onClick={() => setRows([...rows, { label: `${methodTitle(describe, s.method)} (${s.preset})`, method: s.method, preset: s.preset, config: "" }])}>
                    <Plus size={11} />{methodTitle(describe, s.method)} · {s.preset}
                  </button>
                ))}
              </div>
            )}
            {rows.map((row, i) => (
              <div key={i} className="xt-edit-row">
                <Input value={row.label} placeholder={t("Label")} onChange={(e) => setRow(i, { label: e.target.value })} />
                <Select value={row.method} onChange={(v) => setRow(i, { method: v, preset: "" })} options={methods.map((m) => ({ value: m.name, label: m.title || m.name }))} />
                <Select value={row.preset || "-"} onChange={(v) => setRow(i, { preset: v === "-" ? "" : v })}
                  options={[...Object.keys(describe.presets[row.method] ?? {}).map((p) => ({ value: p, label: p, hint: p === describe.methods[row.method]?.default_preset ? t("default") : undefined })), { value: "-", label: t("None") }]} />
                <Button variant="ghost" icon={<Trash2 size={14} />} aria-label={t("Remove")} onClick={() => setRows(rows.filter((_, j) => j !== i))} />
                <Textarea className="mono" placeholder={t("Config overrides (YAML)")} value={row.config} onChange={(e) => setRow(i, { config: e.target.value })} style={{ gridColumn: "1 / -1", minHeight: 84 }} />
              </div>
            ))}
            <div><Button size="sm" icon={<Plus size={13} />} disabled={!methods.length} onClick={() => setRows([...rows, { label: "", method: methods[0]?.name ?? "", preset: "", config: "" }])}>{t("Add row")}</Button></div>
          </div>
        </Field>
        <Field label={t("Shared run settings")}>
          <Textarea className="mono" placeholder={"limit: 200\ngeneration: { batch_size: 8 }"} value={shared} onChange={(e) => setShared(e.target.value)} style={{ minHeight: 84 }} />
        </Field>
      </div>
    </Dialog>
  );
}
