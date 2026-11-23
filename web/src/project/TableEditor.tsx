import * as yaml from "js-yaml";
import { Plus, Sparkles, Table2, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { api, methodTitle, type TableDefinition, type TableView, type TaskRef } from "../api";
import { alignDraftDtype, draftProblems } from "../compose/draftRules";
import { MAX_SPECULATIVE_TOKENS, providerLabel, rolesOf, usesDecoding } from "../compose/Decoding";
import { TaskPicker } from "../compose/TaskPicker";
import { FewshotInput, SamplesInput, TaskSettings, extraSettings, hasFewshot, hasLimit } from "../compose/TaskSettings";
import { useApp } from "../components/common";
import { ModelPicker } from "../components/ModelPicker";
import { useT } from "../i18n";
import { Button, Field, Input, Segmented, Select, Textarea } from "../ui/controls";
import { Dialog, useToast } from "../ui/overlay";

type Row = TableDefinition["rows"][number];
type EditRow = { label: string; method: string; preset: string | null | undefined; config: string; draft: string; k: string; runtime: string; rest: Row };

const BLANK_ROW: Row = { label: "", method: { name: "base" } };

const draftModelOf = (row: Row): Record<string, unknown> | undefined => {
  const model = (row.decoding?.draft as Record<string, any> | undefined)?.model;
  return typeof model === "string" ? { name: model } : model;
};

function toEditRow(row: Row): EditRow {
  const config = row.method.config && Object.keys(row.method.config).length ? yaml.dump(row.method.config) : "";
  const k = row.decoding?.num_speculative_tokens;
  return {
    label: row.label, method: row.method.name, preset: row.method.preset, config,
    draft: (draftModelOf(row)?.name as string | undefined) ?? "",
    k: k === undefined || k === null ? "" : String(k),
    runtime: (row.runtime?.name as string | undefined) ?? "hf",
    rest: row,
  };
}

/** The table row to store for an edited row: the original row with only the edited fields replaced. */
function toDefinitionRow(r: EditRow, decodes: boolean, config: Record<string, unknown>): Row {
  const { rest } = r;
  const row: Row = { ...rest, label: r.label.trim() };
  if (!decodes) {
    const { config: _config, preset: _preset, ...method } = rest.method;
    const keepEmptyConfig = rest.method.config !== undefined && Object.keys(rest.method.config).length === 0;
    return {
      ...row,
      method: {
        ...method, name: r.method,
        ...(r.preset !== undefined ? { preset: r.preset } : {}),
        ...(Object.keys(config).length || keepEmptyConfig ? { config } : {}),
      },
    };
  }
  row.method = rest.method.name === "base" ? rest.method : { name: "base" };
  if (!r.draft) {
    delete row.decoding;
  } else {
    const { num_speculative_tokens: _k, ...decoding } = rest.decoding ?? {};
    const draft = (decoding.draft ?? {}) as Record<string, unknown>;
    const unchanged = draftModelOf(rest)?.name === r.draft;
    row.decoding = {
      ...decoding, mode: "speculative",
      draft: { ...draft, model: unchanged ? draft.model : { name: r.draft } },
      ...(r.k !== "" ? { num_speculative_tokens: Number(r.k) } : {}),
    };
  }
  const { name: _name, ...other } = rest.runtime ?? {};
  if (r.runtime === "hf" && !Object.keys(other).length) delete row.runtime;
  else row.runtime = { ...rest.runtime, name: r.runtime };
  return row;
}

/** The row with its draft precision aligned to the table's single target model on a vllm/sglang provider; other rows unchanged. */
function alignRow(row: Row, models: Parameters<typeof alignDraftDtype>[0], target: { model: Record<string, unknown> }): Row {
  if (!row.decoding || !row.runtime || row.runtime.name === "hf") return row;
  const aligned = alignDraftDtype(models, { ...target, decoding: row.decoding, runtime: row.runtime });
  return aligned.decoding === row.decoding ? row : { ...row, decoding: aligned.decoding };
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

  const decodes = usesDecoding(rolesOf(describe, field));
  const runtimes = Object.entries(describe.runtimes ?? { hf: { label: "transformers", library: null, available: true } });
  const unavailable = useMemo(() => {
    const out: Record<string, string> = {};
    for (const target of models) {
      for (const [code, p] of Object.entries(draftProblems(describe.models, target, {}))) {
        const reason = (models.length > 1 ? `${target}: ` : "") + t(p.text, p.vars);
        out[code] = out[code] ? `${out[code]}; ${reason}` : reason;
      }
    }
    return out;
  }, [models, describe.models, t]);

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
      const sharedValue = parse(shared, t("Shared run settings"));
      const definition: TableDefinition = {
        name: name.trim(),
        field,
        models,
        tasks: tasks.map((x) => (x.mode || x.params ? x : x.name)),
        shared: sharedValue,
        rows: rows.map((r, i) => {
          const row = toDefinitionRow(r, decodes, decodes ? {} : parse(r.config, `${t("Row")} ${i + 1}`));
          if (!decodes || models.length !== 1) return row;
          return alignRow(row, describe.models, { model: { name: models[0], ...(sharedValue.model as object | undefined) } });
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
                  <button key={`${s.method}/${s.preset}`} className="chip suggest mono" onClick={() => setRows([...rows, { label: `${methodTitle(describe, s.method)} (${s.preset})`, method: s.method, preset: s.preset, config: "", draft: "", k: "", runtime: "hf", rest: BLANK_ROW }])}>
                    <Plus size={11} />{methodTitle(describe, s.method)} · {s.preset}
                  </button>
                ))}
              </div>
            )}
            {decodes && rows.length > 0 && (
              <div className="xt-edit-row faint small" style={{ gridTemplateColumns: "1fr 1.3fr 90px auto auto", border: "none", paddingBottom: 0, paddingTop: 0 }}>
                <span>{t("Label")}</span><span>{t("Draft model")}</span><span>k</span><span>{t("Provider")}</span><span />
              </div>
            )}
            {rows.map((row, i) => (
              decodes ? (
                <div key={i} className="xt-edit-row" style={{ gridTemplateColumns: "1fr 1.3fr 90px auto auto" }}>
                  <Input value={row.label} placeholder={t("Label")} onChange={(e) => setRow(i, { label: e.target.value })} />
                  <ModelPicker models={describe.models} value={row.draft} unavailable={unavailable} onChange={(code) => setRow(i, { draft: code })} />
                  <Input type="number" min={1} max={MAX_SPECULATIVE_TOKENS} step={1} value={row.k} placeholder="4" aria-label={t("Draft tokens per step (k)")}
                    onChange={(e) => setRow(i, { k: e.target.value })} />
                  <Segmented value={row.runtime} onChange={(v) => setRow(i, { runtime: v })}
                    options={runtimes.map(([name, r]) => ({
                      value: name,
                      label: providerLabel(describe, name),
                      disabled: !r.available && name !== row.runtime,
                      title: r.available ? undefined : t("{provider} is not available on this machine", { provider: r.label }),
                    }))} />
                  <Button variant="ghost" icon={<Trash2 size={14} />} aria-label={t("Remove")} onClick={() => setRows(rows.filter((_, j) => j !== i))} />
                </div>
              ) : (
              <div key={i} className="xt-edit-row">
                <Input value={row.label} placeholder={t("Label")} onChange={(e) => setRow(i, { label: e.target.value })} />
                <Select value={row.method} onChange={(v) => setRow(i, { method: v, preset: null })} options={methods.map((m) => ({ value: m.name, label: m.title || m.name }))} />
                <Select value={(row.preset === undefined ? describe.methods[row.method]?.default_preset : row.preset) || "-"} onChange={(v) => setRow(i, { preset: v === "-" ? null : v })}
                  options={[...Object.keys(describe.presets[row.method] ?? {}).map((p) => ({ value: p, label: p, hint: p === describe.methods[row.method]?.default_preset ? t("default") : undefined })), { value: "-", label: t("None") }]} />
                <Button variant="ghost" icon={<Trash2 size={14} />} aria-label={t("Remove")} onClick={() => setRows(rows.filter((_, j) => j !== i))} />
                <Textarea className="mono" placeholder={t("Config overrides (YAML)")} value={row.config} onChange={(e) => setRow(i, { config: e.target.value })} style={{ gridColumn: "1 / -1", minHeight: 84 }} />
              </div>
              )
            ))}
            <div><Button size="sm" icon={<Plus size={13} />} disabled={!decodes && !methods.length} onClick={() => setRows([...rows, { label: "", method: decodes ? "base" : methods[0]?.name ?? "", preset: null, config: "", draft: "", k: "", runtime: "hf", rest: BLANK_ROW }])}>{t("Add row")}</Button></div>
          </div>
        </Field>
        <Field label={t("Shared run settings")}>
          <Textarea className="mono" placeholder={decodes ? "hardware: { gpu: RTX-A6000 }\nlimit: 40" : "limit: 200\ngeneration: { batch_size: 8 }"} value={shared} onChange={(e) => setShared(e.target.value)} style={{ minHeight: 84 }} />
        </Field>
      </div>
    </Dialog>
  );
}
