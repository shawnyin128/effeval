import * as yaml from "js-yaml";
import { Table2 } from "lucide-react";
import { useMemo, useState } from "react";
import { api, type Preset, type TableDefinition } from "../api";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { Button, Field, Input, Select } from "../ui/controls";
import { useToast } from "../ui/overlay";

type Spec = Record<string, any>;
type Key = string | number;

/** One variant of an ablation: the swept value and the RunSpec fragment it changes. */
export interface Ablation {
  label: string;
  values: unknown[];
  sweep: Record<string, unknown[]>;
  methodTop: string | null;
  /** True when the sweep varies the whole `preprocess` value. */
  preprocess?: boolean;
  variants: unknown[];
}

/** The preprocess value an ablation value stands for: "none" turns it off, "current" keeps the one set in the method step, a preset name starts from that preset. */
function preprocessVariant(value: unknown, current: Spec | null): unknown {
  if (value === null || value === "none") return null;
  if (value === "current") return current;
  return typeof value === "string" ? { preset: value } : value;
}

const RUN_PARAMS: Key[][] = [["model", "dtype"], ["limit"], ["determinism", "seed"], ["generation", "batch_size"]];

function pathLabel(path: Key[]): string {
  return path.reduce<string>((out, k) => (typeof k === "number" ? `${out}[${k}]` : out ? `${out}.${k}` : String(k)), "");
}

function leaves(value: unknown, prefix: Key[], out: Key[][]): Key[][] {
  if (Array.isArray(value)) value.forEach((v, i) => leaves(v, [...prefix, i], out));
  else if (value && typeof value === "object") Object.entries(value).forEach(([k, v]) => leaves(v, [...prefix, k], out));
  else out.push(prefix);
  return out;
}

function setDeep(target: unknown, path: Key[], value: unknown): unknown {
  if (!path.length) return value;
  const [head, ...rest] = path;
  const copy: any = Array.isArray(target) ? [...target] : { ...(target as object) };
  copy[head] = setDeep(copy[head], rest, value);
  return copy;
}

/** Method config as it will run: explicit values, with the preset filling what the user left unset. */
export function effectiveConfig(spec: Spec, preset: Preset | null): Record<string, unknown> {
  let config: unknown = {};
  for (const [path, item] of Object.entries(preset?.values ?? {})) {
    if (path.startsWith("method.config.")) config = setDeep(config, path.slice("method.config.".length).split("."), item.value);
  }
  return overlay(config, spec.method?.config ?? {}) as Record<string, unknown>;
}

function overlay(base: unknown, top: unknown): unknown {
  if (!top || typeof top !== "object" || Array.isArray(top) || !base || typeof base !== "object" || Array.isArray(base)) return top ?? base;
  const out: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [k, v] of Object.entries(top)) out[k] = v === undefined ? out[k] : overlay(out[k], v);
  return out;
}

/** Turn "one parameter, several values" into a sweep over the parameter's top-level key. */
export function buildAblation(spec: Spec, preset: Preset | null, path: Key[], values: unknown[], pre: Spec | null = null): Ablation {
  const label = pathLabel(path);
  if (path[0] === "preprocess") {
    const variants = path.length === 1 ? values.map((v) => preprocessVariant(v, pre)) : values.map((v) => setDeep(pre, path.slice(1), v));
    return { label, values, sweep: { preprocess: variants }, methodTop: null, preprocess: true, variants };
  }
  if (path[0] === "method" && path[1] === "config") {
    const top = String(path[2]);
    const base = effectiveConfig(spec, preset)[top];
    const variants = values.map((v) => setDeep(base, path.slice(3), v));
    return { label: label.replace(/^method\.config\./, ""), values, sweep: { [`method.config.${top}`]: variants }, methodTop: top, variants };
  }
  return { label, values, sweep: { [path.join(".")]: values }, methodTop: null, variants: values };
}

function parseValues(text: string): unknown[] {
  return text.split(",").map((s) => s.trim()).filter(Boolean).map((s) => yaml.load(s));
}

/** What the user typed for an ablation: the chosen parameter (a JSON-encoded path, "" for none) and the comma-separated values. */
export interface AblationDraft {
  param: string;
  text: string;
}

/** The ablation a draft describes, or null until a parameter and at least two values are given. */
export function deriveAblation(spec: Spec, preset: Preset | null, draft: AblationDraft, pre: Spec | null = null): Ablation | null {
  const values = parseValues(draft.text);
  return draft.param && values.length >= 2 ? buildAblation(spec, preset, JSON.parse(draft.param), values, pre) : null;
}

/** Pick one parameter and list values; the evaluation then runs one variant per value. */
export function AblationSettings({ spec, preset, pre = null, draft, onChange }: {
  spec: Spec;
  preset: Preset | null;
  /** The effective preprocess ({preset?, steps}) the method step describes, or null when there is none. */
  pre?: Spec | null;
  draft: AblationDraft;
  onChange: (draft: AblationDraft) => void;
}) {
  const { projects } = useApp();
  const t = useT();
  const toast = useToast();
  const [project, setProject] = useState("");
  const { param, text } = draft;
  const options = useMemo(() => {
    const config = effectiveConfig(spec, preset);
    const methodPaths = leaves(config, ["method", "config"], []);
    const preprocessPaths = pre ? leaves(pre.steps, ["preprocess", "steps"], []).filter((p) => !["position", "kind", "type"].includes(String(p[p.length - 1]))) : [];
    return [
      ...(pre ? [{ value: JSON.stringify(["preprocess"]), label: t("preprocess (on / off, or a preset)") }] : []),
      ...methodPaths.map((p) => ({ value: JSON.stringify(p), label: pathLabel(p).replace(/^method\.config\./, "") })),
      ...preprocessPaths.map((p) => ({ value: JSON.stringify(p), label: pathLabel(p) })),
      ...RUN_PARAMS.map((p) => ({ value: JSON.stringify(p), label: pathLabel(p) })),
    ];
  }, [spec, preset, pre, t]);
  const ablation = deriveAblation(spec, preset, draft, pre);

  const saveTable = async () => {
    if (!(ablation?.methodTop || ablation?.preprocess) || !project) return;
    const tasks = (spec.tasks?.length ? spec.tasks : (preset?.values.tasks?.value as unknown[]) ?? []) as TableDefinition["tasks"];
    const { model, method, field, ...rest } = spec;
    delete rest.tasks;
    if (ablation.preprocess) delete rest.preprocess;
    delete rest.sweep;
    const definition: TableDefinition = {
      name: `${ablation.label} ablation`,
      field,
      models: [model.name],
      tasks,
      shared: model.dtype ? { ...rest, model: { dtype: model.dtype } } : rest,
      rows: ablation.variants.map((variant, i) => ({
        label: `${ablation.label} = ${JSON.stringify(ablation.values[i])}`,
        method: {
          name: method.name,
          preset: method.preset === "default" ? preset?.name ?? null : method.preset ?? null,
          config: ablation.methodTop ? { ...(method.config ?? {}), [ablation.methodTop]: variant } : { ...(method.config ?? {}) },
        },
        ...(ablation.preprocess ? { preprocess: variant as TableDefinition["rows"][number]["preprocess"] } : {}),
      })),
    };
    try {
      await api.saveTable(project, null, definition);
      toast(t("Saved"), { description: `${t("Project")} · ${project}` });
      window.location.hash = `#/project/${project}`;
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };

  return (
    <>
      <div className="grid-2">
        <Field label={t("Parameter")}>
          <Select value={param || "-"} onChange={(v) => onChange({ param: v === "-" ? "" : v, text })} options={[{ value: "-", label: t("None") }, ...options]} />
        </Field>
        <Field label={t("Values")}>
          <Input className="mono" placeholder={param === '["preprocess"]' ? "none, current, quarot" : "2, 3, 4, 8"} value={text} disabled={!param} onChange={(e) => onChange({ param, text: e.target.value })} />
        </Field>
      </div>
      {ablation && (
        <div className="ablation-preview">
          <div className="row gap-2">
            {ablation.values.map((v, i) => <span key={i} className="chip mono">{ablation.label} = {JSON.stringify(v)}</span>)}
          </div>
          {(ablation.methodTop || ablation.preprocess) && projects.length > 0 && (
            <div className="row gap-2 nowrap-row mt-3">
              <div className="w-md">
                <Select value={project || "-"} onChange={(v) => setProject(v === "-" ? "" : v)} options={[{ value: "-", label: t("Choose a project") }, ...projects.map((p) => ({ value: p.name, label: p.name }))]} />
              </div>
              <Button icon={<Table2 size={14} />} disabled={!project} onClick={saveTable}>{t("Save as experiment table")}</Button>
            </div>
          )}
        </div>
      )}
    </>
  );
}
