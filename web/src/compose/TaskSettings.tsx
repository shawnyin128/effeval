import { useState } from "react";
import type { TaskDef } from "../api";
import { useT } from "../i18n";
import { Checkbox, Input, Segmented, Select } from "../ui/controls";
import { ScalarField } from "./MethodConfig";

type Params = Record<string, unknown>;

/** `params` with `key` set to `value`, or removed when `value` is undefined. */
export function withParam(params: Params | undefined, key: string, value: unknown): Params | undefined {
  const next = { ...params };
  if (value === undefined) delete next[key];
  else next[key] = value;
  return Object.keys(next).length ? next : undefined;
}

/** Whether the task takes a few-shot override. */
export function hasFewshot(task: TaskDef): boolean {
  return !!task.params_schema.properties?.num_fewshot;
}

/** Whether the benchmark can evaluate a subset of its samples. */
export function hasLimit(task: TaskDef): boolean {
  return !!task.params_schema.properties?.limit;
}

/** A `limit` as shown: "10%" for a fraction, the count otherwise. */
export function formatLimit(limit: unknown): string {
  return typeof limit === "number" && limit > 0 && limit < 1 ? `${Number((limit * 100).toPrecision(12))}%` : String(limit);
}

const isRatio = (value: unknown) => typeof value === "number" && value > 0 && value < 1;

/**
 * Samples box for a benchmark's `limit`: a count (integer >= 1) or a percentage (stored as a fraction in (0, 1));
 * empty (or -1) means all samples.
 */
export function SamplesInput({ task, params, invalid, onChange }: {
  task: TaskDef;
  params?: Params;
  invalid?: boolean;
  onChange: (params: Params | undefined) => void;
}) {
  const t = useT();
  const value = params?.limit;
  const [ratio, setRatio] = useState(isRatio(value));
  const asRatio = ratio || isRatio(value);
  const shownValue = value === undefined ? "" : String(asRatio && typeof value === "number" ? Number((value * 100).toPrecision(12)) : value);
  const write = (text: string) => {
    if (text === "") return onChange(withParam(params, "limit", undefined));
    const n = Number(text);
    if (!asRatio) return onChange(withParam(params, "limit", n === -1 ? undefined : n));
    onChange(withParam(params, "limit", n >= 100 ? undefined : Number((n / 100).toPrecision(12))));
  };
  return (
    <div className="samples-input">
      <Segmented value={asRatio ? "ratio" : "count"} onChange={(v) => { setRatio(v === "ratio"); onChange(withParam(params, "limit", undefined)); }}
        options={[{ value: "count", label: t("Count") }, { value: "ratio", label: t("Ratio") }]} />
      <span className="adorned">
        <Input type="number" min={asRatio ? 0 : -1} max={asRatio ? 99.99 : undefined} step={asRatio ? "any" : 1} value={shownValue}
          placeholder={t("All")} title={asRatio ? t("A percentage between 0 and 100") : task.params_schema.properties?.limit?.description} className={`${asRatio ? "has-adornment" : ""} ${invalid ? "invalid" : ""}`}
          onChange={(e) => write(e.target.value)} />
        {asRatio && <span className="adornment">%</span>}
      </span>
    </div>
  );
}

/** Few-shot box for `num_fewshot`: a select when the task accepts a few values, disabled when it accepts none. */
export function FewshotInput({ task, params, onChange }: {
  task: TaskDef;
  params?: Params;
  onChange: (params: Params | undefined) => void;
}) {
  const t = useT();
  const allowed = task.fewshot_allowed;
  const value = params?.num_fewshot;
  const placeholder = task.num_fewshot === null ? t("per subtask") : t("{n} (default)", { n: task.num_fewshot });
  if (allowed !== null && allowed.length === 0) {
    return <Input disabled placeholder={placeholder} title={t("Few-shot override not supported by this task")} />;
  }
  const set = (n: number | undefined) => onChange(withParam(params, "num_fewshot", n === task.num_fewshot ? undefined : n));
  if (allowed !== null && allowed.length > 1) {
    return <Select value={value === undefined ? String(task.num_fewshot ?? allowed[0]) : String(value)} onChange={(v) => set(Number(v))}
      options={allowed.map((n) => ({ value: String(n), label: String(n) }))} />;
  }
  return (
    <Input type="number" min={0} step={1} value={value === undefined ? "" : String(value)} placeholder={placeholder}
      disabled={allowed !== null} title={allowed !== null ? t("Few-shot override not supported by this task") : task.params_schema.properties?.num_fewshot?.description}
      onChange={(e) => set(e.target.value === "" ? undefined : Number(e.target.value))} />
  );
}

/** Sampling box of a generating task: the `do_sample` checkbox, and the `temperature` once it is on. */
export function SamplingInput({ task, params, runSampling, runTemperature, invalid, onChange }: {
  task: TaskDef;
  params?: Params;
  /** Whether the run-level settings sample, which a task without its own choice follows. */
  runSampling: boolean;
  runTemperature?: number;
  invalid?: boolean;
  onChange: (params: Params | undefined) => void;
}) {
  const t = useT();
  const on = (params?.do_sample as boolean | undefined) ?? runSampling;
  const temperature = params?.temperature as number | undefined;
  const fallback = runTemperature ?? (task.params_schema.properties?.temperature?.examples?.[0] as number | undefined);
  const setSampling = (v: boolean) => {
    const next = withParam(params, "do_sample", v === runSampling ? undefined : v);
    onChange(v ? next : withParam(next, "temperature", undefined));
  };
  return (
    <div className="sampling-input">
      <Checkbox checked={on} onChange={setSampling} label={t("Sampling")} />
      {on && (
        <Input type="number" min={0} step="any" value={temperature === undefined ? "" : String(temperature)} placeholder={fallback === undefined ? "" : String(fallback)}
          title={t("Sampling temperature; empty uses the default shown")} aria-label={t("Temperature")} className={invalid ? "invalid" : ""}
          onChange={(e) => onChange(withParam(params, "temperature", e.target.value === "" ? undefined : Number(e.target.value)))} />
      )}
    </div>
  );
}

/** What a setting is called in the UI when its name alone would not say what it controls. */
const PARAM_LABEL: Record<string, string> = {
  seqlen: "PPL window length",
  max_new_tokens: "Max new tokens",
  max_gen_toks: "Max new tokens",
  max_tokens: "Max new tokens",
};

/** Inputs for a benchmark's settings other than `limit`, generated from its params schema. */
export function TaskSettings({ task, params, invalidPaths, pathPrefix, onChange, inline = false, skip = [] }: {
  task: TaskDef;
  params?: Params;
  invalidPaths: Set<string>;
  pathPrefix: string;
  onChange: (params: Params | undefined) => void;
  /** Compact labelled inputs in one row of a table instead of a grid. */
  inline?: boolean;
  /** Settings left out because the caller renders them itself. */
  skip?: string[];
}) {
  const t = useT();
  const fields = Object.entries(task.params_schema.properties ?? {}).filter(([name]) => name !== "limit" && name !== "num_fewshot" && !skip.includes(name));
  return (
    <div className={inline ? "task-inline-params" : "grid-3"}>
      {fields.map(([name, schema]) => (
        <ScalarField key={name} name={name} label={PARAM_LABEL[name] ? t(PARAM_LABEL[name]) : undefined} schema={schema} value={params?.[name]}
          invalid={invalidPaths.has(`${pathPrefix}.${name}`)} onChange={(v) => onChange(withParam(params, name, v))} />
      ))}
    </div>
  );
}

/** Settings of a benchmark other than `limit`. */
export function extraSettings(task: TaskDef): string[] {
  return Object.keys(task.params_schema.properties ?? {}).filter((name) => name !== "limit" && name !== "num_fewshot");
}
