import { Check, Pencil, Plus, RotateCcw, Trash2, X } from "lucide-react";
import { Fragment, useState } from "react";
import type { Architecture, JsonSchema, Manifest, Preset } from "../api";
import { ArchDiagram } from "./ArchDiagram";
import { Button, Field, Input, Segmented, Select } from "../ui/controls";
import { useT } from "../i18n";
import { Badge, Help } from "../ui/display";

type Config = Record<string, unknown>;
type Defs = Record<string, JsonSchema>;
const FORMAT_DEFS: Record<string, string> = { int: "IntFormat", fp: "FpFormat", mx: "MxFormat" };

function resolveRef(schema: JsonSchema, root: JsonSchema): JsonSchema {
  if (schema.$ref) return root.$defs?.[schema.$ref.split("/").pop() as string] ?? schema;
  return schema;
}

function isObject(schema: JsonSchema): boolean {
  return schema.type === "object" || !!schema.properties;
}

/** Config field name as a sentence-case label: "n_samples" -> "N samples". */
function humanize(key: string): string {
  const words = key.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

function presetValue(preset: Preset | null, path: string): { value: unknown; source: string } | undefined {
  return preset?.values[`method.config.${path}`];
}

function formatLabel(quant: Record<string, unknown>): string {
  if (quant.type === "int") return `INT${quant.bits}`;
  if (quant.type === "fp") return String(quant.encoding).toUpperCase();
  if (quant.type === "mx") return `MX${String(quant.element).toUpperCase()}`;
  return String(quant.type);
}

const SITE_SHORT: [RegExp, string][] = [
  [/^linear\..*\.weight$/, "W"], [/^linear\..*\.input$/, "A"], [/^linear\..*\.output$/, "O"],
  [/^attn\.q$/, "Q"], [/^attn\.k$/, "K"], [/^attn\.v$/, "V"], [/^attn\.scores$/, "S"], [/^attn\.probs$/, "P"],
  [/^attn\.out$/, "AO"], [/^lm_head\./, "head"],
];

/** Short tab title of a rule, e.g. "W INT4" or "QK E4M3"; the full description is its tooltip. */
export function ruleLabel(rule: Rule): string {
  const parts = [...new Set(rule.sites.map((s) => SITE_SHORT.find(([re]) => re.test(s))?.[1] ?? s))];
  const where = parts.length === 0 ? "new" : parts.every((p) => p.length === 1 && "QKVSP".includes(p)) ? parts.join("") : parts.join("/");
  return `${where} ${formatLabel(rule.quant)}`;
}

/** A config schema split into its form sections: quantization rules, object-valued groups, and the remaining fields. */
function sectionsOf(manifest: Manifest) {
  const root = manifest.config_schema;
  const properties = Object.entries(root.properties ?? {}).map(([key, raw]) => [key, resolveRef(raw, root)] as const);
  return {
    properties,
    hasRules: manifest.uses.includes("quant_rules") && properties.some(([key]) => key === "rules"),
    groups: properties.filter(([key, schema]) => key !== "rules" && isObject(schema)),
    scalars: properties.filter(([key, schema]) => key !== "rules" && !isObject(schema)),
  };
}

/** Names of the sections the method's config form shows, e.g. ["Quant rules", "Calibration", "Method options"]. */
export function configSections(manifest: Manifest): string[] {
  const { hasRules, groups, scalars } = sectionsOf(manifest);
  return [...(hasRules ? ["Quant rules"] : []), ...groups.map(([key]) => humanize(key)), ...(scalars.length ? ["Method options"] : [])];
}

/** Whether the method takes the quantization rules shared by every method of its Field that `uses: quant_rules`. */
export function hasFieldConfig(manifest: Manifest): boolean {
  return sectionsOf(manifest).hasRules;
}

/** Whether a validation path points into the quantization rules, e.g. "method.config.rules[0].sites". */
/** Error message per spec path, with list indices written as `.0` (the server mixes `rules[0]` and `rules.0`). */
export type Problems = Map<string, string>;

export function normalizePath(path: string): string {
  return path.replace(/\[(\d+)\]/g, ".$1");
}

export function problemAt(problems: Problems, path: string): string | undefined {
  return problems.get(normalizePath(path));
}

/** Messages for `prefix` and every path under it, excluding the paths `except` already shows elsewhere. */
function problemsUnder(problems: Problems, prefix: string, except: (path: string) => boolean = () => false): string[] {
  const base = normalizePath(prefix);
  return [...new Set([...problems].filter(([path]) => (path === base || path.startsWith(`${base}.`)) && !except(path)).map(([, m]) => m))];
}

export function isRulesPath(path: string): boolean {
  return path === "method.config.rules" || path.startsWith("method.config.rules[") || path.startsWith("method.config.rules.");
}

/** Effective quantization rules: the user's own, else the preset's. */
export function effectiveRules(config: Config, preset: Preset | null): Rule[] {
  return (config.rules ?? presetValue(preset, "rules")?.value ?? []) as Rule[];
}

/** Number of config values set by the user rather than inherited from the preset (a group counts each of its fields). */
function changedCount(config: Config): number {
  return Object.values(config).reduce<number>((n, v) => n + (v && typeof v === "object" && !Array.isArray(v) ? Object.keys(v).length : 1), 0);
}

function show(value: unknown): string {
  if (value === undefined || value === null) return "—";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

/** One readable line for a rule, e.g. "linear.*.weight · INT4 · Asymmetric · per channel · layers all". */
export function ruleSentence(rule: Rule, t: (s: string) => string): string {
  const q = rule.quant;
  const target = ruleTarget(rule.sites);
  const parts = [target === "custom" ? rule.sites.join(", ") || "—" : t(TARGETS[target].label), formatLabel(q)];
  if (q.type === "int" && q.symmetric !== undefined) parts.push(q.symmetric ? t("Symmetric") : t("Asymmetric"));
  if (q.type === "mx") parts.push(`${t("Block size")} ${q.block_size}`, String(q.scale).toUpperCase());
  if (q.granularity) parts.push(String(q.granularity).replace("_", " ") + (q.group_size ? ` g${q.group_size}` : "") + (q.group_axis === "token" ? ` ${t("along tokens")}` : ""));
  if (q.residual) parts.push(`${t("Residual tokens")} ${q.residual}`);
  parts.push(`${t("Layers")} ${rule.layers ?? "all"}`);
  return parts.join(" · ");
}

/**
 * A method's config: a read-only summary of the effective values (preset plus changes); "Edit" opens every shown section at once.
 * `part` selects the sections: "field" the quant rules shared across the Field, "method" the method's own groups and options, "all" both.
 */
export function MethodConfig({ manifest, preset, config, onChange, invalidPaths, problems = new Map(), sites, graph, tied, numLayers, part = "all" }: {
  manifest: Manifest;
  preset: Preset | null;
  config: Config;
  onChange: (config: Config) => void;
  invalidPaths: Set<string>;
  problems?: Problems;
  sites: string[];
  /** The model's architecture graph; when present, rule targets can be picked on a diagram. */
  graph?: Architecture["graph"];
  tied?: boolean;
  /** Decoder depth of the model; labels the layer stack on the diagram. */
  numLayers?: number;
  part?: "all" | "method" | "field";
}) {
  const t = useT();
  const [editing, setEditing] = useState(false);
  const root = manifest.config_schema;
  const sections = sectionsOf(manifest);
  const { properties } = sections;
  const hasRules = sections.hasRules && part !== "method";
  const groups = part === "field" ? [] : sections.groups;
  const scalars = part === "field" ? [] : sections.scalars;
  const keys = [...(hasRules ? ["rules"] : []), ...groups.map(([key]) => key), ...scalars.map(([key]) => key)];
  const set = (key: string, value: unknown) => {
    const next = { ...config };
    if (value === undefined) delete next[key];
    else next[key] = value;
    // Keep the editor open while the user works in it, even once the error that opened it is fixed.
    setEditing(true);
    onChange(next);
  };
  if (properties.length === 0) return <p className="faint small">{t("This method has no settings.")}</p>;
  if (keys.length === 0) return <p className="faint small">{t("This method has no settings of its own.")}</p>;
  const rulesPreset = presetValue(preset, "rules");
  const rules = effectiveRules(config, preset);
  const changed = changedCount(Object.fromEntries(Object.entries(config).filter(([key]) => keys.includes(key))));
  const restore = () => onChange(Object.fromEntries(Object.entries(config).filter(([key]) => !keys.includes(key))));
  const hasInvalid = [...invalidPaths].some((p) => p.startsWith("method.config")
    && (part === "all" || (part === "field") === (sections.hasRules && isRulesPath(p))));
  const open = editing || hasInvalid;
  const effective = (key: string, sub?: string) => {
    const own = sub ? (config[key] as Config | undefined)?.[sub] : config[key];
    const path = sub ? `${key}.${sub}` : key;
    const schema = sub ? resolveRef(resolveRef((root.properties ?? {})[key], root).properties?.[sub] ?? {}, root) : resolveRef((root.properties ?? {})[key], root);
    return { value: own ?? presetValue(preset, path)?.value ?? schema.default, changed: own !== undefined };
  };
  const setRules = (next: Rule[]) => set("rules", next);
  const addRule = (target: Target) => setRules([...rules, newRule(target)]);
  // Weights, activations and KV cache each get at most one rule; custom rules are unrestricted.
  const used = new Set(rules.map((r) => ruleTarget(r.sites)).filter((target) => target !== "custom"));
  const takenBy = (index: number) => new Set(rules.filter((_, i) => i !== index).map((r) => ruleTarget(r.sites)).filter((target) => target !== "custom"));

  return (
    <div className="method-config">
      <div className="row method-config-head">
        <span className="small muted">{preset ? t("Values come from preset {name}; hover ? on a field for its source.", { name: preset.name }) : t("No preset: every value is set here.")}</span>
        {changed > 0 && <Badge tone="orange">{t("{n} changed", { n: changed })}</Badge>}
        <span className="spacer" />
        {changed > 0 && preset && <Button variant="link" icon={<RotateCcw size={12} />} onClick={restore}>{t("Restore preset")}</Button>}
        {!hasInvalid && (
          <Button size="sm" variant={open ? "primary" : "secondary"} icon={open ? <Check size={13} /> : <Pencil size={13} />} onClick={() => setEditing(!open)}>
            {open ? t("Done") : t("Edit config")}
          </Button>
        )}
      </div>

      {!open && (
        <dl className="config-summary">
          {hasRules && (
            <>
              <dt>{t("Quant rules")}</dt>
              <dd>
                {rules.length === 0 ? <span className="faint">{t("none")}</span>
                  : rules.map((rule, i) => <div key={i} className={`mono text-xs ${config.rules !== undefined ? "changed" : ""}`}>{ruleSentence(rule, t)}</div>)}
              </dd>
            </>
          )}
          {groups.map(([key, schema]) => (
            <Fragment key={key}>
              <dt>{t(humanize(key))}</dt>
              <dd className="row gap-2">
                {Object.keys(schema.properties ?? {}).map((sub) => {
                  const v = effective(key, sub);
                  return <span key={sub} className={`value-chip ${v.changed ? "changed" : ""}`}><span className="faint">{t(humanize(sub))}</span> <span className="mono">{show(v.value)}</span></span>;
                })}
              </dd>
            </Fragment>
          ))}
          {scalars.length > 0 && (
            <>
              <dt>{t("Method options")}</dt>
              <dd className="row gap-2">
                {scalars.map(([key]) => {
                  const v = effective(key);
                  return <span key={key} className={`value-chip ${v.changed ? "changed" : ""}`}><span className="faint">{t(humanize(key))}</span> <span className="mono">{show(v.value)}</span></span>;
                })}
              </dd>
            </>
          )}
        </dl>
      )}

      {open && (
        <div className="stack">
          {hasRules && (
            <section className="config-section">
              <div className="row config-section-head">
                <h4>{t("Quant rules")}</h4>
                <Help text={root.$defs?.QuantRule?.description} />
                <span className="spacer" />
                {config.rules !== undefined && rulesPreset && <Button variant="link" icon={<RotateCcw size={12} />} onClick={() => set("rules", undefined)}>{t("Restore preset")}</Button>}
              </div>
              {problemAt(problems, "method.config.rules") && <div className="field-error section-error">{problemAt(problems, "method.config.rules")}</div>}
              {rules.map((rule, index) => {
                const base = `method.config.rules.${index}`;
                const elsewhere = problemsUnder(problems, base, (path) => RULE_FIELD_PATH.test(path.slice(base.length)));
                return (
                <div key={index} className={`rule-block ${problemsUnder(problems, base).length ? "invalid" : ""}`}>
                  <div className="row nowrap-row rule-block-head">
                    <span className="strong mono no-shrink">{ruleLabel(rule)}</span>
                    <span className="faint text-xs truncate grow">{ruleSentence(rule, t)}</span>
                    <Button variant="ghost" size="sm" icon={<Trash2 size={13} />} aria-label={t("Delete rule")} title={t("Delete rule")}
                      onClick={() => setRules(rules.filter((_, i) => i !== index))} />
                  </div>
                  {elsewhere.map((message) => <div key={message} className="field-error mb-2">{message}</div>)}
                  <RulePanel rule={rule} sites={sites} graph={graph} tied={tied} numLayers={numLayers} diagramOpen={index === 0} defs={root.$defs ?? {}} taken={takenBy(index)} problems={problems} base={base}
                    onChange={(next) => setRules(rules.map((r, i) => (i === index ? next : r)))} />
                </div>
                );
              })}
              <div className="row gap-2">
                <Button size="sm" icon={<Plus size={13} />} disabled={used.has("weights")} title={used.has("weights") ? t("There is already a rule for this") : undefined} onClick={() => addRule("weights")}>{t("Add weight rule")}</Button>
                <Button size="sm" icon={<Plus size={13} />} disabled={used.has("activations")} title={used.has("activations") ? t("There is already a rule for this") : undefined} onClick={() => addRule("activations")}>{t("Add activation rule")}</Button>
                <Button size="sm" icon={<Plus size={13} />} disabled={used.has("kv")} title={used.has("kv") ? t("There is already a rule for this") : undefined} onClick={() => addRule("kv")}>{t("Add KV cache rule")}</Button>
                <Button size="sm" variant="ghost" icon={<Plus size={13} />} onClick={() => addRule("custom")}>{t("Add custom rule")}</Button>
              </div>
            </section>
          )}
          {groups.map(([key, schema]) => {
            const value = (config[key] ?? {}) as Config;
            return (
              <section key={key} className="config-section">
                <div className="row config-section-head"><h4>{t(humanize(key))}</h4><Help text={schema.description} /></div>
                <div className="grid-3">
                  {Object.entries(schema.properties ?? {}).map(([sub, subSchema]) => (
                    <ScalarField key={sub} name={sub} schema={resolveRef(subSchema, root)} value={value[sub]}
                      preset={presetValue(preset, `${key}.${sub}`)}
                      invalid={invalidPaths.has(`method.config.${key}.${sub}`) || invalidPaths.has(`method.config.${key}`)}
                      error={problemAt(problems, `method.config.${key}.${sub}`) ?? problemAt(problems, `method.config.${key}`)}
                      onChange={(v) => {
                        const next = { ...value };
                        if (v === undefined) delete next[sub];
                        else next[sub] = v;
                        set(key, Object.keys(next).length ? next : undefined);
                      }} />
                  ))}
                </div>
              </section>
            );
          })}
          {scalars.length > 0 && (
            <section className="config-section">
              <div className="row config-section-head"><h4>{t("Method options")}</h4></div>
              <div className="grid-3">
                {scalars.map(([key, schema]) => (
                  <ScalarField key={key} name={key} schema={schema} value={config[key]} preset={presetValue(preset, key)}
                    invalid={invalidPaths.has(`method.config.${key}`)} error={problemAt(problems, `method.config.${key}`)} onChange={(v) => set(key, v)} />
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

/** One input for a schema field: a select for enums and booleans, a comma-separated list for number arrays, else a text or number box. */
export function ScalarField({ name, label: title, schema, value, preset, invalid, error, onChange }: {
  name: string;
  label?: string;
  schema: JsonSchema;
  value: unknown;
  preset?: { value: unknown; source: string };
  invalid: boolean;
  error?: string;
  onChange: (value: unknown) => void;
}) {
  const t = useT();
  const fallback = preset ? String(preset.value) : schema.default !== undefined ? String(schema.default) : "";
  const help = [schema.description, preset && `${t("Source")}: ${preset.source}`].filter(Boolean).join("\n\n");
  const label = <>{title ?? t(humanize(name))}<Help text={help || undefined} /></>;
  const reset = value !== undefined && preset ? (
    <span className="row gap-1"><Badge tone="orange">{t("changed")}</Badge><Button variant="link" onClick={() => onChange(undefined)}>{t("Restore")}</Button></span>
  ) : undefined;
  const kind = schema.type ?? schema.anyOf?.find((s) => s.type && s.type !== "null")?.type;
  const options = schema.enum ?? (kind === "boolean" ? [true, false] : null);
  if (options) {
    return (
      <Field label={label} invalid={invalid} error={error} extra={reset}>
        <Select value={value === undefined ? "" : JSON.stringify(value)} onChange={(v) => onChange(v === "" ? undefined : JSON.parse(v))}
          placeholder={fallback || "—"}
          options={options.map((o) => ({ value: JSON.stringify(o), label: String(o) }))} />
      </Field>
    );
  }
  if (schema.type === "array") {
    return (
      <Field label={label} invalid={invalid} error={error} extra={reset}>
        <ListInput value={value as unknown[] | undefined} placeholder={Array.isArray(schema.default) ? schema.default.join(", ") : ""} onChange={onChange} />
      </Field>
    );
  }
  const numeric = kind === "integer" || kind === "number";
  return (
    <Field label={label} invalid={invalid} error={error} extra={reset}>
      <Input type={numeric ? "number" : "text"} value={value === undefined ? "" : String(value)} placeholder={fallback}
        step={kind === "integer" ? 1 : "any"}
        onChange={(e) => onChange(e.target.value === "" ? undefined : numeric ? Number(e.target.value) : e.target.value)} />
    </Field>
  );
}

/** Comma-separated numbers, committed when the box loses focus; empty means unset. */
function ListInput({ value, placeholder, onChange }: { value?: unknown[]; placeholder: string; onChange: (value: unknown) => void }) {
  const [text, setText] = useState(value?.join(", ") ?? "");
  const commit = () => {
    const parts = text.split(",").map((p) => p.trim()).filter(Boolean);
    onChange(parts.length ? parts.map((p) => (Number.isNaN(Number(p)) ? p : Number(p))) : undefined);
  };
  return <Input value={text} placeholder={placeholder} onChange={(e) => setText(e.target.value)} onBlur={commit}
    onKeyDown={(e) => e.key === "Enter" && commit()} />;
}

export interface Rule {
  sites: string[];
  layers?: string | number | (string | number)[];
  quant: Record<string, unknown>;
}

const DEFAULT_FORMATS: Record<string, Record<string, unknown>> = {
  int: { type: "int", bits: 4, symmetric: false, granularity: "per_channel" },
  fp: { type: "fp", encoding: "e4m3", granularity: "per_token" },
  mx: { type: "mx", element: "e2m1", block_size: 32, scale: "e8m0" },
};
type Target = "weights" | "activations" | "kv" | "custom";

/** Named site sets for the "What to quantize" picker, with the format a new rule of that kind starts from. */
const TARGETS: Record<Exclude<Target, "custom">, { label: string; sites: string[]; quant: Record<string, unknown> }> = {
  weights: { label: "Weights", sites: ["linear.*.weight"], quant: DEFAULT_FORMATS.int },
  activations: { label: "Activations", sites: ["linear.*.input"], quant: { type: "int", bits: 8, symmetric: true, granularity: "per_token" } },
  kv: { label: "KV cache", sites: ["attn.k", "attn.v"], quant: { type: "int", bits: 8, symmetric: true, granularity: "per_token" } },
};

/** The named target whose sites equal `sites` (order-insensitive), else "custom". */
function ruleTarget(sites: string[]): Target {
  const key = [...new Set(sites)].sort().join("|");
  const found = (Object.keys(TARGETS) as (keyof typeof TARGETS)[]).find((name) => [...TARGETS[name].sites].sort().join("|") === key);
  return found ?? "custom";
}

/** The starting format of `type` for a rule on `target`: the target's own default when it uses that type, else the
 * generic default with a granularity that fits (per channel for weights, per token for activations and KV cache). */
function defaultFormat(type: string, target: Target): Record<string, unknown> {
  if (target !== "custom" && TARGETS[target].quant.type === type) return { ...TARGETS[target].quant };
  const quant = { ...DEFAULT_FORMATS[type] };
  if (quant.granularity && target === "weights") quant.granularity = "per_channel";
  if (quant.granularity && (target === "activations" || target === "kv")) quant.granularity = "per_token";
  return quant;
}

function newRule(target: Target): Rule {
  if (target === "custom") return { sites: [], quant: { ...DEFAULT_FORMATS.int } };
  return { sites: [...TARGETS[target].sites], quant: { ...TARGETS[target].quant } };
}

const GRANULARITIES = ["per_tensor", "per_channel", "per_token", "per_group"].map((g) => ({ value: g, label: g.replace("_", " ") }));
const ENCODINGS = ["e5m2", "e4m3", "e3m2", "e2m3", "e2m1"].map((e) => ({ value: e, label: e.toUpperCase() }));

/** Paths below one rule that a field of RulePanel shows itself. */
const RULE_FIELD_PATH = /^\.(sites(\.\d+)?|layers|quant\.\w+)$/;

const globRegex = (pattern: string) => new RegExp(`^${pattern.split("*").map((part) => part.replace(/[.+?^$()|{}[\]\\]/g, "\\$&")).join(".*")}$`);

/** Every quantization site the diagram of `graph` offers: weights, linear inputs, attention operands, lm_head. */
export function diagramSites(graph: NonNullable<Architecture["graph"]>): string[] {
  const out: string[] = [];
  for (const n of graph.nodes) {
    if (n.kind === "linear" && n.canon) out.push(`linear.${n.canon}.weight`, `linear.${n.canon}.input`);
    if (n.out_site) out.push(n.out_site);
  }
  for (const [, , site] of graph.edges) if (site?.startsWith("attn.")) out.push(site);
  if (graph.globals) out.push("lm_head.weight");
  return [...new Set(out)];
}

/** Diagram sites that the rule's site patterns select. */
function selectedSites(patterns: string[], all: string[]): string[] {
  const matchers = patterns.map(globRegex);
  return all.filter((id) => matchers.some((re) => re.test(id)));
}

/** The shortest rule `sites` selecting exactly `selected`: a shortcut pattern when it covers all its sites, else explicit ids;
 * patterns of the old rule that match nothing on the diagram are kept. */
function compressSites(selected: Set<string>, all: string[], old: string[]): string[] {
  const out: string[] = [];
  const covered = new Set<string>();
  for (const pattern of ["linear.*.weight", "linear.*.input"]) {
    const ids = selectedSites([pattern], all);
    if (ids.length > 0 && ids.every((id) => selected.has(id))) {
      out.push(pattern);
      ids.forEach((id) => covered.add(id));
    }
  }
  for (const id of all) if (selected.has(id) && !covered.has(id)) out.push(id);
  const known = old.filter((p) => selectedSites([p], all).length === 0);
  return [...out, ...known];
}

function RulePanel({ rule, sites, graph, tied, numLayers, diagramOpen, taken, problems, base, onChange, defs }: {
  rule: Rule; sites: string[]; graph?: Architecture["graph"]; tied?: boolean; numLayers?: number; diagramOpen: boolean; taken: Set<Target>; problems: Problems; base: string; onChange: (rule: Rule) => void; defs: Defs;
}) {
  const sitesError = problemAt(problems, `${base}.sites`) ?? problemsUnder(problems, `${base}.sites`)[0];
  const t = useT();
  const suggestions = ["linear.*.weight", "linear.*.input", "linear.attn.*.weight", "linear.mlp.*.weight", ...sites];
  const [customPicked, setCustomPicked] = useState(false);
  const [showDiagram, setShowDiagram] = useState(diagramOpen);
  const diagramAll = graph && graph.nodes.length > 0 ? diagramSites(graph) : null;
  const available = new Set(sites);
  const unavailable = diagramAll
    ? Object.fromEntries(diagramAll.filter((id) => sites.length > 0 && !available.has(id)).map((id) => [id, t("This model does not expose this site")]))
    : {};
  const matched = ruleTarget(rule.sites);
  const target: Target = customPicked ? "custom" : matched;
  const pickTarget = (next: string) => {
    setCustomPicked(next === "custom");
    // A named target starts from that target's default format (weights and activations want different grids).
    if (next !== "custom") onChange({ ...rule, ...newRule(next as Target) });
  };
  const sitesHelp = defs.QuantRule?.properties?.sites?.description;
  return (
    <div className="stack gap-3">
      <div className="grid-2">
        <Field label={<>{t("What to quantize")}<Help text={sitesHelp} /></>} error={target === "custom" ? undefined : sitesError}
          help={target === "custom" ? undefined : <span className="mono">{t("Acts on {sites}", { sites: rule.sites.join(", ") })}</span>}>
          <Segmented value={target} onChange={pickTarget} options={[
            ...(["weights", "activations", "kv"] as const).map((value) => ({
              value, label: t(TARGETS[value].label), disabled: taken.has(value),
              title: taken.has(value) ? t("There is already a rule for this") : undefined,
            })),
            { value: "custom", label: t("Custom") },
          ]} />
        </Field>
        <Field label={<>{t("Layers")}<Help text={defs.QuantRule?.properties?.layers?.description} /></>} error={problemAt(problems, `${base}.layers`)}>
          <Input className="mono" value={rule.layers === undefined ? "" : String(rule.layers)} placeholder="all · 0-15 · 3 · re:[02468]"
            onChange={(e) => onChange({ ...rule, layers: e.target.value || undefined })} />
        </Field>
      </div>
      {diagramAll && graph && (
        <Field label={<>{t("Pick on the diagram")}<Help text={t("Click a weight box, an activation arrow or the KV cache to add or remove it from this rule. The shortcuts above select common sets.")} /></>}>
          {showDiagram ? (
            <>
              <ArchDiagram graph={graph} mode="targets" tied={tied} numLayers={numLayers} disabled={unavailable}
                value={selectedSites(rule.sites, diagramAll)}
                onChange={(next) => onChange({ ...rule, sites: compressSites(new Set(next), diagramAll, rule.sites) })} />
              <div><Button size="sm" variant="link" onClick={() => setShowDiagram(false)}>{t("Hide diagram")}</Button></div>
            </>
          ) : <div><Button size="sm" onClick={() => setShowDiagram(true)}>{t("Show diagram")}</Button></div>}
        </Field>
      )}
      {target === "custom" && (
        <Field label={<>{t("Applies to")}<Help text={sitesHelp} /></>} error={sitesError}>
          <datalist id="site-vocabulary">{suggestions.map((s) => <option key={s} value={s} />)}</datalist>
          {rule.sites.length > 0 && (
            <div className="row gap-1">
              {rule.sites.map((s) => (
                <span key={s} className="badge mono site-chip">{s}
                  <button className="btn btn-ghost btn-sm site-chip-remove" aria-label={t("Remove {site}", { site: s })}
                    onClick={() => onChange({ ...rule, sites: rule.sites.filter((x) => x !== s) })}><X size={11} /></button>
                </span>
              ))}
            </div>
          )}
          <Input list="site-vocabulary" className="mono" placeholder={t("Type a site, e.g. attn.q, then Enter")}
            onKeyDown={(e) => {
              const input = e.currentTarget;
              if (e.key === "Enter" && input.value.trim()) {
                onChange({ ...rule, sites: [...new Set([...rule.sites, input.value.trim()])] });
                input.value = "";
              }
            }} />
        </Field>
      )}
      <Field label={t("Number format")} error={problemAt(problems, `${base}.quant.type`)}>
        <Segmented value={String(rule.quant.type)} onChange={(v) => onChange({ ...rule, quant: defaultFormat(v, target) })}
          options={[{ value: "int", label: "INT" }, { value: "fp", label: "FP (eXmY)" }, { value: "mx", label: "MX" }]} />
      </Field>
      <FormatFields quant={rule.quant} defs={defs} sequence={rule.sites.length > 0 && rule.sites.every((s) => s === "attn.k" || s === "attn.v")} error={(field) => problemAt(problems, `${base}.quant.${field}`)} onChange={(quant) => onChange({ ...rule, quant })} />
    </div>
  );
}

function FormatFields({ quant, onChange, defs, error, sequence }: {
  quant: Record<string, unknown>; onChange: (q: Record<string, unknown>) => void; defs: Defs; error: (field: string) => string | undefined;
  sequence: boolean;
}) {
  const t = useT();
  const help = (field: string) => defs[FORMAT_DEFS[String(quant.type)]]?.properties?.[field]?.description;
  const set = (key: string, value: unknown) => {
    const next = { ...quant };
    if (value === undefined || value === "") delete next[key];
    else next[key] = value;
    onChange(next);
  };
  const granularity = (
    <Field label={<>{t("Granularity")}<Help text={help("granularity")} /></>} error={error("granularity")}><Select value={String(quant.granularity)} onChange={(v) => set("granularity", v)} options={GRANULARITIES} /></Field>
  );
  const groupSize = quant.granularity === "per_group" && (
    <Field label={<>{t("Group size")}<Help text={help("group_size")} /></>} error={error("group_size")}><Input type="number" value={String(quant.group_size ?? "")} onChange={(e) => set("group_size", e.target.value ? Number(e.target.value) : undefined)} /></Field>
  );
  const groupAxis = sequence && quant.granularity === "per_group" && (
    <Field label={<>{t("Group axis")}<Help text={help("group_axis")} /></>} error={error("group_axis")}>
      <Select value={String(quant.group_axis ?? "")} onChange={(v) => set("group_axis", v || undefined)} placeholder="channel"
        options={[{ value: "channel", label: t("Features of a token") }, { value: "token", label: t("Tokens of a feature") }]} />
    </Field>
  );
  const residual = sequence && (
    <Field label={<>{t("Residual tokens")}<Help text={help("residual")} /></>} error={error("residual")}><Input type="number" min={1} value={String(quant.residual ?? "")} onChange={(e) => set("residual", e.target.value ? Number(e.target.value) : undefined)} /></Field>
  );
  if (quant.type === "int") {
    return (
      <div className="grid-3">
        <Field label={<>{t("Bits")}<Help text={help("bits")} /></>} error={error("bits")}><Input type="number" min={2} max={8} value={String(quant.bits)} onChange={(e) => set("bits", Number(e.target.value))} /></Field>
        <Field label={<>{t("Grid")}<Help text={help("symmetric")} /></>} error={error("symmetric")}>
          <Select value={String(quant.symmetric)} onChange={(v) => set("symmetric", v === "true")}
            options={[{ value: "false", label: t("Asymmetric") }, { value: "true", label: t("Symmetric") }]} />
        </Field>
        {granularity}
        {groupSize}
        {groupAxis}
        {residual}
        {quant.symmetric === false && (
          <Field label={<>{t("Zero point")}<Help text={help("zero_point")} /></>} error={error("zero_point")}>
            <Select value={String(quant.zero_point ?? "")} onChange={(v) => set("zero_point", v || undefined)} placeholder="integer"
              options={[{ value: "integer", label: t("Integer") }, { value: "float", label: t("Float (min-max offset)") }]} />
          </Field>
        )}
        <Field label={<>{t("Clip ratio")}<Help text={help("clip_ratio")} /></>} error={error("clip_ratio")}><Input type="number" step="0.01" value={String(quant.clip_ratio ?? "")} placeholder="1.0" onChange={(e) => set("clip_ratio", e.target.value ? Number(e.target.value) : undefined)} /></Field>
      </div>
    );
  }
  if (quant.type === "fp") {
    return (
      <div className="grid-3">
        <Field label={<>{t("Encoding")}<Help text={help("encoding")} /></>} error={error("encoding")}><Select value={String(quant.encoding)} onChange={(v) => set("encoding", v)} options={ENCODINGS} /></Field>
        <Field label={<>{t("Special values")}<Help text={help("special")} /></>} error={error("special")}>
          <Select value={String(quant.special ?? "")} onChange={(v) => set("special", v || undefined)} placeholder="auto"
            options={[{ value: "ieee", label: t("IEEE (inf / NaN)") }, { value: "fn", label: t("FN (NaN only)") }, { value: "none", label: t("None (all finite)") }]} />
        </Field>
        {granularity}
        {groupSize}
        {groupAxis}
        {residual}
      </div>
    );
  }
  return (
    <div className="grid-3">
      <Field label={<>{t("Element")}<Help text={help("element")} /></>} error={error("element")}><Select value={String(quant.element)} onChange={(v) => set("element", v)} options={ENCODINGS} /></Field>
      <Field label={<>{t("Block size")}<Help text={help("block_size")} /></>} error={error("block_size")}><Input type="number" value={String(quant.block_size)} onChange={(e) => set("block_size", Number(e.target.value))} /></Field>
      <Field label={<>{t("Scale")}<Help text={help("scale")} /></>} error={error("scale")}>
        <Select value={String(quant.scale)} onChange={(v) => set("scale", v)}
          options={[{ value: "e8m0", label: t("E8M0 · OCP MX") }, { value: "e4m3", label: t("E4M3 · NVFP4-style") }]} />
      </Field>
    </div>
  );
}
