import * as yaml from "js-yaml";
import { ArrowLeft, ArrowRight, CircleAlert, CircleCheck, Download, FileCode2, RotateCcw, Sparkles, Terminal } from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, fieldTitle, methodTitle, type Describe, type Issue, type Rendered, type TaskRef } from "../api";
import { PaperDiff, useApp } from "../components/common";
import { AblationSettings, deriveAblation, type AblationDraft } from "../compose/Ablation";
import { DestinationSelect, useDestination } from "./Projects";
import { useT } from "../i18n";
import { ModelPicker } from "../components/ModelPicker";
import { MethodConfig, effectiveRules, hasFieldConfig, isRulesPath, normalizePath, problemAt, ruleSentence, type Problems } from "../compose/MethodConfig";
import { Decoding, dropIncompatibleDraft, getIn, providerLabel, rolesOf, specDraftProblem, usesDecoding } from "../compose/Decoding";
import { alignDraftDtype, draftProblems } from "../compose/draftRules";
import { Preprocess, effectiveSteps, preprocessLabel, preprocessLines } from "../compose/Preprocess";
import { StepBar, type StepItem } from "../compose/StepBar";
import { TaskConfigStep } from "../compose/TaskConfigStep";
import { TasksStep, selectedTasks } from "../compose/TasksStep";
import { Robustness, repeatsOf, seedsText } from "../compose/Robustness";
import { ReusePanel, WarningsPanel } from "../compose/ReviewChecks";
import { runWarnings } from "../compose/warnings";
import { formatLimit } from "../compose/TaskSettings";
import { MODE_LABEL } from "../compose/taskIndex";
import { Button, Field, Input, Segmented, Select, Textarea } from "../ui/controls";
import { Badge, Card, CommandSteps, IssueList, PageHeader, formatParams, formatSeconds } from "../ui/display";
import { Dialog, useToast } from "../ui/overlay";

type Spec = Record<string, any>;
type StepId = "field" | "model" | "preprocess" | "method" | "fieldConfig" | "tasks" | "taskConfig" | "run" | "ablation" | "review";

const SPEC_KEY = "effeval.compose.v2";
const STEP_KEY = "effeval.compose.step";
const STEP_LABELS: Record<StepId, string> = {
  model: "Model", field: "Field", preprocess: "Model preprocessing", method: "Method", fieldConfig: "Field config", tasks: "Tasks",
  taskConfig: "Task settings", run: "Run settings", ablation: "Ablation", review: "Review",
};
const ALL_STEPS: StepId[] = ["field", "model", "preprocess", "method", "fieldConfig", "tasks", "taskConfig", "run", "ablation", "review"];

/** The wizard's steps for a mode and Field: the preprocessing step only for quantization, the ablation step only in ablation mode. */
function stepsFor(kind: string, field: string): StepId[] {
  return ALL_STEPS.filter((s) => (s !== "preprocess" || field === "quantization") && (s !== "ablation" || kind === "ablation"));
}
const GENERATION = [["max_new_tokens", "Max new tokens"], ["batch_size", "Batch size"], ["temperature", "Temperature"]] as const;

/** A blank spec: no model chosen yet; the first field and its first builtin method with a paper preset. */
/** The method a Field starts with: its first built-in method (preferring one with a default preset), else the Field-independent base. */
function defaultMethod(describe: Describe, field: string): Spec {
  const methods = Object.values(describe.methods).filter((m) => m.field === field && m.builtin && m.name !== "base");
  const method = methods.find((m) => m.default_preset) ?? methods[0];
  if (!method) return { name: "base" };
  return { name: method.name, preset: method.default_preset ? "default" : null, config: {} };
}

function initialSpec(describe: Describe): Spec {
  const field = Object.keys(describe.fields).sort()[0] ?? "";
  return { model: { name: "" }, method: defaultMethod(describe, field), field, tasks: [] };
}

function setIn(obj: Spec, path: string[], value: unknown): Spec {
  const [head, ...rest] = path;
  const next = { ...obj };
  if (rest.length === 0) {
    if (value === undefined || value === "") delete next[head];
    else next[head] = value;
  } else {
    const child = setIn(obj[head] ?? {}, rest, value);
    if (Object.keys(child).length === 0) delete next[head];
    else next[head] = child;
  }
  return next;
}

/** The wizard step a validation path belongs to, or null when no single step owns it. */
/** A plain-language reading of a validation issue: what is wrong and what to do; the server's text otherwise. */
function explain(issue: Issue, t: ReturnType<typeof useT>): string {
  const path = issue.paths[0] ?? "";
  const what = isRulesPath(path) ? t("Quant rules") : path.split(".").pop() ?? path;
  switch (issue.code) {
    case "preset.required":
      return t("No preset is selected, so “{what}” has no value. Set it here, or use a preset again.", { what });
    case "tasks.required":
      return t("No task is selected. Pick at least one task.");
    case "site.unknown":
      return t("This site does not exist in the model. Pick one from the suggestions in the box.");
    case "method.config_invalid":
      return t("Invalid value: {detail}", { detail: issue.message.replace(/^[\w.-]+: /, "") });
    case "model.unknown":
      return t("This model is not registered.");
    case "method.unknown":
      return t("This method is not registered.");
    case "preset.unknown":
      return t("This method has no such preset.");
    case "task.unknown":
      return t("This task is not available.");
    case "method.field_mismatch":
      return t("This method belongs to a different Field.");
    case "method.architecture_unsupported":
      return t("This method does not support the model's architecture.");
    case "decoding.draft_required":
      return t("Pick a draft model.");
    case "decoding.k_required":
      return t("Set how many tokens the draft proposes per step (1 to 16).");
    case "decoding.draft_unknown_model":
      return t("This draft model is not registered.");
    case "decoding.vocab_mismatch":
      return t("The draft model must have the same vocabulary as the target model.");
    case "decoding.draft_method_later":
      return t("The draft runs as a base model: methods and preprocessing on the draft are not supported yet.");
    case "decoding.sampling_unavailable":
      return t("Speculative decoding samples on speed tasks only; set do_sample to false for other tasks.");
    case "decoding.no_effect":
      return t("This task is scored by loglikelihood and never decodes, so speculative decoding has no effect. Pick a generation or speed task.");
    case "decoding.batch_size":
      return t("Speculative and timed runs decode one sequence at a time; set Batch size to 1.");
    case "runtime.sampling_filters":
      return t("SGLang samples plain decoding and draft verification differently when top_k and top_p are both set; set only one.");
    case "runtime.draft_dtype":
      return t("vLLM and SGLang run the draft in the target's precision; give both the same precision.");
    case "runtime.single_gpu":
      return t("vLLM and SGLang runs use one GPU; set the GPU count to 1.");
    case "runtime.method_unsupported":
      return t("vLLM and SGLang run plain checkpoints; methods and preprocessing need the transformers provider.");
    case "runtime.task_unsupported":
      return t("vLLM and SGLang run speed tasks (speed/…) only; use transformers for other tasks.");
    case "speed.hardware_required":
      return t("Timed tasks need a Target GPU, so results are comparable. Set it in Run settings.");
    case "mode.task_unsupported":
      return t("This task cannot run in the chosen mode; pick another mode.");
    default:
      return issue.message;
  }
}

function stepOfPath(path: string, rulesInFieldStep: boolean): StepId | null {
  if (path === "model" || path.startsWith("model.")) return "model";
  if (path === "field") return "field";
  if (rulesInFieldStep && isRulesPath(path)) return "fieldConfig";
  if (path === "preprocess" || path.startsWith("preprocess.")) return "preprocess";
  if (path === "method" || path.startsWith("method.")) return "method";
  if (/^tasks\[\d+\]\.(mode|params)/.test(path)) return "taskConfig";
  if (path === "tasks" || path.startsWith("tasks")) return "tasks";
  if (path === "decoding.draft" || path.startsWith("decoding.draft.")) return "model";
  if (/^(decoding|runtime)(\.|$)/.test(path)) return "fieldConfig";
  if (/^(evaluator|hardware|generation|determinism|reuse|repeats)(\.|$)/.test(path)) return "run";
  if (path === "sweep" || path.startsWith("sweep.")) return "ablation";
  return null;
}

/** One group of the review step: a titled box with a link back to the step that sets it. */
function ReviewGroup({ title, onEdit, children }: { title: ReactNode; onEdit: () => void; children: ReactNode }) {
  const t = useT();
  return (
    <section className="config-section">
      <div className="row config-section-head"><h4>{title}</h4><span className="spacer" /><Button variant="link" onClick={onEdit}>{t("Edit")}</Button></div>
      {children}
    </section>
  );
}

export function Compose() {
  const { describe } = useApp();
  const t = useT();
  const [spec, setSpec] = useState<Spec>(() => JSON.parse(localStorage.getItem(SPEC_KEY) ?? "null") ?? initialSpec(describe));
  const [serverIssues, setIssues] = useState<Issue[]>([]);
  // The user emptied the task list on purpose: with the preset's tasks available, an empty `spec.tasks` would otherwise read as "use the preset's".
  const [tasksCleared, setTasksCleared] = useState(false);
  const [checking, setChecking] = useState(false);
  const toast = useToast();
  const [kind, setKind] = useState<"single" | "ablation">("single");
  const [view, setView] = useState<"form" | "yaml">("form");
  const [yamlText, setYamlText] = useState("");
  const [yamlError, setYamlError] = useState("");
  const [rendered, setRendered] = useState<Rendered | null>(null);
  const [shards, setShards] = useState(1);
  const [to, setTo] = useDestination();
  const [busy, setBusy] = useState(false);
  const [draft, setDraft] = useState<AblationDraft>({ param: "", text: "" });
  const [stepId, setStepId] = useState<StepId>(() => (localStorage.getItem(STEP_KEY) as StepId | null) ?? "field");
  const [visited, setVisited] = useState<Set<StepId>>(() => new Set(ALL_STEPS.slice(0, ALL_STEPS.indexOf(stepId) + 1)));
  // Steps the user has moved away from; their problems are shown, the current step's stay quiet until it is left.
  const [left, setLeft] = useState<Set<StepId>>(() => new Set(ALL_STEPS.slice(0, ALL_STEPS.indexOf(stepId))));
  const previousStep = useRef<StepId | null>(null);
  // The run step's heads-up list stays hidden until the user has left a field there.
  const [runTouched, setRunTouched] = useState(false);

  useEffect(() => {
    localStorage.setItem(SPEC_KEY, JSON.stringify(spec));
    if (!spec.model?.name) return setIssues([]);
    setChecking(true);
    const timer = setTimeout(async () => {
      try {
        const result = await api.validate(spec);
        setIssues(result.issues);
      } catch (e) {
        setIssues([{ code: "request.failed", severity: "error", paths: [], message: (e as Error).message, hint: "" }]);
      } finally {
        setChecking(false);
      }
    }, 350);
    return () => clearTimeout(timer);
  }, [spec]);

  const manifest = describe.methods[spec.method?.name];
  // method.preset: null means no preset; missing or "default" means the method's default preset.
  const presetName = spec.method?.preset === null ? null
    : (spec.method?.preset ?? "default") === "default" ? manifest?.default_preset : spec.method?.preset;
  const preset = presetName ? describe.presets[spec.method.name]?.[presetName] ?? null : null;
  const model = describe.models[spec.model?.name];
  const roles = rolesOf(describe, spec.field);
  const decodingField = usesDecoding(roles);
  const draftRole = roles.find((r) => r.path.startsWith("decoding."));
  const draftName: string = draftRole ? getIn(spec, `${draftRole.path}.name`) ?? "" : "";
  const architecture = model ? describe.architectures[model.architecture] : undefined;
  const update = (path: string, value: unknown) => setSpec((s) => setIn(s, path.split("."), value));
  const engine = !!spec.runtime?.name && spec.runtime.name !== "hf";
  const draftDtype: string | undefined = spec.decoding?.draft?.model?.dtype;
  useEffect(() => {
    if (decodingField) setSpec((s) => alignDraftDtype(describe.models, s));
  }, [decodingField, spec.runtime?.name, spec.model?.name, spec.model?.dtype, draftName]);
  const unavailableDrafts = useMemo(() => Object.fromEntries(
    Object.entries(draftProblems(describe.models, spec.model?.name, {})).map(([code, p]) => [code, t(p.text, p.vars)])), [describe.models, spec.model?.name, t]);
  const repeats = repeatsOf(spec);
  const presetTasks = (preset?.values.tasks?.value as string[] | undefined) ?? [];
  const { tasks, usingPreset: usingPresetTasks } = selectedTasks(spec.tasks, presetTasks, tasksCleared);
  const issues = useMemo<Issue[]>(() => tasksCleared && (spec.tasks ?? []).length === 0
    ? [...serverIssues, { code: "tasks.required", severity: "error", paths: ["tasks"], message: "no tasks given", hint: "" }]
    : serverIssues, [serverIssues, tasksCleared, spec.tasks]);
  const setTasks = (next: TaskRef[]) => { setTasksCleared(next.length === 0); update("tasks", next); };
  const restorePresetTasks = () => { setTasksCleared(false); update("tasks", []); };
  const fieldConfig = !!manifest && hasFieldConfig(manifest);
  const effectivePre = useMemo(() => (spec.preprocess ? { ...(spec.preprocess.preset ? { preset: spec.preprocess.preset } : {}), steps: effectiveSteps(spec.preprocess, describe) } : null),
    [spec.preprocess, describe]);
  const ablation = kind === "ablation" ? deriveAblation(spec, preset, draft, effectivePre) : null;
  const heedWarnings = useMemo(() => runWarnings(spec, tasks, describe, t), [spec, tasks, describe, t]);

  // Model, Field and Method gate the rest; every other step has defaults and can be passed with Next.
  const steps = stepsFor(kind, spec.field);
  const filled: Partial<Record<StepId, boolean>> = { model: roles.every((r) => !!getIn(spec, `${r.path}.name`)), field: !!describe.fields[spec.field], method: !!manifest };
  const reachable = (index: number) => steps.slice(0, index).every((s) => filled[s] !== false);
  const blocking = steps.findIndex((s) => filled[s] === false);
  const storedIndex = steps.indexOf(stepId);
  // A stored step missing here falls back: preprocessing (Field is not quantization) to the method step, ablation (single mode) to review.
  const wanted = storedIndex >= 0 ? storedIndex : steps.indexOf(stepId === "preprocess" ? "method" : "review");
  const current = reachable(wanted) ? wanted : Math.max(0, blocking);
  const step = steps[current];
  // An issue shows once the user has left a step it points at; on the review step every issue shows.
  const shown = issues.filter((i) => i.severity !== "info" && (view === "yaml" || step === "review"
    || i.paths.some((path) => { const owner = stepOfPath(path, fieldConfig); return owner !== null && left.has(owner); })));
  const errors = shown.filter((i) => i.severity === "error");
  const warnings = shown.filter((i) => i.severity === "warning");
  const invalid = useMemo(() => new Set(errors.flatMap((e) => e.paths)), [errors]);
  // The message to show under each field that a shown error points at.
  const problems: Problems = useMemo(() => {
    const map: Problems = new Map();
    for (const error of errors) for (const path of error.paths) if (!map.has(normalizePath(path))) map.set(normalizePath(path), explain(error, t));
    return map;
  }, [errors]);
  const err = (path: string) => problemAt(problems, path);
  const stepIssues = useMemo(() => {
    const out = new Map<StepId, "error" | "warning">();
    for (const issue of [...errors, ...warnings]) {
      for (const path of issue.paths) {
        const owner = stepOfPath(path, fieldConfig);
        if (owner && out.get(owner) !== "error") out.set(owner, issue.severity === "error" ? "error" : "warning");
      }
    }
    return out;
  }, [errors, warnings, fieldConfig]);
  const stepItems: StepItem[] = steps.map((id, index) => ({
    id,
    label: t(STEP_LABELS[id]),
    state: index === current ? "current" : (filled[id] ?? visited.has(id)) && id !== "review" ? "done" : "todo",
    reachable: reachable(index),
    issue: stepIssues.get(id),
  }));

  useEffect(() => {
    localStorage.setItem(STEP_KEY, step);
    setVisited((v) => (v.has(step) ? v : new Set(v).add(step)));
    const from = previousStep.current;
    previousStep.current = step;
    if (from && from !== step) setLeft((l) => (l.has(from) ? l : new Set(l).add(from)));
  }, [step]);

  const goto = (id: StepId) => setStepId(id);
  const canNext = filled[step] !== false && current < steps.length - 1;

  const render = async () => {
    setBusy(true);
    try {
      const result = await api.render(ablation ? { ...spec, sweep: ablation.sweep } : spec, shards, to);
      const failed = result.issues.filter((i) => i.severity === "error");
      if (!result.plan && failed.length) toast(t("{n} conflicts", { n: failed.length }), { tone: "error", issues: failed });
      warnDuplicates(toast, t, result.issues);
      setRendered(result);
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    } finally {
      setBusy(false);
    }
  };

  const switchMode = (next: string) => {
    if (next === "yaml") {
      setYamlText(yaml.dump(spec, { lineWidth: 100 }));
      setYamlError("");
      setView("yaml");
      return;
    }
    setView("form");
    if (next === "single" && step === "ablation") setStepId("review");
    setKind(next as "single" | "ablation");
  };

  // "ready" only when nothing is wrong anywhere, including steps whose problems are not shown yet.
  const hiddenErrors = issues.some((i) => i.severity === "error") && errors.length === 0;
  const status = checking ? <Badge>{t("checking…")}</Badge>
    : errors.length ? <Badge tone="red">{t("{n} conflicts", { n: errors.length })}</Badge>
      : spec.model?.name && !hiddenErrors ? <Badge tone="green">{t("ready")}</Badge> : null;

  const generate = (
    <>
      {(errors.length > 0 || warnings.length > 0) && <IssueList issues={issues} />}
      {errors.length === 0 && spec.model?.name && !checking && (
        <div className="row small muted"><CircleCheck size={15} color="var(--green)" />{t("No conflicts")}</div>
      )}
      <div className="row mt-3">
        <span className="spacer" />
        <Button variant="primary" size="lg" icon={<Terminal size={15} />} disabled={busy || checking || errors.length > 0 || !spec.model?.name} onClick={render}>
          {busy ? t("Generating…") : t("Generate command")}
        </Button>
      </div>
    </>
  );

  const stepTitle = t(STEP_LABELS[step]);
  // Errors of the current step, listed above it; fields they point at are also outlined.
  const stepErrors = step === "review" ? [] : errors.filter((e) => e.paths.some((path) => stepOfPath(path, fieldConfig) === step));
  let body: ReactNode;
  switch (step) {
    case "model":
      body = (
        <Card title={stepTitle}>
          <div className="grid-2">
            {roles.map((role) => {
              const isTarget = role.path === "model";
              const label = roles.length === 1 ? t("Model") : isTarget ? t("Target model") : role.name === "draft" ? t("Draft model") : role.name;
              const chosen: string = getIn(spec, `${role.path}.name`) ?? "";
              return (
                <Field key={role.path} label={label} error={err(`${role.path}.name`) ?? err(role.path)}
                  help={isTarget ? undefined : t("Models that cannot be the draft of this target are greyed out with the reason.")}>
                  <ModelPicker models={describe.models} value={chosen} unavailable={isTarget ? undefined : unavailableDrafts}
                    invalid={invalid.has(`${role.path}.name`)}
                    onChange={(code) => {
                      const next = dropIncompatibleDraft(describe.models, setIn(spec, `${role.path}.name`.split("."), code));
                      if (isTarget && draftName && getIn(next, `${draftRole!.path}.name`) === undefined) {
                        const why = specDraftProblem(describe.models, setIn(spec, `${role.path}.name`.split("."), code));
                        toast(t("Draft model cleared"), { tone: "warning", description: why ? t(why.text, why.vars) : undefined });
                      }
                      setSpec(() => next);
                    }} />
                  {!isTarget && engine && draftDtype && (
                    <div className="small muted mt-3">{t("Draft precision: {dtype} ({provider} runs the draft in the target's precision).", { dtype: draftDtype, provider: providerLabel(describe, spec.runtime?.name) })}</div>
                  )}
                </Field>
              );
            })}
            <Field label={t("Precision")}>
              <Select value={spec.model?.dtype ?? model?.dtype ?? "bfloat16"} disabled={!model}
                onChange={(v) => update("model.dtype", v === model?.dtype ? undefined : v)}
                options={["bfloat16", "float16", "float32"].map((d) => ({ value: d, label: d, hint: d === model?.dtype ? t("native") : undefined }))} />
            </Field>
          </div>
          {model && (
            <div className="row mt-3">
              <Badge mono>{model.checkpoint}</Badge>
              <Badge mono title={model.revision ?? undefined}>{model.revision ? model.revision.slice(0, 10) : t("unpinned")}</Badge>
              <Badge>{model.architecture}</Badge>
              <Badge>{formatParams(model.params)}</Badge>
              {model.tie_word_embeddings && <Badge tone="orange">{t("tied lm_head")}</Badge>}
            </div>
          )}
        </Card>
      );
      break;
    case "field":
      body = (
        <Card title={stepTitle} hint={t("What the evaluation tests")}>
          <div className="choice-grid">
            {Object.values(describe.fields).map((f) => (
              <button key={f.name} type="button" className={`choice ${spec.field === f.name ? "on" : ""}`} aria-pressed={spec.field === f.name}
                onClick={() => {
                  const next = selectField(describe, spec, f.name);
                  setSpec(next);
                  if (next.method !== spec.method) {
                    toast(t("Method changed to {method}", { method: next.method.name }), { tone: "warning", description: t("{from} belongs to a different Field than {field}.", { from: spec.method.name, field: f.title }) });
                  }
                }}>
                <span className="choice-title">{f.title}</span>
                <span className="small muted">{f.description}</span>
              </button>
            ))}
          </div>
        </Card>
      );
      break;
    case "method":
      body = (
        <Card title={stepTitle} hint={fieldTitle(describe, spec.field)}>
          <div className="grid-2">
            <Field label={t("Method")} error={err("method.name")}>
              <Select value={spec.method?.name ?? ""} onChange={(v) => setSpec((s) => ({ ...s, method: { name: v, preset: "default", config: {} } }))}
                options={Object.values(describe.methods).filter((m) => m.field === spec.field || m.field === "any")
                  .map((m) => ({ value: m.name, label: m.title || m.name, hint: m.builtin ? undefined : t("custom") }))} />
            </Field>
            <Field label={t("Preset")} error={err("method.preset")}>
              <Select value={spec.method?.preset === null ? "__none__" : (spec.method?.preset ?? "default") === "default" ? manifest?.default_preset ?? "__none__" : spec.method.preset}
                onChange={(v) => update("method.preset", v === "__none__" ? null : v === manifest?.default_preset ? "default" : v)}
                options={[
                  ...Object.keys(describe.presets[spec.method?.name] ?? {}).map((p) => ({ value: p, label: p, hint: p === manifest?.default_preset ? t("default") : undefined })),
                  { value: "__none__", label: t("None") },
                ]} />
            </Field>
          </div>
          {manifest?.description && <p className="small muted mt-3">{manifest.description}</p>}
          {preset && (
            <div className="row small mt-3 nowrap-row">
              <Sparkles size={14} color="var(--accent)" className="no-shrink" />
              <a href={`https://arxiv.org/abs/${preset.paper.arxiv}`} target="_blank" rel="noreferrer" className="truncate">{preset.paper.title}</a>
            </div>
          )}
          {manifest && (
            <>
              <div className="divider" />
              <MethodConfig part="method" manifest={manifest} preset={preset} config={spec.method?.config ?? {}} invalidPaths={invalid} problems={problems}
                sites={architecture?.sites ?? []} graph={architecture?.graph} tied={model?.tie_word_embeddings} numLayers={model?.shape?.num_layers}
                onChange={(config) => setSpec((s) => ({ ...s, method: { ...s.method, config } }))} />
            </>
          )}
        </Card>
      );
      break;
    case "preprocess":
      body = (
        <Card title={stepTitle} hint={t("Optional: rotation and / or scaling before quantization")}>
          <Preprocess value={spec.preprocess} modelCode={spec.model?.name ?? ""} problems={problems} method={spec.method}
            onPickQuantizer={(name, preset) => {
              setSpec((s) => ({ ...s, method: { name, preset } }));
              toast(t("Quantizer set to {method} · {preset}", { method: methodTitle(describe, name), preset }), { tone: "success", description: t("The preset defines the config; earlier method config values were cleared.") });
            }}
            onChange={(next) => setSpec((s) => { const { preprocess: _drop, ...rest } = s; return next ? { ...rest, preprocess: next } : rest; })} />
        </Card>
      );
      break;
    case "fieldConfig":
      body = (
        <Card title={stepTitle} hint={fieldTitle(describe, spec.field)}>
          {decodingField && <Decoding spec={spec} describe={describe} update={update} err={err} />}
          {decodingField && fieldConfig && <div className="divider" />}
          {fieldConfig ? (
            <MethodConfig part="field" manifest={manifest} preset={preset} config={spec.method?.config ?? {}} invalidPaths={invalid} problems={problems}
              sites={architecture?.sites ?? []} graph={architecture?.graph} tied={model?.tie_word_embeddings} numLayers={model?.shape?.num_layers} onChange={(config) => setSpec((s) => ({ ...s, method: { ...s.method, config } }))} />
          ) : decodingField ? null : (
            <p className="faint small">{t("Method {method} uses no settings shared across the Field {field}; nothing to set here.", { method: methodTitle(describe, spec.method?.name ?? ""), field: spec.field })}</p>
          )}
        </Card>
      );
      break;
    case "tasks":
      body = <TasksStep title={stepTitle} field={spec.field} specTasks={spec.tasks} presetTasks={presetTasks} cleared={tasksCleared} onChange={setTasks} onRestore={restorePresetTasks} />;
      break;
    case "taskConfig":
      body = <TaskConfigStep title={stepTitle} specTasks={spec.tasks} presetTasks={presetTasks} cleared={tasksCleared} invalid={invalid} runMaxNewTokens={spec.generation?.max_new_tokens ?? 512}
        runSampling={spec.generation?.do_sample === true} runTemperature={spec.generation?.temperature || undefined} onChange={setTasks} />;
      break;
    case "run":
      body = (
        <div onBlur={() => setRunTouched(true)}>
        <Card title={stepTitle}>
          <div className="grid-3">
            <Field label={t("Target GPU")} error={err("hardware.gpu")}>
              <Select value={spec.hardware?.gpu ?? ""} onChange={(v) => update("hardware.gpu", v)} placeholder={t("Not specified")}
                options={Object.keys(describe.gpus).map((g) => ({ value: g, label: g }))} />
            </Field>
            <Field label={t("GPUs")} error={err("hardware.count")}><Input type="number" min={1} value={spec.hardware?.count ?? ""} placeholder="1"
              onChange={(e) => update("hardware.count", e.target.value ? Number(e.target.value) : undefined)} /></Field>
            <Field label={t("Reuse results")} error={err("reuse")}>
              <Select value={spec.reuse ?? "auto"} onChange={(v) => update("reuse", v === "auto" ? undefined : v)}
                options={[{ value: "auto", label: t("Any GPU") }, { value: "same_hardware", label: t("Same GPU only") }, { value: "never", label: t("Never") }]} />
            </Field>
          </div>
          <div className="divider" />
          <Robustness spec={spec} setSpec={setSpec} update={update} err={err} />
          <div className="divider" />
          <div className="grid-3">
            <Field label={t("Batch size")} error={err("generation.batch_size")}>
              <Input type="number" min={1} value={spec.generation?.batch_size ?? ""} placeholder="8"
                onChange={(e) => update("generation.batch_size", e.target.value === "" ? undefined : Number(e.target.value))} />
            </Field>
            <Field label={t("Nodes")}>
              <Input type="number" min={1} value={shards} onChange={(e) => setShards(Math.max(1, Number(e.target.value) || 1))} />
            </Field>
            <Field label={t("Upload to")}><DestinationSelect value={to} onChange={setTo} /></Field>
          </div>
          {(runTouched || left.has("run")) && <WarningsPanel warnings={heedWarnings} />}
        </Card>
        </div>
      );
      break;
    case "ablation":
      body = (
        <Card title={stepTitle} hint={t("One variant per value; everything else as set in the previous steps")}>
          <AblationSettings spec={spec} preset={preset} pre={effectivePre} draft={draft} onChange={setDraft} />
        </Card>
      );
      break;
    case "review": {
      const rules = effectiveRules(spec.method?.config ?? {}, preset);
      body = (
        <>
          <Card title={stepTitle} actions={status}>
            <div className="grid-2">
              <ReviewGroup title={t("Model")} onEdit={() => goto("model")}>
                <dl className="kv">
                  <dt>{t("Model")}</dt><dd className="mono">{spec.model?.name || "—"}</dd>
                  <dt>{t("Precision")}</dt><dd className="mono">{spec.model?.dtype ?? model?.dtype ?? "—"}</dd>
                  {model && <><dt>{t("Checkpoint")}</dt><dd className="mono small">{model.checkpoint}</dd></>}
                  {draftRole && <><dt>{t("Draft model")}</dt><dd className="mono">{draftName || "—"}</dd></>}
                  {draftDtype && <><dt>{t("Draft precision")}</dt><dd className="mono">{draftDtype}</dd></>}
                </dl>
              </ReviewGroup>
              <ReviewGroup title={t("Field")} onEdit={() => goto("field")}>
                <dl className="kv"><dt>{t("Field")}</dt><dd>{spec.field || "—"}</dd></dl>
              </ReviewGroup>
              <ReviewGroup title={t("Method")} onEdit={() => goto("method")}>
                <dl className="kv">
                  <dt>{t("Method")}</dt><dd className="mono">{spec.method?.name ? methodTitle(describe, spec.method.name) : "—"}</dd>
                  <dt>{t("Preset")}</dt><dd>{presetName ?? t("None")}</dd>
                  {spec.preprocess && spec.method?.name && <><dt>{t("Run label")}</dt><dd className="mono">{preprocessLabel(spec.preprocess, describe, spec.method.name, t, spec.method.preset)}</dd></>}
                  {preset && <><dt>{t("Paper")}</dt><dd className="small"><a href={`https://arxiv.org/abs/${preset.paper.arxiv}`} target="_blank" rel="noreferrer">{preset.paper.title}</a></dd></>}
                </dl>
              </ReviewGroup>
              {spec.field === "quantization" && (
                <ReviewGroup title={t("Model preprocessing")} onEdit={() => goto("preprocess")}>
                  {!spec.preprocess ? <span className="faint small">{t("none")}</span> : (
                    <div className="stack-sm">
                      {preprocessLines(effectiveSteps(spec.preprocess, describe)).map((line) => <div key={line} className="mono text-xs">{line}</div>)}
                    </div>
                  )}
                </ReviewGroup>
              )}
              <ReviewGroup title={t("Field config")} onEdit={() => goto("fieldConfig")}>
                {decodingField && (
                  <dl className="kv">
                    <dt>{t("Provider")}</dt><dd>{providerLabel(describe, spec.runtime?.name)}</dd>
                    <dt>{t("Draft tokens per step (k)")}</dt><dd className="mono">{spec.decoding?.num_speculative_tokens ?? "—"}</dd>
                  </dl>
                )}
                {decodingField && !fieldConfig ? null : !fieldConfig ? <span className="faint small">{t("none")}</span>
                  : rules.length === 0 ? <span className="faint small">{t("none")}</span>
                    : <div className="stack-sm">{rules.map((rule, i) => <div key={i} className="mono text-xs">{ruleSentence(rule, t)}</div>)}</div>}
              </ReviewGroup>
              <ReviewGroup title={t("Tasks")} onEdit={() => goto("tasks")}>
                {tasks.length === 0 ? <span className="faint small">—</span> : (
                  <div className="stack-sm">
                    {usingPresetTasks && <span className="faint small">{t("from the preset")}</span>}
                    <div className="row gap-2">
                      {tasks.map((task) => (
                        <span key={task.name} className="value-chip"><span className="mono">{task.name}</span>
                          {task.mode && <span className="faint">{MODE_LABEL[task.mode] ?? task.mode}</span>}
                          {task.params?.num_fewshot !== undefined && <span className="faint">{t("Few-shot")} {String(task.params.num_fewshot)}</span>}
                          {task.params?.limit !== undefined && <span className="faint">{t("Samples")} {formatLimit(task.params.limit)}</span>}
                          {task.params?.do_sample === true && <span className="faint">{t("Sampling")}{task.params.temperature !== undefined ? ` ${String(task.params.temperature)}` : ""}</span>}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </ReviewGroup>
              <ReviewGroup title={t("Run settings")} onEdit={() => goto("run")}>
                <dl className="kv">
                  <dt>{t("Target GPU")}</dt><dd>{spec.hardware?.gpu ?? t("Not specified")}{spec.hardware?.count ? ` × ${spec.hardware.count}` : ""}</dd>
                  <dt>{t("Reuse results")}</dt><dd>{spec.reuse ?? "auto"}</dd>
                  {repeats.mode === "vary"
                    ? <><dt>{t("Seeds")}</dt><dd className="mono">{seedsText(repeats.seeds, 20)} · {t("{n} runs", { n: repeats.seeds.length })}</dd></>
                    : <>
                      {spec.determinism?.seed !== undefined && <><dt>{t("Seed")}</dt><dd className="mono">{spec.determinism.seed}</dd></>}
                      {repeats.count > 1 && <><dt>{t("Number of runs")}</dt><dd className="mono">{repeats.count} · {t("same seed")}</dd></>}
                    </>}
                  {GENERATION.filter(([key]) => spec.generation?.[key] !== undefined).map(([key, label]) => (
                    <Fragment key={key}><dt>{t(label)}</dt><dd className="mono">{spec.generation[key]}</dd></Fragment>
                  ))}
                  <dt>{t("Nodes")}</dt><dd className="mono">{shards}</dd>
                  <dt>{t("Upload to")}</dt><dd>{to === "private" ? t("Private") : to === "pool" ? t("Lab pool") : to.replace(/^project:/, `${t("Project")} · `)}</dd>
                </dl>
              </ReviewGroup>
              {kind === "ablation" && (
                <ReviewGroup title={t("Ablation")} onEdit={() => goto("ablation")}>
                  {ablation ? <span className="small"><span className="mono">{ablation.label}</span> · {t("{n} variants", { n: ablation.values.length })}</span>
                    : <span className="faint small">{t("No parameter chosen yet")}</span>}
                </ReviewGroup>
              )}
            </div>
          </Card>
          <Card>
            <WarningsPanel warnings={heedWarnings} />
            <ReusePanel spec={ablation ? { ...spec, sweep: ablation.sweep } : spec} />
          </Card>
          <Card>{generate}</Card>
        </>
      );
      break;
    }
  }

  return (
    <div className="page">
      <PageHeader
        title={t("New evaluation")}
        actions={<>
          {status}
          <Segmented value={view === "yaml" ? "yaml" : kind} onChange={switchMode}
            options={[{ value: "single", label: t("Single evaluation") }, { value: "ablation", label: t("Ablation") }, { value: "yaml", label: t("YAML") }]} />
          <Button variant="ghost" icon={<RotateCcw size={14} />} onClick={() => { setSpec(initialSpec(describe)); setStepId("field"); }}>{t("Reset")}</Button>
        </>}
      />
      {view === "yaml" ? (
        <>
          <Card title={t("RunSpec")}>
            <Textarea style={{ minHeight: 520 }} value={yamlText} onChange={(e) => setYamlText(e.target.value)}
              onBlur={() => {
                try {
                  const parsed = yaml.load(yamlText);
                  if (typeof parsed !== "object" || parsed === null) throw new Error(t("The document must be a mapping"));
                  setSpec(parsed as Spec);
                  setYamlError("");
                } catch (e) {
                  setYamlError((e as Error).message);
                }
              }} />
            {yamlError && <div className="issue error mt-3"><span className="issue-icon"><CircleAlert size={16} /></span><div className="issue-msg">{yamlError}</div></div>}
          </Card>
          <Card>{generate}</Card>
        </>
      ) : (
        <>
          <StepBar steps={stepItems} ariaLabel={t("Steps")} onSelect={(id) => goto(id as StepId)} />
          {stepErrors.length > 0 && (
            <div className="mb-3">
              {stepErrors.map((issue, index) => {
                const plain = explain(issue, t);
                return (
                  <div key={index} className="issue error">
                    <span className="issue-icon"><CircleAlert size={16} /></span>
                    <div className="grow">
                      <div className="issue-msg">{plain}</div>
                      {plain !== issue.message && <div className="issue-code mt-1">{issue.message}</div>}
                      {issue.code === "preset.required" && manifest?.default_preset && (
                        <Button size="sm" className="mt-2" onClick={() => update("method.preset", "default")}>{t("Use preset {name}", { name: manifest.default_preset })}</Button>
                      )}
                      {issue.code === "tasks.required" && step !== "tasks" && (
                        <Button size="sm" className="mt-2" onClick={() => goto("tasks")}>{t("Go to Tasks")}</Button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          {body}
          <div className="row step-nav">
            {current > 0 && <Button icon={<ArrowLeft size={14} />} onClick={() => goto(steps[current - 1])}>{t("Back")}</Button>}
            <span className="spacer" />
            {current < steps.length - 1 && (
              <Button variant="primary" disabled={!canNext} onClick={() => goto(steps[current + 1])}>
                {t("Next")}<ArrowRight size={14} />
              </Button>
            )}
          </div>
        </>
      )}

      <CommandDialog rendered={rendered} onClose={() => setRendered(null)} />
    </div>
  );
}


/** The spec after choosing a Field: only a Field with a draft role keeps a decoding and a provider, and starts as speculative; a method of another Field is replaced by the Field's default method. */
function selectField(describe: Describe, spec: Spec, field: string): Spec {
  const manifest = describe.methods[spec.method?.name];
  const foreign = !!manifest && manifest.field !== field && manifest.field !== "any";
  const { decoding: _drop, ...rest } = setIn(foreign ? { ...spec, method: defaultMethod(describe, field) } : spec, ["field"], field);
  if (usesDecoding(rolesOf(describe, field))) return { ...rest, decoding: spec.decoding ?? { mode: "speculative" } };
  return setIn(rest, ["runtime", "name"], undefined);
}

/** Posts a warning banner when members are already running or have claimed jobs of a rendered plan. */
export function warnDuplicates(toast: ReturnType<typeof useToast>, t: ReturnType<typeof useT>, issues: Issue[]) {
  const busy = issues.filter((i) => i.code === "run.in_progress" || i.code === "run.claimed");
  if (busy.length) toast(t("Someone is already on this"), { tone: "warning", issues: busy, key: "run.duplicate" });
}

/** Save the plan file that a command's write step creates, under the same name and with the same content. */
function downloadPlan(write: string) {
  const lines = write.trimEnd().split("\n");
  const name = /^cat > (\S+) <</.exec(lines[0])?.[1] ?? "effeval-plan.yaml";
  const url = URL.createObjectURL(new Blob([lines.slice(1, -1).join("\n") + "\n"], { type: "text/yaml" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: name });
  link.click();
  URL.revokeObjectURL(url);
}

export function CommandDialog({ rendered, onClose }: { rendered: Rendered | null; onClose: () => void }) {
  const { describe } = useApp();
  const t = useT();
  const [node, setNode] = useState("0");
  const plan = rendered?.plan;
  const main = plan?.jobs.find((j) => j.role === "main");
  const fidelity = main?.fidelity;
  const speculative = !!plan?.jobs.some((j) => j.spec.decoding);
  return (
    <Dialog open={!!rendered} onOpenChange={(open) => !open && onClose()} wide
      icon={<div className="icon-tile accent"><FileCode2 size={18} /></div>}
      footer={rendered && rendered.commands.length > 0 && <Button icon={<Download size={14} />} onClick={() => downloadPlan(rendered.commands[0].write)}>{t("Download config")}</Button>}
      title={plan && plan.jobs.length === 0 ? t("Nothing to run") : t("Command")}>
      {plan && (
        <div className="stack">
          <div className="stack">
            <div>
              <div className="field-label mb-2">{t("Jobs")}</div>
              <table className="list">
                <thead><tr><th className="col-sm">{t("Role")}</th><th>{t("Model")}</th><th>{t("Method")}</th>{speculative && <th>{t("Decoding")}</th>}<th>{t("Task")}</th><th className="col-sm">{t("Evaluator")}</th><th className="col-sm right">{t("Estimate")}</th></tr></thead>
                <tbody>
                  {plan.jobs.map((job) => {
                    const eta = rendered?.estimate?.jobs.find((e) => e.job_id === job.job_id);
                    return (
                      <tr key={job.job_id}>
                        <td><Badge tone={job.role === "main" ? "accent" : undefined}>{t(job.role)}</Badge></td>
                        <td className="small mono">{job.spec.model.name}</td>
                        <td className="small mono">{preprocessLabel(job.spec.preprocess, describe, job.spec.method.name, t, job.spec.method.preset)}</td>
                        {speculative && <td className="small mono">{job.spec.decoding
                          ? `${t("draft")} ${job.spec.decoding.draft.model.name} · k=${job.spec.decoding.num_speculative_tokens} · ${providerLabel(describe, job.spec.runtime?.name)}`
                          : `${t("Autoregressive")} · ${providerLabel(describe, job.spec.runtime?.name)}`}</td>}
                        <td className="small">{job.spec.task.name}</td>
                        <td className="small muted">{job.spec.evaluator.name}</td>
                        <td className="num small muted right" title={eta?.basis}>{eta ? formatSeconds((eta.prepare_s ?? 0) + (eta.measure_s ?? 0)) : "—"}</td>
                      </tr>
                    );
                  })}
                  {plan.reused.map((r) => (
                    <tr key={r.run_key}><td><Badge tone="green">{t("reused")}</Badge></td><td className="small muted" colSpan={5} title={r.reason}>{t(r.role)} · {r.reason}</td></tr>
                  ))}
                </tbody>
              </table>
              {rendered?.estimate && rendered.estimate.total_s !== null && (
                <p className="faint tiny mt-2" title={rendered.estimate.jobs[0]?.basis}>≈ {formatSeconds(rendered.estimate.total_s)}</p>
              )}
            </div>
            <div>
              <div className="field-label mb-2">{t("Paper fidelity")}</div>
              {fidelity?.preset ? (
                <div className="small stack-sm">
                  <div className="row gap-2"><span className="mono">{fidelity.preset}</span><Badge>{t("{n} from paper/code", { n: fidelity.confirmed.length })}</Badge>{fidelity.assumed.length > 0 && <Badge tone="orange">{t("{n} assumed", { n: fidelity.assumed.length })}</Badge>}</div>
                  <PaperDiff deviations={fidelity.deviations} />
                </div>
              ) : <span className="faint small">—</span>}
            </div>
          </div>
          {rendered && rendered.commands.length > 0 && (
            <div>
              {rendered.commands.length > 1 && (
                <div className="mb-3">
                  <Segmented value={node} onChange={setNode} options={rendered.commands.map((_, i) => ({ value: String(i), label: `${t("Node")} ${i + 1}` }))} />
                </div>
              )}
              <CommandSteps command={rendered.commands[Number(node)] ?? rendered.commands[0]} />
            </div>
          )}
        </div>
      )}
    </Dialog>
  );
}
