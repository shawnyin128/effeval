// Non-blocking warnings about a composed RunSpec: each check reads the whole configuration and returns plain sentences.
// To add a check, write a `Check` and append it to CHECKS.
import type { Describe, TaskDef, TaskRef } from "../api";

type Spec = Record<string, any>;
type Translate = (text: string, vars?: Record<string, string | number>) => string;

/** A selected task with the settings that decide how it runs. */
export interface PlannedTask {
  name: string;
  /** The benchmark's own name for evalscope tasks ("evalscope/aime25:x" -> "aime25"), else the task name. */
  benchmark: string;
  mode: string;
  /** The task's own `limit` setting; undefined evaluates every sample. */
  limit: number | undefined;
  /** Effective generation length: the task's own setting, else the run-level one. */
  maxNewTokens: number;
  /** Whether the task samples (its own `do_sample`, else the run-level one); only generation can. */
  sampled: boolean;
}

export interface Context {
  tasks: PlannedTask[];
  repeats: { count: number; seeds: "same" | "vary" };
  useSandbox: boolean;
  t: Translate;
}

export type Check = (ctx: Context) => string[];

/** Generation length below which the long-answer benchmarks may be cut off. */
export const LONG_ANSWER_TOKENS = 4096;
const LONG_ANSWER = /^(aime|math_500|gpqa|live_code_bench)/;
/** evalscope 1.15.0: these benchmarks refuse to score without its sandbox (`use_sandbox`). */
const SANDBOX_REQUIRED = /^(mbpp|mbppplus|bigcodebench)$/;
/** evalscope 1.15.0: these execute the generated code in the job's own process environment unless sandboxed. */
const RUNS_CODE_LOCALLY = /^(humaneval|humanevalplus|live_code_bench)$/;

/** The selected tasks with their effective settings; tasks the catalog does not know are left out. */
export function plannedTasks(spec: Spec, tasks: TaskRef[], describe: Describe): PlannedTask[] {
  const runMax = spec.generation?.max_new_tokens ?? 512;
  return tasks.flatMap((ref) => {
    const def: TaskDef | undefined = describe.tasks[ref.name];
    if (!def) return [];
    const limit = ref.params?.limit;
    const target = ref.name.slice(def.evaluator.length + 1).split(":")[0];
    return [{
      name: ref.name,
      benchmark: def.evaluator === "evalscope" ? target : ref.name,
      mode: ref.mode ?? def.default_mode,
      limit: typeof limit === "number" && limit > 0 ? limit : undefined,
      maxNewTokens: Number(ref.params?.max_new_tokens ?? runMax),
      sampled: (ref.mode ?? def.default_mode) === "generation" && (ref.params?.do_sample ?? spec.generation?.do_sample) === true,
    }];
  });
}

export function contextOf(spec: Spec, tasks: TaskRef[], describe: Describe, t: Translate): Context {
  return {
    tasks: plannedTasks(spec, tasks, describe),
    repeats: Array.isArray(spec.repeats?.seeds)
      ? { count: spec.repeats.seeds.length, seeds: "vary" }
      : { count: spec.repeats?.count ?? 1, seeds: "same" },
    useSandbox: !!spec.evaluator?.passthrough?.evalscope?.use_sandbox,
    t,
  };
}

const limitText = (limit: number) => (limit < 1 ? `${Number((limit * 100).toPrecision(12))}%` : String(limit));

function list(items: string[]): string {
  return items.length > 3 ? `${items.slice(0, 3).join(", ")} +${items.length - 3}` : items.join(", ");
}

/** Repeats with a different seed each time: which tasks cannot vary (likelihood scoring, greedy decoding). */
const repeatsCannotVary: Check = ({ tasks, repeats, t }) => {
  if (repeats.count < 2 || repeats.seeds !== "vary") return [];
  const fixed = tasks.filter((task) => !task.sampled).map((task) => task.name);
  const varying = tasks.filter((task) => task.sampled).map((task) => task.name);
  if (fixed.length === 0) return [];
  return varying.length === 0
    ? [t("Repeated runs will give identical results: {tasks} use likelihood scoring or greedy decoding, so the seed changes nothing. Only generation tasks with Sampling on vary.", { tasks: list(fixed) })]
    : [t("Repeats of {fixed} will be identical (likelihood scoring or greedy decoding); only {varying} will show a spread.", { fixed: list(fixed), varying: list(varying) })];
};

/** Repeats with the same seed: identical by construction unless something in the run is non-deterministic. */
const sameSeedRepeats: Check = ({ tasks, repeats, t }) => {
  if (repeats.count < 2 || repeats.seeds !== "same" || tasks.length === 0) return [];
  const sampledTasks = tasks.filter((task) => task.sampled).map((task) => task.name);
  return sampledTasks.length > 0
    ? [t("Every repeat uses the same seed, so even the sampled tasks ({tasks}) reproduce the same samples; only GPU non-determinism can differ.", { tasks: list(sampledTasks) })]
    : [t("Repeats with the same seed are identical by construction here (likelihood scoring or greedy decoding). Use a different seed each run and turn Sampling on for a generation task to see a spread.")];
};

const limitedSamples: Check = ({ tasks, t }) => {
  const limited = tasks.filter((task) => task.limit !== undefined);
  return limited.length === 0 ? [] : [
    t("Only a subset of samples is evaluated ({tasks}), so these numbers are not comparable with full-set or paper results.",
      { tasks: list(limited.map((task) => `${task.name} · ${limitText(task.limit!)}`)) }),
  ];
};

const truncatedAnswers: Check = ({ tasks, t }) => {
  const short = tasks.filter((task) => task.mode === "generation" && LONG_ANSWER.test(task.benchmark.split("/").pop()!) && task.maxNewTokens < LONG_ANSWER_TOKENS);
  return short.length === 0 ? [] : [
    t("{tasks} usually need long answers; with Max new tokens below {n} they may be cut off mid-answer and score low.",
      { tasks: list(short.map((task) => `${task.name} · ${task.maxNewTokens}`)), n: LONG_ANSWER_TOKENS }),
  ];
};

const codeExecution: Check = ({ tasks, useSandbox, t }) => {
  if (useSandbox) return [];
  const names = (pattern: RegExp) => tasks.filter((task) => pattern.test(task.benchmark)).map((task) => task.name);
  const needsSandbox = names(SANDBOX_REQUIRED);
  const local = names(RUNS_CODE_LOCALLY);
  return [
    ...(needsSandbox.length ? [t("{tasks} cannot be scored without evalscope's sandbox (use_sandbox, needs Docker); the run will fail unless the evaluator passthrough enables it.", { tasks: list(needsSandbox) })] : []),
    ...(local.length ? [t("{tasks} execute model-generated code directly on the machine that runs the job (evalscope's default, no sandbox).", { tasks: list(local) })] : []),
  ];
};

export const CHECKS: Check[] = [repeatsCannotVary, sameSeedRepeats, limitedSamples, truncatedAnswers, codeExecution];

/** Every warning for the composed configuration, in check order. */
export function runWarnings(spec: Spec, tasks: TaskRef[], describe: Describe, t: Translate): string[] {
  const ctx = contextOf(spec, tasks, describe, t);
  return CHECKS.flatMap((check) => check(ctx));
}
