// Pure helpers behind the task picker: the catalog holds thousands of tasks, so they are indexed once.
import type { Describe, TaskDef } from "../api";

export type Source = "lm_eval" | "evalscope" | "custom";
export const SOURCES: Source[] = ["lm_eval", "evalscope", "custom"];

export interface TaskIndex {
  /** Tasks that are nobody's subtask, by source, sorted by name. */
  roots: Record<Source, TaskDef[]>;
  /** Direct subtasks of every group, sorted by name. */
  children: Map<string, TaskDef[]>;
  byName: Map<string, TaskDef>;
  /** Every indexed task by source, subtasks included. */
  counts: Record<Source, number>;
}

/** Where a task comes from: its library, or `custom` for everything else (native, kernel, third-party evaluators). */
export function sourceOf(task: TaskDef): Source {
  return task.evaluator === "lm_eval" || task.evaluator === "evalscope" ? task.evaluator : "custom";
}

const byName = (a: TaskDef, b: TaskDef) => a.name.localeCompare(b.name);

/** The category that holds perplexity tasks; everything else is a downstream task. */
export const PPL_CATEGORY = "language_modeling";

/** Wording of an evaluation mode in the UI. */
export const MODE_LABEL: Record<string, string> = { loglikelihood: "Likelihood", generation: "Generation" };

/** A library metric name as shown: filter suffix and module path dropped ("exact_match,strict-match" -> "exact_match"). */
export function metricLabel(name: string): string {
  return name.split(",")[0].split("/").pop() ?? name;
}

/** A task's name as `[evaluator, target]`. */
export function splitName(task: TaskDef): [string, string] {
  return [task.evaluator, task.name.slice(task.evaluator.length + 1)];
}

/** Whether the task can run in `mode` ("all" accepts every task). */
export const inMode = (task: TaskDef, mode: string) => mode === "all" || task.modes.includes(mode);

/** Index the tasks that can run in one of `allowedModes`. */
export function buildIndex(describe: Describe, allowedModes: string[]): TaskIndex {
  const tasks = Object.values(describe.tasks).filter((t) => t.modes.some((m) => allowedModes.includes(m)));
  const lookup = new Map(tasks.map((t) => [t.name, t]));
  const nested = new Set<string>();
  const children = new Map<string, TaskDef[]>();
  for (const task of tasks) {
    const members = task.subtasks.map((n) => lookup.get(n)).filter((t): t is TaskDef => !!t).sort(byName);
    if (members.length) children.set(task.name, members);
    members.forEach((m) => nested.add(m.name));
  }
  const roots: Record<Source, TaskDef[]> = { lm_eval: [], evalscope: [], custom: [] };
  for (const task of tasks) if (!nested.has(task.name)) roots[sourceOf(task)].push(task);
  SOURCES.forEach((s) => roots[s].sort(byName));
  const counts: Record<Source, number> = { lm_eval: 0, evalscope: 0, custom: 0 };
  for (const task of tasks) counts[sourceOf(task)] += 1;
  return { roots, children, byName: lookup, counts };
}

export interface SearchResult {
  /** Best matches (at most `limit`), most relevant first; subtasks of a matching group are left out (open the group to see them). */
  matches: string[];
  /** Top-level tasks to list, in order of their best match. */
  roots: string[];
  /** Matches and every group above them. */
  visible: Set<string>;
  /** Groups to show open because a match sits inside them while they do not match themselves. */
  expanded: Set<string>;
}

/** Words researchers type for what a task measures or is called, mapped to the terms they stand for. */
const ALIASES: Record<string, string[]> = {
  ppl: ["perplexity", "wikitext"],
  perplexity: ["ppl", "wikitext"],
  wiki: ["wikitext"],
  wiki2: ["wikitext2", "wikitext"],
  code: ["humaneval", "mbpp", "livecodebench", "live_code_bench"],
  coding: ["humaneval", "mbpp", "livecodebench", "live_code_bench"],
  math: ["gsm8k", "math_500", "aime", "amc"],
  mc: ["mmlu", "arc", "hellaswag"],
};

/** Whether `q` begins a word of `name` (words are separated by anything but letters and digits): "ppl" is in "wikitext2_ppl" but not in "applied". */
const startsWord = (name: string, q: string) => {
  const at = (from: number): boolean => {
    const i = name.indexOf(q, from);
    return i < 0 ? false : i === 0 || !/[a-z0-9]/.test(name[i - 1]) || at(i + 1);
  };
  return at(0);
};

/** Relevance of `task` to the lower-case query `q` (lower is better), or -1 when it does not match. */
function score(task: TaskDef, q: string, aliases: string[], categoryTitle: string): number {
  const name = task.name.toLowerCase();
  const target = task.target.toLowerCase();
  if (target === q || name === q) return 0;
  if (name.startsWith(q) || target.startsWith(q)) return 1;
  if (startsWord(name, q)) return 2;
  if (aliases.some((a) => target === a || target.startsWith(a))) return 2.2;
  if (aliases.some((a) => name.includes(a))) return 2.5;
  if (name.includes(q)) return 2.8;
  if (task.tags.some((t) => t.toLowerCase().includes(q))) return 3;
  if (task.metrics.some((m) => metricLabel(m.name).toLowerCase() === q) || categoryTitle === q) return 3.5;
  return task.description.toLowerCase().includes(q) ? 4 : -1;
}

/** Tasks (of `source`, when given) whose name, tags or description contain `query` (case-insensitive), ranked, with their ancestor groups. */
export function search(index: TaskIndex, query: string, source?: Source, limit = 200, titles: Record<string, string> = {}, pinned: ReadonlySet<string> = new Set()): SearchResult {
  const q = query.trim().toLowerCase();
  const aliases = ALIASES[q] ?? [];
  const scores = new Map<string, number>();
  for (const task of index.byName.values()) {
    if (source && sourceOf(task) !== source) continue;
    const s = score(task, q, aliases, (titles[task.category] ?? "").toLowerCase());
    if (s >= 0) scores.set(task.name, pinned.has(task.name) ? s - 0.1 : s);
  }
  const parents = new Map<string, string[]>();
  for (const [group, members] of index.children) for (const m of members) parents.set(m.name, [...(parents.get(m.name) ?? []), group]);
  const hasMatchingAncestor = (name: string): boolean => (parents.get(name) ?? []).some((p) => scores.has(p) || hasMatchingAncestor(p));
  const isGroup = (n: string) => (index.children.has(n) ? 0 : 1);
  const matches = [...scores.keys()]
    .filter((n) => !hasMatchingAncestor(n))
    .sort((a, b) => scores.get(a)! - scores.get(b)! || isGroup(a) - isGroup(b) || a.length - b.length || a.localeCompare(b))
    .slice(0, limit);
  const visible = new Set<string>();
  const expanded = new Set<string>();
  const roots: string[] = [];
  const climb = (name: string): string => {
    const up = parents.get(name) ?? [];
    for (const parent of up) {
      expanded.add(parent);
      visible.add(parent);
    }
    return up.length ? climb(up[0]) : name;
  };
  for (const name of matches) {
    visible.add(name);
    const root = climb(name);
    if (!roots.includes(root)) roots.push(root);
  }
  return { matches, roots, visible, expanded };
}

/** The server's common tasks that can run in one of `allowedModes`. */
export function common(describe: Describe, allowedModes: string[]): TaskDef[] {
  return describe.common_tasks
    .map((n) => describe.tasks[n])
    .filter((t): t is TaskDef => !!t && t.modes.some((m) => allowedModes.includes(m)));
}
