import { useMemo, type ReactNode } from "react";
import type { TaskDef, TaskRef } from "../api";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { Card } from "../ui/display";
import { TaskPicker } from "./TaskPicker";

/**
 * Tasks the evaluation runs. With no tasks chosen the server runs the preset's tasks, so they count as selected until
 * the user picks, unless the user emptied the list on purpose (`cleared`).
 * Returns the effective list and whether it comes from the preset.
 */
export function selectedTasks(specTasks: TaskRef[] | undefined, presetTasks: string[], cleared = false): { tasks: TaskRef[]; usingPreset: boolean } {
  const usingPreset = (specTasks ?? []).length === 0 && presetTasks.length > 0 && !cleared;
  return { tasks: usingPreset ? presetTasks.map((name) => ({ name })) : specTasks ?? [], usingPreset };
}

/** Task selection for a Field: narrowed by what is measured, then by scoring mode and kind; preset tasks carry a "paper" badge. */
export function TasksStep({ title, field, runtime, sampling, specTasks, presetTasks, cleared, onChange, onRestore }: {
  title: ReactNode;
  field: string;
  /** The provider of the run ("hf" when unset). */
  runtime?: string;
  /** Whether speculative verification samples (SpS) rather than verifies greedily. */
  sampling?: boolean;
  specTasks: TaskRef[] | undefined;
  presetTasks: string[];
  cleared: boolean;
  onChange: (tasks: TaskRef[]) => void;
  onRestore: () => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const { tasks, usingPreset } = selectedTasks(specTasks, presetTasks, cleared);
  const specdecodeOnly = field === "speculative_decoding" && ((runtime ?? "hf") !== "hf" || !!sampling);
  const keep = useMemo(() => field === "speculative_decoding"
    ? (task: TaskDef) => task.evaluator === "specdecode" || (!specdecodeOnly && task.modes.includes("generation"))
    : undefined, [field, specdecodeOnly]);
  const whyExcluded = (task: TaskDef | undefined) => {
    if (!keep || !task || keep(task)) return null;
    return (runtime ?? "hf") !== "hf"
      ? t("vLLM and SGLang run specdecode tasks (specdecode/…) only; use transformers for other tasks.")
      : t("Speculative sampling (SpS) runs on specdecode tasks only; switch Verification to Greedy for other tasks.");
  };
  return (
    <Card title={title} hint={`${t("{n} selected", { n: tasks.length })}${usingPreset ? ` · ${t("from the preset")}` : ""}`}>
      <TaskPicker allowedModes={describe.fields[field]?.modes ?? []} selected={tasks} presetTasks={presetTasks} onChange={onChange} onRestore={onRestore} keep={keep} whyExcluded={whyExcluded} />
    </Card>
  );
}
