import type { ReactNode } from "react";
import type { TaskRef } from "../api";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { Field, Input, Select } from "../ui/controls";
import { Badge, Card } from "../ui/display";
import { FewshotInput, SamplesInput, SamplingInput, TaskSettings, extraSettings, hasFewshot, hasLimit, withParam } from "./TaskSettings";
import { selectedTasks } from "./TasksStep";
import { MODE_LABEL } from "./taskIndex";

const GENERATION_SETTINGS = ["max_new_tokens", "do_sample", "temperature"];

/** Per-task settings (scoring mode, few-shot, samples and the benchmark's own settings), one row per selected task. */
export function TaskConfigStep({ title, specTasks, presetTasks, cleared, invalid, runMaxNewTokens, runSampling, runTemperature, onChange }: {
  title: ReactNode;
  specTasks: TaskRef[] | undefined;
  presetTasks: string[];
  cleared: boolean;
  invalid: Set<string>;
  /** The run-level generation length (Run settings); what a Generation task uses without its own. */
  runMaxNewTokens: number;
  /** Whether the run-level generation settings sample (only a YAML or MCP spec sets this). */
  runSampling: boolean;
  runTemperature?: number;
  onChange: (tasks: TaskRef[]) => void;
}) {
  const { describe } = useApp();
  const t = useT();
  const { tasks, usingPreset } = selectedTasks(specTasks, presetTasks, cleared);
  const modeLabel = (m: string) => MODE_LABEL[m] ?? m;

  return (
    <Card title={title} hint={`${t("{n} selected", { n: tasks.length })}${usingPreset ? ` · ${t("from the preset")}` : ""}`}>
      {tasks.length === 0 ? <p className="faint small">{t("No tasks selected yet; choose them in the previous step.")}</p> : (
        <div className="task-config-scroll"><table className="list task-config">
          <colgroup><col style={{ width: 160 }} /><col style={{ width: 92 }} /><col style={{ width: 116 }} /><col style={{ width: 222 }} /><col style={{ width: 116 }} /><col style={{ width: 150 }} /><col /></colgroup>
          <thead><tr><th>{t("Task")}</th><th>{t("Scoring")}</th><th>{t("Few-shot")}</th><th>{t("Samples")}</th><th>{t("Max new tokens")}</th><th>{t("Sampling")}</th><th>{t("Other settings")}</th></tr></thead>
          <tbody>
            {tasks.map((entry, index) => {
              const task = describe.tasks[entry.name];
              if (!task) {
                return <tr key={entry.name}><td className="mono text-sm">{entry.name}</td><td colSpan={6} className="faint small">{t("Unknown task")}</td></tr>;
              }
              const mode = entry.mode ?? task.default_mode;
              const setEntry = (patch: Partial<TaskRef>) => {
                const next: TaskRef = { ...entry, ...patch };
                if (!next.mode || next.mode === task.default_mode) delete next.mode;
                if (!next.params) delete next.params;
                onChange(tasks.map((x, i) => (i === index ? next : x)));
              };
              const maxNew = entry.params?.max_new_tokens as number | undefined;
              const setParams = (params: Record<string, unknown> | undefined) => setEntry({ params });
              return (
                <tr key={task.name}>
                  <td className="wrap">
                    <span className="task-name-cell">
                      <span className="mono fade" title={task.name}>{task.name}</span>
                      {presetTasks.includes(task.name) && <Badge tone="accent">{t("paper")}</Badge>}
                    </span>
                  </td>
                  <td className="wrap">
                    {task.modes.length > 1 ? (
                      <Field invalid={invalid.has(`tasks[${index}].mode`)}>
                        <Select value={mode} onChange={(m) => setEntry({ mode: m })}
                          options={task.modes.map((m) => ({ value: m, label: modeLabel(m), hint: m === task.default_mode ? t("default") : undefined }))} />
                      </Field>
                    ) : <Badge tone={mode === "generation" ? "orange" : undefined} title={t("How the task is scored")}>{modeLabel(mode)}</Badge>}
                  </td>
                  <td className="wrap">{hasFewshot(task) ? <FewshotInput task={task} params={entry.params} onChange={setParams} /> : <span className="num muted">{task.num_fewshot}</span>}</td>
                  <td className="wrap">{hasLimit(task) ? <SamplesInput task={task} params={entry.params} invalid={invalid.has(`tasks[${index}].params.limit`)} onChange={setParams} /> : <span className="faint small">—</span>}</td>
                  <td className="wrap">
                    {task.params_schema.properties?.max_new_tokens ? (
                      <Input type="number" min={1} step={1} value={maxNew === undefined ? "" : String(maxNew)} placeholder={t("default {n}", { n: runMaxNewTokens })}
                        title={t("Longest answer the model may generate for this task; empty uses the default")}
                        className={invalid.has(`tasks[${index}].params.max_new_tokens`) ? "invalid" : ""}
                        onChange={(e) => setParams(withParam(entry.params, "max_new_tokens", e.target.value === "" ? undefined : Number(e.target.value)))} />
                    ) : <span className="faint small">—</span>}
                  </td>
                  <td className="wrap">
                    {task.params_schema.properties?.do_sample ? (
                      <SamplingInput task={task} params={entry.params} runSampling={runSampling} runTemperature={runTemperature}
                        invalid={invalid.has(`tasks[${index}].params.temperature`)} onChange={setParams} />
                    ) : <span className="faint small">—</span>}
                  </td>
                  <td className="wrap">
                    {extraSettings(task).filter((k) => !GENERATION_SETTINGS.includes(k)).length > 0
                      ? <TaskSettings task={task} params={entry.params} invalidPaths={invalid} pathPrefix={`tasks[${index}].params`} onChange={setParams} skip={GENERATION_SETTINGS} inline />
                      : <span className="faint small">—</span>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table></div>
      )}
    </Card>
  );
}
