import { Gauge } from "lucide-react";
import { useMemo, useState } from "react";
import type { TaskDef } from "../api";
import { useApp } from "../components/common";
import { TaskPicker } from "../compose/TaskPicker";
import { FewshotInput, SamplesInput, TaskSettings, extraSettings, hasFewshot, hasLimit } from "../compose/TaskSettings";
import { useT } from "../i18n";
import { Field } from "../ui/controls";
import { Badge, Card, CodeBlock, PageHeader } from "../ui/display";
import { Dialog } from "../ui/overlay";

const EVALUATOR_TEMPLATE = `# pyproject.toml of your package
[project.entry-points."effeval.evaluators"]
my_evaluator = "my_package.evaluator:EVALUATOR"

# my_package/evaluator.py
class MyEvaluator:
    name = "my_evaluator"
    def list_tasks(self): ...          # TaskListing of the tasks your library can run, or None
    def run(self, handle, job, task, workdir): ...   # QualityResult
EVALUATOR = MyEvaluator()`;

const SYNC = `pip install -e .          # on a compute node, then re-install so entry points register
effeval task sync          # an admin uploads the listing: tasks appear here as my_evaluator/<task>`;

/** Every task the catalog knows, browsable by source and group, with its settings and how to add evaluators. */
export function Benchmarks() {
  const { describe } = useApp();
  const t = useT();
  const [open, setOpen] = useState<TaskDef | null>(null);
  const modes = useMemo(() => [...new Set(Object.values(describe.tasks).flatMap((x) => x.modes))], [describe]);
  const total = Object.keys(describe.tasks).length;

  return (
    <div className="page">
      <PageHeader title={t("Benchmarks")} />
      <Card title={t("Available")} hint={String(total)}>
        <TaskPicker allowedModes={modes} selected={[]} readOnly onOpen={setOpen} />
      </Card>

      <Card title={t("Evaluators")}>
        <table className="list">
          <thead><tr><th className="col-md">{t("Name")}</th><th>{t("Library")}</th><th>{t("Description")}</th></tr></thead>
          <tbody>
            {Object.values(describe.evaluators).map((e) => (
              <tr key={e.name}>
                <td className="mono">{e.name}</td>
                <td className="small">{e.library ? `${e.library} ${e.library_version}` : <span className="faint">—</span>}</td>
                <td className="small muted wrap">{e.description}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card title={t("Add an evaluator")}>
        <div className="command-steps">
          <div className="step-label"><span className="step-num">1</span>{t("Implement the Evaluator interface and announce it")}</div>
          <CodeBlock text={EVALUATOR_TEMPLATE} language="python" />
          <div className="step-label"><span className="step-num">2</span>{t("Install and upload its task list")}<span className="faint tiny">{t("native-style tasks with their own settings still go in core/benchmarks/")}</span></div>
          <CodeBlock text={SYNC} language="bash" />
        </div>
      </Card>

      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)} wide title={<span className="mono">{open?.name}</span>}
        icon={<div className="icon-tile accent"><Gauge size={18} /></div>}>
        {open && <BenchmarkDetail task={open} />}
      </Dialog>
    </div>
  );
}

function BenchmarkDetail({ task }: { task: TaskDef }) {
  const { describe } = useApp();
  const t = useT();
  const [params, setParams] = useState<Record<string, unknown> | undefined>(undefined);
  return (
    <div className="stack">
      <p className="small" style={{ margin: 0 }}>{task.description}</p>
      <dl className="kv">
        <dt>{t("Evaluator")}</dt><dd><span className="mono">{task.evaluator}</span>{describe.evaluators[task.evaluator]?.library ? ` · ${describe.evaluators[task.evaluator].library} ${describe.evaluators[task.evaluator].library_version}` : ""}</dd>
        <dt>{t("Task")}</dt><dd className="mono">{task.target}</dd>
        <dt>{t("Version")}</dt><dd>{task.version}</dd>
        <dt>{t("Category")}</dt><dd>{t(describe.categories[task.category]?.label ?? task.category)}</dd>
        <dt>{t("Mode")}</dt><dd>{task.modes.join(", ")}</dd>
        <dt>{t("Few-shot")}</dt><dd>{task.num_fewshot ?? t("per subtask")}{task.fewshot_allowed ? ` (${task.fewshot_allowed.length ? task.fewshot_allowed.join(", ") : t("fixed")})` : ""}</dd>
        <dt>{t("Metrics")}</dt><dd>{task.metrics.map((m, i) => <Badge key={m.name} tone={i === 0 ? "accent" : undefined}>{m.name}{m.higher_is_better === null ? "" : m.higher_is_better ? " ↑" : " ↓"}</Badge>)}</dd>
        {task.subtasks.length > 0 && <><dt>{t("Subtasks")}</dt><dd className="small">{task.subtasks.length}</dd></>}
        {task.tags.length > 0 && <><dt>{t("Tags")}</dt><dd>{task.tags.map((x) => <Badge key={x}>{x}</Badge>)}</dd></>}
        <dt>{t("Baseline")}</dt><dd>{task.reference ? t("compared against the unmodified model") : t("measured inline")}</dd>
      </dl>
      {(hasLimit(task) || hasFewshot(task) || extraSettings(task).length > 0) && (
        <div>
          <div className="field-label mb-2">{t("Settings in New evaluation")}</div>
          <div className="stack-sm">
            {hasFewshot(task) && <div className="w-md"><Field label={t("Few-shot")}><FewshotInput task={task} params={params} onChange={setParams} /></Field></div>}
            {hasLimit(task) && <div className="w-md"><Field label={t("Samples")}><SamplesInput task={task} params={params} onChange={setParams} /></Field></div>}
            {extraSettings(task).length > 0 && <TaskSettings task={task} params={params} invalidPaths={new Set()} pathPrefix="" onChange={setParams} />}
          </div>
        </div>
      )}
    </div>
  );
}
