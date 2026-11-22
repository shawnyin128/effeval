import { CircleCheck, TriangleAlert } from "lucide-react";
import { useEffect, useState } from "react";
import { api, rowMethodTitle, type ReusePreviewJob, type Row } from "../api";
import { RecordPanel } from "../components/RecordPanel";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { formatMetric } from "../report/metrics";
import { Button } from "../ui/controls";
import { Badge } from "../ui/display";
import { Dialog } from "../ui/overlay";

/** Non-blocking heads-ups about the composed configuration, one sentence each. */
export function WarningsPanel({ warnings }: { warnings: string[] }) {
  const t = useT();
  if (warnings.length === 0) return null;
  return (
    <div className="mt-3">
      <div className="field-label mb-2">{t("Heads-up")}</div>
      {warnings.map((text) => (
        <div key={text} className="issue warning">
          <span className="issue-icon"><TriangleAlert size={16} /></span>
          <div className="issue-msg">{text}</div>
        </div>
      ))}
    </div>
  );
}

/** Which planned jobs already have a stored result, per the server (read-only; nothing is saved). */
export function ReusePanel({ spec }: { spec: Record<string, any> }) {
  const { describe } = useApp();
  const t = useT();
  const [jobs, setJobs] = useState<ReusePreviewJob[] | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const [record, setRecord] = useState<Row | null>(null);
  const body = JSON.stringify(spec);

  useEffect(() => {
    setJobs(null);
    setFailed(null);
    let current = true;
    const timer = setTimeout(async () => {
      try {
        const result = await api.reusePreview(JSON.parse(body));
        if (current) setJobs(result.jobs);
      } catch (e) {
        if (current) setFailed((e as Error).message);
      }
    }, 300);
    return () => { current = false; clearTimeout(timer); };
  }, [body]);

  if (failed) return <p className="small muted mt-3">{t("Could not check for existing results")}: {failed}</p>;
  if (jobs === null) return <p className="small muted mt-3">{t("Checking for existing results…")}</p>;
  if (jobs.length === 0) return null;
  const reused = jobs.filter((j) => j.status === "reused");
  const toRun = jobs.length - reused.length;
  const existing = jobs.filter((j) => j.record !== null);
  const open = async (id: string) => {
    try {
      setRecord(await api.record(id));
    } catch {
      setRecord(null);
    }
  };
  const when = (iso: string) => iso.slice(0, 16).replace("T", " ");
  return (
    <div className="mt-3">
      <div className="field-label mb-2">{t("Existing results")}</div>
      {existing.length > 0 && toRun === 0 ? (
        <div className="issue info">
          <span className="issue-icon"><CircleCheck size={16} color="var(--green)" /></span>
          <div className="issue-msg"><b>{t("Every result already exists ({n}); they will be reused and nothing will run.", { n: reused.length })}</b></div>
        </div>
      ) : (
        <p className="small">{t("{n} existing results will be reused (not run again); {m} need to run.", { n: reused.length, m: toRun })}</p>
      )}
      {existing.length > 0 && (
        <table className="list mt-2">
          <thead><tr><th>{t("Task")}</th><th>{t("Method")}</th><th className="col-sm">{t("Seed")}</th><th>{t("Run by")}</th><th>{t("When")}</th><th>{t("Result")}</th><th>{t("Status")}</th></tr></thead>
          <tbody>
            {existing.map((job) => {
              const stored = job.record!;
              const primary = describe.tasks[job.task]?.metrics[0]?.name ?? Object.keys(stored.metrics)[0];
              return (
                <tr key={job.run_key}>
                  <td className="small wrap">{job.task}{job.role === "reference" && <> <Badge>{t("reference")}</Badge></>}</td>
                  <td className="small mono">{rowMethodTitle(describe, job)}</td>
                  <td className="num small">{job.seed}</td>
                  <td className="small">{stored.owner ?? "—"}</td>
                  <td className="small muted">{when(stored.created_at)}</td>
                  <td className="small">
                    <Button variant="link" onClick={() => open(stored.record_id)} title={stored.record_id}>
                      {primary && stored.metrics[primary] !== undefined ? formatMetric(describe, primary, stored.metrics[primary], job.task) : stored.record_id}
                    </Button>
                  </td>
                  <td className="small wrap">
                    {job.status === "reused" ? <Badge tone="green">{t("Reused")}</Badge>
                      : <><Badge tone="orange">{t("Will run again")}</Badge><div className="faint tiny">{job.reason}</div></>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <Dialog open={record !== null} onOpenChange={(o) => !o && setRecord(null)} wide title={record?.record_id ?? ""}>
        {record && <RecordPanel row={record} onClose={() => setRecord(null)} />}
      </Dialog>
    </div>
  );
}
