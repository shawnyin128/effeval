import { RotateCcw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, rowMethodTitle, type Row } from "../api";
import { formatTime, Uploader, useApp } from "../components/common";
import { useT } from "../i18n";
import { useLive } from "../live";
import { Button } from "../ui/controls";
import { Badge, Card, Empty, PageHeader } from "../ui/display";
import { useToast } from "../ui/overlay";

export function Trash() {
  const { describe, me } = useApp();
  const t = useT();
  const toast = useToast();
  const [rows, setRows] = useState<Row[] | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const load = useCallback(() => api.trashList().then(setRows), []);
  useLive(["records"], load);
  useEffect(() => {
    load();
  }, [load]);
  // Emptying only ever purges the viewer's own records; an admin also sees everyone else's trash here.
  const mine = (rows ?? []).filter((r) => r.owner === me.username);
  const others = (rows ?? []).length - mine.length;
  const [emptying, setEmptying] = useState(false);
  const emptyMine = async () => {
    const results = await Promise.allSettled(mine.map((r) => api.purge(r.record_id)));
    const failed = results.filter((r) => r.status === "rejected") as PromiseRejectedResult[];
    setEmptying(false);
    if (failed.length) toast(t("{n} could not be deleted", { n: failed.length }), { tone: "error", description: (failed[0].reason as Error).message });
    else toast(t("Deleted permanently {n}", { n: results.length }));
    load();
  };
  const act = async (fn: () => Promise<unknown>, done: string) => {
    try {
      await fn();
      toast(done);
      setConfirming(null);
      load();
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
    }
  };
  return (
    <div className="page">
      <PageHeader title={t("Trash")} actions={mine.length > 0 && <>
        {others > 0 && <span className="small muted">{t("{n} of other users are kept", { n: others })}</span>}
        {emptying
          ? <>
            <Button variant="danger" icon={<Trash2 size={14} />} onClick={emptyMine}>{t("Delete {n} of mine permanently", { n: mine.length })}</Button>
            <Button variant="ghost" onClick={() => setEmptying(false)}>{t("Cancel")}</Button>
          </>
          : <Button variant="danger" icon={<Trash2 size={14} />} onClick={() => setEmptying(true)}>{t("Empty my trash")}</Button>}
      </>} />
      <Card>
        {rows !== null && rows.length === 0 ? <Empty icon={<Trash2 size={20} />} title={t("Trash is empty")} /> : (
          <table className="list">
            <thead><tr><th className="col-lg">{t("Model")}</th><th className="col-sm">{t("Method")}</th><th className="col-md">{t("Task")}</th><th>{t("Metrics")}</th><th className="col-md">{t("Uploaded by")}</th><th className="col-md">{t("Deleted at")}</th><th className="col-lg" /></tr></thead>
            <tbody>
              {(rows ?? []).map((r) => (
                <tr key={r.record_id}>
                  <td className="mono small">{r.model}</td>
                  <td className="small">{rowMethodTitle(describe, r)}</td>
                  <td className="small">{r.task}</td>
                  <td className="mono small">{Object.entries(r.metrics).map(([k, v]) => `${k} ${v}`).join(" · ")}</td>
                  <td className="small"><Uploader row={r} /></td>
                  <td className="small muted">
                    {r.trashed && formatTime(r.trashed.changed_at)}
                    {r.trashed?.superseded_by && <> <Badge>{t("replaced by redo")}</Badge></>}
                  </td>
                  <td className="actions">
                    <Button variant="link" icon={<RotateCcw size={13} />} onClick={() => act(() => api.restore(r.record_id), t("Restored"))}>{t("Restore")}</Button>
                    {confirming === r.record_id
                      ? <Button variant="link" className="danger" onClick={() => act(() => api.purge(r.record_id), t("Deleted permanently"))}>{t("Confirm delete")}</Button>
                      : <Button variant="link" className="danger" onClick={() => setConfirming(r.record_id)}>{t("Delete permanently")}</Button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
