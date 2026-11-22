import { GitCompareArrows } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type SampleDiff as Diff } from "../api";
import { useT } from "../i18n";
import { Button, Segmented, Select } from "../ui/controls";
import { Badge } from "../ui/display";
import { Dialog, useToast } from "../ui/overlay";

const PAGE = 100;
const FLIP_LABEL: Record<string, string> = {
  to_wrong: "Right → wrong",
  to_right: "Wrong → right",
  both_right: "Both right",
  both_wrong: "Both wrong",
};

function show(value: unknown): string {
  if (value === null || value === undefined) return "—";
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** Side-by-side view of a record's samples against its reference, filtered by how correctness moved. */
export function SampleDiffDialog({ recordId, open, onOpenChange }: { recordId: string; open: boolean; onOpenChange: (open: boolean) => void }) {
  const t = useT();
  const toast = useToast();
  const [flip, setFlip] = useState("changed");
  const [subtask, setSubtask] = useState("");
  const [diff, setDiff] = useState<Diff | null>(null);

  const load = async (offset: number) => {
    try {
      const page = await api.sampleDiff(recordId, { flip, subtask: subtask || undefined, offset, limit: PAGE });
      setDiff((prev) => (offset && prev ? { ...page, samples: [...prev.samples, ...page.samples] } : page));
    } catch (e) {
      toast(t("Action failed"), { tone: "error", description: (e as Error).message });
      onOpenChange(false);
    }
  };
  useEffect(() => {
    if (open) load(0);
  }, [open, flip, subtask, recordId]);

  const subtasks = Object.keys(diff?.subtasks ?? {}).filter(Boolean).sort();
  return (
    <Dialog open={open} onOpenChange={onOpenChange} wide title={t("Compare samples")}
      icon={<div className="icon-tile accent"><GitCompareArrows size={18} /></div>}>
      {diff && (
        <div className="stack">
          <div className="row gap-3 nowrap-row">
            <Segmented value={flip} onChange={setFlip} options={[
              { value: "changed", label: `${t("Changed")} ${diff.counts.to_wrong + diff.counts.to_right}` },
              { value: "to_wrong", label: `${t(FLIP_LABEL.to_wrong)} ${diff.counts.to_wrong}` },
              { value: "to_right", label: `${t(FLIP_LABEL.to_right)} ${diff.counts.to_right}` },
              { value: "all", label: t("All") },
            ]} />
            {subtasks.length > 1 && (
              <div className="w-md">
                <Select value={subtask || "-"} onChange={(v) => setSubtask(v === "-" ? "" : v)}
                  options={[{ value: "-", label: t("All subtasks") }, ...subtasks.map((s) => ({ value: s, label: `${s} · ${diff.subtasks[s].to_wrong}↓ ${diff.subtasks[s].to_right}↑` }))]} />
              </div>
            )}
            <span className="spacer" />
            <span className="small muted">
              {t("Aligned {a} of {m} / {r} samples", { a: diff.coverage.aligned, m: diff.coverage.main, r: diff.coverage.reference })}
            </span>
          </div>
          <div className="pick-table">
            <table className="list">
              <thead><tr><th>{t("Sample")}</th><th>{t("Target")}</th><th>{t("Baseline answer")}</th><th>{t("Method answer")}</th><th /></tr></thead>
              <tbody>
                {diff.samples.map((s) => (
                  <tr key={s.id}>
                    <td className="mono small">{s.id}</td>
                    <td className="mono small wrap">{show(s.target)}</td>
                    <td className="mono small wrap">{show(s.reference)}</td>
                    <td className="mono small wrap">{show(s.main)}</td>
                    <td><Badge tone={s.flip === "to_wrong" ? "red" : s.flip === "to_right" ? "green" : undefined}>{t(FLIP_LABEL[s.flip])}</Badge></td>
                  </tr>
                ))}
                {diff.samples.length === 0 && <tr><td colSpan={5} className="faint small center" style={{ padding: 24 }}>{t("No match")}</td></tr>}
              </tbody>
            </table>
          </div>
          {diff.samples.length < diff.total && (
            <div><Button size="sm" onClick={() => load(diff.samples.length)}>{t("Show {n} more", { n: Math.min(PAGE, diff.total - diff.samples.length) })}</Button></div>
          )}
        </div>
      )}
    </Dialog>
  );
}
