import * as yaml from "js-yaml";
import { createContext, useContext } from "react";
import type { Describe, Deviation, Me, Problem, Project, Row } from "../api";
import { useT } from "../i18n";
import { Badge } from "../ui/display";
import { Tip } from "../ui/overlay";

export interface AppContextValue {
  me: Me;
  describe: Describe;
  reloadDescribe: () => Promise<void>;
  projects: Project[];
  reloadProjects: () => Promise<void>;
}

export const AppContext = createContext<AppContextValue | null>(null);

const PROBLEM_LABEL: Record<Problem, string> = {
  redo: "marked for redo",
  trashed: "in trash",
  retracted: "retracted",
  reference_trashed: "baseline in trash",
};

/** Orange "!" whose tooltip says why a result is invalid; renders nothing when there are no problems. */
export function ProblemMark({ problems }: { problems: Problem[] }) {
  const t = useT();
  if (!problems.length) return null;
  const text = t("Invalid: {reasons}", { reasons: problems.map((p) => t(PROBLEM_LABEL[p])).join(", ") });
  return <Tip content={text}><span className="problem-mark" aria-label={text}>!</span></Tip>;
}

export function useApp(): AppContextValue {
  const value = useContext(AppContext);
  if (value === null) throw new Error("useApp outside AppContext");
  return value;
}

export function FidelityBadges({ fidelity }: { fidelity: Row["fidelity"] }) {
  const t = useT();
  if (!fidelity.preset) return <Badge>{t("no preset")}</Badge>;
  return (
    <span className="row gap-1 nowrap-row">
      {fidelity.deviations.length
        ? <Badge tone="orange" title={fidelity.deviations.map((d) => d.path).join(", ")}>{fidelity.deviations.length === 1 ? t("1 deviation") : t("{n} deviations", { n: fidelity.deviations.length })}</Badge>
        : <Badge tone="accent" title={fidelity.preset}>{t("paper config")}</Badge>}
      {fidelity.assumed.length > 0 && <Badge tone="orange">{t("{n} assumed", { n: fidelity.assumed.length })}</Badge>}
      {fidelity.repro === "pass" && <Badge tone="green">{t("repro")} ✓</Badge>}
      {fidelity.repro === "fail" && <Badge tone="red">{t("repro")} ✗</Badge>}
    </span>
  );
}

function showValue(value: unknown): string {
  if (Array.isArray(value) && value.every((v) => typeof v !== "object" || (v && "name" in v && Object.keys(v).length <= 2))) {
    return value.map((v) => (v && typeof v === "object" ? String((v as { name: unknown }).name) : String(v))).join(", ");
  }
  if (value && typeof value === "object") return yaml.dump(value, { flowLevel: 3, lineWidth: 60 }).trimEnd();
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** Lists each preset deviation as its path with the paper value above the actual one, or a "matches the paper" badge when there are none. */
export function PaperDiff({ deviations }: { deviations: Deviation[] }) {
  const t = useT();
  if (!deviations.length) return <div><Badge tone="accent">{t("matches the paper")}</Badge></div>;
  return (
    <div className="diff-list">
      {deviations.map((d) => (
        <div key={d.path} className="diff-item">
          <div className="mono diff-path">{d.path}</div>
          <dl className="diff-values">
            <dt>{t("Paper")}</dt><dd className="mono diff-paper">{showValue(d.preset)}</dd>
            <dt>{t("Actual")}</dt><dd className="mono">{showValue(d.actual)}</dd>
          </dl>
        </div>
      ))}
    </div>
  );
}

/** Small avatar plus username; the tooltip says how and when the record was uploaded and from which machine. */
export function Uploader({ row, compact = false }: { row: Row; compact?: boolean }) {
  const t = useT();
  if (!row.owner) return <span className="faint">—</span>;
  const env = (row.record as { env?: { hostname?: string; gpu?: string } | null }).env;
  const where = [env?.hostname, row.gpu].filter(Boolean).join(" · ");
  const title = `${row.owner} · ${t("via {via}", { via: row.submitted_via })} · ${formatTime(row.created_at)}${where ? ` · ${where}` : ""}`;
  return (
    <span className="uploader" title={title}>
      <span className="avatar xs">{initials(row.owner)}</span>
      {!compact && <span>{row.owner}</span>}
    </span>
  );
}

export function formatTime(iso: string): string {
  return iso.slice(0, 16).replace("T", " ");
}

export function initials(name: string): string {
  return name.slice(0, 2).toUpperCase();
}
