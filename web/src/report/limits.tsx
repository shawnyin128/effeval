// Marking of results computed on a subset: the sample limit a record was run with, read from the record's own spec.

/** A fractional limit as a percentage ("10%"); a count as is. */
export function formatLimit(limit: number): string {
  return limit > 0 && limit < 1 ? `${Number((limit * 100).toPrecision(12))}%` : String(limit);
}

/** A sample limit: a count, a fraction in (0, 1), or null for the full set (stored as absent or -1). */
export type Limit = number | null;

export type Translate = (key: string, vars?: Record<string, string | number>) => string;

/** The limit stored in a frozen job spec. */
export function specLimit(spec: Record<string, any>): Limit {
  const limit = spec.task?.limit;
  return limit === undefined || limit === null || limit === -1 ? null : limit;
}

/** "first 50 samples" for a count, "10% of samples" for a fraction. */
export function limitText(limit: number, t: Translate): string {
  return limit > 0 && limit < 1 ? t("{p} of samples", { p: formatLimit(limit) }) : t("first {n} samples", { n: limit });
}

/** Stable text identity of a limit, or of a per-task limit map. */
export function limitKey(value: Limit | [string, Limit][]): string {
  return JSON.stringify(value);
}

export interface LimitSummary {
  /** Distinct limits among the results of a column, in first-seen order. */
  limits: Limit[];
  /** The results do not all share one limit (a full-set result counts as a limit of its own). */
  mixed: boolean;
  /** At least one result was run on a subset. */
  limited: boolean;
}

export function summarizeLimits(values: Limit[]): LimitSummary {
  const limits = [...new Set(values)];
  return { limits, mixed: limits.length > 1, limited: limits.some((l) => l !== null) };
}

/** Column header text of a summary: the shared limit, or "mixed"; "" when every result is the full set. */
export function summaryText(summary: LimitSummary, t: Translate): string {
  if (!summary.limited) return "";
  if (summary.mixed) return t("Mixed sample counts");
  return limitText(summary.limits[0] as number, t);
}

/** Every limit of a summary spelled out, for tooltips. */
export function summaryDetail(summary: LimitSummary, t: Translate): string {
  return summary.limits.map((l) => (l === null ? t("all samples") : limitText(l, t))).join(" / ");
}

/** English sentence fragment for a limit in a paper footnote: "the first 50 samples" or "10% of the samples". */
export function plainLimit(limit: number): string {
  return limit > 0 && limit < 1 ? `${formatLimit(limit)} of the samples` : `the first ${limit} samples`;
}

/** The "†" footnote of a paper table naming each row evaluated on a subset; null when none is.
 *  `uniform` rows ran every task on one limit (row names grouped by limit); `perTask` rows list their own limits. */
export function rowLimitFootnote(uniform: { name: string; limit: number }[], perTask: { name: string; limits: [string, Limit][] }[]): string | null {
  if (!uniform.length && !perTask.length) return null;
  const byLimit = new Map<number, string[]>();
  uniform.forEach(({ name, limit }) => byLimit.set(limit, [...(byLimit.get(limit) ?? []), name]));
  const parts = [...byLimit].map(([limit, names]) => (byLimit.size === 1 && !perTask.length ? plainLimit(limit) : `${plainLimit(limit)} (${names.join(", ")})`));
  perTask.forEach(({ name, limits }) => parts.push(`${name}: ${limits.map(([task, l]) => `${task} ${l === null ? "all samples" : plainLimit(l)}`).join(", ")}`));
  return `† Evaluated on a subset of the samples: ${parts.join("; ")}.`;
}

/** The "†" footnote of a paper table naming each column evaluated on a subset; null when none is. */
export function limitFootnote(columns: { label: string; limit?: LimitSummary }[]): string | null {
  const items = columns.filter((c) => c.limit?.limited).map((c) => {
    const s = c.limit as LimitSummary;
    return `${c.label}: ${s.mixed ? `mixed sample counts (${s.limits.map((l) => (l === null ? "all samples" : plainLimit(l))).join(", ")}), † marks the subset results` : plainLimit(s.limits[0] as number)}`;
  });
  return items.length ? `† Evaluated on a subset of the samples. ${items.join("; ")}.` : null;
}

/** Warning-toned tag for a column header when its results were run on a subset. */
export function LimitTag({ summary, t }: { summary: LimitSummary; t: Translate }) {
  const text = summaryText(summary, t);
  if (!text) return null;
  return <span className="limit-tag" title={summaryDetail(summary, t)}>{text}</span>;
}

/** Small mark on one result computed on a subset, shown where its column mixes limits. */
export function CellLimit({ limit, t }: { limit: Limit | undefined; t: Translate }) {
  if (limit === null || limit === undefined) return null;
  return <sup className="cell-limit" title={limitText(limit, t)}>{limit > 0 && limit < 1 ? formatLimit(limit) : limit}</sup>;
}
