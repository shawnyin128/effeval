import { useState } from "react";
import { useT } from "../i18n";
import { Field, Input, Segmented } from "../ui/controls";

type Spec = Record<string, any>;

const MAX_SHOWN_SEEDS = 6;
const MAX_RUNS = 100;

/** How a spec repeats its runs: a seed list (one run per seed) or one seed run `count` times. */
export type Repeats = { mode: "same"; seed: number; count: number } | { mode: "vary"; seeds: number[] };

export function repeatsOf(spec: Spec): Repeats {
  const list = spec.repeats?.seeds;
  if (Array.isArray(list)) return { mode: "vary", seeds: list };
  return { mode: "same", seed: spec.determinism?.seed ?? 0, count: spec.repeats?.count ?? 1 };
}

/** The seeds in order, at most `limit` of them (the rest elided). */
export function seedsText(seeds: number[], limit = MAX_SHOWN_SEEDS): string {
  return seeds.slice(0, limit).join(", ") + (seeds.length > limit ? ", …" : "");
}

export type SeedProblem = { kind: "empty" | "blank" | "format" | "duplicate" | "limit"; token: string };

/** Parses comma-separated seeds (whitespace around items and one trailing comma allowed); reports the first bad item ("blank" for an empty item between commas). */
export function parseSeeds(text: string): { seeds: number[]; problem?: SeedProblem } {
  const tokens = text.split(",").map((token) => token.trim());
  if (tokens[tokens.length - 1] === "") tokens.pop();
  if (tokens.length === 0) return { seeds: [], problem: { kind: "empty", token: "" } };
  const seeds: number[] = [];
  for (const token of tokens) {
    if (token === "") return { seeds: [], problem: { kind: "blank", token } };
    if (!/^\d+$/.test(token) || !Number.isSafeInteger(Number(token))) return { seeds: [], problem: { kind: "format", token } };
    if (seeds.includes(Number(token))) return { seeds: [], problem: { kind: "duplicate", token } };
    seeds.push(Number(token));
  }
  if (seeds.length > MAX_RUNS) return { seeds: [], problem: { kind: "limit", token: "" } };
  return { seeds };
}

/** Switching to a seed list starts it from the current seed (and count); switching back keeps its first seed and its length. */
export function switchRepeatMode(spec: Spec, mode: "same" | "vary"): Spec {
  const now = repeatsOf(spec);
  if (now.mode === mode) return spec;
  const { repeats: _repeats, determinism, ...rest } = spec;
  const others = { ...(determinism ?? {}) };
  delete others.seed;
  const withDeterminism = (extra: Record<string, unknown>) => {
    const merged = { ...others, ...extra };
    return Object.keys(merged).length ? { ...rest, determinism: merged } : rest;
  };
  if (now.mode === "same") {
    const seeds = Array.from({ length: Math.min(now.count, MAX_RUNS) }, (_, k) => now.seed + k);
    return { ...withDeterminism({}), repeats: { seeds } };
  }
  const base = withDeterminism(now.seeds.length ? { seed: now.seeds[0] } : {});
  return now.seeds.length > 1 ? { ...base, repeats: { count: now.seeds.length } } : base;
}

/** Comma-separated seeds in a text box; the list is empty while the text has a problem, which is shown once the box loses focus. */
function SeedList({ seeds, onChange, serverError }: { seeds: number[]; onChange: (seeds: number[]) => void; serverError?: string }) {
  const t = useT();
  const [text, setText] = useState(seeds.join(", "));
  const [left, setLeft] = useState(false);
  const { problem } = parseSeeds(text);
  const message = problem === undefined ? serverError : problem.kind === "empty" ? t("Add at least one seed")
    : problem.kind === "blank" ? t("Empty entry between commas")
      : problem.kind === "format" ? t("'{seed}' is not a whole number, 0 or more", { seed: problem.token })
        : problem.kind === "duplicate" ? t("Seed {seed} is repeated", { seed: problem.token })
          : t("At most {n} seeds", { n: MAX_RUNS });
  return (
    <Field label={t("Seeds")} error={left || problem === undefined ? message : undefined}>
      <Input className="mono" value={text} placeholder="3, 5, 7" aria-label={t("Seeds")}
        onChange={(e) => { setText(e.target.value); onChange(parseSeeds(e.target.value).seeds); }}
        onBlur={() => setLeft(true)} />
    </Field>
  );
}

/** Run settings section for repeated runs: one seed run several times, or an explicit list of seeds with one run each. */
export function Robustness({ spec, setSpec, update, err }: {
  spec: Spec;
  setSpec: (change: (spec: Spec) => Spec) => void;
  update: (path: string, value: unknown) => void;
  err: (path: string) => string | undefined;
}) {
  const t = useT();
  const repeats = repeatsOf(spec);
  const seedsError = err("repeats.seeds") ?? err("repeats");
  const hint = repeats.mode === "vary"
    ? repeats.seeds.length === 0 ? ""
      : repeats.seeds.length === 1 ? t("Runs once with seed {seed}", { seed: repeats.seeds[0] })
        : t("Will run with seeds {seeds}", { seeds: seedsText(repeats.seeds) })
    : repeats.count < 2 ? t("Runs once with seed {seed}", { seed: repeats.seed })
      : t("Will run {n} times, all with seed {seed}", { n: repeats.count, seed: repeats.seed });
  return (
    <>
      <div className="field-label mb-3">{t("Robustness")}</div>
      <div className="grid-3">
        {repeats.mode === "vary" ? (
          <SeedList seeds={repeats.seeds} serverError={seedsError} onChange={(seeds) => update("repeats.seeds", seeds)} />
        ) : (
          <Field label={t("Seed")} error={err("determinism.seed")}>
            <Input type="number" value={spec.determinism?.seed ?? ""} placeholder="0"
              onChange={(e) => update("determinism.seed", e.target.value === "" ? undefined : Number(e.target.value))} />
          </Field>
        )}
        <Field label={t("Number of runs")} error={repeats.mode === "same" ? err("repeats.count") : undefined}>
          {repeats.mode === "vary"
            ? <div className="readonly-value mono">{t("{n} runs", { n: repeats.seeds.length })}</div>
            : <Input type="number" min={1} max={MAX_RUNS} step={1} value={spec.repeats?.count ?? ""} placeholder="1"
              onChange={(e) => update("repeats.count", e.target.value === "" || Number(e.target.value) < 2 ? undefined : Number(e.target.value))} />}
        </Field>
        <Field label={t("Seed across runs")}>
          <Segmented value={repeats.mode} onChange={(v) => setSpec((s) => switchRepeatMode(s, v === "vary" ? "vary" : "same"))}
            options={[{ value: "same", label: t("Same seed") }, { value: "vary", label: t("New seed each run") }]} />
        </Field>
      </div>
      {hint && <p className="small muted mt-2">{hint}</p>}
    </>
  );
}
