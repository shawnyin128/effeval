import { useEffect, useState } from "react";
import type { Describe, ModelEntry } from "../api";
import { useT } from "../i18n";
import { draftProblem, type DraftProblem } from "./draftRules";
import { Field, Input, Segmented } from "../ui/controls";

type Spec = Record<string, any>;

export const MAX_SPECULATIVE_TOKENS = 16;
export const DEFAULT_SPS_TEMPERATURE = 0.7;

/** Whether the spec verifies the draft by sampling (SpS) rather than greedily. */
export const isSampling = (spec: Spec): boolean => spec.generation?.do_sample === true;

/** The seeds a text lists ("0, 1"): non-negative integers separated by commas or spaces; null when some token is not one. */
export function parseSeeds(text: string): number[] | null {
  const tokens = text.split(/[,\s]+/).filter(Boolean);
  return tokens.every((x) => /^\d+$/.test(x)) ? tokens.map(Number) : null;
}

/** The spec's verification as one line: "Greedy" or "SpS (T=0.7, top_p=0.9, top_k=50, seeds 0, 1)". */
export function verificationText(spec: Spec, t: (key: string) => string): string {
  if (!isSampling(spec)) return t("Greedy");
  const g = spec.generation ?? {};
  const parts = [`T=${g.temperature ?? "—"}`];
  if (g.top_p !== undefined) parts.push(`top_p=${g.top_p}`);
  if (g.top_k !== undefined) parts.push(`top_k=${g.top_k}`);
  if (Array.isArray(spec.repeats?.seeds) && spec.repeats.seeds.length) parts.push(`${t("seeds")} ${spec.repeats.seeds.join(", ")}`);
  return `SpS (${parts.join(", ")})`;
}

/** A model a Field evaluates: its role name ("target", "draft") and the RunSpec path of its model reference. */
export type Role = { name: string; path: string };

/** The model roles of a Field; a single target when the server lists none. */
export function rolesOf(describe: Describe, field: string): Role[] {
  return describe.fields[field]?.roles ?? [{ name: "target", path: "model" }];
}

/** Whether the Field compares decoding schemes: some role lives under `decoding`. */
export function usesDecoding(roles: Role[]): boolean {
  return roles.some((r) => r.path.startsWith("decoding."));
}

/** The value at a dotted path of a spec, or undefined. */
export function getIn(spec: Spec, path: string): any {
  return path.split(".").reduce((node, key) => (node === undefined || node === null ? undefined : node[key]), spec);
}

/** The provider's display name ("hf" is the transformers provider); the identifier itself for an unknown one. */
export function providerLabel(describe: Describe, name: string | undefined): string {
  const key = name ?? "hf";
  return describe.runtimes?.[key]?.label ?? (key === "hf" ? "transformers" : key);
}

/** Why the spec's draft model cannot pair with its target model, or null (also when there is no draft). */
export function specDraftProblem(models: Record<string, ModelEntry>, spec: Spec): DraftProblem | null {
  const draft = spec.decoding?.draft?.model?.name;
  return draft ? draftProblem(models, spec.model?.name, draft, {}) : null;
}

/** The spec without its draft model when the draft cannot pair with the target; the same spec otherwise. */
export function dropIncompatibleDraft(models: Record<string, ModelEntry>, spec: Spec): Spec {
  if (!specDraftProblem(models, spec)) return spec;
  const { draft: _drop, ...decoding } = spec.decoding;
  return { ...spec, decoding };
}

/** Decoding settings of a Field config step: the provider, the tokens the draft proposes per step, and how the target verifies them (Greedy or SpS with its sampling settings). */
export function Decoding({ spec, describe, update, err }: {
  spec: Spec;
  describe: Describe;
  update: (path: string, value: unknown) => void;
  err: (path: string) => string | undefined;
}) {
  const t = useT();
  const runtimes = Object.entries(describe.runtimes ?? { hf: { label: "transformers", library: null, available: true } });
  const missing = runtimes.filter(([, r]) => !r.available).map(([, r]) => r.label);
  const engine = !!spec.runtime?.name && spec.runtime.name !== "hf";
  const provider = providerLabel(describe, spec.runtime?.name);
  const sampling = isSampling(spec);
  const sgl = spec.runtime?.name === "sglang";
  const number = (path: string) => (e: { target: { value: string } }) => update(path, e.target.value === "" ? undefined : Number(e.target.value));
  const g = spec.generation ?? {};
  const temperatureBad = sampling && !(typeof g.temperature === "number" && g.temperature > 0);
  const topPBad = g.top_p !== undefined && !(g.top_p > 0 && g.top_p <= 1);
  const topKBad = g.top_k !== undefined && !(Number.isInteger(g.top_k) && g.top_k >= 1);
  const seedList: number[] = spec.repeats?.seeds ?? [];
  const [seedText, setSeedText] = useState(seedList.join(", "));
  const seedsBad = parseSeeds(seedText) === null;
  useEffect(() => {
    const typed = parseSeeds(seedText);
    if (typed !== null && typed.join(",") !== seedList.join(",")) setSeedText(seedList.join(", "));
  }, [seedList.join(",")]);
  const setVerification = (v: string) => {
    if (v === "sps") {
      update("generation.do_sample", true);
      if (g.temperature === undefined) update("generation.temperature", DEFAULT_SPS_TEMPERATURE);
    } else {
      update("generation.do_sample", false);
      for (const key of ["temperature", "top_p", "top_k"]) update(`generation.${key}`, undefined);
    }
  };
  const targetDtype: string | undefined = spec.model?.dtype ?? describe.models[spec.model?.name]?.dtype;
  return (
    <>
      <div className="grid-2">
        <Field label={t("Provider")} error={err("runtime.name") ?? err("runtime")}>
          <Segmented value={spec.runtime?.name ?? "hf"} onChange={(v) => update("runtime.name", v === "hf" ? undefined : v)}
            options={runtimes.map(([name, r]) => ({
              value: name,
              label: r.label,
              disabled: !r.available,
              title: r.available ? undefined : t("{provider} is not available on this machine", { provider: r.label }),
            }))} />
          {missing.length > 0 && <div className="small muted mt-3">{t("Not set up on this machine: {providers}.", { providers: missing.join(", ") })}</div>}
        </Field>
        <Field label={t("Draft tokens per step (k)")} error={err("decoding.num_speculative_tokens")}>
          <Input type="number" min={1} max={MAX_SPECULATIVE_TOKENS} step={1} value={spec.decoding?.num_speculative_tokens ?? ""} placeholder="4"
            onChange={(e) => update("decoding.num_speculative_tokens", e.target.value === "" ? undefined : Number(e.target.value))} />
        </Field>
      </div>
      <p className="small muted mt-3">
        {t("The draft proposes k tokens per step and the target verifies them; results are compared with the target's own autoregressive decoding.")}
      </p>
      <div className="divider" />
      <Field label={t("Verification")} error={err("generation.do_sample") ?? err("decoding.sampling")}>
        <Segmented value={sampling ? "sps" : "greedy"} onChange={setVerification} options={[
          { value: "greedy", label: "Greedy", title: t("The target accepts a draft token when it is the target's own argmax; output equals greedy decoding") },
          { value: "sps", label: t("Speculative sampling (SpS)"), title: t("Draft tokens are accepted by rejection sampling; the output distribution equals sampling from the target") },
        ]} />
        <div className="small muted mt-3">
          {sampling
            ? t("SpS runs on specdecode tasks only.")
            : t("Greedy verification: specdecode tasks and generation tasks (accuracy checks) can run.")}
        </div>
      </Field>
      {sampling && (
        <>
          <div className="grid-2 mt-3">
            <Field label={t("Temperature")} error={err("generation.temperature") ?? (temperatureBad ? t("Required for SpS; must be greater than 0.") : undefined)}>
              <Input type="number" min={0} step="any" value={g.temperature ?? ""} placeholder={String(DEFAULT_SPS_TEMPERATURE)}
                className={temperatureBad ? "invalid" : ""} onChange={number("generation.temperature")} />
            </Field>
            <Field label={t("Seeds")} error={err("repeats.seeds") ?? (seedsBad ? t("Seeds are whole numbers separated by commas, e.g. 0, 1.") : undefined)}>
              <Input value={seedText} placeholder="0, 1" className={seedsBad ? "invalid" : ""}
                title={t("One run per seed; empty runs once with the determinism seed")}
                onChange={(e) => {
                  setSeedText(e.target.value);
                  const parsed = parseSeeds(e.target.value);
                  if (parsed) update("repeats.seeds", parsed.length ? parsed : undefined);
                }} />
            </Field>
            <Field label="top_p" error={err("generation.top_p") ?? (topPBad ? t("top_p must be in (0, 1]; empty turns it off.") : undefined)}>
              <Input type="number" min={0} max={1} step="any" value={g.top_p ?? ""} placeholder={t("off")}
                className={topPBad ? "invalid" : ""} onChange={number("generation.top_p")} />
            </Field>
            <Field label="top_k" error={err("generation.top_k") ?? (topKBad ? t("top_k must be a whole number of at least 1; empty turns it off.") : undefined)}>
              <Input type="number" min={1} step={1} value={g.top_k ?? ""} placeholder={t("off")}
                className={topKBad ? "invalid" : ""} onChange={number("generation.top_k")} />
            </Field>
          </div>
          <div className="small muted mt-3">
            {engine && <div>{spec.runtime.name === "vllm"
              ? t("vLLM drafts at the temperature only; top_p and top_k apply to the target's verification.")
              : t("SGLang drafts greedily; the target verifies by sampling.")}</div>}
            {sgl && g.top_p !== undefined && g.top_k !== undefined && <div className="error-text">{t("SGLang samples plain decoding and draft verification differently when top_k and top_p are both set; set only one.")}</div>}
          </div>
        </>
      )}
      {engine && (
        <div className="small muted mt-3">
          <div>{t("{provider} runs specdecode tasks (specdecode/…) only.", { provider })}</div>
          <div>{t("One GPU per run.")}</div>
          <div>{t("No effeval methods or preprocessing: plain checkpoints.")}</div>
          {targetDtype && <div>{t("The draft runs in the target's precision: {dtype}.", { dtype: targetDtype })}</div>}
          <div>{t("Not reported: TTFT, peak memory, draft time share.")}</div>
        </div>
      )}
    </>
  );
}
