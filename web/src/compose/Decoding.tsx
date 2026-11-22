import type { Describe, ModelEntry } from "../api";
import { useT } from "../i18n";
import { Field, Input, Segmented } from "../ui/controls";

type Spec = Record<string, any>;

export const MAX_SPECULATIVE_TOKENS = 16;

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

/** Models a draft may be: those with the target's vocabulary (all of them while the target's vocabulary is unknown). */
export function draftChoices(models: Record<string, ModelEntry>, target: string | undefined): Record<string, ModelEntry> {
  const vocab = target ? models[target]?.shape?.vocab : undefined;
  if (vocab === undefined) return models;
  return Object.fromEntries(Object.entries(models).filter(([, m]) => m.shape?.vocab === vocab));
}

/** The spec without its draft model when the draft no longer shares the target's vocabulary; the same spec otherwise. */
export function dropIncompatibleDraft(models: Record<string, ModelEntry>, spec: Spec): Spec {
  const draft = spec.decoding?.draft?.model?.name;
  if (!draft || draftChoices(models, spec.model?.name)[draft]) return spec;
  const { draft: _drop, ...decoding } = spec.decoding;
  return { ...spec, decoding };
}

/** Decoding settings of a Field config step: the provider and the tokens the draft proposes per step. */
export function Decoding({ spec, describe, update, err }: {
  spec: Spec;
  describe: Describe;
  update: (path: string, value: unknown) => void;
  err: (path: string) => string | undefined;
}) {
  const t = useT();
  const runtimes = Object.entries(describe.runtimes ?? { hf: { label: "transformers", library: null, available: true } });
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
        </Field>
        <Field label={t("Draft tokens per step (k)")} error={err("decoding.num_speculative_tokens")}>
          <Input type="number" min={1} max={MAX_SPECULATIVE_TOKENS} step={1} value={spec.decoding?.num_speculative_tokens ?? ""} placeholder="4"
            onChange={(e) => update("decoding.num_speculative_tokens", e.target.value === "" ? undefined : Number(e.target.value))} />
        </Field>
      </div>
      <p className="small muted mt-3">
        {t("The draft proposes k tokens per step and the target verifies them; results are compared with the target's own autoregressive decoding.")}
      </p>
    </>
  );
}
