import type { ModelEntry } from "../api";

type Spec = Record<string, any>;

/** An English message (an i18n key) and the values for its placeholders. */
export type DraftProblem = { text: string; vars: Record<string, string> };

/** Why `draft` cannot be the draft of `target`, mirroring the server's validation; null when it would accept it. */
export function draftProblem(
  models: Record<string, ModelEntry>,
  target: string | undefined,
  draft: string,
  opts: { runtime?: string; targetDtype?: string; draftDtype?: string; runtimeLabel?: string },
): DraftProblem | null {
  const targetEntry = target ? models[target] : undefined;
  if (!target || !targetEntry) return null;
  const draftEntry = models[draft];
  if (!draftEntry) return { text: "This model is not registered.", vars: {} };
  const unshaped = !targetEntry.shape ? target : !draftEntry.shape ? draft : undefined;
  if (unshaped !== undefined) return { text: "Cannot check the vocabulary: {model} has no recorded shape.", vars: { model: unshaped } };
  const dv = draftEntry.shape!.vocab;
  const tv = targetEntry.shape!.vocab;
  if (dv !== tv) {
    return { text: "Different vocabulary: {draft} has {dv} tokens, {target} has {tv}.", vars: { draft, dv: String(dv), target, tv: String(tv) } };
  }
  const draftDtype = opts.draftDtype ?? draftEntry.dtype;
  const targetDtype = opts.targetDtype ?? targetEntry.dtype;
  if (opts.runtime && opts.runtime !== "hf" && draftDtype !== targetDtype) {
    return {
      text: "{provider} runs the draft in the target's precision ({target_dtype}); this draft is {draft_dtype}.",
      vars: { provider: opts.runtimeLabel ?? opts.runtime, target_dtype: targetDtype, draft_dtype: draftDtype },
    };
  }
  return null;
}

/** draftProblem for every registered model, keyed by code, only the ones with a problem. */
export function draftProblems(
  models: Record<string, ModelEntry>,
  target: string | undefined,
  opts: Parameters<typeof draftProblem>[3],
): Record<string, DraftProblem> {
  const out: Record<string, DraftProblem> = {};
  for (const code of Object.keys(models)) {
    const problem = draftProblem(models, target, code, opts);
    if (problem) out[code] = problem;
  }
  return out;
}

/** The spec with decoding.draft.model.dtype set to the target's precision on vllm/sglang when the draft's own precision differs, and removed otherwise (also on hf); the draft model may be a bare name or a {name, dtype?} object; the same object when nothing changes. */
export function alignDraftDtype(models: Record<string, ModelEntry>, spec: Spec): Spec {
  const draftModel = spec.decoding?.draft?.model;
  if (!draftModel) return spec;
  const ref: Spec = typeof draftModel === "string" ? { name: draftModel } : draftModel;
  const targetDtype: string | undefined = spec.model?.dtype ?? models[spec.model?.name]?.dtype;
  const engine = !!spec.runtime?.name && spec.runtime.name !== "hf";
  const own = models[ref.name]?.dtype;
  const want = engine && targetDtype !== undefined && targetDtype !== own ? targetDtype : undefined;
  if (ref.dtype === want) return spec;
  const { dtype: _drop, ...model } = ref;
  const nextModel = want === undefined ? model : { ...model, dtype: want };
  return { ...spec, decoding: { ...spec.decoding, draft: { ...spec.decoding.draft, model: nextModel } } };
}
