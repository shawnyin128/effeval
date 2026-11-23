"""Hugging Face transformers runtime: load a model and apply a method to produce an InProcess handle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effeval.core.manifest import MethodManifest
from effeval.core.registry import manifest_from_impl, registered, resolve_entry
from effeval.core.spec import JobSpec, ResolvedMethod, ResolvedModel, ResolvedPreprocess, RuntimeSpec
from effeval.core.types import InProcess
from effeval.runtime.speculative import (
    GreedyAcceptance,
    ModelDrafter,
    SpeculativeDecoder,
    StaticCacheSettings,
)


class ManifestMismatch(RuntimeError):
    """The method code installed on this node does not match the manifest the plan was validated against."""


def load_model(model: ResolvedModel, runtime: RuntimeSpec) -> InProcess:
    """Load one checkpoint and its tokenizer."""
    loaded = AutoModelForCausalLM.from_pretrained(
        model.checkpoint,
        revision=model.revision,
        dtype=getattr(torch, model.dtype),
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation=runtime.attn_implementation,
    )
    loaded.eval()
    tokenizer = AutoTokenizer.from_pretrained(model.checkpoint, revision=model.revision)
    return InProcess(model=loaded, tokenizer=tokenizer)


def load(spec: JobSpec) -> InProcess:
    """Load the job's target checkpoint."""
    return load_model(spec.model, spec.runtime)


def method_function(manifest: MethodManifest) -> tuple[Any, type | None]:
    """Import the manifest's entry and confirm the installed code still has the manifest's config schema.

    Returns (function, pydantic config class or None). Raises ManifestMismatch on schema drift.
    """
    fn = resolve_entry(manifest.entry)
    impl = registered(fn)
    if impl is None:
        return fn, None
    installed = manifest_from_impl(impl, builtin=manifest.builtin)
    if installed.schema_hash != manifest.schema_hash or installed.version != manifest.version:
        raise ManifestMismatch(
            f"method {manifest.name!r}: installed code has version {installed.version} / "
            f"{installed.schema_hash}, the manifest says {manifest.version} / {manifest.schema_hash}. "
            "Re-export and re-register the manifest."
        )
    return fn, impl.config_model


PREPROCESS_ENTRIES = (
    "effeval.methods.preprocess.engine:make_context",
    "effeval.methods.preprocess.engine:apply",
)


def preprocess_summary(preprocess: Any, applied: Any) -> dict[str, Any]:
    """What `apply` did, compactly: entries per step, source versions, artifact keys and the edit log."""
    return {
        "steps": [
            {
                "kind": step.kind,
                "entries": [
                    entry.model_dump(mode="json", include={"position", "layers", "mode", "source"})
                    for entry in step.entries
                ],
            }
            for step in preprocess.steps
        ],
        "versions": dict(preprocess.versions),
        "artifacts": {name: ref.key for name, ref in applied.artifacts.items()},
        "edits": applied.edits,
    }


@torch.no_grad()
def _apply(
    handle: InProcess,
    model: ResolvedModel,
    method: ResolvedMethod,
    preprocess: ResolvedPreprocess | None,
    manifest: MethodManifest,
) -> InProcess:
    fn, config = None, None
    if manifest.entry is not None:
        fn, config_model = method_function(manifest)
        config = method.config if config_model is None else config_model.model_validate(method.config)
    summary = None
    if preprocess is not None:
        make_context, apply = (resolve_entry(entry) for entry in PREPROCESS_ENTRIES)
        weight = handle.model.get_input_embeddings().weight
        context = make_context(
            handle.model,
            handle.tokenizer,
            {
                "checkpoint": model.checkpoint,
                "revision": model.revision,
                "dtype": str(weight.dtype),
                "device": weight.device.type,
            },
            downstream=getattr(config, "rules", None),
        )
        summary = preprocess_summary(preprocess, apply(handle.model, preprocess, context))
    if fn is None:
        return replace(handle, preprocess_applied=summary)
    result = fn(handle.model, handle.tokenizer, config)
    if not isinstance(result, torch.nn.Module):
        raise TypeError(f"method {manifest.name!r} returned {type(result).__name__}; expected the model")
    return InProcess(model=result, tokenizer=handle.tokenizer, preprocess_applied=summary)


def prepare_role(
    model: ResolvedModel,
    method: ResolvedMethod,
    preprocess: ResolvedPreprocess | None,
    runtime: RuntimeSpec,
    manifest: MethodManifest,
) -> InProcess:
    """Load one role's checkpoint, apply its preprocessing (if any), then its method.

    The preprocessing sees the method's quantizer rules as `ctx.downstream` (None when the method has none).
    The returned handle's `preprocess_applied` summarises what the preprocessing did (None without any).
    """
    return _apply(load_model(model, runtime), model, method, preprocess, manifest)


def prepare(spec: JobSpec, manifest: MethodManifest) -> InProcess:
    """`prepare_role` for the job's target role."""
    return _apply(load(spec), spec.model, spec.method, spec.preprocess, manifest)


def decode_settings(runtime: RuntimeSpec) -> StaticCacheSettings | None:
    """The static-cache settings every decoder model of a job uses; None decodes with the dynamic cache.

    The single place that decides cache kind, graphs and capacity: the target of both arms and the draft
    all take their settings from here.
    """
    if not runtime.cuda_graphs:
        return None
    return StaticCacheSettings(runtime.static_cache_tokens, True)


def prepare_speculative(
    spec: JobSpec, manifests: Mapping[str, MethodManifest]
) -> tuple[InProcess, dict[str, Any] | None]:
    """Prepare the target and the draft role of a speculative job through `prepare_role`.

    Returns (the target handle with `decoder` set to a prepared SpeculativeDecoder, the draft's
    preprocessing summary or None). The decoder's `prepare()` result (cache allocation and graph capture of
    both models) is left on `decoder.prepared`. Raises ValueError when the two vocabularies differ and
    NotImplementedError for sampling verification.
    """
    decoding = spec.decoding
    if decoding is None:
        raise ValueError("prepare_speculative needs a job with decoding")
    if decoding.verification != "greedy":
        raise NotImplementedError(f"{decoding.verification} verification is not implemented")
    target = prepare_role(spec.model, spec.method, spec.preprocess, spec.runtime, manifests[spec.method.name])
    draft = decoding.draft
    drafted = prepare_role(
        draft.model, draft.method, draft.preprocess, spec.runtime, manifests[draft.method.name]
    )
    if target.tokenizer.get_vocab() != drafted.tokenizer.get_vocab():
        raise ValueError(
            f"draft {draft.model.name} and target {spec.model.name} have different tokenizer vocabularies"
        )
    if target.model.config.vocab_size != drafted.model.config.vocab_size:
        raise ValueError(
            f"draft {draft.model.name} and target {spec.model.name} have different config.vocab_size "
            f"({drafted.model.config.vocab_size} vs {target.model.config.vocab_size})"
        )
    settings = decode_settings(spec.runtime)
    decoder = SpeculativeDecoder(
        target.model,
        ModelDrafter(drafted.model, settings),
        decoding.num_speculative_tokens,
        GreedyAcceptance(),
        settings=settings,
    )
    decoder.prepare()
    return replace(target, decoder=decoder), drafted.preprocess_applied


def stop_token_ids(tokenizer: Any, generation_config: Any) -> frozenset[int]:
    """The ids that end a generation: the tokenizer's `eos_token_id` plus the generation config's
    `eos_token_id` (an int or a list), either of which may be absent."""
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    config_eos = getattr(generation_config, "eos_token_id", None)
    if config_eos is not None:
        ids.update(config_eos if isinstance(config_eos, (list, tuple)) else [config_eos])
    return frozenset(ids)
