"""Hugging Face transformers runtime: load a model and apply a method to produce an InProcess handle."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from effeval.core.manifest import MethodManifest
from effeval.core.registry import manifest_from_impl, registered, resolve_entry
from effeval.core.spec import JobSpec
from effeval.core.types import InProcess


class ManifestMismatch(RuntimeError):
    """The method code installed on this node does not match the manifest the plan was validated against."""


def load(spec: JobSpec) -> InProcess:
    model = AutoModelForCausalLM.from_pretrained(
        spec.model.checkpoint,
        revision=spec.model.revision,
        dtype=getattr(torch, spec.model.dtype),
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation=spec.runtime.attn_implementation,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(spec.model.checkpoint, revision=spec.model.revision)
    return InProcess(model=model, tokenizer=tokenizer)


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
def prepare(spec: JobSpec, manifest: MethodManifest) -> InProcess:
    """Load the base model, apply the job's preprocessing (if any), then its method.

    The preprocessing sees the method's quantizer rules as `ctx.downstream` (None when the method has none).
    The returned handle's `preprocess_applied` summarises what the preprocessing did (None without any).
    """
    handle = load(spec)
    fn, config = None, None
    if manifest.entry is not None:
        fn, config_model = method_function(manifest)
        config = (
            spec.method.config if config_model is None else config_model.model_validate(spec.method.config)
        )
    summary = None
    if spec.preprocess is not None:
        make_context, apply = (resolve_entry(entry) for entry in PREPROCESS_ENTRIES)
        weight = handle.model.get_input_embeddings().weight
        context = make_context(
            handle.model,
            handle.tokenizer,
            {
                "checkpoint": spec.model.checkpoint,
                "revision": spec.model.revision,
                "dtype": str(weight.dtype),
                "device": weight.device.type,
            },
            downstream=getattr(config, "rules", None),
        )
        summary = preprocess_summary(spec.preprocess, apply(handle.model, spec.preprocess, context))
    if fn is None:
        return replace(handle, preprocess_applied=summary)
    model = fn(handle.model, handle.tokenizer, config)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"method {manifest.name!r} returned {type(model).__name__}; expected the model")
    return InProcess(model=model, tokenizer=handle.tokenizer, preprocess_applied=summary)
