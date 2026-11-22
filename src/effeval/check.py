"""Checks that need the compute node itself: GPU identity, memory, method code, the model's architecture
layout, and site matches on the model.

Models are instantiated on the meta device, so these checks download configs but no weights.
"""

from __future__ import annotations

import importlib

import torch
from transformers import AutoConfig

from effeval.core import evaluator as evaluators
from effeval.core.catalog import Catalog
from effeval.core.fragments import QuantRule
from effeval.core.spec import JobSpec, Plan, ResolvedMethod, ResolvedModel
from effeval.core.types import Issue, Severity
from effeval.ops import introspect, sites
from effeval.runtime import hf

MEMORY_HEADROOM = 0.9


def _error(code: str, message: str, hint: str = "") -> Issue:
    return Issue(code=code, severity=Severity.ERROR, message=message, hint=hint)


def _kernel_issues(name: str) -> list[Issue]:
    """Whether this node can run kernel `name`: a CUDA GPU of capability 8.0+ and the kernel's package."""
    if not torch.cuda.is_available():
        return [_error("kernel.no_gpu", f"benchmarking {name} needs a CUDA GPU; none is visible")]
    major, minor = torch.cuda.get_device_capability(0)
    if major < 8:
        return [
            _error("kernel.gpu_too_old", f"{name} needs compute capability 8.0+, this GPU is {major}.{minor}")
        ]
    try:
        importlib.import_module(name)
    except ImportError as exc:
        return [
            _error(
                "kernel.not_installed",
                f"cannot import {name}: {exc}",
                hint="pip install git+https://github.com/IST-DASLab/marlin" if name == "marlin" else "",
            )
        ]
    return []


def _roles(job_spec: JobSpec) -> list[tuple[ResolvedModel, ResolvedMethod]]:
    """The (model, method) of every role the job loads: the target, then the draft if there is one."""
    roles = [(job_spec.model, job_spec.method)]
    if job_spec.decoding is not None:
        roles.append((job_spec.decoding.draft.model, job_spec.decoding.draft.method))
    return roles


def check(plan: Plan, catalog: Catalog) -> list[Issue]:
    issues: list[Issue] = []
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    visible = torch.cuda.device_count() if device else 0
    for job in plan.jobs:
        wanted = job.spec.hardware.gpu
        if wanted is None:
            continue
        spec = catalog.gpu(wanted)
        if device is None:
            issues.append(_error("hardware.missing", f"plan targets {wanted} but no CUDA device is visible"))
        elif spec is None or device not in spec.device_names:
            issues.append(
                _error(
                    "hardware.mismatch",
                    f"plan targets {wanted} but this node has {device!r}",
                    hint="re-render for this GPU, or add the device name to hwdb if it is the same GPU",
                )
            )
        if visible < job.spec.hardware.count:
            issues.append(
                _error("hardware.count", f"plan needs {job.spec.hardware.count} GPU(s), {visible} visible")
            )
        break  # every job of a plan targets the same hardware

    for job in plan.jobs:
        task = catalog.tasks[job.spec.task.name]
        if task.benchmark is None:
            continue
        issues.extend(_kernel_issues(task.benchmark.kernel))
        break

    checked: set[str] = set()
    for job in plan.jobs:
        spec = job.spec.evaluator
        if spec.name in checked:
            continue
        checked.add(spec.name)
        try:
            evaluators.load(spec.name)
        except (LookupError, TypeError, ValueError, ImportError) as exc:
            issues.append(
                _error(
                    "evaluator.unavailable",
                    f"evaluator {spec.name!r} cannot be loaded on this node: {exc}",
                    hint="install the effeval compute extra and re-install effeval so entry points register",
                )
            )
            continue
        mismatch = evaluators.library_issue(spec.library)
        if mismatch is not None:
            issues.append(
                _error(
                    "evaluator.library_mismatch",
                    f"{spec.name}: {mismatch}",
                    hint="install the pinned version, or re-plan with a listing synced from this node",
                )
            )

    seen: set[tuple[str, str | None, str]] = set()
    for job in plan.jobs:
        for model, method in _roles(job.spec):
            issues.extend(_role_issues(model, method, seen, catalog, device, visible))
    return issues


def _role_issues(
    model: ResolvedModel,
    method: ResolvedMethod,
    seen: set[tuple[str, str | None, str]],
    catalog: Catalog,
    device: str | None,
    visible: int,
) -> list[Issue]:
    """Manifest, architecture, memory and site checks of one (model, method) role, once per pair."""
    issues: list[Issue] = []
    key = (model.name, model.revision, method.model_dump_json())
    if key in seen:
        return issues
    seen.add(key)
    manifest = catalog.manifests[method.name]
    if manifest.entry is not None:
        try:
            hf.method_function(manifest)
        except hf.ManifestMismatch as exc:
            issues.append(_error("method.manifest_mismatch", str(exc)))
        except (ImportError, AttributeError, ValueError) as exc:
            issues.append(
                _error(
                    "method.entry_unavailable",
                    f"cannot import {manifest.entry}: {exc}",
                    hint="install the method's package on this node",
                )
            )
    config = AutoConfig.from_pretrained(model.checkpoint, revision=model.revision)
    live = introspect.extract_architecture(config, source=model.checkpoint)
    if live.name != model.architecture:
        issues.append(
            _error(
                "model.architecture_mismatch",
                f"{model.name} is registered as {model.architecture} but its checkpoint builds {live.name}",
            )
        )
        return issues
    stored = catalog.architectures[live.name]
    if live.model_dump(exclude={"extracted_with"}) != stored.model_dump(exclude={"extracted_with"}):
        issues.append(
            _error(
                "architecture.layout_changed",
                f"transformers {live.extracted_with['transformers']} lays out {live.name} "
                f"differently from the stored table (extracted with "
                f"{stored.extracted_with['transformers']})",
                hint="re-extract with `effeval arch extract` and review the diff",
            )
        )
        return issues
    meta = introspect.meta_model(config)
    weight_bytes = catalog.models[model.name].params * (4 if model.dtype == "float32" else 2)
    if device is not None:
        free = sum(torch.cuda.mem_get_info(i)[0] for i in range(visible))
        if weight_bytes > MEMORY_HEADROOM * free:
            issues.append(
                _error(
                    "memory.insufficient",
                    f"{model.name} weights need {weight_bytes / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free",
                )
            )
    if "quant_rules" in manifest.uses:
        rules = [QuantRule.model_validate(r) for r in method.config["rules"]]
        try:
            _, hits = sites.resolve(meta, rules)
        except (sites.SiteError, ValueError) as exc:
            issues.append(_error("site.unmatched", str(exc)))
            return issues
        for hit in hits:
            layers = f"layers {hit.layers}" if hit.layers else "lm_head"
            issues.append(
                Issue(
                    code="site.hits",
                    severity=Severity.INFO,
                    message=f"{manifest.name} rule {hit.rule}: {hit.locations} locations, "
                    f"{layers}, sites {hit.sites}",
                )
            )
    return issues
