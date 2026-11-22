"""Model-free time estimates for a plan (architecture.md §9.3). Runs on the server; needs no GPU.

Priority of evidence: measured timings of earlier records on the same GPU, then a roofline prior.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable

from pydantic import BaseModel

from effeval.core.catalog import Catalog, GpuSpec, sample_count
from effeval.core.spec import Job, Plan
from effeval.core.types import Mode
from effeval.plan import model_group_key

PREFILL_MFU = 0.35  # assumed fraction of dense peak FLOPs reached by HF prefill
DECODE_BW_EFFICIENCY = 0.6  # assumed fraction of HBM bandwidth reached by HF decode
LOAD_GB_PER_S = 2.0  # assumed checkpoint load speed

History = Callable[[Job, str], list[tuple[float, float]]]
"""(job, gpu name) -> measured (prepare_s, measure_s) of comparable earlier records."""


class JobEstimate(BaseModel):
    job_id: str
    role: str
    task: str
    prepare_s: float | None
    measure_s: float | None
    basis: str


class Estimate(BaseModel):
    hardware: str | None
    jobs: list[JobEstimate]
    total_s: float | None
    notes: list[str]


def _roofline(
    job: Job, catalog: Catalog, gpu: GpuSpec, params: float, first_in_group: bool
) -> tuple[float, float | None]:
    spec = job.spec
    bytes_per_param = 4 if spec.model.dtype == "float32" else 2
    weight_bytes = params * bytes_per_param
    flops = gpu.bf16_dense_tflops * 1e12 * PREFILL_MFU * gpu_count(job)
    bandwidth = gpu.hbm_tb_per_s * 1e12 * DECODE_BW_EFFICIENCY * gpu_count(job)

    draft = spec.decoding.draft if spec.decoding is not None else None
    draft_params = catalog.models[draft.model.name].params if draft is not None else 0.0
    draft_bytes = draft_params * (4 if draft is not None and draft.model.dtype == "float32" else 2)

    prepare = 0.0
    if first_in_group:
        prepare = (weight_bytes + draft_bytes) / (LOAD_GB_PER_S * 1e9)
        calibration = spec.method.config.get("calibration")
        if spec.method.name == "gptq" and isinstance(calibration, dict):
            tokens = calibration["n_samples"] * calibration["seqlen"]
            prepare += 2 * (2 * params * tokens) / flops  # two passes per layer: collect Hessians, re-run

    size = catalog.tasks[spec.task.name].size
    if size is None:
        return prepare, None
    limit = spec.task.limit
    if isinstance(limit, float):
        samples = sample_count(limit, size.samples)
    else:
        samples = size.samples if limit is None else min(size.samples, limit * size.subtasks)
    if spec.task.mode == Mode.LOGLIKELIHOOD:
        measure = 2 * params * samples * size.choices * size.prompt_tokens / flops
    elif spec.task.mode in (Mode.GENERATION, Mode.SPEED):
        prefill = 2 * (params + draft_params) * samples * size.prompt_tokens / flops
        output = min(size.output_tokens, spec.generation.max_new_tokens)
        steps = samples * output / spec.generation.batch_size
        round_bytes = weight_bytes
        if spec.decoding is not None:
            # worst case: every round accepts nothing, so a round yields one token but runs k draft steps
            round_bytes += spec.decoding.num_speculative_tokens * draft_bytes
        measure = prefill + steps * round_bytes / bandwidth
    else:
        raise ValueError(f"unsupported mode {spec.task.mode!r}")
    return prepare, measure


def gpu_count(job: Job) -> int:
    return job.spec.hardware.count


def estimate(plan: Plan, catalog: Catalog, history: History | None) -> Estimate:
    notes: list[str] = []
    gpu_names = {j.spec.hardware.gpu for j in plan.jobs}
    gpu_name = next(iter(gpu_names)) if len(gpu_names) == 1 else None
    gpu = catalog.gpu(gpu_name) if gpu_name else None
    if gpu is None:
        notes.append("set hardware.gpu to a GPU from hwdb to get an estimate")
    seen_groups: set[str] = set()
    estimates = []
    for job in plan.jobs:
        group = model_group_key(job)
        first = group not in seen_groups
        seen_groups.add(group)
        measured = history(job, gpu.name) if (history is not None and gpu is not None) else []
        if measured:
            prepare = statistics.median(m[0] for m in measured) if first else 0.0
            measure = statistics.median(m[1] for m in measured)
            basis = f"median of {len(measured)} earlier record(s) on {gpu.name}"
        elif gpu is not None:
            params = catalog.models[job.spec.model.name].params
            prepare, measure = _roofline(job, catalog, gpu, params, first)
            basis = (
                f"roofline prior: {params:.3g} params, {PREFILL_MFU:.0%} of peak FLOPs, "
                f"{DECODE_BW_EFFICIENCY:.0%} of HBM bandwidth (assumed)"
                + (", speculative decoding at one token per round (upper bound)" if job.spec.decoding else "")
                if measure is not None
                else f"no size prior for {job.spec.task.name}; measured history only"
            )
        else:
            prepare = measure = None
            basis = "no estimate: set hardware.gpu"
        estimates.append(
            JobEstimate(
                job_id=job.job_id,
                role=job.role,
                task=job.spec.task.name,
                prepare_s=prepare,
                measure_s=measure,
                basis=basis,
            )
        )
    totals = [
        e.prepare_s + e.measure_s for e in estimates if e.prepare_s is not None and e.measure_s is not None
    ]
    total = sum(totals) if len(totals) == len(estimates) else None
    notes.append("queueing for a node is outside effeval and not estimated")
    return Estimate(hardware=gpu.name if gpu else None, jobs=estimates, total_s=total, notes=notes)
