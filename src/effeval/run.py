"""Execute a plan on a compute node: prepare each model once, run its jobs, and deliver records.

Delivery follows the connectivity `doctor` measured: online pushes every record as soon as its job
finishes; offline (or after a failed push) appends to a local bundle file that can be pasted into the UI
or pushed later.
"""

from __future__ import annotations

import gc
import gzip
import importlib.metadata
import platform
import shutil
import socket
import subprocess
import time
import uuid
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import torch
from pydantic import BaseModel

from effeval.core import bundle as bundle_io
from effeval.core import evaluator as evaluators
from effeval.core.catalog import Catalog
from effeval.core.spec import Job, JobSpec, Plan
from effeval.core.types import (
    EngineHandle,
    EnvFingerprint,
    Fidelity,
    Handle,
    InProcess,
    QualityResult,
    Record,
    utc_now,
)
from effeval.metrics import derived
from effeval.plan import model_group_key
from effeval.runtime import determinism, hf
from effeval.runtime.engines.prepare import prepare_engine
from effeval.store.http import ClientError, HttpClient

PACKAGES = ("effeval", "torch", "transformers", "datasets", "accelerate", "lm_eval", "evalscope", "marlin")


class RunOutcome(BaseModel):
    plan_id: str
    pushed: list[str]
    bundle_path: str | None
    package_path: str | None = None
    failed: list[str] = []
    switched_offline: str | None


def fingerprint(catalog: Catalog, attn_implementation: str | None) -> EnvFingerprint:
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    known = catalog.gpu_for_device(device) if device else None
    driver = None
    if shutil.which("nvidia-smi"):
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        driver = (
            query.stdout.strip().splitlines()[0] if query.returncode == 0 and query.stdout.strip() else None
        )
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return EnvFingerprint(
        hostname=socket.gethostname(),
        python=platform.python_version(),
        gpu=known.name if known else device,
        gpu_count=torch.cuda.device_count() if device else 0,
        driver=driver,
        cuda=torch.version.cuda,
        packages=packages,
        effeval_git_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        attn_implementation=attn_implementation,
    )


def attention_used(protocol: dict[str, Any], default: str | None) -> str | None:
    """The attention function the decoder measured with (`decode_cache.target.attention`), else `default`."""
    target = (protocol.get("decode_cache") or {}).get("target") or {}
    return target.get("attention", default)


def with_repro(job: Job, result: QualityResult, catalog: Catalog) -> Fidelity:
    """Compare against the paper's reported value when the job is paper-faithful and the paper reports it."""
    fidelity = job.fidelity.model_copy(deep=True)
    if fidelity.preset is None or fidelity.deviations or job.spec.task.limit is not None:
        return fidelity
    method, name = fidelity.preset.split("/", 1)
    preset = catalog.presets.get(method, {}).get(name)
    for reported in preset.reported if preset else []:
        if (
            reported.model == job.spec.model.name
            and reported.task == job.spec.task.name
            and reported.metric in result.metrics
        ):
            gap = abs(Decimal(result.metrics[reported.metric]) - Decimal(reported.value))
            fidelity.repro = "pass" if gap <= Decimal(reported.tolerance) else "fail"
    return fidelity


RECORD_FILE = "record.json"


def saved_record(out_dir: Path, job_id: str) -> Record | None:
    """The record an earlier run of this job left under `out_dir`, if any."""
    for path in sorted(out_dir.glob(f"*/{job_id}/{RECORD_FILE}")):
        return Record.model_validate_json(path.read_text(encoding="utf-8"))
    return None


def write_samples(result: QualityResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(s.model_dump_json() for s in result.samples) + "\n"
    path.write_bytes(gzip.compress(lines.encode("utf-8"), mtime=0))


def prepare_models(spec: JobSpec, catalog: Catalog, workdir: Path) -> tuple[Handle, dict[str, object]]:
    """Prepare the models a job's spec calls for.

    Returns the handle and the protocol entries preparation adds.

    The provider and the decoding mode of the spec choose the path: an engine runtime starts its worker with
    its files in `workdir`; transformers prepares target and draft together for speculative decoding, only
    the target for plain decoding.
    """
    if spec.runtime.name != "hf":
        return prepare_engine(spec, workdir)
    if spec.decoding is not None:
        handle, draft_applied = hf.prepare_speculative(spec, catalog.manifests)
        return handle, {} if draft_applied is None else {"draft_preprocess_applied": draft_applied}
    return hf.prepare(spec, catalog.manifests[spec.method.name]), {}


def _groups(jobs: list[Job]) -> list[list[Job]]:
    groups: dict[str, list[Job]] = {}
    for job in jobs:
        groups.setdefault(model_group_key(job), []).append(job)
    return list(groups.values())


def local_path(uri: str) -> Path | None:
    """Filesystem path of a file:// artifact URI, or None for other URIs."""
    return Path(unquote(urlparse(uri).path)) if uri.startswith("file://") else None


def upload_artifacts(client: HttpClient, record: Record) -> Record:
    """Upload the local per-sample file (if it still exists) and point the record at the server copy."""
    path = local_path(record.artifacts.get("per_sample", ""))
    if path is None:
        return record
    if not path.exists():
        return record.model_copy(update={"artifacts": {}})
    uri = client.upload_artifact(path)
    return record.model_copy(update={"artifacts": {**record.artifacts, "per_sample": uri}})


def deliver(client: HttpClient, record: Record, to: str = "private") -> Record:
    record = upload_artifacts(client, record)
    client.push(bundle_io.make_bundle([record]), to=to)
    return record


def run_plan(
    plan: Plan,
    catalog: Catalog,
    *,
    client: HttpClient | None,
    out_dir: Path,
    owner: str | None,
    log: Callable[[str], None],
    to: str = "private",
    keep_going: bool = False,
) -> RunOutcome:
    """Run every job, uploading each result to `to` ("private", "pool" or "project:<name>").

    Jobs whose result an earlier run left in `out_dir` are not run again; their saved record is reused. A
    failure (model preparation or evaluation) is reported by heartbeat with its stage; it is raised at once
    unless `keep_going`, in which case the remaining jobs run and `RunOutcome.failed` lists the failures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / f"{plan.plan_id}.bundle.json"
    offline: list[Record] = []
    pushed: list[str] = []
    switched: str | None = None
    initial_client = client

    def keep_offline(record: Record) -> None:
        offline.append(record)
        bundle_path.write_text(bundle_io.dumps(bundle_io.make_bundle(offline)), encoding="utf-8")

    def beat(job: Job, status: str, message: str = "") -> None:
        nonlocal client, switched
        if client is None:
            return
        try:
            client.heartbeat(plan.plan_id, job.job_id, status, message)
        except ClientError as exc:
            switched = f"heartbeat failed: {exc}"
            log(f"server unreachable ({exc}); continuing offline, results go to {bundle_path}")
            client = None

    def store_result(record: Record) -> None:
        nonlocal client, switched
        if client is not None:
            try:
                pushed.append(deliver(client, record, to).record_id)
                return
            except ClientError as exc:
                switched = f"push failed: {exc}"
                log(f"push failed ({exc}); continuing offline, results go to {bundle_path}")
                client = None
        keep_offline(record)

    def give_up(jobs: list[Job], stage: str, exc: Exception) -> None:
        for job in jobs:
            beat(job, "failed", f"{stage}: {type(exc).__name__}: {exc}")
        if not keep_going:
            raise exc
        failed.extend(f"{job.spec.task.name} ({job.job_id}) {stage}: {exc}" for job in jobs)

    def run_job(
        job: Job, handle: Handle | None, attn: str | None, prepare_s: float, prepared: dict[str, object]
    ) -> None:
        determinism.apply(job.spec.determinism)
        beat(job, "running")
        task = catalog.tasks[job.spec.task.name]
        workdir = out_dir / plan.plan_id / job.job_id
        log(f"[{job.role}] {task.name} ({job.spec.task.mode.value}) via {job.spec.evaluator.name}")
        start = time.perf_counter()
        raw_before = handle.client.raw_counters() if isinstance(handle, EngineHandle) else None
        try:
            result = evaluators.load(job.spec.evaluator.name).run(handle, job.spec, task, workdir)
        except Exception as exc:
            give_up([job], "evaluate", exc)
            return
        measure_s = time.perf_counter() - start
        try:
            evaluators.check_result_metrics(task, result.metrics)
        except ValueError as exc:
            beat(job, "failed", str(exc))
            raise
        extra = [m for m in job.spec.metrics if m not in result.metrics]
        metrics = {**result.metrics, **derived.compute(extra, result.samples)}
        protocol = {**result.protocol, **prepared}
        if raw_before is not None:
            raw_after = handle.client.raw_counters()
            protocol["engine"] = {
                **prepared["engine"],
                "raw_counters": {name: raw_after[name] - raw_before[name] for name in raw_after},
            }
        if isinstance(handle, InProcess) and handle.preprocess_applied is not None:
            protocol["preprocess_applied"] = handle.preprocess_applied
        samples_path = workdir / "per_sample.jsonl.gz"
        write_samples(result, samples_path)
        record = Record(
            record_id="rec_" + uuid.uuid4().hex,
            job_id=job.job_id,
            run_key=job.run_key,
            reference_key=job.reference_key,
            role=job.role,
            compare=job.compare,
            spec=job.spec.model_dump(mode="json"),
            metrics=metrics,
            protocol=protocol,
            env=fingerprint(catalog, attention_used(protocol, attn)),
            fidelity=with_repro(job, result, catalog),
            artifacts={
                "per_sample": samples_path.resolve().as_uri(),
                bundle_io.SHA_KEY: bundle_io.file_sha256(samples_path.read_bytes()),
            },
            produces=sorted(result.produces, key=lambda a: a.value),
            owner=owner,
            timings={"prepare_s": prepare_s, "measure_s": measure_s},
            created_at=utc_now(),
        )
        log(f"  {metrics}")
        (workdir / RECORD_FILE).write_text(record.model_dump_json(), encoding="utf-8")
        store_result(record)
        beat(job, "done")

    failed: list[str] = []
    for group in _groups(plan.jobs):
        pending = []
        for job in group:
            saved = saved_record(out_dir, job.job_id)
            if saved is None:
                pending.append(job)
                continue
            log(f"[{job.role}] {job.spec.task.name}: finished in an earlier run, reusing its local result")
            store_result(saved)
            beat(job, "done", "reused local result")
        missing = [j for j in pending if j.spec.task.name not in catalog.tasks]
        for job in missing:
            name = job.spec.task.name
            give_up(
                [job],
                "task",
                LookupError(
                    f"task {name!r} is not in this machine's task catalog; a speed task needs its prompt set "
                    "file in the directory named by EFFEVAL_PROMPT_SETS here"
                ),
            )
        pending = [j for j in pending if j not in missing]
        benchmarks = [j for j in pending if catalog.tasks[j.spec.task.name].benchmark is not None]
        pending = [j for j in pending if j not in benchmarks]
        for job in benchmarks:
            run_job(job, None, None, 0.0, {})
        if not pending:
            continue
        first = pending[0]
        draft = None if first.spec.decoding is None else first.spec.decoding.draft
        determinism.apply(first.spec.determinism)
        log(
            f"preparing {first.spec.model.name} with {first.spec.method.name}"
            + ("" if draft is None else f" and draft {draft.model.name} with {draft.method.name}")
        )
        engine_dir = out_dir / plan.plan_id / f"engine-{first.job_id}"
        start = time.perf_counter()
        try:
            handle, prepared = prepare_models(first.spec, catalog, engine_dir)
        except Exception as exc:
            give_up(pending, "prepare", exc)
            continue
        prepare_s = time.perf_counter() - start
        attn = handle.model.config._attn_implementation if isinstance(handle, InProcess) else None
        try:
            for index, job in enumerate(pending):
                run_job(job, handle, attn, prepare_s if index == 0 else 0.0, prepared)
        except BaseException:
            if isinstance(handle, EngineHandle):
                try:
                    handle.client.close()
                except Exception as close_exc:
                    log(f"closing the engine failed: {type(close_exc).__name__}: {close_exc}")
            raise
        if isinstance(handle, EngineHandle):
            handle.client.close()
        del handle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if offline and initial_client is not None and client is None:
        log("retrying upload of results kept offline")
        try:
            for record in offline:
                pushed.append(deliver(initial_client, record, to).record_id)
            log("retry succeeded; the bundle file is kept as a local copy")
        except ClientError as exc:
            log(f"retry failed ({exc}); paste {bundle_path} into the UI Import page or run `effeval push`")
    package_path = None
    if offline:
        package_path = out_dir / f"{plan.plan_id}.results.tar.gz"
        bundle_io.write_package(package_path, bundle_io.make_bundle(offline))
    return RunOutcome(
        plan_id=plan.plan_id,
        pushed=pushed,
        bundle_path=str(bundle_path) if offline else None,
        package_path=None if package_path is None else str(package_path),
        switched_offline=switched,
        failed=failed,
    )
