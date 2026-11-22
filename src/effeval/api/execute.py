"""Compute-node operations: connectivity check, plan execution, pushes, manifest export, direct evaluation."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml
from pydantic import BaseModel

from effeval.check import check
from effeval.core import bundle as bundle_io
from effeval.core import evaluator as evaluators
from effeval.core.catalog import Catalog, builtin_catalog
from effeval.core.manifest import MethodManifest, schema_hash
from effeval.core.registry import discover, manifest_from_impl, registered, resolve_entry
from effeval.core.spec import Plan
from effeval.core.types import InProcess, Issue, Record, Severity, utc_now
from effeval.plan import freeze, shard
from effeval.run import (
    RunOutcome,
    deliver,
    fingerprint,
    run_plan,
    upload_artifacts,
    with_repro,
    write_samples,
)
from effeval.store.http import ClientError, HttpClient


class Doctor(BaseModel):
    server: str | None
    online: bool
    reason: str
    username: str | None
    gpu: str | None
    gpu_count: int


def doctor(server: str | None, token: str | None, timeout: float = 3.0) -> tuple[Doctor, HttpClient | None]:
    """Measure whether results can be pushed from this machine right now. Never assumes connectivity."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    count = torch.cuda.device_count() if gpu else 0

    def report(online: bool, reason: str, username: str | None = None) -> Doctor:
        return Doctor(
            server=server, online=online, reason=reason, username=username, gpu=gpu, gpu_count=count
        )

    if server is None:
        return report(False, "no server configured (--server, EFFEVAL_SERVER or `effeval login`)"), None
    client = HttpClient(server, token)
    try:
        client.health(timeout=timeout)
    except ClientError as exc:
        return report(False, str(exc)), None
    if token is None:
        return report(False, "server reachable but not logged in (`effeval login` or EFFEVAL_TOKEN)"), None
    try:
        me = client.me()
    except ClientError as exc:
        return report(False, f"server reachable but the token was rejected: {exc}"), None
    return report(True, f"server reachable, logged in as {me['username']}", me["username"]), client


def load_document(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"{path} does not contain a mapping")
    return doc


def plan_from_document(
    doc: dict[str, Any], client: HttpClient | None, owner: str | None, server: str | None
) -> tuple[Plan | None, list[Issue]]:
    """A frozen plan (rendered by the UI) is used as is; a RunSpec is planned here, with reuse if online."""
    if "plan_version" in doc:
        return Plan.model_validate(doc), []
    catalog = builtin_catalog()
    if client is not None:
        catalog = catalog.with_manifests(client.manifests()).with_models(client.models())
    return freeze(doc, catalog, owner=owner, server=server, lookup=client)


def run_document(
    doc: dict[str, Any],
    *,
    doctor_report: Doctor,
    client: HttpClient | None,
    out_dir: Path,
    shard_spec: tuple[int, int] | None,
    confirm: Callable[[Plan, list[Issue]], bool],
    log: Callable[[str], None],
    to: str = "private",
    keep_going: bool = False,
) -> tuple[RunOutcome | None, list[Issue]]:
    """Plan (if needed), check on this node, confirm, run, uploading results to `to`.

    Returns (outcome or None if stopped, issues).
    """
    plan, issues = plan_from_document(doc, client, doctor_report.username, doctor_report.server)
    if plan is None:
        return None, issues
    if shard_spec is not None:
        plan = shard(plan, *shard_spec)
    catalog: Catalog = (
        builtin_catalog()
        .with_task_listings(list(plan.listings.values()))
        .with_manifests(list(plan.manifests.values()))
        .with_models(list(plan.models.values()))
    )
    issues = issues + check(plan, catalog)
    if any(i.severity == Severity.ERROR for i in issues) or not confirm(plan, issues):
        return None, issues
    outcome = run_plan(
        plan,
        catalog,
        client=client,
        out_dir=out_dir,
        owner=doctor_report.username,
        log=log,
        to=to,
        keep_going=keep_going,
    )
    return outcome, issues


def push_file(path: Path, client: HttpClient, to: str = "private") -> dict[str, Any]:
    """Push an offline bundle to `to`, uploading per-sample files that still exist on this machine.

    Returns the server's import result; records already stored (e.g. pasted earlier) come back as
    duplicates and keep the artifacts they were first stored with.
    """
    bundle = bundle_io.loads(path.read_text(encoding="utf-8"))
    records = [upload_artifacts(client, record) for record in bundle.records]
    return client.push(bundle_io.make_bundle(records), to=to)


def export_manifest(entry: str) -> MethodManifest:
    """Manifest of a function decorated with @method. Raises ValueError for undecorated functions."""
    impl = registered(resolve_entry(entry))
    if impl is None:
        raise ValueError(f"{entry} is not decorated with @effeval.core.registry.method")
    return manifest_from_impl(impl, builtin=False)


def discovered_manifests() -> list[MethodManifest]:
    """Manifests of every method installed packages announce under the `effeval.methods` entry point group."""
    return [manifest_from_impl(impl, builtin=False) for impl in discover()]


def sync_methods(client: HttpClient) -> list[MethodManifest]:
    """Upload the manifest of every discovered method to the server; returns what was uploaded."""
    manifests = discovered_manifests()
    for manifest in manifests:
        client.register_manifest(manifest)
    return manifests


def evaluate(
    handle: InProcess,
    *,
    model: str,
    tasks: list[str | dict[str, Any]],
    field: str,
    identity: dict[str, Any] | None = None,
    out_dir: Path = Path("effeval-results"),
    client: HttpClient | None = None,
) -> list[Record]:
    """Evaluate a model object you already prepared (architecture.md §5.4 path 1).

    `model` is the registered model code the handle was built from (see `effeval model add`). `tasks` are
    names or {"name", "mode", "params"} entries, e.g. {"name": "lm_eval/mmlu", "params": {"limit": 100}}.
    `identity` = {"method": name, "version": ..., "config": {...}} makes the records reusable and comparable;
    without it they are stored but never reused. Reference runs are not executed here: run the unmodified
    model with method `none` separately and the server links them through the reference RunKey.
    """
    name = (identity or {}).get("method", "direct")
    open_schema = {"type": "object"}
    manifest = MethodManifest(
        name=name,
        version=str((identity or {}).get("version", "unversioned")),
        field=field,
        runtime="hf",
        entry=None,
        config_schema=open_schema,
        schema_hash=schema_hash(open_schema),
    )
    catalog = builtin_catalog().with_manifests([manifest])
    if client is not None:
        catalog = catalog.with_models(client.models())
    raw = {
        "model": {"name": model, "dtype": str(next(handle.model.parameters()).dtype).replace("torch.", "")},
        "method": {"name": name, "preset": None, "config": (identity or {}).get("config", {})},
        "field": field,
        "tasks": tasks,
        "reuse": "never",
    }
    plan, issues = freeze(raw, catalog, owner=None, server=None, lookup=None)
    if plan is None:
        raise ValueError("; ".join(f"{i.code}: {i.message}" for i in issues if i.severity == Severity.ERROR))
    records = []
    for job in (j for j in plan.jobs if j.role == "main"):
        task = catalog.tasks[job.spec.task.name]
        result = evaluators.load(job.spec.evaluator.name).run(handle, job.spec, task, out_dir / job.job_id)
        evaluators.check_result_metrics(task, result.metrics)
        samples_path = out_dir / job.job_id / "per_sample.jsonl.gz"
        write_samples(result, samples_path)
        record = Record(
            record_id="rec_" + uuid.uuid4().hex,
            job_id=job.job_id,
            run_key=job.run_key,
            reference_key=job.reference_key,
            role="main",
            compare=job.compare,
            spec=job.spec.model_dump(mode="json"),
            metrics=result.metrics,
            protocol=result.protocol,
            env=fingerprint(catalog, handle.model.config._attn_implementation),
            fidelity=with_repro(job, result, catalog),
            artifacts={"per_sample": samples_path.resolve().as_uri()},
            produces=sorted(result.produces, key=lambda a: a.value),
            reusable=identity is not None,
            submitted_via="api",
            created_at=utc_now(),
        )
        records.append(deliver(client, record) if client is not None else record)
    return records
