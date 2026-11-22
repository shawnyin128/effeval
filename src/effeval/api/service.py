"""Operations shared by the server, MCP and the CLI. Torch-free: never imports execution code.

Every function takes the store and the acting user explicitly; HTTP handlers are thin wrappers around these.
"""

from __future__ import annotations

import functools
import json
import math
import os
import re
import shlex
import typing
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx
import yaml
from pydantic import BaseModel, Field, ValidationError

from effeval.core import bundle as bundle_io
from effeval.core.bundle import Bundle
from effeval.core.catalog import (
    Catalog,
    EvaluatorDef,
    ModelEntry,
    TaskDef,
    builtin_catalog,
    model_code,
    task_defs_from_listing,
)
from effeval.core.fragments import FRAGMENTS
from effeval.core.graph import positions
from effeval.core.listing import TaskListing
from effeval.core.manifest import MethodManifest, schema_hash
from effeval.core.preprocess import (
    DATA_SOURCES,
    NO_PAPER,
    RESERVED_SOURCES,
    SOURCE_POSITIONS,
    SOURCE_VERSIONS,
    Preprocess,
    PreprocessPresetDef,
    ResolvedPreprocess,
    RotationSource,
    ScalingSource,
    builtin_preprocess_presets,
    preprocess_label,
    preprocess_title,
    resolve_preprocess,
)
from effeval.core.spec import Job, JobSpec, Plan, RunSpec, run_key
from effeval.core.types import Fidelity, Issue, Record, Sample, Severity, SubmittedVia, Visibility, utc_now
from effeval.metrics import derived
from effeval.metrics.derived import DERIVED, applicable
from effeval.metrics.quality import COMPARATIVE
from effeval.plan import freeze, plan_for_jobs, shard
from effeval.preflight import Estimate, estimate
from effeval.store.sqlite import SqliteStore
from effeval.validate import validate


class Actor(BaseModel):
    username: str
    role: str
    can_publish: bool


class ServiceError(Exception):
    """A request the service refuses; `status` is the HTTP status the server should use."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


_LISTED: dict[str, tuple[str, tuple[EvaluatorDef, dict[str, TaskDef]]]] = {}
"""Converted task listing by evaluator as (digest, conversion); only the latest digest is kept."""
_DESCRIBED_TASKS: dict[tuple[tuple[str, str], ...], dict[str, Any]] = {}
"""The describe `tasks` payload by the digests of the stored listings; only the latest is kept."""


def _listed(store: SqliteStore) -> list[tuple[EvaluatorDef, dict[str, TaskDef]]]:
    base = builtin_catalog()
    converted = []
    for name, digest in sorted(store.task_listing_digests().items()):
        if name not in _LISTED or _LISTED[name][0] != digest:
            _LISTED[name] = (
                digest,
                task_defs_from_listing(store.task_listing(name), base.metrics, base.categories),
            )
        converted.append(_LISTED[name][1])
    return converted


def task_catalog(store: SqliteStore) -> Catalog:
    """The builtin catalog with every evaluator's stored task listing applied (no manifests or models)."""
    catalog = builtin_catalog()
    for evaluator, tasks in _listed(store):
        catalog = catalog.with_listed(evaluator, tasks)
    return catalog


def catalog_for(store: SqliteStore, viewer: str | None) -> Catalog:
    catalog = task_catalog(store).with_manifests([m for m, _owner, _vis in store.manifests(viewer)])
    return catalog.with_models([entry for entry, _by in store.models()])


def _described_tasks(store: SqliteStore, catalog: Catalog) -> dict[str, Any]:
    key = tuple(sorted(store.task_listing_digests().items()))
    if key not in _DESCRIBED_TASKS:
        _DESCRIBED_TASKS.clear()
        _DESCRIBED_TASKS[key] = {n: t.model_dump(mode="json") for n, t in catalog.tasks.items()}
    return _DESCRIBED_TASKS[key]


def _method_title(catalog: Catalog, name: str | None) -> str:
    manifest = catalog.manifests.get(name)
    return manifest.title if manifest is not None and manifest.title else str(name)


def _preprocess_of(raw: Any) -> ResolvedPreprocess | None:
    """The frozen preprocess of a stored spec (JobSpec dump), None when the spec has none."""
    return None if raw is None else ResolvedPreprocess.model_validate(raw)


def _preset_steps(preset: PreprocessPresetDef) -> Preprocess:
    unsupported = [path for path in preset.values if path != "preprocess.steps"]
    if unsupported:
        raise ValueError(f"preprocess preset {preset.name!r} sets unsupported paths {unsupported}")
    return Preprocess.model_validate({"steps": preset.values["preprocess.steps"].value})


def _preprocess_presets() -> dict[str, dict[str, Any]]:
    """Each preprocess preset with its resolved steps (all four parts, defaults filled) and versions."""
    out = {}
    for name, preset in builtin_preprocess_presets().items():
        resolved = resolve_preprocess(_preset_steps(preset))
        out[name] = {
            "name": preset.name,
            "title": preset.title,
            "description": preset.description,
            "paper": preset.paper.model_dump(mode="json"),
            "steps": [step.model_dump(mode="json") for step in resolved.steps],
            "versions": resolved.versions,
        }
    return out


@functools.cache
def _preprocess_sources() -> dict[str, dict[str, Any]]:
    """Per kind and source type: JSON schema, algorithm version, whether it needs calibration data, whether it
    is implemented (the reserved "trained" source is not), the position patterns it can be used at (None: any)
    and the positions among them that have no published method (`no_paper`)."""
    out: dict[str, dict[str, Any]] = {}
    for kind, union in (("rotation", RotationSource), ("scaling", ScalingSource)):
        out[kind] = {}
        for cls in typing.get_args(typing.get_args(union)[0]):
            source_type = cls.model_fields["type"].default
            out[kind][source_type] = {
                "version": SOURCE_VERSIONS[kind][source_type],
                "needs_data": source_type in DATA_SOURCES,
                "implemented": source_type not in RESERVED_SOURCES,
                "positions": SOURCE_POSITIONS.get(kind, {}).get(source_type),
                "no_paper": NO_PAPER.get(kind, {}).get(source_type, []),
                "schema": cls.model_json_schema(),
            }
    return out


def describe(store: SqliteStore, viewer: str | None) -> dict[str, Any]:
    """Everything a UI form or an agent needs: schemas, methods, presets, tasks, fields, evaluators, GPUs.

    `architectures[name].graph` is the effective layer graph; `models[code].positions` is the preprocessing
    position table of the model's shape (`core.graph.positions`); `preprocess_presets` and
    `preprocess_sources` describe model preprocessing.
    """
    catalog = catalog_for(store, viewer)
    graphs = {name: a.effective_graph() for name, a in catalog.architectures.items()}

    def model_view(m: ModelEntry) -> dict[str, Any]:
        graph = graphs.get(m.architecture)
        table = None if graph is None or m.shape is None else positions(graph, m.shape)
        return {**m.model_dump(mode="json"), "positions": table}

    return {
        "runspec_schema": RunSpec.model_json_schema(),
        "methods": {name: m.model_dump(mode="json") for name, m in catalog.manifests.items()},
        "presets": {
            m: {n: p.model_dump(mode="json") for n, p in by.items()} for m, by in catalog.presets.items()
        },
        "tasks": _described_tasks(store, catalog),
        "task_catalogs": store.task_listing_summaries(),
        "common_tasks": [n for n in catalog.common_tasks if n in catalog.tasks],
        "fields": {n: f.model_dump(mode="json") for n, f in catalog.fields.items()},
        "evaluators": {n: e.model_dump(mode="json") for n, e in catalog.evaluators.items()},
        "gpus": {n: g.model_dump(mode="json") for n, g in catalog.gpus.items()},
        "comparative_metrics": {n: c.description for n, c in COMPARATIVE.items()},
        "models": {code: model_view(m) for code, m in catalog.models.items()},
        "architectures": {
            name: {
                **a.model_dump(mode="json", exclude={"graph"}),
                "graph": None if graphs[name] is None else graphs[name].model_dump(mode="json"),
                "sites": a.site_names(),
            }
            for name, a in catalog.architectures.items()
        },
        "preprocess_presets": _preprocess_presets(),
        "preprocess_sources": _preprocess_sources(),
        "fragments": {name: prop for name, (prop, _adapter) in FRAGMENTS.items()},
        "metrics": {n: m.model_dump(mode="json") for n, m in catalog.metrics.items()},
        "categories": {n: c.model_dump(mode="json") for n, c in catalog.categories.items()},
        "averages": [a.model_dump(mode="json") for a in catalog.averages],
        "derived_metrics": sorted(DERIVED),
    }


def validate_spec(store: SqliteStore, viewer: str | None, raw: dict[str, Any]) -> list[Issue]:
    return validate(raw, catalog_for(store, viewer))


def _history(store: SqliteStore, viewer: str | None):
    def lookup(job: Job, gpu: str) -> list[tuple[float, float]]:
        records = store.visible_records(
            viewer,
            where="model = ? AND method_name = ? AND task_name = ? AND gpu = ?",
            params=(job.spec.model.name, job.spec.method.name, job.spec.task.name, gpu),
        )
        return [
            (r.timings["prepare_s"], r.timings["measure_s"])
            for r in records
            if {"prepare_s", "measure_s"} <= r.timings.keys()
        ]

    return lookup


class Rendered(BaseModel):
    plan: Plan | None
    issues: list[Issue]
    estimate: Estimate | None
    commands: list[Command]


class Command(BaseModel):
    """Shell steps for one node: write the frozen plan, run it, and push results if the run was offline."""

    write: str
    run: str
    push: str


RESULTS_DIR = "effeval-results"


def render_command(
    plan: Plan, server: str | None, shard_index: int | None, shard_count: int | None, to: str = "private"
) -> Command:
    """The plan as a heredoc, `effeval run` uploading to `to`, and the `effeval push` of its package."""
    name = f"effeval-{plan.plan_id[5:17]}.yaml"
    marker = "EFFEVAL_PLAN_" + plan.plan_id[5:17].upper()
    body = yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    if marker in body:
        raise ValueError("heredoc marker collides with plan content")
    target = ["--server", server] if server else []
    destination = ["--to", to] if to != "private" else []
    run = ["effeval", "run", "--config", name, "--out", RESULTS_DIR, *target]
    if shard_count and shard_count > 1:
        run += ["--shard", f"{shard_index}/{shard_count}"]
    push = ["effeval", "push", f"{RESULTS_DIR}/{plan.plan_id}.results.tar.gz", *target, *destination]
    return Command(
        write=f"cat > {name} <<'{marker}'\n{body}{marker}\n",
        run=shlex.join([*run, *destination]) + "\n",
        push=shlex.join(push) + "\n",
    )


def render(
    store: SqliteStore,
    actor: Actor,
    raw: dict[str, Any],
    server: str | None,
    shards: int = 1,
    to: str = "private",
) -> Rendered:
    """Freeze `raw` into a plan, estimate it and render one command per shard that uploads to `to`."""
    check_destination(store, actor, to)
    catalog = catalog_for(store, actor.username)
    plan, issues = freeze(raw, catalog, owner=actor.username, server=server, lookup=store)
    if plan is None:
        return Rendered(plan=None, issues=issues, estimate=None, commands=[])
    issues = issues + _duplicate_issues(store, actor, {j.run_key: _job_label(catalog, j) for j in plan.jobs})
    store.save_plan(plan.plan_id, actor.username, plan.model_dump_json())
    est = estimate(plan, catalog, _history(store, actor.username))
    if not plan.jobs:
        return Rendered(plan=plan, issues=issues, estimate=est, commands=[])
    if shards < 1:
        raise ServiceError(400, "shards must be >= 1")
    commands = [
        render_command(shard(plan, k, shards) if shards > 1 else plan, server, k, shards, to)
        for k in range(1, shards + 1)
    ]
    return Rendered(plan=plan, issues=issues, estimate=est, commands=commands)


def reuse_preview(store: SqliteStore, actor: Actor, raw: dict[str, Any]) -> dict[str, Any]:
    """Read-only: which jobs `raw` would plan, and which of them a stored result already covers.

    Returns {"issues", "jobs"}; each job has role, run_key, model, method, label, task, seed, repeat and a
    status:
    "reused" (a stored result is used), "run" (nothing stored), or "not_reused" (a stored result exists but
    the spec's `reuse` setting rules it out; `reason` says why). A job with a stored result carries `record`
    (record_id, owner, created_at, gpu, metrics). Nothing is saved.
    """
    catalog = catalog_for(store, actor.username)
    planned, issues = freeze(raw, catalog, owner=actor.username, server=None, lookup=None)
    if planned is None:
        return {"issues": [i.model_dump(mode="json") for i in issues], "jobs": []}
    covered, _ = freeze(raw, catalog, owner=actor.username, server=None, lookup=store)
    reused = {r.run_key for r in covered.reused}
    policy = raw.get("reuse", "auto")
    gpu = raw.get("hardware", {}).get("gpu")
    jobs = []
    for job in planned.jobs:
        need = frozenset().union(*(COMPARATIVE[c].requires for c in job.compare))
        record = store.find_reusable(job.run_key, viewer=actor.username, need=need, hardware=None)
        status, reason = "run", None
        if record is not None and job.run_key in reused:
            status = "reused"
        elif record is not None:
            status = "not_reused"
            if policy == "never":
                reason = "reuse is set to Never"
            elif gpu is None:
                reason = "reuse is limited to the same GPU but no target GPU is set"
            else:
                reason = (
                    f"the stored result ran on {record.env.gpu if record.env else 'an unknown GPU'}, "
                    f"not {gpu}"
                )
        jobs.append(
            {
                "role": job.role,
                "run_key": job.run_key,
                "model": job.spec.model.name,
                "method": job.spec.method.name,
                "label": _job_display(catalog, job),
                "task": job.spec.task.name,
                "seed": job.spec.determinism.seed,
                "repeat": job.spec.repeat,
                "status": status,
                "reason": reason,
                "record": None
                if record is None
                else {
                    "record_id": record.record_id,
                    "owner": record.owner,
                    "created_at": record.created_at,
                    "gpu": record.env.gpu if record.env else None,
                    "metrics": record.metrics,
                },
            }
        )
    return {"issues": [i.model_dump(mode="json") for i in issues], "jobs": jobs}


class ImportResult(BaseModel):
    imported: list[str]
    duplicates: list[str]
    comparisons_updated: list[str]
    redone: list[str] = []
    evidence_attached: list[str] = []
    unknown_metrics: list[str] = Field(
        default_factory=list,
        description="Metric names metrics.yaml does not declare; shown with default format.",
    )


def backfill(store: SqliteStore, record: Record) -> list[str]:
    """Compute the derived metrics a record should carry but lacks, from per-sample outputs the server holds.

    Wanted: the task's default derived metrics plus those the job asked for. Returns the names stored.
    """
    catalog = task_catalog(store)
    task = catalog.tasks.get(record.spec.get("task", {}).get("name"))
    if task is None:
        return []
    have = set(record.metrics) | set(store.derived(record.record_id))
    available = store.available_artifacts(record)
    missing = [
        n
        for n in dict.fromkeys([*task.derived, *record.spec.get("metrics", [])])
        if n in DERIVED and n not in have and applicable(n, available, task.sample_scores)
    ]
    if not missing:
        return []
    values = derived.compute(missing, store.load_samples(record.artifacts["per_sample"]) or [])
    store.put_derived(record.record_id, values)
    return sorted(values)


def backfill_all(store: SqliteStore) -> int:
    """Run `backfill` over every stored record; returns how many metric values were added."""
    return sum(len(backfill(store, record)) for record in store.all_records())


DESTINATION_HELP = "destination must be 'private', 'pool' or 'project:<name>'"


def check_destination(store: SqliteStore, actor: Actor, to: str) -> None:
    """Raise ServiceError unless the actor may upload to `to` ("private", "pool" or "project:<name>")."""
    if to == "pool" and not actor.can_publish:
        raise ServiceError(403, "this token cannot publish to the lab pool; upload privately or to a project")
    if to.startswith("project:"):
        _member_project(store, actor.username, to[len("project:") :])
    elif to not in {"private", "pool"}:
        raise ServiceError(400, DESTINATION_HELP)


def _known_comparators(record: Record) -> list[str]:
    """The record's stored comparator names that still exist; names of removed comparators are dropped."""
    return [name for name in record.compare if name in COMPARATIVE]


def _compute_comparisons(store: SqliteStore, main: Record) -> bool:
    """(Re)compute comparative metrics of `main` against the newest visible reference record."""
    compare = _known_comparators(main)
    if main.reference_key is None or not compare:
        return False
    reference = next(
        iter(
            store.visible_records(
                main.owner,
                where="run_key = ? AND visibility != 'retracted' AND record_id NOT IN"
                " (SELECT record_id FROM record_states)",
                params=(main.reference_key,),
            )
        ),
        None,
    )
    if reference is None:
        return False
    main_samples = store.load_samples(main.artifacts.get("per_sample", "")) or []
    ref_samples = store.load_samples(reference.artifacts.get("per_sample", "")) or []
    available = store.available_artifacts(main) & store.available_artifacts(reference)
    catalog = task_catalog(store)
    task_name = main.spec.get("task", {}).get("name")
    directional = {
        k for k in {*main.metrics, *reference.metrics} if catalog.metric_direction(task_name, k) is not None
    }
    main_metrics = {k: v for k, v in main.metrics.items() if k in directional}
    ref_metrics = {k: v for k, v in reference.metrics.items() if k in directional}
    values: dict[str, str] = {}
    for name in compare:
        metric = COMPARATIVE[name]
        if metric.requires <= available:
            values.update(metric.compute(main_metrics, ref_metrics, main_samples, ref_samples))
    store.replace_comparisons(main.record_id, reference.record_id, values)
    return True


def _verify_evidence(catalog: Catalog, record: Record, payload: bytes, samples: list[Sample]) -> None:
    expected = record.artifacts.get(bundle_io.SHA_KEY)
    if expected is not None:
        if bundle_io.file_sha256(payload) != expected:
            raise ServiceError(
                400, f"per-sample file does not match record {record.record_id} (checksum differs)"
            )
        return
    task = catalog.tasks.get(record.spec.get("task", {}).get("name"))
    candidates = [m.name for m in task.metrics] if task else []
    primary = next((m for m in [*candidates, *record.metrics] if m in DERIVED and m in record.metrics), None)
    if primary is None:
        raise ServiceError(
            400, f"record {record.record_id} has no file checksum and its metric cannot be rechecked"
        )
    recomputed = DERIVED[primary].compute(samples)
    if recomputed is None or not math.isclose(
        float(recomputed), float(record.metrics[primary]), rel_tol=1e-6
    ):
        raise ServiceError(
            400,
            f"per-sample file gives {primary} = {recomputed}, "
            f"record {record.record_id} says {record.metrics[primary]}",
        )


def attach_evidence(store: SqliteStore, actor: Actor, record_id: str, uri: str) -> list[str]:
    """Give a stored record the per-sample file it was measured with, then recompute what depended on it.

    The file must match the record's checksum (or, for records without one, reproduce its primary metric).
    Returns the records whose comparisons changed. ServiceError 409 if the record already has server evidence.
    """
    record = _owned(store, actor, record_id)
    if store.available_artifacts(record):
        raise ServiceError(409, f"record {record_id} already has its per-sample file on the server")
    payload = store.artifact_bytes(uri)
    _verify_evidence(task_catalog(store), record, payload, store.load_samples(uri) or [])
    store.set_artifact(record_id, uri)
    record = store.record(record_id)
    backfill(store, record)
    updated = [record_id] if _compute_comparisons(store, record) else []
    for main in store.records_referencing(record.run_key):
        if main.record_id != record_id and _compute_comparisons(store, main):
            updated.append(main.record_id)
    return updated


def import_package(store: SqliteStore, actor: Actor, payload: bytes, to: str = "private") -> ImportResult:
    """Import a results package: verify and store each per-sample file, then import the bundle with them."""
    try:
        bundle, files = bundle_io.read_package(payload)
    except bundle_io.BundleError as exc:
        raise ServiceError(400, str(exc)) from exc
    records = []
    for record in bundle.records:
        data = files.get(record.job_id)
        if data is None:
            records.append(record)
            continue
        try:
            uri = store.save_artifact(data, actor.username)
        except (ValueError, OSError) as exc:
            raise ServiceError(400, f"per-sample file of {record.job_id} is malformed: {exc}") from exc
        _verify_evidence(task_catalog(store), record, data, store.load_samples(uri) or [])
        records.append(record.model_copy(update={"artifacts": {**record.artifacts, "per_sample": uri}}))
    return import_bundle(store, actor, bundle_io.make_bundle(records), "paste", to)


def import_bundle(
    store: SqliteStore, actor: Actor, bundle: Bundle, via: SubmittedVia, to: str = "private"
) -> ImportResult:
    """Store every record as owned by the actor, send it to `to`, then refresh affected comparisons.

    `to` is "private", "pool" (publish) or "project:<name>" (link to that project; records already stored are
    linked too). A new record whose RunKey matches a record the actor marked for redo replaces it: the old one
    moves to the trash and the new one takes over its project links and, with publish rights, its visibility.
    """
    check_destination(store, actor, to)
    imported, duplicates, updated, redone, attached = [], [], [], [], []
    for record in bundle.records:
        if record.owner not in (None, actor.username):
            raise ServiceError(
                403, f"record {record.record_id} belongs to {record.owner}, not {actor.username}"
            )
        owned = record.model_copy(
            update={"owner": actor.username, "visibility": Visibility.PRIVATE, "submitted_via": via}
        )
        if not store.insert_record(owned):
            duplicates.append(record.job_id)
            stored = store.record(store.record_id_for_job(record.job_id))
            if store.available_artifacts(owned) and not store.available_artifacts(stored):
                updated.extend(attach_evidence(store, actor, stored.record_id, owned.artifacts["per_sample"]))
                attached.append(stored.record_id)
            continue
        imported.append(owned.record_id)
        backfill(store, owned)
        for old in store.redo_pending(owned.run_key, actor.username):
            store.link_records_from(old.record_id, owned.record_id, actor.username)
            if old.visibility == Visibility.PUBLIC and actor.can_publish:
                store.set_visibility(owned.record_id, Visibility.PUBLIC)
            store.set_state(old.record_id, "trashed", actor.username, superseded_by=owned.record_id)
            redone.append(old.record_id)
        if _compute_comparisons(store, owned):
            updated.append(owned.record_id)
        for main in store.records_referencing(owned.run_key):
            if main.record_id != owned.record_id and _compute_comparisons(store, main):
                updated.append(main.record_id)
    stored = imported + [rid for rid in map(store.record_id_for_job, duplicates) if rid is not None]
    if to == "pool":
        for record_id in stored:
            if store.record(record_id).visibility == Visibility.PRIVATE:
                store.set_visibility(record_id, Visibility.PUBLIC)
    elif to.startswith("project:"):
        store.link_records(to[len("project:") :], stored, actor.username)
    catalog = task_catalog(store)

    def declared(record: Record, metric: str) -> bool:
        task = catalog.tasks.get(record.spec.get("task", {}).get("name"))
        return metric in catalog.metrics or (task is not None and metric in {m.name for m in task.metrics})

    unknown = sorted({m for r in bundle.records for m in r.metrics if not declared(r, m)})
    return ImportResult(
        imported=imported,
        duplicates=duplicates,
        comparisons_updated=sorted(set(updated)),
        redone=redone,
        evidence_attached=attached,
        unknown_metrics=unknown,
    )


def _problems(
    store: SqliteStore, record: Record, state: dict[str, Any] | None, reference: str | None
) -> list[str]:
    problems = []
    if state is not None:
        problems.append(state["state"])
    if record.visibility == Visibility.RETRACTED:
        problems.append("retracted")
    ref_state = None if reference is None else store.state(reference)
    if ref_state is not None and ref_state["state"] == "trashed":
        problems.append("reference_trashed")
    return problems


def _reported_metrics(record: Record) -> dict[str, str]:
    """The metrics the evaluator itself reported; those the job asked effeval to derive are left out.

    The native evaluator's metrics are its own results, so all of them are kept.
    """
    if record.spec.get("evaluator", {}).get("name") == "native":
        return dict(record.metrics)
    asked = set(record.spec.get("metrics", []))
    return {k: v for k, v in record.metrics.items() if k not in asked}


def row(store: SqliteStore, catalog: Catalog, record: Record, viewer: str | None = None) -> dict[str, Any]:
    """Flat table row: fixed columns plus the full record for detail views and path filters.

    `label` is the display name "<preprocess title>[*] + <method title>" ("QuaRot + GPTQ"), the method title
    without preprocessing; `preprocess` is the frozen preprocess or None, `display["preprocess"]` its title.

    `state` is "active", "redo" or "trashed"; `problems` lists why the row may be invalid ("redo",
    "trashed", "retracted", "reference_trashed"); `projects` lists the viewer's projects that link the record.
    """
    state = store.state(record.record_id)
    comparisons = store.comparisons(record.record_id)
    mine = {p["name"] for p in store.projects(viewer)} if viewer is not None else set()
    spec = record.spec
    manifest = catalog.manifests.get(spec.get("method", {}).get("name"))
    display = {}
    for path in manifest.display if manifest else []:
        display[path] = spec.get("method", {}).get("config", {}).get(path)
    fidelity = record.fidelity
    preprocess = _preprocess_of(spec.get("preprocess"))
    if preprocess is not None:
        display["preprocess"] = preprocess_title(preprocess)
    return {
        "record_id": record.record_id,
        "created_at": record.created_at,
        "owner": record.owner,
        "visibility": record.visibility.value,
        "field": spec.get("field"),
        "model": spec.get("model", {}).get("name"),
        "dtype": spec.get("model", {}).get("dtype"),
        "method": spec.get("method", {}).get("name"),
        "preset": spec.get("method", {}).get("preset"),
        "preprocess": None if preprocess is None else preprocess.model_dump(mode="json"),
        "label": preprocess_label(preprocess, _method_title(catalog, spec.get("method", {}).get("name"))),
        "task": spec.get("task", {}).get("name"),
        "mode": spec.get("task", {}).get("mode"),
        "evaluator": spec.get("evaluator", {}).get("name"),
        "seed": spec.get("determinism", {}).get("seed"),
        "repeat": spec.get("repeat", 0),
        "gpu": record.env.gpu if record.env else None,
        "role": record.role,
        "metrics": _reported_metrics(record),
        "derived": {},
        "comparisons": {"reference": comparisons["reference"], "values": {}},
        "fidelity": {
            "preset": fidelity.preset,
            "confirmed": len(fidelity.confirmed),
            "assumed": list(fidelity.assumed),
            "deviations": [d.model_dump(mode="json") for d in fidelity.deviations],
            "repro": fidelity.repro,
        },
        "display": display,
        "annotation": store.annotation(record.record_id),
        "submitted_via": record.submitted_via,
        "run_key": record.run_key,
        "state": "active" if state is None else state["state"],
        "problems": _problems(store, record, state, comparisons["reference"]),
        "projects": [p for p in store.projects_of(record.record_id) if p in mine],
        "record": record.model_dump(mode="json"),
    }


def query(
    store: SqliteStore, viewer: str | None, scope: str, filters: dict[str, str]
) -> list[dict[str, Any]]:
    """Rows of one scope, narrowed by `filters` (row key or record path = value).

    scope: "mine" (the viewer's records), "pool" (public and retracted records of everyone) or
    "project:<name>" (records linked to that project, trashed ones included so they can be flagged).
    Raises ServiceError 400 on an unknown scope, 404/403 for a missing or foreign project.
    """
    if scope == "mine":
        records = [r for r in store.visible_records(viewer) if r.owner == viewer]
    elif scope == "pool":
        records = [r for r in store.visible_records(viewer) if r.visibility != Visibility.PRIVATE]
    elif scope.startswith("project:"):
        records = store.project_records(_member_project(store, viewer, scope[len("project:") :])["name"])
    else:
        raise ServiceError(400, f"scope must be 'mine', 'pool' or 'project:<name>', got {scope!r}")
    catalog = catalog_for(store, viewer)
    rows = [row(store, catalog, r, viewer) for r in records]
    for key, value in filters.items():
        rows = [r for r in rows if _match(r, key, value)]
    return rows


def _match(row_: dict[str, Any], key: str, value: str) -> bool:
    node: Any = row_ if key in row_ else row_["record"]
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return str(node) == value


def _owned(store: SqliteStore, actor: Actor, record_id: str) -> Record:
    record = store.record(record_id)
    if record is None:
        raise ServiceError(404, f"no record {record_id}")
    if record.owner != actor.username and actor.role != "admin":
        raise ServiceError(403, "only the owner (or an admin) can change this record")
    return record


def publish(store: SqliteStore, actor: Actor, record_id: str) -> None:
    record = _owned(store, actor, record_id)
    if not actor.can_publish:
        raise ServiceError(
            403, "this token cannot publish; publish from the UI or use a token with publish rights"
        )
    if record.visibility == Visibility.RETRACTED:
        raise ServiceError(409, "retracted records stay retracted")
    store.set_visibility(record_id, Visibility.PUBLIC)


def retract(store: SqliteStore, actor: Actor, record_id: str) -> None:
    record = _owned(store, actor, record_id)
    if record.visibility != Visibility.PUBLIC:
        raise ServiceError(409, "only public records can be retracted; private records are simply not shared")
    store.set_visibility(record_id, Visibility.RETRACTED)


def annotate(
    store: SqliteStore, actor: Actor, record_id: str, tags: list[str], note: str, status: str
) -> None:
    _owned(store, actor, record_id)
    if status not in {"unreviewed", "verified", "suspect"}:
        raise ServiceError(400, "status must be unreviewed, verified or suspect")
    store.set_annotation(record_id, tags, note, status, actor.username)


def dependents(store: SqliteStore, actor: Actor, record_id: str) -> dict[str, Any]:
    """What breaks if the record goes: records comparing against it (with owners) and projects linking it."""
    _owned(store, actor, record_id)
    return {
        "comparisons": [
            {"record_id": r.record_id, "owner": r.owner, "model": r.spec.get("model", {}).get("name")}
            for r in store.comparing_against(record_id)
        ],
        "projects": store.projects_of(record_id),
    }


def trash(store: SqliteStore, actor: Actor, record_id: str) -> None:
    """Move a record to the trash; it disappears from every view except project warnings and the trash."""
    _owned(store, actor, record_id)
    store.set_state(record_id, "trashed", actor.username)


def restore(store: SqliteStore, actor: Actor, record_id: str) -> None:
    """Bring a trashed record back as active. ServiceError 409 if it is not in the trash."""
    _owned(store, actor, record_id)
    state = store.state(record_id)
    if state is None or state["state"] != "trashed":
        raise ServiceError(409, "only trashed records can be restored")
    store.clear_state(record_id)


def purge(store: SqliteStore, actor: Actor, record_id: str) -> None:
    """Delete a trashed record for good. ServiceError 409 if it is not in the trash."""
    _owned(store, actor, record_id)
    state = store.state(record_id)
    if state is None or state["state"] != "trashed":
        raise ServiceError(409, "move the record to the trash first")
    store.purge(record_id)


def trashed(store: SqliteStore, actor: Actor) -> list[dict[str, Any]]:
    """Trash rows: the actor's trashed records, or everyone's for an admin, with who trashed them and when."""
    catalog = catalog_for(store, actor.username)
    items = store.trashed(None if actor.role == "admin" else actor.username)
    return [{**row(store, catalog, record, actor.username), "trashed": state} for record, state in items]


def redo(
    store: SqliteStore, actor: Actor, record_id: str, server: str | None, to: str = "private"
) -> Rendered:
    """Mark a record for redo and return the command that re-runs its job.

    The record stays (greyed) until a result with the same RunKey is imported, which replaces it. Raises
    ServiceError 409 for trashed records or when the stored job no longer has the RunKey it was stored under.
    """
    record = _owned(store, actor, record_id)
    state = store.state(record_id)
    if state is not None and state["state"] == "trashed":
        raise ServiceError(409, "restore the record before redoing it")
    check_destination(store, actor, to)
    spec = JobSpec.model_validate(record.spec)
    if run_key(spec) != record.run_key:
        raise ServiceError(
            409, "this record's identity changed with this effeval version; re-run it from Compose"
        )
    job = Job(
        job_id="",
        run_key=record.run_key,
        role=record.role,
        reference_key=record.reference_key,
        compare=_known_comparators(record),
        fidelity=record.fidelity,
        spec=spec,
    )
    reference = None
    if record.reference_key is not None:
        stored = next(
            iter(store.visible_records(actor.username, where="run_key = ?", params=(record.reference_key,))),
            None,
        )
        if stored is not None:
            reference = Job(
                job_id="",
                run_key=stored.run_key,
                role="reference",
                reference_key=None,
                compare=[],
                fidelity=Fidelity(),
                spec=JobSpec.model_validate(stored.spec),
            )
    catalog = catalog_for(store, actor.username)
    references = {} if reference is None else {reference.run_key: reference}
    plan = plan_for_jobs(
        [job], references, catalog, owner=actor.username, server=server, lookup=store, force=True
    )
    store.save_plan(plan.plan_id, actor.username, plan.model_dump_json())
    store.set_state(record_id, "redo", actor.username)
    est = estimate(plan, catalog, _history(store, actor.username))
    return Rendered(
        plan=plan, issues=[], estimate=est, commands=[render_command(plan, server, None, None, to)]
    )


def cancel_redo(store: SqliteStore, actor: Actor, record_id: str) -> None:
    """Clear a redo mark. ServiceError 409 if the record is not marked for redo."""
    _owned(store, actor, record_id)
    state = store.state(record_id)
    if state is None or state["state"] != "redo":
        raise ServiceError(409, "this record is not marked for redo")
    store.clear_state(record_id)


PROJECT_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _member_project(store: SqliteStore, viewer: str | None, name: str) -> dict[str, Any]:
    project = store.project(name)
    if project is None:
        raise ServiceError(404, f"no project {name!r}")
    if viewer not in project["members"]:
        raise ServiceError(403, f"you are not a member of project {name!r}")
    return project


def _managed_project(store: SqliteStore, actor: Actor, name: str) -> dict[str, Any]:
    project = store.project(name)
    if project is None:
        raise ServiceError(404, f"no project {name!r}")
    if actor.username != project["created_by"] and actor.role != "admin":
        raise ServiceError(403, "only the project creator (or an admin) can change the project")
    return project


def create_project(store: SqliteStore, actor: Actor, name: str, description: str) -> dict[str, Any]:
    """Create a project with the actor as creator and first member.

    Names are lower-case: letters, digits, '.', '_' and '-'. ServiceError 400 on a bad name, 409 if taken.
    """
    if not PROJECT_NAME.match(name):
        raise ServiceError(400, "project names use lower-case letters, digits, '.', '_' and '-'")
    try:
        store.create_project(name, description, actor.username)
    except ValueError as exc:
        raise ServiceError(409, str(exc)) from exc
    return store.project(name)


def projects(store: SqliteStore, actor: Actor) -> list[dict[str, Any]]:
    """Projects the actor belongs to."""
    return store.projects(actor.username)


def project(store: SqliteStore, actor: Actor, name: str) -> dict[str, Any]:
    return _member_project(store, actor.username, name)


def update_project(
    store: SqliteStore, actor: Actor, name: str, new_name: str | None, description: str | None
) -> dict[str, Any]:
    """Change a project's description and/or rename it (creator or admin). ServiceError 400 on a bad name, 409
    if the new name is taken."""
    _managed_project(store, actor, name)
    if new_name is not None and new_name != name and not PROJECT_NAME.match(new_name):
        raise ServiceError(400, "project names use lower-case letters, digits, '.', '_' and '-'")
    if description is not None:
        store.set_project_description(name, description)
    if new_name is not None and new_name != name:
        try:
            store.rename_project(name, new_name)
        except ValueError as exc:
            raise ServiceError(409, str(exc)) from exc
        name = new_name
    return store.project(name)


def delete_project(store: SqliteStore, actor: Actor, name: str) -> None:
    """Delete a project, its tables, snapshots, claims and record links (creator or admin); records stay."""
    _managed_project(store, actor, name)
    store.delete_project(name)


def add_member(store: SqliteStore, actor: Actor, name: str, username: str) -> None:
    _managed_project(store, actor, name)
    if store.user(username) is None:
        raise ServiceError(404, f"no user {username!r}")
    store.add_member(name, username)


def remove_member(store: SqliteStore, actor: Actor, name: str, username: str) -> None:
    project_ = _managed_project(store, actor, name)
    if username == project_["created_by"]:
        raise ServiceError(409, "the project creator cannot be removed")
    store.remove_member(name, username)


def link(store: SqliteStore, actor: Actor, name: str, record_ids: list[str]) -> None:
    """Link records the actor can see (their own, or lab pool ones) to a project they belong to."""
    _member_project(store, actor.username, name)
    for record_id in record_ids:
        record = store.record(record_id)
        if record is None or not store.can_view(actor.username, record_id):
            raise ServiceError(404, f"no record {record_id}")
        if record.owner != actor.username and record.visibility == Visibility.PRIVATE:
            raise ServiceError(403, f"record {record_id} is private to {record.owner}")
    store.link_records(name, record_ids, actor.username)


def unlink(store: SqliteStore, actor: Actor, name: str, record_id: str) -> None:
    _member_project(store, actor.username, name)
    store.unlink_record(name, record_id)


def _job_display(catalog: Catalog, job: Job) -> str:
    """The job's method with its preprocessing, as "<preprocess title>[*] + <method title>"."""
    return preprocess_label(job.spec.preprocess, _method_title(catalog, job.spec.method.name))


def _job_label(catalog: Catalog, job: Job) -> str:
    spec = job.spec
    method = spec.method.name if spec.preprocess is None else _job_display(catalog, job)
    return f"{spec.model.name} · {method} · {spec.task.name}"


def _duplicate_issues(store: SqliteStore, actor: Actor, labels: dict[str, str]) -> list[Issue]:
    """Warnings for jobs another member is already running (by heartbeat) or has claimed in a project."""
    activity = store.activity()
    issues = []
    for key, claims in store.claims(list(labels)).items():
        for claim in claims:
            if claim["claimed_by"] != actor.username:
                issues.append(
                    Issue(
                        code="run.claimed",
                        severity=Severity.WARNING,
                        message=(
                            f"{labels[key]}: claimed by {claim['claimed_by']} in project {claim['project']}"
                        ),
                        hint="check with them before running it again",
                    )
                )
    for key, label in labels.items():
        beat = activity.get(key)
        if beat is not None and beat["status"] == "running" and beat["owner"] != actor.username:
            issues.append(
                Issue(
                    code="run.in_progress",
                    severity=Severity.WARNING,
                    message=f"{label}: {beat['owner']} is already running this job (since {beat['at'][:16]})",
                    hint="its result will be reusable once it is uploaded",
                )
            )
    return issues


class TableRow(BaseModel):
    label: str = ""
    method: dict[str, Any] = Field(description="name, and optionally preset and config, as in a RunSpec.")
    preprocess: str | dict[str, Any] | None = Field(
        default=None,
        description="Model preprocessing: a preset name or a RunSpec `preprocess` ({preset?, steps?}).",
    )


class ExperimentTable(BaseModel):
    """A planned grid: for every model, one row per method config and one column per task."""

    name: str = Field(min_length=1)
    field: str
    models: list[str] = Field(min_length=1)
    tasks: list[str | dict[str, Any]] = Field(min_length=1)
    rows: list[TableRow] = Field(min_length=1)
    shared: dict[str, Any] = Field(default_factory=dict, description="RunSpec settings every row shares.")


def _merged(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in top.items():
        out[key] = (
            _merged(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
        )
    return out


def _preprocess_dict(value: str | dict[str, Any]) -> dict[str, Any]:
    """A `preprocess` given as a preset name, as the {"preset": name} a RunSpec takes."""
    return {"preset": value} if isinstance(value, str) else value


def _freeze_row(
    table: ExperimentTable, row: TableRow, model: str, tasks: list[Any], catalog: Catalog, owner: str
) -> tuple[Plan | None, list[Issue]]:
    top: dict[str, Any] = {
        "model": {"name": model},
        "method": row.method,
        "field": table.field,
        "tasks": tasks,
    }
    if row.preprocess is not None:
        top["preprocess"] = _preprocess_dict(row.preprocess)
    raw = _merged(table.shared, top)
    if "preprocess" in raw:
        raw["preprocess"] = _preprocess_dict(raw["preprocess"])
    return freeze(raw, catalog, owner=owner, server=None, lookup=None)


def _add_references(baselines: dict[str, dict[str, Any]], plan: Plan, catalog: Catalog) -> None:
    """Put the plan's reference jobs into the baseline rows, one row per reference config."""
    for job in plan.jobs:
        if job.role == "main":
            continue
        group = job.spec.model_copy(update={"task": None, "evaluator": None}).model_dump_json()
        base = baselines.setdefault(
            group,
            {
                "key": f"base{len(baselines)}",
                "label": job.spec.method.name,
                "title": _job_display(catalog, job),
                "method": {"name": job.spec.method.name},
                "preprocess": None,
                "baseline": True,
                "issues": [],
                "jobs": {},
                "invalid": {},
            },
        )
        runs = base["jobs"].setdefault(job.spec.task.name, [])
        if job.run_key not in {j.run_key for j in runs}:
            runs.append(job)


def _table_jobs(table: ExperimentTable, catalog: Catalog, owner: str) -> list[dict[str, Any]]:
    """Frozen jobs of every cell, per model: a baseline row (the references), then the table's rows.

    A cell's jobs are one per repeat (a single job without repeats), in repeat order.

    A row that cannot be planned as a whole is planned task by task; each task that still fails lands in the
    row's `invalid` (task -> error issues) instead of its `jobs`.
    """
    blocks = []
    for model in table.models:
        baselines: dict[str, dict[str, Any]] = {}
        rows = []
        for index, row_def in enumerate(table.rows):
            row = {
                "key": f"row{index}",
                "label": row_def.label,
                "title": "",
                "method": row_def.method,
                "preprocess": row_def.preprocess,
                "baseline": False,
                "issues": [],
                "jobs": {},
                "invalid": {},
            }
            plan, issues = _freeze_row(table, row_def, model, table.tasks, catalog, owner)
            per_task = [(None, plan, issues)]
            if plan is None:
                per_task = [
                    (task, *_freeze_row(table, row_def, model, [task], catalog, owner))
                    for task in table.tasks
                ]
            for task, plan, issues in per_task:
                warnings = [i for i in issues if i.severity == Severity.WARNING]
                row["issues"] += [i for i in warnings if i not in row["issues"]]
                if plan is None:
                    name = task if isinstance(task, str) else task["name"]
                    row["invalid"][name] = [i for i in issues if i.severity == Severity.ERROR]
                    continue
                _add_references(baselines, plan, catalog)
                for job in plan.jobs:
                    if job.role == "main":
                        row["jobs"].setdefault(job.spec.task.name, []).append(job)
            planned = next((j for runs in row["jobs"].values() for j in runs), None)
            method = row_def.method.get("name", "?")
            row["title"] = (
                _method_title(catalog, method) if planned is None else _job_display(catalog, planned)
            )
            row["label"] = row_def.label or (
                method if planned is None or planned.spec.preprocess is None else row["title"]
            )
            rows.append(row)
        blocks.append({"model": model, "rows": [*baselines.values(), *rows]})
    return blocks


def _cell(
    store: SqliteStore,
    actor: Actor,
    job: Job,
    need: frozenset,
    done: dict[str, Record],
    activity: dict[str, dict[str, Any]],
    claims: dict[str, list[dict[str, Any]]],
    project: str,
) -> dict[str, Any]:
    cell: dict[str, Any] = {"run_key": job.run_key, "role": job.role, "status": "pending"}
    record = done.get(job.run_key)
    if record is not None:
        cell["status"] = "done"
    else:
        record = store.find_reusable(job.run_key, viewer=actor.username, need=need, hardware=None)
        beat = activity.get(job.run_key)
        if record is not None:
            cell["status"] = "reusable"
        elif beat is not None and beat["status"] in ("running", "failed"):
            cell["status"] = beat["status"]
    if record is not None:
        cell.update(
            record_id=record.record_id,
            metrics=record.metrics,
            owner=record.owner,
            visibility=record.visibility.value,
            at=record.created_at,
            limit=record.spec["task"].get("limit"),
        )
    if job.run_key in activity:
        cell["activity"] = activity[job.run_key]
    claim = next((c for c in claims.get(job.run_key, []) if c["project"] == project), None)
    if claim is not None:
        cell["claim"] = {"by": claim["claimed_by"], "at": claim["claimed_at"]}
    return cell


_STATUS_PRECEDENCE = ("running", "failed", "pending", "reusable")


def _spread(cells: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Per metric over the cells that have it: exact mean (decimal string) and {"std", "n"} (sample std).

    A metric that is not a finite number in every such cell is left out. `std` is None when n is 1.
    """
    names = list(dict.fromkeys(m for c in cells for m in c.get("metrics", {})))
    means: dict[str, str] = {}
    spread: dict[str, dict[str, Any]] = {}
    for name in names:
        try:
            values = [Decimal(c["metrics"][name]) for c in cells if name in c.get("metrics", {})]
        except InvalidOperation:
            continue
        if not all(v.is_finite() for v in values):
            continue
        n = len(values)
        mean = sum(values, Decimal(0)) / n
        std = None
        if n > 1:
            std = (sum(((v - mean) ** 2 for v in values), Decimal(0)) / (n - 1)).sqrt()
        means[name] = format(mean, "f")
        spread[name] = {"std": None if std is None else format(std, "f"), "n": n}
    return means, spread


def _repeated_cell(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """One table cell from the cells of a job's repeats.

    A single run is returned unchanged. Otherwise `metrics` holds the mean over the repeats that have a
    result, `spread` the sample std and n per metric, and `repeats` the per-repeat cells; the status is "done"
    only when every repeat is done, else the most active missing one (running, failed, pending, reusable).
    """
    if len(parts) == 1:
        return parts[0]
    states = [p["status"] for p in parts]
    status = (
        "done" if all(s == "done" for s in states) else next(s for s in _STATUS_PRECEDENCE if s in states)
    )
    cell: dict[str, Any] = {"run_key": parts[0]["run_key"], "role": parts[0]["role"], "status": status}
    with_record = [p for p in parts if "record_id" in p]
    if with_record:
        first = with_record[0]
        cell.update(
            record_id=first["record_id"], owner=first["owner"], visibility=first["visibility"], at=first["at"]
        )
        cell["metrics"], cell["spread"] = _spread(with_record)
    activity = next((p["activity"] for p in parts if p.get("activity", {}).get("status") == "running"), None)
    activity = activity or next((p["activity"] for p in parts if "activity" in p), None)
    if activity is not None:
        cell["activity"] = activity
    claim = next((p["claim"] for p in parts if "claim" in p), None)
    if claim is not None:
        cell["claim"] = claim
    cell["repeats"] = parts
    return cell


def _table_view(store: SqliteStore, actor: Actor, project: str, stored: dict[str, Any]) -> dict[str, Any]:
    table = ExperimentTable.model_validate_json(stored["body"])
    catalog = catalog_for(store, actor.username)
    blocks = _table_jobs(table, catalog, actor.username)
    done: dict[str, Record] = {}
    for record in store.project_records(project):
        if store.state(record.record_id) is None:
            done.setdefault(record.run_key, record)
    jobs = [job for b in blocks for row in b["rows"] for runs in row["jobs"].values() for job in runs]
    needs: dict[str, frozenset] = {}
    for job in jobs:
        need = frozenset().union(*(COMPARATIVE[c].requires for c in job.compare))
        needs[job.run_key] = needs.get(job.run_key, frozenset()) | need
        if job.reference_key is not None:
            needs[job.reference_key] = needs.get(job.reference_key, frozenset()) | need
    activity = store.activity()
    claims = store.claims([j.run_key for j in jobs])
    counts: dict[str, int] = {}
    for block in blocks:
        for row in block["rows"]:
            cells = {}
            for task, runs in row.pop("jobs").items():
                cells[task] = _repeated_cell(
                    [
                        _cell(store, actor, job, needs[job.run_key], done, activity, claims, project)
                        for job in runs
                    ]
                )
            for task, errors in row.pop("invalid").items():
                cells[task] = {"status": "invalid", "issues": [i.model_dump(mode="json") for i in errors]}
            for cell in cells.values():
                counts[cell["status"]] = counts.get(cell["status"], 0) + 1
            row["cells"] = cells
            row["issues"] = [i.model_dump(mode="json") for i in row["issues"]]
    tasks = [t if isinstance(t, str) else t["name"] for t in table.tasks]
    return {
        "table_id": stored["table_id"],
        "created_by": stored["created_by"],
        "updated_at": stored["updated_at"],
        "definition": table.model_dump(mode="json"),
        "tasks": tasks,
        "blocks": blocks,
        "counts": counts,
    }


def _stored_table(store: SqliteStore, project: str, table_id: str) -> dict[str, Any]:
    found = next((t for t in store.tables(project) if t["table_id"] == table_id), None)
    if found is None:
        raise ServiceError(404, f"no table {table_id} in project {project!r}")
    return found


def tables(store: SqliteStore, actor: Actor, project: str) -> list[dict[str, Any]]:
    """Every experiment table of the project with the live status of each cell.

    Cell status: "done" (a result is in the project), "reusable" (a stored result could be added), "running" /
    "failed" (latest heartbeat of a job with that RunKey), else "pending"; claims and activity ride along.
    """
    _member_project(store, actor.username, project)
    return [_table_view(store, actor, project, t) for t in store.tables(project)]


def save_table(
    store: SqliteStore, actor: Actor, project: str, table_id: str | None, definition: dict[str, Any]
) -> dict[str, Any]:
    """Create (table_id None) or replace an experiment table. ServiceError 400 on an invalid definition."""
    _member_project(store, actor.username, project)
    try:
        table = ExperimentTable.model_validate(definition)
    except ValidationError as exc:
        raise ServiceError(400, f"invalid table: {exc}") from exc
    try:
        table_id = store.put_table(project, table_id, table.model_dump_json(), actor.username)
    except KeyError as exc:
        raise ServiceError(404, f"no table {table_id} in project {project!r}") from exc
    return _table_view(store, actor, project, _stored_table(store, project, table_id))


def delete_table(store: SqliteStore, actor: Actor, project: str, table_id: str) -> None:
    _member_project(store, actor.username, project)
    if not store.delete_table(project, table_id):
        raise ServiceError(404, f"no table {table_id} in project {project!r}")


def table_command(
    store: SqliteStore,
    actor: Actor,
    project: str,
    table_id: str,
    run_keys: list[str],
    server: str | None,
    *,
    force: bool,
) -> Rendered:
    """Command that runs the chosen cells and uploads the results to the project.

    Without `force`, cells covered by a stored result are reused instead of run; with it (explicit re-run)
    they run again. References the chosen cells compare against come along unless reusable. The issues warn
    about cells another member is running or has claimed. ServiceError 400 for RunKeys outside the table.
    """
    _member_project(store, actor.username, project)
    table = ExperimentTable.model_validate_json(_stored_table(store, project, table_id)["body"])
    catalog = catalog_for(store, actor.username)
    jobs = {
        job.run_key: job
        for block in _table_jobs(table, catalog, actor.username)
        for row in block["rows"]
        for runs in row["jobs"].values()
        for job in runs
    }
    unknown = [k for k in run_keys if k not in jobs]
    if unknown:
        raise ServiceError(400, f"not cells of this table: {unknown}")
    chosen = [jobs[k] for k in dict.fromkeys(run_keys)]
    references = {k: j for k, j in jobs.items() if j.role == "reference"}
    plan = plan_for_jobs(
        chosen, references, catalog, owner=actor.username, server=server, lookup=store, force=force
    )
    issues = _duplicate_issues(store, actor, {j.run_key: _job_label(catalog, j) for j in plan.jobs})
    store.save_plan(plan.plan_id, actor.username, plan.model_dump_json())
    est = estimate(plan, catalog, _history(store, actor.username))
    commands = [render_command(plan, server, None, None, f"project:{project}")] if plan.jobs else []
    return Rendered(plan=plan, issues=issues, estimate=est, commands=commands)


def _job_summary(jobs: list[dict[str, Any]], catalog: Catalog) -> str:
    def method(job: dict[str, Any]) -> str:
        spec = job["spec"]
        preprocess = _preprocess_of(spec.get("preprocess"))
        if preprocess is None:
            return spec["method"]["name"]
        return preprocess_label(preprocess, _method_title(catalog, spec["method"]["name"]))

    def names(values: list[str]) -> str:
        values = list(dict.fromkeys(values))
        return values[0] if len(values) == 1 else f"{values[0]} +{len(values) - 1}" if values else "—"

    main = [j for j in jobs if j["role"] == "main"]
    return " · ".join(
        [
            names([j["spec"]["model"]["name"] for j in main]),
            names([method(j) for j in main]),
            names([j["spec"]["task"]["name"] for j in main]),
        ]
    )


def runs(store: SqliteStore, actor: Actor) -> list[dict[str, Any]]:
    """The actor's plans, newest first, each with a title (custom or a model · method · task summary) and the
    record each job produced so far (job_id -> record_id)."""
    out = []
    catalog = catalog_for(store, actor.username)
    for run in store.plans(actor.username):
        jobs = run["plan"]["jobs"]
        run["summary"] = _job_summary(jobs, catalog)
        run["records"] = {j["job_id"]: rid for j in jobs if (rid := store.record_id_for_job(j["job_id"]))}
        out.append(run)
    return out


def _own_plan(store: SqliteStore, actor: Actor, plan_id: str) -> str:
    found = store.plan(plan_id)
    if found is None:
        raise ServiceError(404, f"no plan {plan_id}")
    if found[0] != actor.username and actor.role != "admin":
        raise ServiceError(403, "only the plan's owner (or an admin) can change it")
    return found[1]


def rename_run(store: SqliteStore, actor: Actor, plan_id: str, title: str) -> None:
    """Give a plan a custom title; an empty title brings back the automatic summary."""
    _own_plan(store, actor, plan_id)
    store.set_plan_title(plan_id, title.strip() or None)


def delete_run(store: SqliteStore, actor: Actor, plan_id: str) -> None:
    """Remove a plan from Runs; results it already produced are kept."""
    _own_plan(store, actor, plan_id)
    store.delete_plan(plan_id)


def run_command(
    store: SqliteStore, actor: Actor, plan_id: str, server: str | None, to: str = "private"
) -> Command:
    """The commands of a stored plan, unchanged, for running it (again) on a node."""
    return render_command(Plan.model_validate_json(_own_plan(store, actor, plan_id)), server, None, None, to)


def resume_run(
    store: SqliteStore, actor: Actor, plan_id: str, server: str | None, to: str = "private"
) -> Rendered:
    """Command that runs the stored plan's jobs that have no stored result yet, under their original job ids.

    Keeping the ids lets the runner skip jobs whose results are already on disk in the same --out directory.
    ServiceError 404 for unknown plans, 403 for plans of someone else.
    """
    found = store.plan(plan_id)
    if found is None:
        raise ServiceError(404, f"no plan {plan_id}")
    owner, body = found
    if owner != actor.username and actor.role != "admin":
        raise ServiceError(403, "only the plan's owner (or an admin) can resume it")
    check_destination(store, actor, to)
    original = Plan.model_validate_json(body)
    remaining = [j for j in original.jobs if store.record_id_for_job(j.job_id) is None]
    plan = original.model_copy(
        update={
            "plan_id": "plan_" + uuid.uuid4().hex,
            "created_at": utc_now(),
            "jobs": remaining,
            "reused": [],
        }
    )
    store.save_plan(plan.plan_id, actor.username, plan.model_dump_json())
    catalog = catalog_for(store, actor.username)
    est = estimate(plan, catalog, _history(store, actor.username))
    commands = [render_command(plan, server, None, None, to)] if remaining else []
    return Rendered(plan=plan, issues=[], estimate=est, commands=commands)


def freeze_table(store: SqliteStore, actor: Actor, project: str, table_id: str) -> dict[str, Any]:
    """Store the table exactly as it is now: per cell its record, reference record, values and comparisons.

    It also keeps, per column, the task's primary metric and category as they are now, plus the metric
    display definitions, averages and categories in use, so its exports never change with the catalog.
    """
    _member_project(store, actor.username, project)
    view = _table_view(store, actor, project, _stored_table(store, project, table_id))
    catalog = task_catalog(store)
    unknown_tasks = sorted(t for t in view["tasks"] if t not in catalog.tasks)
    if unknown_tasks:
        raise ServiceError(409, f"table references unknown tasks {unknown_tasks}")
    used: set[str] = set()
    for block in view["blocks"]:
        for row in block["rows"]:
            for cell in row["cells"].values():
                if cell["status"] == "done":
                    comparisons = store.comparisons(cell["record_id"])
                    cell["reference"] = comparisons["reference"]
                    cell["comparisons"] = {}
                    used.update(cell["metrics"])
    view["columns"] = []
    for task_name in view["tasks"]:
        task = catalog.tasks[task_name]
        category = catalog.categories[task.category]
        used.add(task.metrics[0].name)
        view["columns"].append(
            {
                "task": task_name,
                "metric": task.metrics[0].name,
                "category": task.category,
                "band": category.label,
                "order": category.order,
            }
        )
    view["metrics"] = {
        n: catalog.metrics[n].model_dump(mode="json") for n in sorted(used) if n in catalog.metrics
    }
    view["averages"] = [a.model_dump(mode="json") for a in catalog.averages]
    view["categories"] = {n: c.model_dump(mode="json") for n, c in catalog.categories.items()}
    snapshot_id = store.put_snapshot(project, table_id, json.dumps(view), actor.username)
    return store.snapshot(project, snapshot_id)


def snapshots(store: SqliteStore, actor: Actor, project: str) -> list[dict[str, Any]]:
    _member_project(store, actor.username, project)
    return store.snapshots(project)


def snapshot(store: SqliteStore, actor: Actor, project: str, snapshot_id: str) -> dict[str, Any]:
    _member_project(store, actor.username, project)
    found = store.snapshot(project, snapshot_id)
    if found is None:
        raise ServiceError(404, f"no snapshot {snapshot_id} in project {project!r}")
    return found


def rename_snapshot(store: SqliteStore, actor: Actor, project: str, snapshot_id: str, title: str) -> None:
    """Name a frozen version (any member); an empty title brings back the table's name."""
    _member_project(store, actor.username, project)
    if not store.set_snapshot_title(project, snapshot_id, title.strip() or None):
        raise ServiceError(404, f"no snapshot {snapshot_id} in project {project!r}")


def delete_snapshot(store: SqliteStore, actor: Actor, project: str, snapshot_id: str) -> None:
    """Delete a frozen version. ServiceError 403 unless the actor froze it or is an admin, 404 if unknown."""
    found = snapshot(store, actor, project, snapshot_id)
    if found["created_by"] != actor.username and actor.role != "admin":
        raise ServiceError(403, f"only {found['created_by']} or an admin can delete this version")
    store.delete_snapshot(project, snapshot_id)


def claim(store: SqliteStore, actor: Actor, project: str, run_keys: list[str]) -> None:
    """Claim cells (by RunKey) so members see who runs them. ServiceError 409 if someone else holds one."""
    _member_project(store, actor.username, project)
    try:
        store.claim(project, run_keys, actor.username)
    except ValueError as exc:
        raise ServiceError(409, str(exc)) from exc


def release(store: SqliteStore, actor: Actor, project: str, run_key: str) -> None:
    """Drop a claim; only its holder or an admin may. ServiceError 404 if the cell is not claimed."""
    _member_project(store, actor.username, project)
    held = store.claims([run_key]).get(run_key, [])
    mine = next((c for c in held if c["project"] == project), None)
    if mine is None:
        raise ServiceError(404, "this cell is not claimed")
    if mine["claimed_by"] != actor.username and actor.role != "admin":
        raise ServiceError(403, f"claimed by {mine['claimed_by']}; only they (or an admin) can release it")
    store.release(project, run_key)


FLIPS = ("to_wrong", "to_right", "both_right", "both_wrong")


def _subtask(sample_id: str) -> str:
    return sample_id.rsplit(":", 1)[0] if ":" in sample_id else ""


def _answer(sample: Sample) -> Any:
    if sample.prediction is not None:
        return sample.prediction
    if sample.choice_loglikelihoods:
        return max(range(len(sample.choice_loglikelihoods)), key=lambda i: sample.choice_loglikelihoods[i])
    return None


def sample_diff(
    store: SqliteStore,
    viewer: str | None,
    record_id: str,
    against: str | None,
    *,
    flip: str = "changed",
    subtask: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """Align a record's samples with its reference's and list them by how correctness moved.

    `against` defaults to the reference its comparisons used. `flip` is "changed" (to_wrong or to_right),
    "all" or one of FLIPS. Returns coverage, counts per flip and subtask, and one page of aligned samples.
    ServiceError 404 for invisible records, 409 when a per-sample file is missing on the server.
    """
    if flip not in ("changed", "all", *FLIPS):
        raise ServiceError(400, f"flip must be changed, all or one of {list(FLIPS)}")
    against = against or store.comparisons(record_id)["reference"]
    records = {}
    for rid in (record_id, against):
        found = None if rid is None else store.record(rid)
        if found is None or not store.can_view(viewer, rid):
            raise ServiceError(
                404, f"no record {rid}" if rid else "this record has no reference to compare against"
            )
        if not store.available_artifacts(found):
            raise ServiceError(409, f"record {rid} has no per-sample file on the server")
        records[rid] = found
    main = store.load_samples(records[record_id].artifacts["per_sample"]) or []
    ref = {s.id: s for s in store.load_samples(records[against].artifacts["per_sample"]) or []}
    aligned = []
    for m in main:
        r = ref.get(m.id)
        if r is None or "correct" not in m.scores or "correct" not in r.scores:
            continue
        now, before = m.scores["correct"] > 0.5, r.scores["correct"] > 0.5
        kind = (
            "both_right"
            if now and before
            else "both_wrong"
            if not now and not before
            else "to_right"
            if now
            else "to_wrong"
        )
        aligned.append((kind, m, r))
    counts = {k: sum(1 for kind, _, _ in aligned if kind == k) for k in FLIPS}
    wanted = {"changed": {"to_wrong", "to_right"}, "all": set(FLIPS)}.get(flip, {flip})
    picked = [
        (k, m, r) for k, m, r in aligned if k in wanted and (subtask is None or _subtask(m.id) == subtask)
    ]
    subtasks: dict[str, dict[str, int]] = {}
    for kind, m, _ in aligned:
        subtasks.setdefault(_subtask(m.id), dict.fromkeys(FLIPS, 0))[kind] += 1
    return {
        "record_id": record_id,
        "against": against,
        "coverage": {"main": len(main), "reference": len(ref), "aligned": len(aligned)},
        "counts": counts,
        "subtasks": subtasks,
        "total": len(picked),
        "samples": [
            {
                "id": m.id,
                "subtask": _subtask(m.id),
                "flip": kind,
                "target": m.target,
                "main": _answer(m),
                "reference": _answer(r),
            }
            for kind, m, r in picked[offset : offset + limit]
        ],
    }


HUB = "https://huggingface.co"
HUB_DTYPES = ("bfloat16", "float16", "float32")


def _hub_get(path: str) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = httpx.get(f"{HUB}{path}", headers=headers, timeout=15, follow_redirects=True)
    except httpx.HTTPError as exc:
        raise ServiceError(
            502, f"the server cannot reach huggingface.co ({exc}); fill in the form by hand instead"
        ) from exc
    if response.status_code in (401, 403, 404):
        raise ServiceError(
            404,
            f"huggingface.co has no accessible {path.split('/resolve/')[0]} (HTTP {response.status_code})",
        )
    if response.status_code >= 400:
        raise ServiceError(502, f"huggingface.co answered HTTP {response.status_code} for {path}")
    return response.json()


def model_from_hub(repo: str, revision: str | None) -> ModelEntry:
    """Registry entry for a Hugging Face repo, pinned to the commit `revision` (default: main) resolves to.

    Read from the hub API and the repo's config.json, so the server needs no model code. ServiceError 502 when
    the hub is unreachable, 404 for unknown or private repos, 400 for architectures without a layout table.
    """
    info = _hub_get(f"/api/models/{repo}" + (f"/revision/{revision}" if revision else ""))
    sha = info["sha"]
    config = _hub_get(f"/{repo}/resolve/{sha}/config.json")
    architecture = (config.get("architectures") or [None])[0]
    if architecture not in builtin_catalog().architectures:
        raise ServiceError(
            400, f"{repo} is a {architecture}, which has no layout table yet; run `effeval arch extract` once"
        )
    dtype = config.get("torch_dtype") or config.get("dtype")
    if dtype not in HUB_DTYPES:
        raise ServiceError(400, f"{repo} declares native dtype {dtype!r}; expected one of {list(HUB_DTYPES)}")
    params = (info.get("safetensors") or {}).get("total")
    if not params:
        raise ServiceError(
            400, f"{repo} publishes no safetensors parameter count; register it with `effeval model add`"
        )
    return ModelEntry(
        code=model_code(repo),
        checkpoint=repo,
        revision=sha,
        architecture=architecture,
        dtype=dtype,
        params=params,
        tie_word_embeddings=bool(config.get("tie_word_embeddings", True)),
        description=f"{repo} at {sha[:10]}",
    )


def storage(store: SqliteStore, actor: Actor) -> dict[str, Any]:
    """Where per-sample files live on the server disk: the lab's shared directory and the actor's own."""
    return {
        "shared": {**store.storage_usage(store.SHARED), "editable": actor.role == "admin"},
        "personal": {**store.storage_usage(actor.username), "editable": True},
    }


def set_storage(store: SqliteStore, actor: Actor, *, shared: bool, path: str) -> int:
    """Move the shared (admins only) or the actor's storage to `path`; returns how many files moved.

    ServiceError 403 for non-admins changing the shared path, 400 for relative or unwritable paths.
    """
    if shared and actor.role != "admin":
        raise ServiceError(403, "only admins can move the shared storage")
    try:
        return store.set_storage(store.SHARED if shared else actor.username, path)
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc


def register_model(store: SqliteStore, actor: Actor, entry: ModelEntry) -> ModelEntry:
    """Add a model code for the whole lab. Builtin codes and existing codes cannot be redefined."""
    builtin = builtin_catalog()
    if entry.code in builtin.models:
        if builtin.models[entry.code] != entry:
            raise ServiceError(
                409, f"{entry.code!r} is a builtin model code for {builtin.models[entry.code].checkpoint}"
            )
        return entry
    if entry.architecture not in builtin.architectures:
        raise ServiceError(400, f"architecture {entry.architecture} has no layout; extract it first")
    try:
        store.add_model(entry, actor.username)
    except ValueError as exc:
        raise ServiceError(409, str(exc)) from exc
    return entry


def register_task_listing(store: SqliteStore, actor: Actor, listing: TaskListing) -> dict[str, Any]:
    """Store `listing` as the lab-wide task catalog of its evaluator (admins only; the last upload wins).

    Raises ServiceError: 403 for non-admins, 409 for a builtin evaluator's name, 400 for a listing that is
    inconsistent with the catalog's metric and category registries.
    """
    if actor.role != "admin":
        raise ServiceError(403, "only admins can upload a task catalog")
    base = builtin_catalog()
    known = base.evaluators.get(listing.evaluator.name)
    if known is not None and known.source == "builtin":
        raise ServiceError(409, f"{listing.evaluator.name!r} is a builtin evaluator name")
    try:
        task_defs_from_listing(listing, base.metrics, base.categories)
    except ValueError as exc:
        raise ServiceError(400, str(exc)) from exc
    store.put_task_listing(listing, actor.username)
    return {"evaluator": listing.evaluator.name, "tasks": len(listing.tasks), "digest": listing.digest()}


def register_manifest(store: SqliteStore, actor: Actor, manifest: MethodManifest) -> MethodManifest:
    if manifest.name in builtin_catalog().manifests:
        raise ServiceError(409, f"{manifest.name!r} is a builtin method name")
    if manifest.builtin:
        raise ServiceError(400, "uploaded manifests cannot be marked builtin")
    if manifest.schema_hash != schema_hash(manifest.config_schema):
        raise ServiceError(400, "schema_hash does not match config_schema")
    try:
        store.put_manifest(manifest, actor.username)
    except PermissionError as exc:
        raise ServiceError(403, str(exc)) from exc
    return manifest


def issues_have_errors(issues: list[Issue]) -> bool:
    return any(i.severity == Severity.ERROR for i in issues)
