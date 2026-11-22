"""Freeze a RunSpec into a Plan: one job per task, derived reference jobs, and reuse of existing records."""

from __future__ import annotations

import copy
import uuid
from typing import Any

from effeval.core.canonical import canonical_json
from effeval.core.catalog import DEFAULT_SAMPLING_TEMPERATURE, Catalog, builtin_catalog
from effeval.core.engines import ENGINES
from effeval.core.preprocess import resolve_preprocess
from effeval.core.spec import (
    GenerationSpec,
    Job,
    JobSpec,
    Plan,
    ResolvedDecoding,
    ResolvedEvaluator,
    ResolvedMethod,
    ResolvedRole,
    ResolvedTask,
    ReusedJob,
    run_key,
    runtime_identity,
)
from effeval.core.types import Artifact, Deviation, Fidelity, Issue, Severity, utc_now
from effeval.metrics.quality import COMPARATIVE, why_not_applicable
from effeval.store.base import ReuseLookup
from effeval.validate import Resolution, expand_sweep, job_metrics, normalize, resolve


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict) and key != "method":
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _job_fidelity(res: Resolution, task: str) -> Fidelity:
    """Return the run's fidelity narrowed to one job.

    A ``tasks`` deviation is kept only when ``task`` is outside the preset's tasks. Benchmark settings that
    differ from the benchmark's defaults are deviations at ``tasks.<task>.<setting>``.
    """
    fidelity = res.fidelity.model_copy(deep=True)
    plan = next(p for p in res.tasks if p.task.name == task)
    defaults = plan.task.settings({}).model_dump(mode="json")
    changed = {**plan.params, **({} if plan.limit is None else {"limit": plan.limit})}
    fidelity.deviations += [
        Deviation(path=f"tasks.{task}.{name}", preset=defaults[name], actual=value)
        for name, value in changed.items()
    ]
    if plan.num_fewshot != plan.task.num_fewshot:
        fidelity.deviations.append(
            Deviation(path=f"tasks.{task}.num_fewshot", preset=plan.task.num_fewshot, actual=plan.num_fewshot)
        )
    deviation = next((d for d in fidelity.deviations if d.path == "tasks"), None)
    if deviation is None or res.preset is None:
        return fidelity
    fidelity.deviations.remove(deviation)
    if task in normalize("tasks", deviation.preset):
        bucket = fidelity.assumed if res.preset.values["tasks"].source == "assumed" else fidelity.confirmed
        bucket.append("tasks")
    else:
        fidelity.deviations.append(Deviation(path="tasks", preset=deviation.preset, actual=task))
    return fidelity


def _task_generation(run: GenerationSpec, task_plan: Any) -> GenerationSpec:
    """The run-level generation settings with this task's own overrides applied."""
    update: dict[str, Any] = {}
    if task_plan.max_new_tokens is not None:
        update["max_new_tokens"] = task_plan.max_new_tokens
    if task_plan.do_sample is not None:
        update["do_sample"] = task_plan.do_sample
        update["temperature"] = (
            task_plan.temperature or run.temperature or DEFAULT_SAMPLING_TEMPERATURE
            if task_plan.do_sample
            else None
        )
    return run.model_copy(update=update)


def _resolved_decoding(res: Resolution, generation: GenerationSpec) -> ResolvedDecoding | None:
    """The frozen decoding of a speculative run; None for autoregressive decoding."""
    decoding = res.spec.decoding
    if decoding is None or decoding.mode != "speculative":
        return None
    draft = decoding.draft
    return ResolvedDecoding(
        mode="speculative",
        draft=ResolvedRole(
            model=res.draft_model,
            method=ResolvedMethod(
                name=draft.method.name,
                version=res.draft_manifest.version,
                preset=None,
                config=draft.method.config,
            ),
            preprocess=None if draft.preprocess is None else resolve_preprocess(draft.preprocess),
        ),
        num_speculative_tokens=decoding.num_speculative_tokens,
        verification="sampling" if generation.do_sample else "greedy",
    )


def _main_jobs(res: Resolution) -> list[tuple[JobSpec, Any]]:
    """One (job, its task plan) per task and run.

    With `repeats.seeds` each listed seed is one run and a job is identified by its seed alone; otherwise
    `repeats.count` runs share determinism.seed and are told apart by their repeat index.
    """
    spec = res.spec
    repeats = spec.repeats
    if repeats.seeds is not None:
        runs = [(seed, 0) for seed in repeats.seeds]
    else:
        runs = [(spec.determinism.seed, repeat) for repeat in range(repeats.count or 1)]
    jobs = []
    for task_plan in res.tasks:
        for seed, repeat in runs:
            generation = _task_generation(spec.generation, task_plan)
            job = JobSpec(
                model=res.model,
                method=ResolvedMethod(
                    name=spec.method.name,
                    version=res.manifest.version,
                    preset=None if res.preset is None else res.preset.name,
                    config=spec.method.config,
                ),
                preprocess=None if spec.preprocess is None else resolve_preprocess(spec.preprocess),
                field=spec.field,
                runtime=spec.runtime,
                runtime_library=None if spec.runtime.name == "hf" else ENGINES[spec.runtime.name].library,
                decoding=_resolved_decoding(res, generation),
                task=ResolvedTask(
                    name=task_plan.task.name,
                    version=task_plan.task.version,
                    mode=task_plan.mode,
                    num_fewshot=task_plan.num_fewshot,
                    limit=task_plan.limit,
                    params=task_plan.params,
                ),
                evaluator=ResolvedEvaluator(
                    name=task_plan.evaluator.name,
                    version=task_plan.evaluator.version,
                    library=None
                    if task_plan.evaluator.library is None
                    else f"{task_plan.evaluator.library}=={task_plan.evaluator.library_version}",
                    passthrough=spec.evaluator.passthrough.get(task_plan.evaluator.name, {}),
                ),
                generation=generation,
                determinism=spec.determinism.model_copy(update={"seed": seed}),
                hardware=spec.hardware,
                metrics=job_metrics(task_plan.task, task_plan.produces, spec.metrics),
                repeat=repeat,
            )
            jobs.append((job, task_plan))
    return jobs


def _reference_of(main: JobSpec, res: Resolution, catalog: Catalog) -> JobSpec | None:
    """Apply the field's reference overrides; the method part is re-resolved against its manifest."""
    overrides = copy.deepcopy(res.field_def.reference)
    method_override = overrides.pop("method", None)
    doc = _merge(main.model_dump(mode="json"), overrides)
    if method_override is not None:
        manifest = catalog.manifests[method_override["name"]]
        doc["method"] = ResolvedMethod(
            name=manifest.name,
            version=manifest.version,
            preset=None,
            config=method_override.get("config", {}),
        ).model_dump(mode="json")
    reference = JobSpec.model_validate(doc)
    return None if run_key(reference) == run_key(main) else reference


def _applicable(names: list[str], produces: frozenset[Artifact], sample_scores: list[str]) -> list[str]:
    return [n for n in names if why_not_applicable(n, produces, sample_scores) is None]


def freeze(
    raw: dict[str, Any],
    catalog: Catalog,
    *,
    owner: str | None,
    server: str | None,
    lookup: ReuseLookup | None,
) -> tuple[Plan | None, list[Issue]]:
    """Return (plan, issues). The plan is None when any combination has an error."""
    resolutions = [resolve(combo, catalog) for combo in expand_sweep(raw)]
    issues = [i for r in resolutions for i in r.issues]
    if any(i.severity == Severity.ERROR for i in issues):
        return None, issues

    jobs: dict[str, Job] = {}
    needs: dict[str, frozenset[Artifact]] = {}
    for res in resolutions:
        for main, task_plan in _main_jobs(res):
            referenced = task_plan.task.reference
            compare = (
                _applicable(res.compare, task_plan.produces, task_plan.task.sample_scores)
                if referenced
                else []
            )
            if main.generation.do_sample and "output_match" in compare:
                compare.remove("output_match")
                issues.append(
                    Issue(
                        code="compare.output_match_dropped",
                        severity=Severity.INFO,
                        paths=["generation.do_sample"],
                        message=f"output_match is not computed for {main.task.name}: sampled outputs "
                        "are not expected to match the reference's",
                    )
                )
            key = run_key(main)
            reference = _reference_of(main, res, catalog) if referenced else None
            ref_key = None if reference is None else run_key(reference)
            if key in jobs:
                merged = list(dict.fromkeys([*jobs[key].spec.metrics, *main.metrics]))
                jobs[key] = jobs[key].model_copy(
                    update={"spec": jobs[key].spec.model_copy(update={"metrics": merged})}
                )
            jobs.setdefault(
                key,
                Job(
                    job_id="",
                    run_key=key,
                    role="main",
                    reference_key=ref_key,
                    compare=compare,
                    fidelity=_job_fidelity(res, main.task.name),
                    spec=main,
                ),
            )
            needs[key] = needs.get(key, frozenset()) | frozenset().union(
                *(COMPARATIVE[c].requires for c in compare)
            )
            if reference is not None:
                jobs.setdefault(
                    ref_key,
                    Job(
                        job_id="",
                        run_key=ref_key,
                        role="reference",
                        reference_key=None,
                        compare=[],
                        fidelity=Fidelity(),
                        spec=reference,
                    ),
                )
                needs[ref_key] = needs.get(ref_key, frozenset()) | needs[key]

    policy = resolutions[0].spec.reuse
    reused: list[ReusedJob] = []
    to_run: list[Job] = []
    for key, job in jobs.items():
        record = None
        if lookup is not None and policy != "never":
            hardware = job.spec.hardware.gpu if policy == "same_hardware" else None
            if policy == "same_hardware" and hardware is None:
                issues.append(
                    Issue(
                        code="reuse.no_hardware",
                        severity=Severity.WARNING,
                        paths=["hardware.gpu"],
                        message="reuse: same_hardware needs hardware.gpu; nothing is reused",
                    )
                )
            else:
                record = lookup.find_reusable(key, viewer=owner, need=needs[key], hardware=hardware)
        if record is not None:
            gpu = record.env.gpu if record.env else "unknown GPU"
            reused.append(
                ReusedJob(
                    run_key=key,
                    role=job.role,
                    record_id=record.record_id,
                    reason=f"same RunKey, record by {record.owner} on {gpu} at {record.created_at}",
                )
            )
        else:
            to_run.append(job.model_copy(update={"job_id": "job_" + uuid.uuid4().hex}))
    if lookup is None:
        issues.append(
            Issue(
                code="reuse.unavailable",
                severity=Severity.INFO,
                message="no result store reachable: nothing can be reused, every job will run",
            )
        )

    return _assemble(list(jobs.values()), to_run, reused, issues, catalog, owner=owner, server=server), issues


def _assemble(
    jobs: list[Job],
    to_run: list[Job],
    reused: list[ReusedJob],
    issues: list[Issue],
    catalog: Catalog,
    *,
    owner: str | None,
    server: str | None,
) -> Plan:
    return Plan(
        plan_id="plan_" + uuid.uuid4().hex,
        created_at=utc_now(),
        owner=owner,
        server=server,
        jobs=_ordered(to_run),
        reused=reused,
        notes=[i for i in issues if i.severity != Severity.ERROR],
        manifests={
            name: catalog.manifests[name]
            for job in jobs
            for name in _method_names(job.spec)
            if not catalog.manifests[name].builtin
        },
        models={
            name: catalog.models[name]
            for job in jobs
            for name in _model_names(job.spec)
            if name not in builtin_catalog().models
        },
        listings={
            name: catalog.listing_for(name, sorted(used))
            for name, used in _listed_tasks_used(jobs, catalog).items()
        },
    )


def _method_names(spec: JobSpec) -> list[str]:
    """The methods of a job's target and, when speculative, of its draft."""
    return [spec.method.name] + ([] if spec.decoding is None else [spec.decoding.draft.method.name])


def _model_names(spec: JobSpec) -> list[str]:
    """The models of a job: its target and, when speculative, its draft."""
    return [spec.model.name] + ([] if spec.decoding is None else [spec.decoding.draft.model.name])


def _listed_tasks_used(jobs: list[Job], catalog: Catalog) -> dict[str, set[str]]:
    """Evaluator name -> names of the tasks the jobs use, for evaluators whose tasks come from a listing."""
    used: dict[str, set[str]] = {}
    for job in jobs:
        if catalog.evaluators[job.spec.evaluator.name].source == "listing":
            used.setdefault(job.spec.evaluator.name, set()).add(job.spec.task.name)
    return used


def plan_for_jobs(
    jobs: list[Job],
    references: dict[str, Job],
    catalog: Catalog,
    *,
    owner: str | None,
    server: str | None,
    lookup: ReuseLookup,
    force: bool,
) -> Plan:
    """Return a plan for already frozen `jobs`.

    Each main job's reference (looked up in `references` by RunKey) is reused when a stored record has the
    artifacts the comparisons need, and runs alongside otherwise. With `force`, every job in `jobs` runs
    even if a stored record covers it; without, such jobs are reused too. Raises KeyError if a job's method
    or model is unknown to `catalog`.
    """
    needs: dict[str, frozenset[Artifact]] = {}
    wanted: dict[str, Job] = {}
    for job in jobs:
        wanted.setdefault(job.run_key, job)
        need = frozenset().union(*(COMPARATIVE[c].requires for c in job.compare))
        needs[job.run_key] = needs.get(job.run_key, frozenset()) | need
        if job.reference_key is not None and job.reference_key in references:
            needs[job.reference_key] = needs.get(job.reference_key, frozenset()) | need
    extra = {key: references[key] for key in needs if key not in wanted}
    to_run: list[Job] = []
    reused: list[ReusedJob] = []
    for key, job in {**wanted, **extra}.items():
        record = None
        if key in extra or not force:
            record = lookup.find_reusable(key, viewer=owner, need=needs[key], hardware=None)
        if record is None:
            to_run.append(job.model_copy(update={"job_id": "job_" + uuid.uuid4().hex}))
        else:
            reused.append(
                ReusedJob(
                    run_key=key,
                    role=job.role,
                    record_id=record.record_id,
                    reason=f"same RunKey, record by {record.owner} at {record.created_at}",
                )
            )
    every = [*wanted.values(), *extra.values()]
    return _assemble(every, to_run, reused, [], catalog, owner=owner, server=server)


def model_group_key(job: Job) -> str:
    """Jobs with the same model-preparation config share one prepared model on the compute node."""
    spec = job.spec
    if spec.runtime.name == "hf":
        runtime = spec.runtime.model_dump_json()
    else:
        runtime = canonical_json(runtime_identity(spec))
    key = f"{spec.model.model_dump_json()}|{spec.method.model_dump_json()}|{runtime}"
    if spec.preprocess is not None:
        key += f"|{spec.preprocess.model_dump_json(include={'steps', 'versions'})}"
    if spec.decoding is not None:
        draft = spec.decoding.draft
        key += f"|decoding:{draft.model.model_dump_json()}|{draft.method.model_dump_json()}"
        if draft.preprocess is not None:
            key += f"|{draft.preprocess.model_dump_json(include={'steps', 'versions'})}"
        key += f"|{spec.decoding.num_speculative_tokens}|{spec.decoding.verification}"
    return key


def _ordered(jobs: list[Job]) -> list[Job]:
    """Group jobs by prepared model so each model is loaded once; references first within a model."""
    return sorted(jobs, key=lambda j: (model_group_key(j), j.role != "reference", j.spec.task.name))


def shard(plan: Plan, index: int, count: int) -> Plan:
    """Keep the model groups assigned to shard `index` (1-based) of `count`; a group never spans shards."""
    if not 1 <= index <= count:
        raise ValueError(f"shard must satisfy 1 <= k <= N, got {index}/{count}")
    groups: list[str] = []
    for job in plan.jobs:
        key = model_group_key(job)
        if key not in groups:
            groups.append(key)
    mine = {g for i, g in enumerate(groups) if i % count == index - 1}
    return plan.model_copy(update={"jobs": [j for j in plan.jobs if model_group_key(j) in mine]})
