"""lm-evaluation-harness evaluator: lists the harness's own tasks and runs them on an in-process model.

HFLM(pretrained=<model object>) + simple_evaluate(model=...); task names are the harness's own names.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import lm_eval
from lm_eval import simple_evaluate
from lm_eval.api.registry import is_higher_better
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager

from effeval.core.catalog import TaskDef
from effeval.core.listing import EvaluatorInfo, TaskListing
from effeval.core.spec import JobSpec
from effeval.core.types import Artifact, Capabilities, HandleKind, InProcess, Mode, QualityResult, utc_now
from effeval.evaluators._lm_eval_format import listed_tasks, metrics_from_results, samples_from_rows

DESCRIPTION = (
    "EleutherAI lm-evaluation-harness via HFLM(pretrained=<model object>) and simple_evaluate(model=...)."
)


def _index() -> dict[str, dict[str, Any]]:
    return {
        name: {"kind": entry.kind.name.lower(), "cfg": entry.cfg, "tags": sorted(entry.tags)}
        for name, entry in TaskManager().task_index.items()
    }


class LmEval:
    name = "lm_eval"

    def list_tasks(self) -> TaskListing:
        tasks, skipped = listed_tasks(_index(), is_higher_better)
        left_out = ", ".join(f"{n.replace('_', ' ')}: {c}" for n, c in sorted(skipped.items()))
        return TaskListing(
            evaluator=EvaluatorInfo(
                name=self.name,
                version="2",
                description=f"{DESCRIPTION} Left out of the listing: {left_out}.",
                capabilities=Capabilities(
                    modes={Mode.LOGLIKELIHOOD, Mode.GENERATION}, handles={HandleKind.IN_PROCESS}
                ),
                library="lm-eval",
                library_version=importlib.metadata.version("lm-eval"),
            ),
            generated_at=utc_now(),
            tasks=tasks,
        )

    def run(self, handle: InProcess, job: JobSpec, task: TaskDef, workdir: Path) -> QualityResult:
        lm = HFLM(pretrained=handle.model, tokenizer=handle.tokenizer, batch_size=job.generation.batch_size)
        gen_kwargs: dict[str, Any] | None = None
        if job.task.mode == Mode.GENERATION:
            gen_kwargs = {
                "do_sample": job.generation.do_sample,
                "max_gen_toks": job.generation.max_new_tokens,
            }
            if job.generation.temperature is not None:
                gen_kwargs["temperature"] = job.generation.temperature
        seed = job.determinism.seed
        results = simple_evaluate(
            model=lm,
            tasks=[task.target],
            num_fewshot=job.task.num_fewshot,
            limit=job.task.limit,
            log_samples=True,
            gen_kwargs=gen_kwargs,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
            **job.evaluator.passthrough,
        )
        metrics = metrics_from_results(results["results"][task.target])
        primary = task.metrics[0].name
        primary_metric, _, primary_filter = primary.partition(",")
        samples = samples_from_rows(
            results["samples"],
            primary_filter or "none",
            Artifact.CHOICE_LOGLIKELIHOODS in task.produces[job.task.mode],
            [n for n in task.sample_scores if n != "correct"],
            primary_metric if "correct" in task.sample_scores else None,
            job.task.mode == Mode.GENERATION,
        )
        subtasks = {
            name: metrics_from_results(values)
            for name, values in results["results"].items()
            if name != task.target
        }
        return QualityResult(
            metrics=metrics,
            samples=samples,
            protocol={
                "backend": "lm_eval",
                "backend_version": lm_eval.__version__,
                "library": f"lm-eval=={importlib.metadata.version('lm-eval')}",
                "task": task.target,
                "num_fewshot": job.task.num_fewshot,
                "n_samples": results.get("n-samples", {}),
                "gen_kwargs": gen_kwargs,
                "subtasks": subtasks,
            },
            produces=task.produces[job.task.mode],
        )


EVALUATOR = LmEval()
