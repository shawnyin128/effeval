"""EvalScope evaluator: lists EvalScope's own benchmarks and runs them on an in-process model.

The model is wrapped as a ModelAPI and passed to TaskConfig(model=...).
EvalScope rounds report scores to 4 decimals (report.py `normalize_score`), so the aggregate is recomputed
from the per-sample review files (micro mean of the backend metric, as EvalScope's Metric.score) at full
precision.
"""

from __future__ import annotations

import copy
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import evalscope
from evalscope import TaskConfig, run_task
from evalscope.api.model import GenerateConfig, ModelAPI
from evalscope.api.registry import BENCHMARK_REGISTRY
from evalscope.models.modelscope import ModelScopeAPI, message_content_to_string

from effeval.core.catalog import TaskDef
from effeval.core.listing import EvaluatorInfo, TaskListing
from effeval.core.spec import JobSpec
from effeval.core.types import Capabilities, HandleKind, InProcess, Mode, QualityResult, utc_now
from effeval.evaluators._evalscope_format import (
    benchmark_of,
    listed_tasks,
    metrics_from_reviews,
    review_files,
)

MODEL_NAME = "effeval_model"


class InProcessModelAPI(ModelScopeAPI):
    """ModelScopeAPI minus checkpoint loading; generation, chat templating and batching stay EvalScope's."""

    def __init__(self, model, tokenizer, model_name: str = MODEL_NAME):
        ModelAPI.__init__(self, model_name=model_name, base_url=None, api_key=None, config=GenerateConfig())
        tokenizer = copy.deepcopy(
            tokenizer
        )  # EvalScope mutates padding settings; keep the caller's untouched
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("tokenizer has neither pad nor eos token; EvalScope batching needs one")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.model = model
        self.tokenizer = tokenizer
        self.device = str(model.device)
        self.torch_dtype = model.dtype
        self.chat_template = None
        self.tokenizer_call_args = {}
        self.enable_thinking = None

    def ms_chat(self, messages, tools) -> str:
        """The prompt text of `messages`; EvalScope's version hands pydantic messages to transformers 5."""
        if tools:
            raise NotImplementedError("tool calling is not supported by the in-process model")
        plain = [
            {"role": m.role, "content": m.content} for m in message_content_to_string(copy.deepcopy(messages))
        ]
        if self.tokenizer.chat_template is None:
            return "".join(f"{m['role']}: {m['content']}\n" for m in plain)
        return self.tokenizer.apply_chat_template(plain, add_generation_prompt=True, tokenize=False)


def _meta_dict(meta: Any) -> dict[str, Any]:
    primary = meta.primary_metric
    return {
        "name": meta.name,
        "output_types": list(meta.output_types),
        "subset_list": list(meta.subset_list),
        "few_shot_num": meta.few_shot_num,
        "few_shot_mode": meta.few_shot_mode,
        "allowed_few_shot_nums": None
        if meta.allowed_few_shot_nums is None
        else list(meta.allowed_few_shot_nums),
        "metric_list": list(meta.metric_list),
        "primary_metric": None if primary is None else getattr(primary, "name", primary),
        "description": meta.description,
        "pretty_name": meta.pretty_name,
        "tags": list(meta.tags),
        "evaluation_version": meta.evaluation_version,
    }


class EvalScope:
    name = "evalscope"

    def list_tasks(self) -> TaskListing:
        metas = [_meta_dict(meta) for _, meta in BENCHMARK_REGISTRY.items()]
        return TaskListing(
            evaluator=EvaluatorInfo(
                name=self.name,
                version="2",
                description=(
                    "EvalScope native backend with the in-process model wrapped as a ModelAPI. It only "
                    "generates; multiple choice is scored by extracting the answer from generated text. "
                    "Aggregates are recomputed from per-sample review files."
                ),
                capabilities=Capabilities(modes={Mode.GENERATION}, handles={HandleKind.IN_PROCESS}),
                library="evalscope",
                library_version=importlib.metadata.version("evalscope"),
            ),
            generated_at=utc_now(),
            tasks=listed_tasks(metas),
        )

    def run(self, handle: InProcess, job: JobSpec, task: TaskDef, workdir: Path) -> QualityResult:
        benchmark = benchmark_of(task.target)
        dataset_args: dict = {}
        if job.task.num_fewshot is not None:
            dataset_args["few_shot_num"] = job.task.num_fewshot
        if task.subsets is not None:
            dataset_args["subset_list"] = task.subsets
        generation = {
            "do_sample": job.generation.do_sample,
            "max_tokens": job.generation.max_new_tokens,
            "batch_size": job.generation.batch_size,
        }
        if job.generation.temperature is not None:
            generation["temperature"] = job.generation.temperature
        config = TaskConfig(
            model=InProcessModelAPI(handle.model, handle.tokenizer),
            model_id=MODEL_NAME,
            datasets=[benchmark],
            dataset_args={benchmark: dataset_args},
            dataset_hub="huggingface",
            limit=job.task.limit,
            eval_batch_size=job.generation.batch_size,
            generation_config=generation,
            work_dir=str(workdir),
            no_timestamp=True,
            seed=job.determinism.seed,
            **job.evaluator.passthrough,
        )
        run_task(config)

        review_dir = workdir / "reviews" / MODEL_NAME
        reviews = [
            (subset, json.loads(line))
            for subset, path in review_files(review_dir, benchmark, task.subsets)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        primary = task.metrics[0].name
        metrics, samples = metrics_from_reviews(reviews, primary, "correct" in task.sample_scores)
        return QualityResult(
            metrics=metrics,
            samples=samples,
            protocol={
                "backend": "evalscope",
                "backend_version": evalscope.__version__,
                "library": f"evalscope=={importlib.metadata.version('evalscope')}",
                "task": benchmark,
                "subsets": task.subsets,
                "metric": primary,
                "num_fewshot": job.task.num_fewshot,
                "aggregate": "micro mean of per-sample scores recomputed from review files",
                "batch_composition": "dynamic: EvalScope groups requests that arrive within 2 s; "
                "deterministic only with batch_size 1",
            },
            produces=task.produces[Mode.GENERATION],
        )


EVALUATOR = EvalScope()
