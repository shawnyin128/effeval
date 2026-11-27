"""Speed evaluator: per-turn generation timing on a prompt set, for autoregressive and speculative decoding.

Tasks are `specdecode/<prompt set>`.
Protocol after github.com/hemingkx/Spec-Bench `evaluation/eval.py`: warmup
runs of the first selected sample (all its turns, `warmup` times), then each sample's turns in order (a later
turn continues the conversation with the decoded, stripped previous answer), each turn timed from before the
backend call to after it, with the device synchronized at both ends. Tokenization, `SpeedBackend.reset` and
model loading are not timed. Decoding goes through the `SpeedBackend` interface
(`effeval.evaluators.speed_backend`); metrics a backend cannot provide are omitted.

Per-sample scores: `latency_ms` is the sample's total time over all its turns, `tokens_per_s` its new tokens
over that time, `ttft_ms` the time to the first token of turn 1.

Prompt set format, JSONL with one object per line (blank lines are skipped, unknown keys are rejected):

    {"id": "q1", "category": "qa", "turns": ["first user message", "optional follow-up"]}

- `id` (str, unique in the set): identifies the sample; speedup and output match pair samples by it.
- `category` (str): metrics are also reported per category, and the `categories` and `limit` task params
  select by it.
- `turns` (non-empty list of str): user messages. Later turns continue the conversation with the model's own
  previous answer.
- `system` (str, optional): system message placed first.
- `max_new_tokens` (int >= 1, optional): generated-token budget per turn of this sample,
  overriding the task's.

Built-in sets are manifests in `core/data/prompt_sets`; a user set is `<name>.jsonl` in the directory named by
`EFFEVAL_PROMPT_SETS` and becomes the task `specdecode/<name>`. The set's content hash is part of the task
version,
so editing a set makes its results new ones.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import effeval
from effeval.core.catalog import TaskDef, sample_count
from effeval.core.promptsets import Prompt, load_prompt_set
from effeval.core.spec import JobSpec
from effeval.core.types import Artifact, EngineHandle, InProcess, QualityResult, Sample, metric_str
from effeval.evaluators.speed_backend import Generation, SpeedBackend

_COUNTERS = ("rounds", "proposed", "accepted", "draft_ms", "decode_ms")

SPEC_BENCH_DEVIATIONS = (
    "chat template: the model's own (Qwen3, enable_thinking from the task), not Spec-Bench's vicuna template",
    "dtype: the run's model dtype instead of Spec-Bench's float16",
    "determinism: deterministic algorithms on and TF32 off, which Spec-Bench does not set",
    "draft length: a fixed num_speculative_tokens k instead of the HF assisted-generation schedule",
    "output match: equality of generated token ids over all turns, where Spec-Bench compares answer text",
)


def _mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def summarize(samples: list[Sample]) -> dict[str, str]:
    """Metrics over `samples`: mean per-sample tokens/s, latency and TTFT, and the speculative counters.

    The acceptance rate is accepted / proposed drafts, mean accepted tokens is 1 + accepted / rounds, and the
    draft time share is draft time over decoding time (omitted when that is zero). A metric is omitted when
    any sample lacks a score it needs, i.e. when the backend did not report it.
    """
    scores = [s.scores for s in samples]

    def has(*names: str) -> bool:
        return all(name in s for s in scores for name in names)

    def total(name: str) -> float:
        return math.fsum(s[name] for s in scores)

    out = {
        "throughput_tok_s": metric_str(_mean([s["tokens_per_s"] for s in scores])),
        "latency_ms": metric_str(_mean([s["latency_ms"] for s in scores])),
    }
    if has("ttft_ms"):
        out["ttft_ms"] = metric_str(_mean([s["ttft_ms"] for s in scores]))
    if has("proposed", "accepted"):
        proposed = total("proposed")
        out["acceptance_rate"] = metric_str(total("accepted") / proposed if proposed else 0.0)
    if has("rounds", "accepted"):
        rounds = total("rounds")
        out["mean_accepted_tokens"] = metric_str(1 + total("accepted") / rounds if rounds else 0.0)
    if has("draft_ms", "decode_ms") and total("decode_ms"):
        out["draft_time_share"] = metric_str(total("draft_ms") / total("decode_ms"))
    return out


def select(prompts: list[Prompt], categories: list[str] | None, limit: int | float | None) -> list[Prompt]:
    """The first `limit` prompts of each of `categories` in file order, categories in the order given.

    `categories` None means all of the set's, in order of first appearance; `limit` is a count, a fraction
    (rounded up) or None for all. Raises ValueError on a category the set does not have.
    """
    present = list(dict.fromkeys(p.category for p in prompts))
    wanted = present if categories is None else list(dict.fromkeys(categories))
    unknown = [c for c in wanted if c not in present]
    if unknown:
        raise ValueError(f"unknown categories {unknown}; the prompt set has {present}")
    selected = []
    for category in wanted:
        members = [p for p in prompts if p.category == category]
        selected.extend(members[: sample_count(limit, len(members))])
    return selected


def _total(generations: list[Generation], name: str) -> float | None:
    values = [getattr(g, name) for g in generations]
    return None if any(v is None for v in values) else math.fsum(values)


def _prompt_ids(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> list[int]:
    """Token ids of the chat `messages` followed by the assistant generation prompt."""
    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=enable_thinking, tokenize=False
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def measure(
    backend: SpeedBackend,
    prompts: list[Prompt],
    max_new_tokens: int,
    enable_thinking: bool,
    warmup: int,
) -> list[Sample]:
    """Generate every prompt's turns with `backend`, timed per turn; returns one Sample per prompt.

    Before the timed samples, all turns of `prompts[0]` are generated `warmup` times. `backend.reset()` runs
    before every generation and after warmup, outside the timed region; peak memory restarts after warmup.
    """
    tokenizer = backend.tokenizer

    def generate(messages: list[dict[str, str]], budget: int) -> tuple[Generation, float]:
        ids = _prompt_ids(tokenizer, messages, enable_thinking)
        backend.reset()
        backend.synchronize()
        start = time.perf_counter()
        generation = backend.generate(ids, budget)
        backend.synchronize()
        elapsed = time.perf_counter() - start
        return generation, elapsed if generation.seconds is None else generation.seconds

    def converse(prompt: Prompt) -> list[tuple[Generation, float, str]]:
        """The prompt's turns in order: (generation, seconds, answer text) per turn."""
        budget = prompt.max_new_tokens or max_new_tokens
        messages = [{"role": "system", "content": prompt.system}] if prompt.system else []
        turns = []
        for content in prompt.turns:
            messages.append({"role": "user", "content": content})
            generation, elapsed = generate(messages, budget)
            text = tokenizer.decode(generation.output_ids, skip_special_tokens=True).strip()
            messages.append({"role": "assistant", "content": text})
            turns.append((generation, elapsed, text))
        return turns

    for _ in range(warmup):
        converse(prompts[0])
    backend.reset()
    backend.reset_peak_memory()

    samples = []
    for prompt in prompts:
        turns = converse(prompt)
        generations = [generation for generation, _, _ in turns]
        output_ids = [generation.output_ids for generation in generations]
        seconds = math.fsum(elapsed for _, elapsed, _ in turns)
        new_tokens = sum(len(ids) for ids in output_ids)
        scores = {
            "tokens_per_s": new_tokens / seconds,
            "latency_ms": seconds * 1000.0,
            "new_tokens": float(new_tokens),
            "sampled": float(backend.sampled),
        }
        if generations[0].ttft_ms is not None:
            scores["ttft_ms"] = generations[0].ttft_ms
        for name in _COUNTERS:
            if (value := _total(generations, name)) is not None:
                scores[name] = value
        samples.append(
            Sample(
                id=prompt.id,
                category=prompt.category,
                prediction=[text for _, _, text in turns],
                scores=scores,
                output_ids=output_ids,
            )
        )
    return samples


def make_backend(handle: InProcess | EngineHandle | None, job: JobSpec) -> SpeedBackend:
    """The `SpeedBackend` for a model handle; raises TypeError for anything else."""
    if isinstance(handle, InProcess):
        from effeval.evaluators._speed_inprocess import InProcessBackend  # imports torch

        return InProcessBackend(handle, job)
    if isinstance(handle, EngineHandle):
        from effeval.evaluators._speed_engine import EngineBackend

        return EngineBackend(handle, job)
    raise TypeError(f"speed needs a model handle, got {type(handle).__name__}")


class Speed:
    name = "specdecode"

    def list_tasks(self) -> None:
        return None

    def run(
        self, handle: InProcess | EngineHandle | None, job: JobSpec, task: TaskDef, workdir: Path
    ) -> QualityResult:
        return evaluate(make_backend(handle, job), job, task)


def evaluate(backend: SpeedBackend, job: JobSpec, task: TaskDef) -> QualityResult:
    """Run the task's prompt set on `backend`.

    Raises ValueError when the set's content differs from the one the job was planned with.
    """
    settings = task.settings(job.task.params)
    prompt_set = load_prompt_set(task.target)
    if prompt_set.info.version != job.task.version:
        raise ValueError(
            f"prompt set {task.target!r} changed since the job was planned: version "
            f"{prompt_set.info.version}, planned {job.task.version}"
        )
    prompts = select(prompt_set.prompts, settings.categories, job.task.limit)
    max_new_tokens = job.generation.max_new_tokens
    samples = measure(backend, prompts, max_new_tokens, settings.enable_thinking, settings.warmup)
    metrics = summarize(samples)
    if (peak := backend.peak_memory_gb()) is not None:
        metrics["peak_memory_gb"] = metric_str(peak)
    categories = {}
    for name in dict.fromkeys(p.category for p in prompts):
        selected = [s for s in samples if s.category == name]
        categories[name] = {"samples": len(selected), **summarize(selected)}
    described = backend.describe()
    protocol = {
        "backend": "speed",
        "backend_version": effeval.__version__,
        "prompt_set": {
            "name": prompt_set.info.name,
            "origin": prompt_set.info.origin,
            "version": prompt_set.info.version,
            "sha256": prompt_set.sha256,
            **prompt_set.source,
        },
        "enable_thinking": settings.enable_thinking,
        "chat_template_sha256": hashlib.sha256(
            json.dumps(backend.tokenizer.chat_template, sort_keys=True).encode()
        ).hexdigest(),
        "warmup": settings.warmup,
        "timing": described["timing"],
        "device": described["device"],
        "decoder": described["decoder"],
        "max_new_tokens": max_new_tokens,
        "temperature": described["temperature"],
        "sampling": described["sampling"],
        "categories": categories,
    }
    if "decode_cache" in described:
        protocol["decode_cache"] = described["decode_cache"]
    if prompt_set.info.name == "spec_bench":
        protocol["deviations_from_spec_bench"] = [
            d
            for d in SPEC_BENCH_DEVIATIONS
            if described.get("torch_determinism", True) or not d.startswith("determinism:")
        ]
    return QualityResult(
        metrics=metrics,
        samples=samples,
        protocol=protocol,
        produces=frozenset({Artifact.PER_SAMPLE}),
    )


EVALUATOR = Speed()
