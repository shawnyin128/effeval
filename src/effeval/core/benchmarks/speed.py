from pydantic import Field

from effeval.core.catalog import SampleLimit, TaskDef, TaskMetric, TaskSize
from effeval.core.promptsets import PromptSetInfo, builtin_set_infos, user_set_infos
from effeval.core.types import Artifact, Mode


class Params(SampleLimit):
    categories: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Categories of the prompt set to run, in this order; empty runs all of them. `limit` "
        "applies to each.",
    )
    max_new_tokens: int = Field(
        default=1024, ge=1, description="Generated tokens per turn, at most; a sample's own value wins."
    )
    do_sample: bool | None = Field(
        default=None, description="Sample instead of decoding greedily; empty keeps the run-level setting."
    )
    temperature: float | None = Field(
        default=None,
        gt=0.0,
        description="Sampling temperature; only with do_sample. Empty uses the run-level one.",
    )
    enable_thinking: bool = Field(
        default=False, description="Qwen3 chat template switch for thinking mode, the same in every arm."
    )
    warmup: int = Field(
        default=3,
        ge=0,
        description="Untimed runs of the first selected sample, all its turns, before timing.",
    )


def task_for(info: PromptSetInfo) -> TaskDef:
    """The task `specdecode/<set name>` that measures generation speed on the prompt set `info`."""
    return TaskDef(
        name=f"specdecode/{info.name}",
        evaluator="specdecode",
        target=info.name,
        version=info.version,
        category="speculative_decoding",
        description=(
            f"{info.description} Generated with the model's chat template at batch size 1. Reports tokens "
            "per second, latency, peak memory and, for speculative decoding, the acceptance rate and mean "
            "accepted tokens per round. Speedup and output match come from comparing a run with its "
            "autoregressive reference."
        ),
        modes=[Mode.SPEED],
        default_mode=Mode.SPEED,
        num_fewshot=0,
        fewshot_allowed=[],
        produces={Mode.SPEED: frozenset({Artifact.PER_SAMPLE})},
        metrics=[
            TaskMetric(name="throughput_tok_s", higher_is_better=True),
            TaskMetric(name="latency_ms", higher_is_better=False),
            TaskMetric(name="ttft_ms", higher_is_better=False),
            TaskMetric(name="peak_memory_gb", higher_is_better=False),
            TaskMetric(name="acceptance_rate", higher_is_better=True),
            TaskMetric(name="mean_accepted_tokens", higher_is_better=True),
            TaskMetric(name="draft_time_share", higher_is_better=False),
        ],
        sample_scores=["tokens_per_s", "latency_ms", "ttft_ms", "new_tokens", "sampled"],
        reference=True,
        size=TaskSize(samples=info.samples, subtasks=info.categories, prompt_tokens=300, output_tokens=400),
        params=Params,
    )


def user_tasks() -> list[TaskDef]:
    """Tasks of the prompt sets in the user directory (`EFFEVAL_PROMPT_SETS`)."""
    return [task_for(info) for info in user_set_infos().values()]


BENCHMARKS = [task_for(info) for info in builtin_set_infos().values()]
