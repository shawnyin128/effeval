from pydantic import Field

from effeval.core.catalog import NativeTaskDef, SampleCount, SampleLimit, TaskDef, TaskMetric, TaskSize
from effeval.core.types import Artifact, Mode


class Params(SampleLimit):
    limit: SampleCount = Field(
        default=-1,
        description="Windows to evaluate: an integer >= 1 for a count, a number in (0, 1) for that fraction of "
        "the windows, -1 for every window.",
    )
    seqlen: int = Field(default=2048, ge=8, description="Tokens per window; the GPTQ paper uses 2048.")


BENCHMARK = TaskDef(
    name="native/wikitext2_ppl",
    evaluator="native",
    target="wikitext2_ppl",
    version="2",
    category="language_modeling",
    description=(
        "WikiText-2 test perplexity, GPTQ protocol (arXiv 2210.17323 §A.2.1): the split is joined with two "
        "line breaks, tokenized once, cut into non-overlapping windows of `seqlen` tokens, and "
        "ppl = exp(mean token NLL)."
    ),
    modes=[Mode.LOGLIKELIHOOD],
    default_mode=Mode.LOGLIKELIHOOD,
    num_fewshot=0,
    produces={Mode.LOGLIKELIHOOD: frozenset({Artifact.PER_SAMPLE})},
    metrics=[TaskMetric(name="ppl", higher_is_better=False)],
    sample_scores=["nll"],
    size=TaskSize(samples=1, prompt_tokens=290000),  # total test tokens, assumed for BPE tokenizers
    native=NativeTaskDef(
        protocol="gptq_ppl",
        dataset="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="test",
        text_column="text",
        joiner="\n\n",
        use_fast_tokenizer=False,
    ),
    params=Params,
)
