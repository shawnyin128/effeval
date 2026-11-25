from effeval.core.benchmarks.wikitext2_ppl import Params
from effeval.core.catalog import NativeTaskDef, TaskDef, TaskMetric, TaskSize
from effeval.core.types import Artifact, Mode

BENCHMARK = TaskDef(
    name="native/c4_ppl",
    evaluator="native",
    target="c4_ppl",
    version="1",
    category="language_modeling",
    description=(
        "C4 perplexity, CodeQuant protocol: the first 1100 documents of the allenai/c4 `en` validation "
        "shard 0 are joined with a space, tokenized once, cut to the first 524288 tokens (256 x 2048) and "
        "into non-overlapping windows of `seqlen` tokens, and ppl = exp(mean token NLL)."
    ),
    modes=[Mode.LOGLIKELIHOOD],
    default_mode=Mode.LOGLIKELIHOOD,
    num_fewshot=0,
    produces={Mode.LOGLIKELIHOOD: frozenset({Artifact.PER_SAMPLE})},
    metrics=[TaskMetric(name="ppl", higher_is_better=False)],
    sample_scores=["nll"],
    size=TaskSize(samples=1, prompt_tokens=524288),  # the fixed 256 x 2048 token cap
    native=NativeTaskDef(
        protocol="gptq_ppl",
        dataset="allenai/c4",
        dataset_config=None,
        split="validation",
        text_column="text",
        joiner=" ",
        use_fast_tokenizer=False,
        data_files="en/c4-validation.00000-of-00008.json.gz",
        max_documents=1100,
        max_tokens=256 * 2048,
    ),
    params=Params,
)
