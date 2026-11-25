from effeval.core.benchmarks.wikitext2_ppl import Params
from effeval.core.catalog import NativeTaskDef, TaskDef, TaskMetric, TaskSize
from effeval.core.types import Artifact, Mode

BENCHMARK = TaskDef(
    name="native/wikitext2_preprocessed_ppl",
    evaluator="native",
    target="wikitext2_preprocessed_ppl",
    version="1",
    category="language_modeling",
    description=(
        "Perplexity on the Self-GRIT preprocessed WikiText-2 test split (1835 rows; not the 4358-row "
        "Salesforce split of native/wikitext2_ppl), as in the CodeQuant evaluation: the split is joined "
        "with two line breaks, tokenized once, cut into non-overlapping windows of `seqlen` tokens, and "
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
        dataset="Self-GRIT/wikitext-2-raw-v1-preprocessed",
        dataset_config=None,
        split="test",
        text_column="text",
        joiner="\n\n",
        use_fast_tokenizer=False,
    ),
    params=Params,
)
