"""The CodeQuant perplexity tasks: native/c4_ppl and native/wikitext2_preprocessed_ppl."""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from catalogs import CATALOG  # noqa: E402
from effeval.core.catalog import NativeTaskDef  # noqa: E402
from effeval.core.spec import (  # noqa: E402
    DeterminismSpec,
    GenerationSpec,
    HardwareSpec,
    JobSpec,
    ResolvedEvaluator,
    ResolvedMethod,
    ResolvedModel,
    ResolvedTask,
    RuntimeSpec,
)
from effeval.core.types import InProcess  # noqa: E402
from effeval.evaluators.native import EVALUATOR  # noqa: E402


class IdRecorder(torch.nn.Module):
    """A model that returns zero logits and keeps every window of token ids it was called with."""

    def __init__(self, vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 2)
        self.windows = []
        self.vocab = vocab

    def get_input_embeddings(self):
        return self.embed

    def forward(self, ids, use_cache=False):
        self.windows.append(ids.clone())
        return transformers.modeling_outputs.CausalLMOutput(logits=torch.zeros(*ids.shape, self.vocab))


def run_task(task, limit, seqlen, tokenizer_name="gpt2"):
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    model = IdRecorder(len(tokenizer))
    spec = JobSpec(
        model=ResolvedModel(
            name="gpt2",
            checkpoint=tokenizer_name,
            revision=None,
            architecture="GPT2LMHeadModel",
            dtype="float32",
        ),
        method=ResolvedMethod(name="base", version="1", preset=None, config={}),
        field="quantization",
        runtime=RuntimeSpec(),
        task=ResolvedTask(
            name=task.name,
            version=task.version,
            mode=task.default_mode,
            num_fewshot=0,
            limit=limit,
            params={"seqlen": seqlen},
        ),
        evaluator=ResolvedEvaluator(name="native", version="1", library=None, passthrough={}),
        generation=GenerationSpec(max_new_tokens=4, batch_size=1),
        determinism=DeterminismSpec(),
        hardware=HardwareSpec(),
    )
    result = EVALUATOR.run(InProcess(model=model, tokenizer=tokenizer), spec, task, None)
    return result, torch.cat(model.windows, dim=1)


def tokenizer():
    return transformers.AutoTokenizer.from_pretrained("gpt2", use_fast=False)


def official_text(load):
    """Return `load()`, or skip the test when the Hub data is neither cached nor reachable."""
    try:
        return load()
    except OSError as error:
        pytest.skip(f"dataset unavailable offline: {error}")
    except ValueError as error:
        if "Couldn't find cache" not in str(error):
            raise
        pytest.skip(f"dataset unavailable offline: {error}")


@pytest.mark.compute
@pytest.mark.network
def test_preprocessed_wikitext_ids_equal_the_official_construction():
    from datasets import load_dataset

    data = official_text(lambda: load_dataset("Self-GRIT/wikitext-2-raw-v1-preprocessed", split="test"))
    salesforce = official_text(lambda: load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"))
    task = CATALOG.tasks["native/wikitext2_preprocessed_ppl"]
    result, seen = run_task(task, 3, 2048)
    official = tokenizer()("\n\n".join(data["text"]), return_tensors="pt").input_ids
    assert len(data) == 1835
    assert result.protocol["tokens"] == official.numel()
    assert torch.equal(seen, official[:, : 3 * 2048])
    other = tokenizer()("\n\n".join(salesforce["text"]), return_tensors="pt").input_ids
    assert other.numel() != official.numel()


@pytest.mark.compute
@pytest.mark.network
def test_c4_ids_equal_the_official_construction():
    import itertools

    from datasets import load_dataset

    def first_rows():
        stream = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        return list(itertools.islice(stream, 1100))

    rows = official_text(first_rows)
    task = CATALOG.tasks["native/c4_ppl"]
    result, seen = run_task(task, 3, 2048)
    official = tokenizer()(" ".join(row["text"] for row in rows), return_tensors="pt").input_ids[:, :524288]
    assert result.protocol["tokens"] == official.numel() == 524288
    assert torch.equal(seen, official[:, : 3 * 2048])


@pytest.fixture
def jsonl(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("".join(json.dumps({"text": f"document {i} " + "word " * 40}) + "\n" for i in range(10)))
    return str(path)


def documents(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in handle]


def fixture_task(jsonl, **fields):
    base = CATALOG.tasks["native/wikitext2_ppl"]
    native = NativeTaskDef(
        protocol="gptq_ppl",
        dataset="json",
        dataset_config=None,
        split="train",
        text_column="text",
        joiner=" | ",
        use_fast_tokenizer=False,
        data_files=jsonl,
        **fields,
    )
    return base.model_copy(update={"native": native})


@pytest.mark.compute
def test_native_task_honours_data_files_max_documents_and_token_cap(jsonl):
    texts = documents(jsonl)
    full = tokenizer()(" | ".join(texts[:4]), return_tensors="pt").input_ids
    result, seen = run_task(fixture_task(jsonl, max_documents=4), None, 16)
    assert result.protocol["tokens"] == full.numel()
    assert torch.equal(seen, full[:, : full.numel() // 16 * 16])
    result, seen = run_task(fixture_task(jsonl, max_documents=4, max_tokens=48), None, 16)
    assert result.protocol["tokens"] == 48 and torch.equal(seen, full[:, :48])
    assert result.protocol["max_documents"] == 4 and result.protocol["max_tokens"] == 48
    result, seen = run_task(fixture_task(jsonl, max_documents=4, max_tokens=48), None, 8)
    assert result.protocol["tokens"] == 48 and torch.equal(seen, full[:, :48])


@pytest.mark.compute
def test_existing_ppl_protocol_record_is_unchanged_without_the_new_fields(jsonl):
    result, _ = run_task(fixture_task(jsonl), None, 16)
    assert not {"max_documents", "max_tokens"} & set(result.protocol)
    full = tokenizer()(" | ".join(documents(jsonl)), return_tensors="pt").input_ids
    assert result.protocol["tokens"] == full.numel()
