"""Integration tests against the real frameworks with a tiny random model. Need network for datasets."""

import json
import math
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from catalogs import CATALOG, LISTING_DIR  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
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
from effeval.core.types import Artifact  # noqa: E402

pytestmark = [pytest.mark.compute, pytest.mark.network]


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-qwen3")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def job(model_dir, task, limit, *, catalog=CATALOG, num_fewshot=..., dtype="float32"):
    definition = catalog.tasks[task]
    evaluator = catalog.evaluators[definition.evaluator]
    mode = definition.default_mode
    return JobSpec(
        model=ResolvedModel(
            name="tiny", checkpoint=model_dir, revision=None, architecture="Qwen3ForCausalLM", dtype=dtype
        ),
        method=ResolvedMethod(name="base", version="1", preset=None, config={}),
        field="quantization",
        runtime=RuntimeSpec(),
        task=ResolvedTask(
            name=task,
            version="1",
            mode=mode,
            num_fewshot=definition.num_fewshot if num_fewshot is ... else num_fewshot,
            limit=limit,
        ),
        evaluator=ResolvedEvaluator(name=evaluator.name, version="1", library=None, passthrough={}),
        generation=GenerationSpec(max_new_tokens=4, batch_size=2),
        determinism=DeterminismSpec(),
        hardware=HardwareSpec(),
    )


def handle(model_dir):
    from effeval.runtime import hf

    return hf.load(job(model_dir, "lm_eval/arc_challenge", 1))


def test_lm_eval_loglikelihood_returns_full_precision_and_choice_logliks(tiny_model_dir, tmp_path):
    from effeval.evaluators.lm_eval import EVALUATOR

    spec = job(tiny_model_dir, "lm_eval/arc_challenge", 3)
    result = EVALUATOR.run(handle(tiny_model_dir), spec, CATALOG.tasks["lm_eval/arc_challenge"], tmp_path)
    assert {"acc", "acc_norm"} <= set(result.metrics)
    assert len(result.samples) == 3
    assert all(s.choice_loglikelihoods and len(s.choice_loglikelihoods) >= 3 for s in result.samples)
    assert float(result.metrics["acc"]) == sum(s.scores["correct"] for s in result.samples) / 3
    assert Artifact.CHOICE_LOGLIKELIHOODS in result.produces


def test_evalscope_generation_recomputes_aggregate_from_reviews(tiny_model_dir, tmp_path):
    from effeval.evaluators.evalscope import EVALUATOR

    spec = job(tiny_model_dir, "evalscope/gsm8k", 2)
    result = EVALUATOR.run(handle(tiny_model_dir), spec, CATALOG.tasks["evalscope/gsm8k"], tmp_path)
    assert len(result.samples) == 2
    assert all(s.id.startswith("main:") for s in result.samples)
    assert float(result.metrics["accuracy"]) == sum(s.scores["correct"] for s in result.samples) / 2


def test_native_gptq_ppl_matches_manual_window_computation(tiny_model_dir, tmp_path):
    from effeval.evaluators.native import EVALUATOR

    task = CATALOG.tasks["native/wikitext2_ppl"]
    spec = job(tiny_model_dir, "native/wikitext2_ppl", 2)
    spec = spec.model_copy(update={"task": spec.task.model_copy(update={"params": {"seqlen": 64}})})
    result = EVALUATOR.run(handle(tiny_model_dir), spec, task, tmp_path)
    assert (result.protocol["windows"], result.protocol["seqlen"]) == (2, 64)
    nll = [s.scores["nll"] for s in result.samples]
    mean_nll = sum(nll) / len(nll)
    expected = math.exp(mean_nll)
    # The window losses are float32 values, exact in float64. The evaluator sums them in float32 (each scaled
    # by seqlen), divides and exponentiates in float32; every operation adds at most u = 2**-24 relative
    # error, the exponent's absolute error (about (n + 2) * u * mean_nll) becomes ppl's relative error, and
    # exp itself adds about 2u.
    u = torch.finfo(torch.float32).eps / 2
    tolerance = expected * ((len(nll) + 2) * mean_nll + 2) * u
    assert abs(float(result.metrics["ppl"]) - expected) <= tolerance


FIXTURE_FIELDS = ("version", "mode", "produces", "num_fewshot", "fewshot_allowed", "metrics", "tags")


@pytest.mark.parametrize("name", ["lm_eval", "evalscope"])
def test_fixture_listings_match_the_installed_libraries(name):
    """Drift guard: the hand-kept fixture entries equal what the library lists today."""
    from effeval.core import evaluator

    live = {t.name: t for t in evaluator.load(name).list_tasks().tasks}
    fixture = json.loads((LISTING_DIR / f"{name}.json").read_text(encoding="utf-8"))
    for entry in fixture["tasks"]:
        listed = live[entry["name"]].model_dump(mode="json")
        assert {k: listed[k] for k in FIXTURE_FIELDS} == {k: entry[k] for k in FIXTURE_FIELDS}, entry["name"]


def test_lm_eval_lists_its_common_tasks():
    from effeval.evaluators.lm_eval import EVALUATOR

    names = {t.name for t in EVALUATOR.list_tasks().tasks}
    assert {"mmlu", "gsm8k", "arc_challenge", "wikitext", "hellaswag", "winogrande", "piqa"} <= names


REAL_MODEL = os.environ.get("EFFEVAL_TEST_MODEL", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="module")
def real_handle():
    from effeval.runtime import hf

    return hf.load(job(REAL_MODEL, "lm_eval/arc_challenge", 1, dtype="bfloat16"))


def test_lm_eval_applies_the_jobs_num_fewshot_to_the_prompts(real_handle, tmp_path, monkeypatch):
    from effeval.evaluators import lm_eval as adapter

    harness_results = []
    simple_evaluate = adapter.simple_evaluate

    def spy(**kwargs):
        harness_results.append(simple_evaluate(**kwargs))
        return harness_results[-1]

    monkeypatch.setattr(adapter, "simple_evaluate", spy)
    task = CATALOG.tasks["lm_eval/arc_challenge"]
    shots = {}
    for num_fewshot in (0, 3):
        spec = job(REAL_MODEL, task.name, 5, num_fewshot=num_fewshot, dtype="bfloat16")
        result = adapter.EVALUATOR.run(real_handle, spec, task, tmp_path)
        harness = harness_results[-1]
        context = harness["samples"]["arc_challenge"][0]["arguments"][0][0]
        shots[num_fewshot] = (harness["n-shot"]["arc_challenge"], context.count("Answer:"))
        print("lm_eval arc_challenge", num_fewshot, "shots:", result.metrics, shots[num_fewshot])
    assert shots == {0: (0, 1), 3: (3, 4)}


def test_evalscope_subset_child_task_runs_only_its_subset_with_the_jobs_fewshot(real_handle, tmp_path):
    from effeval.evaluators import evalscope as adapter

    catalog = builtin_catalog()
    task = catalog.tasks["evalscope/arc:ARC-Challenge"]
    assert task.subsets == ["ARC-Challenge"]
    spec = job(REAL_MODEL, task.name, 5, catalog=catalog, num_fewshot=2, dtype="bfloat16")
    result = adapter.EVALUATOR.run(real_handle, spec, task, tmp_path)
    print("evalscope arc:ARC-Challenge 2-shot limit 5:", result.metrics, result.protocol["subsets"])
    assert {s.id.split(":")[0] for s in result.samples} == {"ARC-Challenge"}
    assert len(result.samples) == 5
    assert float(result.metrics["accuracy"]) == sum(s.scores["correct"] for s in result.samples) / 5
    reviews = sorted((tmp_path / "reviews" / adapter.MODEL_NAME).glob("arc_*.jsonl"))
    assert [path.name for path in reviews] == ["arc_ARC-Challenge.jsonl"]
    for line in reviews[0].read_text(encoding="utf-8").splitlines():
        assert json.loads(line)["messages"][0]["content"].count("\nANSWER: ") == 2


def test_native_gptq_ppl_takes_a_fraction_of_the_windows(tiny_model_dir, tmp_path):
    from effeval.evaluators.native import EVALUATOR

    task = CATALOG.tasks["native/wikitext2_ppl"]
    spec = job(tiny_model_dir, "native/wikitext2_ppl", 0.001)
    spec = spec.model_copy(update={"task": spec.task.model_copy(update={"params": {"seqlen": 64}})})
    result = EVALUATOR.run(handle(tiny_model_dir), spec, task, tmp_path)
    total = result.protocol["tokens"] // 64
    assert result.protocol["windows"] == math.ceil(total * 0.001) and 1 <= result.protocol["windows"] < total


class LogitRecorder:
    """Delegates to a model and keeps every (input ids, logits) pair it saw."""

    def __init__(self, model):
        self.model = model
        self.calls = []

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def __call__(self, ids, **kwargs):
        out = self.model(ids, **kwargs)
        self.calls.append((ids, out.logits))
        return out


def test_native_ppl_computes_the_loss_from_float32_logits_of_a_bfloat16_model(tiny_model_dir, tmp_path):
    from effeval.core.types import InProcess
    from effeval.evaluators.native import EVALUATOR
    from effeval.runtime import hf

    task = CATALOG.tasks["native/wikitext2_ppl"]
    spec = job(tiny_model_dir, "native/wikitext2_ppl", 3, dtype="bfloat16")
    spec = spec.model_copy(update={"task": spec.task.model_copy(update={"params": {"seqlen": 64}})})
    loaded = hf.load(spec)
    recorder = LogitRecorder(loaded.model)
    result = EVALUATOR.run(InProcess(model=recorder, tokenizer=loaded.tokenizer), spec, task, tmp_path)
    assert len(recorder.calls) == 3
    loss_fn = torch.nn.CrossEntropyLoss()
    for (ids, logits), sample in zip(recorder.calls, result.samples, strict=True):
        assert logits.dtype == torch.bfloat16
        labels = ids[:, 1:].reshape(-1)
        shifted = logits[:, :-1, :].reshape(-1, logits.size(-1))
        in_float32 = float(loss_fn(shifted.float(), labels))
        in_bfloat16 = float(loss_fn(shifted, labels))
        assert in_float32 != in_bfloat16, "the test needs a window whose bfloat16 loss is rounded"
        assert sample.scores["nll"] == in_float32
