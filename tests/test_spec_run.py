"""Running speculative jobs: role loading in the transformers runtime, the run path and the node checks."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from catalogs import CATALOG  # noqa: E402
from effeval import check as node_check  # noqa: E402
from effeval import run as run_module  # noqa: E402
from effeval.core import evaluator as evaluators  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.spec import (  # noqa: E402
    DeterminismSpec,
    GenerationSpec,
    HardwareSpec,
    JobSpec,
    ResolvedDecoding,
    ResolvedEvaluator,
    ResolvedMethod,
    ResolvedModel,
    ResolvedRole,
    ResolvedTask,
    RuntimeSpec,
)
from effeval.core.types import Artifact, Mode, QualityResult, Sample  # noqa: E402
from effeval.ops.introspect import model_entry  # noqa: E402
from effeval.plan import freeze  # noqa: E402
from effeval.runtime import hf  # noqa: E402
from effeval.runtime.speculative import GreedyAcceptance, ModelDrafter, SpeculativeDecoder  # noqa: E402
from test_spec_schema import SPEED_TASK, _catalog, _speed_comparatives, spec  # noqa: E402, F401

BASE = ResolvedMethod(name="base", version="1", preset=None, config={})
RTN = ResolvedMethod(
    name="rtn",
    version="1",
    preset=None,
    config={
        "rules": [
            {
                "sites": ["linear.*.weight"],
                "quant": {"type": "int", "bits": 4, "symmetric": False, "granularity": "per_channel"},
            }
        ]
    },
)


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained("gpt2")


def save_tiny(path, tokenizer, *, hidden, config_vocab=None, extra_token=False, seed=0):
    """Save a tiny Qwen3 (and a gpt2 tokenizer, optionally with one more token) under `path`."""
    config_vocab = len(tokenizer) if config_vocab is None else config_vocab
    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=config_vocab,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    transformers.AutoModelForCausalLM.from_config(config).save_pretrained(path)
    saved = transformers.AutoTokenizer.from_pretrained("gpt2")
    if extra_token:
        saved.add_tokens(["<extra>"])
    saved.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory, tokenizer):
    root = tmp_path_factory.mktemp("spec-run")
    return {
        "target": save_tiny(root / "target", tokenizer, hidden=32, seed=0),
        "draft": save_tiny(root / "draft", tokenizer, hidden=16, seed=1),
        "wider_vocab": save_tiny(root / "wider", tokenizer, hidden=16, config_vocab=len(tokenizer) + 8),
        "other_tokenizer": save_tiny(root / "other-tok", tokenizer, hidden=16, extra_token=True),
    }


def resolved(path):
    return ResolvedModel(
        name=path, checkpoint=path, revision=None, architecture="Qwen3ForCausalLM", dtype="float32"
    )


def job_spec(target, draft, *, draft_method=BASE, k=3, verification="greedy"):
    evaluator = CATALOG.evaluators["lm_eval"]
    return JobSpec(
        model=resolved(target),
        method=BASE,
        field="speculative_decoding",
        runtime=RuntimeSpec(),
        decoding=ResolvedDecoding(
            mode="speculative",
            draft=ResolvedRole(model=resolved(draft), method=draft_method),
            num_speculative_tokens=k,
            verification=verification,
        ),
        task=ResolvedTask(name="lm_eval/gsm8k", version="1", mode=Mode.GENERATION, num_fewshot=0, limit=2),
        evaluator=ResolvedEvaluator(name=evaluator.name, version="1", library=None, passthrough={}),
        generation=GenerationSpec(max_new_tokens=4, batch_size=1),
        determinism=DeterminismSpec(),
        hardware=HardwareSpec(),
    )


def weights(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def test_prepare_role_applies_the_role_method(dirs):
    manifests = builtin_catalog().manifests
    plain = hf.prepare_role(resolved(dirs["draft"]), BASE, None, RuntimeSpec(), manifests["base"])
    quantized = hf.prepare_role(resolved(dirs["draft"]), RTN, None, RuntimeSpec(), manifests["rtn"])
    assert plain.decoder is None
    assert not torch.equal(weights(plain.model), weights(quantized.model))


def test_prepare_speculative_applies_the_draft_method_and_sets_the_decoder(dirs):
    manifests = builtin_catalog().manifests
    spec = job_spec(dirs["target"], dirs["draft"], draft_method=RTN, k=3)
    handle, draft_applied = hf.prepare_speculative(spec, manifests)
    assert isinstance(handle.decoder, SpeculativeDecoder)
    assert handle.decoder.k == 3
    assert isinstance(handle.decoder.drafter, ModelDrafter)
    assert isinstance(handle.decoder.rule, GreedyAcceptance)
    assert draft_applied is None
    plain_draft = hf.prepare_role(resolved(dirs["draft"]), BASE, None, RuntimeSpec(), manifests["base"])
    drafter_model = handle.decoder.drafter.model.model
    assert not torch.equal(weights(drafter_model), weights(plain_draft.model))
    plain_target = hf.prepare(spec, manifests["base"])
    assert torch.equal(weights(handle.model), weights(plain_target.model))
    assert handle.decoder.target.model is handle.model


@pytest.mark.parametrize("draft", ["wider_vocab", "other_tokenizer"])
def test_a_draft_with_another_vocabulary_is_rejected(dirs, draft):
    spec = job_spec(dirs["target"], dirs[draft])
    with pytest.raises(ValueError, match="vocab"):
        hf.prepare_speculative(spec, builtin_catalog().manifests)


def test_sampling_verification_is_not_implemented_yet(dirs):
    spec = job_spec(dirs["target"], dirs["draft"], verification="sampling")
    with pytest.raises(NotImplementedError, match="sampling"):
        hf.prepare_speculative(spec, builtin_catalog().manifests)


class Recorder:
    """Stand-in evaluator that remembers the handle of every job it runs."""

    name = "speed"

    def __init__(self):
        self.handles = {}

    def run(self, handle, job, task, workdir):
        self.handles[(job.task.name, job.decoding is not None)] = handle
        return QualityResult(
            metrics={"throughput_tok_s": "1.0"},
            samples=[Sample(id="0", scores={"tokens_per_s": 1.0})],
            protocol={},
            produces=frozenset({Artifact.PER_SAMPLE}),
        )


@pytest.fixture
def catalog_and_plan(dirs):
    target, _ = model_entry(dirs["target"], None)
    draft, _ = model_entry(dirs["draft"], None)
    second = SPEED_TASK.model_copy(update={"name": "speed/second"})
    catalog = _catalog(models={target.code: target, draft.code: draft})
    catalog = catalog.model_copy(update={"tasks": {**catalog.tasks, second.name: second}})
    raw = spec(
        hardware={"gpu": "H100-SXM"},
        model={"name": target.code, "dtype": "float32"},
        tasks=[SPEED_TASK.name, second.name],
        decoding={
            "mode": "speculative",
            "draft": {"model": {"name": draft.code, "dtype": "float32"}},
            "num_speculative_tokens": 3,
        },
    )
    plan, issues = freeze(raw, catalog, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return catalog, plan


def test_a_speculative_plan_prepares_once_per_group_and_hands_each_job_its_decoder(
    monkeypatch, tmp_path, catalog_and_plan
):
    catalog, plan = catalog_and_plan
    recorder = Recorder()
    monkeypatch.setattr(evaluators, "load", lambda name: recorder)
    calls = []
    for name in ("prepare", "prepare_speculative"):
        original = getattr(hf, name)
        monkeypatch.setattr(
            hf, name, lambda *args, _original=original, _name=name: calls.append(_name) or _original(*args)
        )
    run_module.run_plan(plan, catalog, client=None, out_dir=tmp_path, owner=None, log=lambda line: None)
    assert sorted(calls) == ["prepare", "prepare_speculative"]
    assert len(recorder.handles) == 4
    speculative = [h for (task, spec), h in recorder.handles.items() if spec]
    plain = [h for (task, spec), h in recorder.handles.items() if not spec]
    assert len(speculative) == 2 and speculative[0] is speculative[1]
    assert isinstance(speculative[0].decoder, SpeculativeDecoder)
    assert all(h.decoder is None for h in plain)


def test_a_job_whose_task_is_missing_from_the_catalog_fails_alone_with_a_clear_message(
    monkeypatch, tmp_path, catalog_and_plan
):
    catalog, plan = catalog_and_plan
    smaller = catalog.model_copy(
        update={"tasks": {n: t for n, t in catalog.tasks.items() if n != "speed/second"}}
    )
    recorder = Recorder()
    monkeypatch.setattr(evaluators, "load", lambda name: recorder)

    outcome = run_module.run_plan(
        plan, smaller, client=None, out_dir=tmp_path, owner=None, log=lambda line: None, keep_going=True
    )
    assert len(outcome.failed) == 2  # the missing task's main and reference jobs
    assert all("speed/second" in f and "not in this machine's task catalog" in f for f in outcome.failed)
    assert {task for task, _ in recorder.handles} == {SPEED_TASK.name}  # the other task still ran

    with pytest.raises(LookupError, match="not in this machine's task catalog"):
        run_module.run_plan(
            plan, smaller, client=None, out_dir=tmp_path / "strict", owner=None, log=lambda line: None
        )


def test_the_record_keeps_the_target_attention_and_the_job_spec(monkeypatch, tmp_path, catalog_and_plan):
    catalog, plan = catalog_and_plan
    monkeypatch.setattr(evaluators, "load", lambda name: Recorder())
    outcome = run_module.run_plan(
        plan, catalog, client=None, out_dir=tmp_path, owner=None, log=lambda line: None
    )
    assert outcome.failed == []
    records = [run_module.saved_record(tmp_path, job.job_id) for job in plan.jobs]
    main = [r for r in records if r.role == "main"]
    assert main and all(r.spec["decoding"]["num_speculative_tokens"] == 3 for r in main)
    assert all(r.env.attn_implementation == "sdpa" for r in records)


def test_an_engine_runtime_is_not_runnable_yet(dirs):
    spec = job_spec(dirs["target"], dirs["draft"]).model_copy(update={"runtime": RuntimeSpec(name="vllm")})
    with pytest.raises(NotImplementedError, match="vllm"):
        run_module.prepare_models(spec, builtin_catalog())


def test_node_checks_cover_the_draft_role(monkeypatch, catalog_and_plan):
    catalog, plan = catalog_and_plan
    seen = []
    original = transformers.AutoConfig.from_pretrained

    def record(checkpoint, **kwargs):
        seen.append(checkpoint)
        return original(checkpoint, **kwargs)

    monkeypatch.setattr(node_check.AutoConfig, "from_pretrained", record)
    node_check.check(plan, catalog)
    main = next(j for j in plan.jobs if j.role == "main").spec
    assert set(seen) == {main.model.checkpoint, main.decoding.draft.model.checkpoint}
    assert len(seen) == 2
