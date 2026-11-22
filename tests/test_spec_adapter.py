"""GenerateProxy and SpeculativeHFLM: HF-style generate calls routed through a speculative decoder."""

from __future__ import annotations

import copy

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("lm_eval")

from lm_eval.models.huggingface import HFLM  # noqa: E402

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.evaluators.lm_eval import SpeculativeHFLM  # noqa: E402
from effeval.runtime.speculative import GreedyAcceptance, ModelDrafter, SpeculativeDecoder  # noqa: E402
from effeval.runtime.speculative.adapter import GenerateProxy, decoding_metrics  # noqa: E402

VOCAB = 97


def make_tokenizer():
    from tokenizers import Tokenizer, models

    vocab = {f"a{i}": i for i in range(VOCAB - 1)}
    vocab["\n\n"] = VOCAB - 1
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="a0"))
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer)


@pytest.fixture(scope="module")
def target():
    return build_tiny_qwen3().double()


@pytest.fixture(scope="module")
def draft(target):
    draft = copy.deepcopy(target)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for p in draft.parameters():
            p += torch.randn(p.shape, generator=generator, dtype=p.dtype) * 0.05 * p.std()
    return draft


@pytest.fixture
def proxy(target, draft):
    target.generation_config.eos_token_id = None
    decoder = SpeculativeDecoder(target, ModelDrafter(draft), 3, GreedyAcceptance())
    return GenerateProxy(decoder, target, make_tokenizer())


def prompt(length: int = 6, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB - 1, (1, length), generator=generator)


def hf_generate(target, ids, **kwargs):
    with torch.inference_mode():
        return target.generate(ids, do_sample=False, pad_token_id=None, **kwargs)


def test_proxy_exposes_target_attributes(proxy, target):
    assert proxy.device == target.device
    assert proxy.dtype == torch.float64
    assert proxy.config is target.config
    assert proxy.generation_config is target.generation_config


def test_generate_matches_hf_greedy_for_max_new_tokens_and_max_length(proxy, target):
    ids = prompt()
    assert torch.equal(proxy.generate(ids, max_new_tokens=20), hf_generate(target, ids, max_new_tokens=20))
    assert torch.equal(proxy.generate(ids, max_length=17), hf_generate(target, ids, max_length=17))


def test_generate_stops_after_eos_from_generation_config(proxy, target):
    ids = prompt()
    full = hf_generate(target, ids, max_new_tokens=20)
    eos = int(full[0, ids.shape[1] + 4])
    target.generation_config.eos_token_id = [eos, 95]
    expected = hf_generate(target, ids, max_new_tokens=20)
    got = proxy.generate(ids, max_new_tokens=20)
    assert expected.shape[1] < full.shape[1]
    assert torch.equal(got, expected)


def test_return_dict_has_sequences_and_no_logits(proxy, target):
    ids = prompt()
    out = proxy.generate(ids, max_new_tokens=8, return_dict_in_generate=True, use_cache=True, pad_token_id=0)
    assert torch.equal(out.sequences, hf_generate(target, ids, max_new_tokens=8))
    assert out.logits is None


def test_unsupported_arguments_fail_fast(proxy):
    ids = prompt()
    with pytest.raises(TypeError):
        proxy.generate(ids, max_new_tokens=4, repetition_penalty=1.2)
    with pytest.raises(ValueError):
        proxy.generate(torch.cat([ids, ids]), max_new_tokens=4)
    with pytest.raises(ValueError):
        proxy.generate(ids, attention_mask=torch.tensor([[0, 1, 1, 1, 1, 1]]), max_new_tokens=4)
    with pytest.raises(ValueError):
        proxy.generate(ids, max_new_tokens=4, num_return_sequences=2)
    with pytest.raises(ValueError):
        proxy.generate(ids, max_new_tokens=4, top_k=5)
    with pytest.raises(ValueError):
        proxy.generate(ids, max_new_tokens=4, output_logits=True)
    with pytest.raises(ValueError):
        proxy.generate(ids)


def test_stats_accumulate_over_calls(proxy):
    ids = prompt()
    proxy.generate(ids, max_new_tokens=10)
    first = proxy.stats.new_tokens
    proxy.generate(ids, max_new_tokens=10)
    assert first == 10
    assert proxy.stats.new_tokens == 20
    assert proxy.stats.rounds > 0
    assert len(proxy.stats.accept_lengths) == proxy.stats.rounds
    metrics = decoding_metrics(proxy.stats)
    assert float(metrics["mean_accepted_tokens"]) == pytest.approx(
        1 + proxy.stats.accepted / proxy.stats.rounds
    )
    assert float(metrics["acceptance_rate"]) == pytest.approx(proxy.stats.accepted / proxy.stats.proposed)


def test_hflm_speculative_generate_matches_base(proxy, target):
    tokenizer = proxy.tokenizer
    base = HFLM(pretrained=target, tokenizer=tokenizer, batch_size=1)
    spec = SpeculativeHFLM(pretrained=target, tokenizer=tokenizer, batch_size=1, proxy=proxy)
    context = prompt(5, seed=3)
    free = base._model_generate(context, context.shape[1] + 24, stop=[])
    stop_text = tokenizer.decode(free[0, context.shape[1] + 6 : context.shape[1] + 7])
    for kwargs in ({}, {"do_sample": False}):
        want = base._model_generate(context, context.shape[1] + 24, stop=[stop_text], **kwargs)
        got = spec._model_generate(context, context.shape[1] + 24, stop=[stop_text], **kwargs)
        assert want.shape[1] < free.shape[1], "the stop string must cut the generation"
        assert torch.equal(got, want)
