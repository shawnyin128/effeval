"""Grouped decode attention of the static cache: closeness to SDPA, float64 equality, gating, graphs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from effeval.core.fragments import IntFormat  # noqa: E402
from effeval.ops import online  # noqa: E402
from effeval.runtime.speculative import static_attention  # noqa: E402
from effeval.runtime.speculative.state import CachedModel  # noqa: E402
from effeval.runtime.speculative.static import StaticCachedModel, StaticCacheSettings  # noqa: E402
from effeval.runtime.speculative.static_attention import (  # noqa: E402
    EAGER_NAME,
    SDPA_NAME,
    grouped_attention,
    static_sdpa_attention,
)
from test_spec_loop import tiny_qwen3  # noqa: E402
from test_static_cache import (  # noqa: E402
    GRAPHS,
    SMALL,
    assert_blocks_equal,
    attach,
    cuda_model,
    drive,
    drive_shapes,
    gpu,
    greedy,
    make_prompt,
)

UNGROUPED = StaticCacheSettings(capacity=96, cuda_graphs=False, min_bucket=16, grouped_attention=False)


def reference(query, key, value, mask, scaling):
    """SDPA over key/value repeated to the query heads: the math the grouped path must reproduce."""
    groups = query.shape[1] // key.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(
        query,
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
        attn_mask=mask,
        scale=scaling,
    ).transpose(1, 2)


def make_inputs(queries, span, dtype=torch.float64, heads=6, kv_heads=2, dim=8, seed=0):
    """Random query [1, heads, queries, dim], key/value [1, kv_heads, span, dim], causal boolean mask."""
    generator = torch.Generator().manual_seed(seed)
    q, k, v = (
        torch.randn(1, n, length, dim, generator=generator).to(dtype)
        for n, length in ((heads, queries), (kv_heads, span), (kv_heads, span))
    )
    position = span - queries
    allowed = torch.arange(span)[None, :] <= (position + torch.arange(queries))[:, None]
    return q, k, v, allowed[None, None]


@pytest.mark.parametrize("queries", [1, 3, 5, 16])
@pytest.mark.parametrize("additive", [False, True])
def test_grouped_attention_equals_sdpa_in_float64(queries, additive):
    module = torch.nn.Identity()
    q, k, v, allowed = make_inputs(queries, 40)
    mask = torch.zeros(allowed.shape, dtype=q.dtype).masked_fill(~allowed, torch.finfo(q.dtype).min)
    out, weights = grouped_attention(module, q, k, v, mask if additive else allowed, 0.3, 0.0)
    assert weights is None and out.shape == (1, queries, 6, 8)
    assert torch.allclose(out, reference(q, k, v, allowed, 0.3), atol=1e-12, rtol=0)


def test_grouped_attention_default_scaling_is_one_over_sqrt_dim():
    module = torch.nn.Identity()
    q, k, v, allowed = make_inputs(2, 20)
    out = grouped_attention(module, q, k, v, allowed, None, 0.0)[0]
    assert torch.allclose(out, reference(q, k, v, allowed, 8**-0.5), atol=1e-12, rtol=0)


def test_grouped_attention_hides_masked_rows():
    module = torch.nn.Identity()
    q, k, v, allowed = make_inputs(4, 40)
    poisoned_v = v.clone()
    poisoned_v[:, :, 39:] = 1e4
    clean = grouped_attention(module, q, k, v, allowed, 0.3, 0.0)[0]
    got = grouped_attention(module, q, k, poisoned_v, allowed, 0.3, 0.0)[0]
    assert torch.allclose(got[:, :3], clean[:, :3], atol=1e-12, rtol=0)
    assert not torch.allclose(got[:, 3:], clean[:, 3:], atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_attention_is_close_to_sdpa_in_low_precision(dtype):
    module = torch.nn.Identity()
    q, k, v, allowed = make_inputs(5, 64, dtype=dtype)
    got = grouped_attention(module, q, k, v, allowed, 0.35, 0.0)[0].double()
    want = reference(q.double(), k.double(), v.double(), allowed, 0.35)
    assert (got - want).abs().max() < (2e-6 if dtype == torch.float32 else 3e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32, torch.float64])
def test_softmax_runs_in_float32_or_wider(monkeypatch, dtype):
    """The softmax dtype is checked at the call: outputs cannot tell float32 from input-dtype softmax.

    Measured on CPU and CUDA (bf16, scores scaled x1 to x16, 64 and 512 keys): the output of a softmax in the
    input dtype equals the output of a float32 softmax rounded to bf16 bit for bit, because both accumulate in
    float32 and round once, so no output tolerance separates the two.
    """
    seen = []
    real = torch.nn.functional.softmax

    def spy(x, dim=None, _stacklevel=3, dtype=None):
        seen.append(x.dtype if dtype is None else dtype)
        return real(x, dim=dim, dtype=dtype)

    monkeypatch.setattr(torch.nn.functional, "softmax", spy)
    q, k, v, allowed = make_inputs(3, 20, dtype=dtype)
    out = grouped_attention(torch.nn.Identity(), q, k, v, allowed, 0.35, 0.0)[0]
    assert seen == [torch.promote_types(dtype, torch.float32)]
    assert out.dtype == dtype


def test_mask_less_multi_query_and_long_calls_take_the_sdpa_path(monkeypatch):
    module = torch.nn.Identity()
    q, k, v, allowed = make_inputs(5, 5)
    calls = []
    monkeypatch.setattr(
        static_attention,
        "sdpa_attention_forward",
        lambda *args, **kwargs: calls.append(1) or (torch.zeros(1), None),
    )
    static_sdpa_attention(module, q, k, v, None, scaling=0.3)
    assert calls == [1]
    static_sdpa_attention(module, q[:, :, :1], k, v, None, scaling=0.3)
    assert calls == [1]
    static_sdpa_attention(module, q.repeat(1, 1, 4, 1), k, v, allowed.repeat(1, 1, 4, 1), scaling=0.3)
    assert calls == [1, 1]


def test_static_cached_model_switches_plain_attention_and_records_the_previous_one():
    model = tiny_qwen3()
    static = StaticCachedModel(model, SMALL)
    assert static.previous_attn_implementation == "sdpa"
    assert model.config._attn_implementation == SDPA_NAME
    assert static.prepare([(1, 1)])["attention"] == SDPA_NAME
    ungrouped_model = tiny_qwen3()
    other = StaticCachedModel(ungrouped_model, UNGROUPED)
    assert ungrouped_model.config._attn_implementation == "sdpa"
    assert other.prepare([(1, 1)])["attention"] == "sdpa"
    eager_model = tiny_qwen3()
    eager_model.set_attn_implementation("eager")
    StaticCachedModel(eager_model, SMALL)
    assert eager_model.config._attn_implementation == EAGER_NAME


def test_two_static_models_over_one_model_keep_the_grouped_attention():
    model = tiny_qwen3()
    first = StaticCachedModel(model, SMALL)
    second = StaticCachedModel(model, SMALL)
    assert model.config._attn_implementation == SDPA_NAME
    assert first.prepare([(1, 1)])["attention"] == second.prepare([(1, 1)])["attention"] == SDPA_NAME


def test_grouped_function_runs_in_every_decode_forward(monkeypatch):
    calls = []
    real = static_attention.grouped_attention
    monkeypatch.setattr(
        static_attention, "grouped_attention", lambda *a, **k: calls.append(a[1].shape[2]) or real(*a, **k)
    )
    static = StaticCachedModel(tiny_qwen3(layers=2), SMALL)
    seq = make_prompt(9)
    static.feed(seq, None, 1)
    static.feed(torch.cat([seq, torch.tensor([[4]])], dim=1), None, 1)
    assert calls == [9, 9, 1, 1]


def test_logits_equal_the_dynamic_path_for_single_multi_query_and_long_prefill_calls():
    long_settings = StaticCacheSettings(capacity=512, cuda_graphs=False, min_bucket=64)
    for prompt_length, settings in ((9, SMALL), (300, long_settings)):
        prompt = make_prompt(prompt_length)
        with torch.inference_mode():
            want = drive(CachedModel(tiny_qwen3()), prompt)
        got = drive(StaticCachedModel(tiny_qwen3(), settings), prompt)
        assert_blocks_equal(got, want, atol=1e-10)
        assert [b.argmax(-1).tolist() for b in got] == [b.argmax(-1).tolist() for b in want]


def test_eager_model_logits_equal_the_dynamic_eager_path():
    dynamic_model = tiny_qwen3()
    dynamic_model.set_attn_implementation("eager")
    model = tiny_qwen3()
    model.set_attn_implementation("eager")
    prompt = make_prompt(12)
    with torch.inference_mode():
        want = drive(CachedModel(dynamic_model), prompt)
    # eager attention softmaxes in float32, the grouped path in float64: equal to the 1e-5 level
    assert_blocks_equal(drive(StaticCachedModel(model, SMALL), prompt), want, atol=1e-5)


def test_greedy_tokens_equal_hf_generate_after_the_switch():
    model = tiny_qwen3()
    prompt = make_prompt(11)
    with torch.inference_mode():
        want = model.generate(prompt, max_new_tokens=30, do_sample=False, use_cache=True)[0, 11:].tolist()
    static = StaticCachedModel(model, SMALL)
    assert model.config._attn_implementation == SDPA_NAME
    assert greedy(static, prompt, 30) == want
    with torch.inference_mode():
        again = model.generate(prompt, max_new_tokens=30, do_sample=False, use_cache=True)[0, 11:].tolist()
    assert again == want


def test_sites_keep_their_attention_function_and_stay_quantizable():
    model = tiny_qwen3()
    fmt = IntFormat(type="int", bits=4, symmetric=False, granularity="per_token")
    attachment = attach(model, "attn.k", fmt)
    try:
        static = StaticCachedModel(model, SMALL)
        assert model.config._attn_implementation == "effeval_sdpa"
        assert static.prepare([(1, 1)])["attention"] == "effeval_sdpa"
        prompt = make_prompt(14)
        quantized = greedy(static, prompt, 20)
        with torch.inference_mode():
            assert greedy(CachedModel(model), prompt, 20) == quantized
    finally:
        attachment.remove()
    assert greedy(StaticCachedModel(tiny_qwen3(), UNGROUPED), make_prompt(14), 20) != quantized


def test_online_ops_on_attention_operands_keep_their_attention_function():
    model = tiny_qwen3()
    heads_q = torch.linalg.qr(torch.randn(4, 32, 32, generator=torch.Generator().manual_seed(2))).Q.double()
    heads_k = torch.linalg.qr(torch.randn(2, 32, 32, generator=torch.Generator().manual_seed(1))).Q.double()
    attachment = online.register(
        model,
        [
            online.OnlineOp("attn.q", online.HeadwiseRightMultiply(heads_q)),
            online.OnlineOp("attn.k", online.HeadwiseRightMultiply(heads_k)),
        ],
    )
    try:
        static = StaticCachedModel(model, SMALL)
        assert model.config._attn_implementation == "effeval_sdpa"
        assert static.prepare([(1, 1)])["attention"] == "effeval_sdpa"
        prompt = make_prompt(10)
        with torch.inference_mode():
            want = drive(CachedModel(model), prompt)
        assert_blocks_equal(drive(static, prompt), want, atol=1e-9)
    finally:
        attachment.remove()


@gpu
def test_graph_with_grouped_attention_equals_eager_static_and_dynamic_in_float64():
    prompt = make_prompt(250)
    settings = dict(capacity=1024, min_bucket=256)
    with torch.inference_mode():
        want = greedy(CachedModel(cuda_model()), prompt, 40)
    eager = StaticCachedModel(cuda_model(), StaticCacheSettings(cuda_graphs=False, **settings))
    assert greedy(eager, prompt, 40) == want
    graphed = StaticCachedModel(cuda_model(), StaticCacheSettings(cuda_graphs=True, **settings))
    graphed.prepare([(1, 1)])
    assert graphed.model.config._attn_implementation == SDPA_NAME
    assert greedy(graphed, prompt, 40) == want


@gpu
def test_graph_rollback_schedule_with_grouped_attention_matches_dynamic_in_float64():
    prompt = make_prompt(14)
    with torch.inference_mode():
        want = drive(CachedModel(cuda_model()), prompt)
    graphed = StaticCachedModel(cuda_model(), GRAPHS)
    graphed.prepare(drive_shapes() | {(1, 1)})
    assert_blocks_equal(drive(graphed, prompt), want)


@gpu
def test_bfloat16_graph_replay_is_bitwise_equal_to_the_eager_forward_with_grouped_attention():
    model = cuda_model().to(torch.bfloat16)
    graphed = StaticCachedModel(model, GRAPHS)
    graphed.prepare(drive_shapes() | {(1, 1)})
    eager = StaticCachedModel(
        cuda_model().to(torch.bfloat16), StaticCacheSettings(capacity=96, cuda_graphs=False, min_bucket=16)
    )
    prompt = make_prompt(14)
    for got, want in zip(drive(graphed, prompt), drive(eager, prompt), strict=True):
        assert torch.equal(got, want)


@gpu
def test_bfloat16_grouped_attention_is_close_to_sdpa_on_cuda():
    module = torch.nn.Identity()
    for queries in (1, 5):
        inputs = make_inputs(queries, 640, dtype=torch.bfloat16, heads=8, dim=64)
        q, k, v, allowed = (t.cuda() for t in inputs)
        got = grouped_attention(module, q, k, v, allowed, 0.125, 0.0)[0]
        want = reference(q, k, v, allowed, 0.125)
        assert (got.float() - want.float()).abs().max() < 2e-2
