"""StaticCachedModel: equality with the dynamic CachedModel, rollback, buckets, hook gating, CUDA graphs."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.compute

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.cache_utils import StaticLayer  # noqa: E402

from effeval.core.fragments import FpFormat, IntFormat, MxFormat, QuantRule  # noqa: E402
from effeval.ops import online, sites  # noqa: E402
from effeval.runtime.speculative.state import CachedModel  # noqa: E402
from effeval.runtime.speculative.static import (  # noqa: E402
    StaticCachedModel,
    StaticCacheSettings,
    _BucketedStaticLayer,
    buckets,
)
from test_spec_loop import VOCAB, tiny_qwen3  # noqa: E402

requires_gpu = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def gpu(fn):
    for mark in requires_gpu:
        fn = mark(fn)
    return fn


SMALL = StaticCacheSettings(capacity=96, cuda_graphs=False, min_bucket=16)
EXTRAS = [5, 1, 3, 1, 4, 0, 2, 1, 5, 1, 3, 2, 1, 4, 0, 5]
PER_CHANNEL = IntFormat(type="int", bits=4, symmetric=False, granularity="per_channel")


def make_prompt(length: int, seed: int = 3):
    return torch.randint(0, VOCAB, (1, length), generator=torch.Generator().manual_seed(seed))


def drive(cm, prompt):
    """Prefill, then rounds that feed a varying number of extra tokens and roll all of them back.

    Returns the list of logits blocks. Rounds alternate large and small feeds so rejected rows of an earlier
    round lie beyond the tokens present in a later one.
    """
    generator = torch.Generator().manual_seed(11)
    seq = prompt.to(cm.device)
    blocks = [cm.feed(seq[:, :-1], None, 1)]
    for count in EXTRAS:
        extra = torch.randint(0, VOCAB, (1, count), generator=generator) if count else None
        logits = cm.feed(seq, extra, count + 1)
        blocks.append(logits)
        seq = torch.cat([seq, logits[0].argmax().view(1, 1)], dim=1)
        cm.rollback(seq.shape[1] - 1)
    return blocks


def drive_shapes():
    return {(count + 1, count + 1) for count in EXTRAS}


def assert_blocks_equal(a, b, atol=1e-10):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape
        assert torch.allclose(x.cpu(), y.cpu(), atol=atol)


def greedy(cm, prompt, steps):
    seq = prompt.to(cm.device)
    tokens = []
    logits = cm.feed(seq, None, 1)
    for _ in range(steps):
        nxt = logits[-1].argmax().view(1, 1)
        tokens.append(int(nxt))
        seq = torch.cat([seq, nxt], dim=1)
        logits = cm.feed(seq, None, 1)
    return tokens


def attach(model, site, quant):
    return sites.attach(model, [QuantRule(sites=[site], layers="all", quant=quant)])


def test_buckets():
    assert buckets(4096, 256) == [256, 512, 1024, 2048, 4096]
    assert buckets(1000, 256) == [256, 512, 1000]
    assert buckets(256, 256) == [256]
    assert buckets(100, 256) == [100]


def test_feed_matches_cached_model():
    model = tiny_qwen3()
    prompt = make_prompt(9)
    static = StaticCachedModel(model, SMALL)
    dynamic = CachedModel(model)
    with torch.inference_mode():
        seq = prompt
        got = [static.feed(seq, None, 1)]
        want = [dynamic.feed(seq, None, 1)]
        for _ in range(5):
            seq = torch.cat([seq, want[-1][-1].argmax().view(1, 1)], dim=1)
            got.append(static.feed(seq, None, 1))
            want.append(dynamic.feed(seq, None, 1))
    for g, w in zip(got, want, strict=True):
        assert g.dtype == torch.float64 and g.shape == w.shape
        assert torch.allclose(g, w, atol=1e-12)
    assert static.cached == dynamic.cached == 14


def test_extra_tokens_and_keep_rows():
    model = tiny_qwen3()
    seq = make_prompt(7, seed=2)
    static = StaticCachedModel(model, SMALL)
    static.feed(seq[:, :3], None, 1)
    logits = static.feed(seq[:, :4], seq[:, 4:7], 4)
    with torch.inference_mode():
        fresh = model(input_ids=seq).logits[0, -4:]
    assert static.cached == 7
    assert logits.shape == (4, VOCAB)
    assert torch.allclose(logits, fresh, atol=1e-10)


def test_rollback_then_refeed_matches_fresh_forward():
    model = tiny_qwen3()
    seq = make_prompt(8, seed=1)
    static = StaticCachedModel(model, SMALL)
    static.feed(seq[:, :5], None, 1)
    static.feed(seq[:, :8], None, 1)
    assert static.cached == 8
    static.rollback(6)
    assert static.cached == 6
    logits = static.feed(seq[:, :7], None, 1)
    with torch.inference_mode():
        fresh = model(input_ids=seq[:, :7]).logits[0, -1:]
    assert torch.allclose(logits, fresh, atol=1e-10)
    with pytest.raises(ValueError):
        static.rollback(9)


def test_reset_starts_a_fresh_generation():
    model = tiny_qwen3()
    seq = make_prompt(8)
    static = StaticCachedModel(model, SMALL)
    first = static.feed(seq, None, 1)
    static.reset()
    assert static.cached == 0
    assert torch.equal(static.feed(seq, None, 1), first)


def test_feed_past_capacity_raises():
    model = tiny_qwen3()
    static = StaticCachedModel(model, StaticCacheSettings(capacity=16, cuda_graphs=False, min_bucket=8))
    static.feed(make_prompt(16), None, 1)
    with pytest.raises(ValueError, match="capacity"):
        static.feed(make_prompt(17), None, 1)
    with pytest.raises(ValueError, match="capacity"):
        static.reserve(17)
    static.reserve(16)


def test_feed_rejects_nothing_new_and_bad_keep():
    static = StaticCachedModel(tiny_qwen3(), SMALL)
    seq = make_prompt(5)
    static.feed(seq, None, 1)
    with pytest.raises(ValueError, match="uncached"):
        static.feed(seq, None, 1)
    with pytest.raises(ValueError, match="keep"):
        static.feed(make_prompt(6), None, 2)


def test_greedy_decode_across_buckets_matches_cached_model():
    model = tiny_qwen3()
    prompt = make_prompt(14)
    with torch.inference_mode():
        want = greedy(CachedModel(model), prompt, 40)
    assert greedy(StaticCachedModel(model, SMALL), prompt, 40) == want


def test_rollback_schedule_matches_cached_model():
    model = tiny_qwen3()
    prompt = make_prompt(14)
    static = StaticCachedModel(model, SMALL)
    with torch.inference_mode():
        want = drive(CachedModel(model), prompt)
    assert_blocks_equal(drive(static, prompt), want)


def test_prepare_without_graphs_only_allocates():
    static = StaticCachedModel(tiny_qwen3(), SMALL)
    info = static.prepare([(1, 1), (3, 3)])
    assert info == {
        "graphs": 0,
        "capture_ms": 0.0,
        "buckets": [16, 32, 64, 96],
        "capacity": 96,
        "attention": "effeval_static_sdpa",
    }
    assert static.cached == 0
    assert all(layer.is_initialized and not layer.keys.any() for layer in static.cache.layers)


def test_eager_attention_model_matches_dynamic():
    model = tiny_qwen3()
    model.set_attn_implementation("eager")
    prompt = make_prompt(14)
    with torch.inference_mode():
        want = drive(CachedModel(model), prompt)
    got = drive(StaticCachedModel(model, SMALL), prompt)
    # eager attention softmaxes in float32, so the padded span rounds differently at the 1e-6 level
    assert_blocks_equal(got, want, atol=1e-5)
    assert [b.argmax(-1).tolist() for b in got] == [b.argmax(-1).tolist() for b in want]


def test_stale_rows_past_cached_do_not_change_logits():
    model = tiny_qwen3()
    seq = make_prompt(9, seed=5)
    clean = StaticCachedModel(model, SMALL)
    clean.feed(seq[:, :6], None, 1)
    expected = clean.feed(seq[:, :7], None, 1)
    poisoned = StaticCachedModel(model, SMALL)
    poisoned.feed(seq[:, :6], None, 1)
    poisoned.feed(seq[:, :9], None, 1)
    poisoned.rollback(6)
    with torch.inference_mode():
        for layer in poisoned.cache.layers:
            layer.keys[:, :, 6:] = 1e4
            layer.values[:, :, 6:] = 1e4
    assert torch.equal(poisoned.feed(seq[:, :7], None, 1), expected)


KEY_AXIS_FORMATS = {
    "int4_per_channel": PER_CHANNEL,
    "int4_per_tensor": IntFormat(type="int", bits=4, symmetric=False, granularity="per_tensor"),
    "int4_symmetric_per_channel": IntFormat(type="int", bits=4, symmetric=True, granularity="per_channel"),
    "fp_e4m3_per_tensor": FpFormat(type="fp", encoding="e4m3", granularity="per_tensor"),
    "mx_e4m3_scale": MxFormat(type="mx", element="e2m1", block_size=16, scale="e4m3"),
}


@pytest.mark.parametrize("site", ["attn.k", "attn.v"])
@pytest.mark.parametrize("name", list(KEY_AXIS_FORMATS))
def test_key_axis_quantizers_match_dynamic_through_rollbacks(name, site):
    model = tiny_qwen3()
    attachment = attach(model, site, KEY_AXIS_FORMATS[name])
    try:
        prompt = make_prompt(14)
        with torch.inference_mode():
            want = drive(CachedModel(model), prompt)
        assert_blocks_equal(drive(StaticCachedModel(model, SMALL), prompt), want, atol=1e-9)
    finally:
        attachment.remove()


def test_per_token_kv_quantizer_and_online_ops_match_dynamic():
    model = tiny_qwen3()
    attachment = attach(
        model, "attn.k", IntFormat(type="int", bits=4, symmetric=False, granularity="per_token")
    )
    try:
        prompt = make_prompt(14)
        with torch.inference_mode():
            want = drive(CachedModel(model), prompt)
        assert_blocks_equal(drive(StaticCachedModel(model, SMALL), prompt), want, atol=1e-9)
    finally:
        attachment.remove()


def test_zero_rows_are_what_makes_per_channel_exact():
    model = tiny_qwen3()
    attachment = attach(model, "attn.k", PER_CHANNEL)
    try:
        static = StaticCachedModel(model, SMALL)
        assert static._zero_stale
        static.feed(make_prompt(10), None, 1)
        static.rollback(4)
        assert all(
            not layer.keys[:, :, 4:].any() and layer.keys[:, :, :4].any() for layer in static.cache.layers
        )
    finally:
        attachment.remove()


def test_zero_rows_not_written_without_key_axis_sites():
    static = StaticCachedModel(tiny_qwen3(), SMALL)
    assert not static._zero_stale
    static.feed(make_prompt(10), None, 1)
    static.rollback(4)
    assert all(layer.keys[:, :, 4:10].any() for layer in static.cache.layers)


def greedy_logits(cm, prompt, steps):
    """Reset, then decode `steps` greedy tokens one at a time; returns the last logits row."""
    cm.reset()
    seq = prompt
    logits = cm.feed(seq, None, 1)
    for _ in range(steps):
        seq = torch.cat([seq, logits[-1].argmax().view(1, 1)], dim=1)
        logits = cm.feed(seq, None, 1)
    return logits


def test_reset_zeroes_the_rows_of_the_previous_sample_for_key_axis_quantizers():
    model = tiny_qwen3()
    attachment = attach(model, "attn.k", PER_CHANNEL)
    try:
        static = StaticCachedModel(model, StaticCacheSettings(capacity=96, cuda_graphs=False, min_bucket=16))
        long_prompt, short_prompt = make_prompt(40, seed=0), make_prompt(6, seed=1)
        with torch.inference_mode():
            greedy_logits(static, long_prompt, 20)
            static.reset()
            assert all(not layer.keys.any() and not layer.values.any() for layer in static.cache.layers)
            got = greedy_logits(static, short_prompt, 5)
            want = greedy_logits(CachedModel(model), short_prompt, 5)
        assert torch.allclose(got, want, atol=1e-9)
    finally:
        attachment.remove()


def test_sequence_options_are_rejected():
    model = tiny_qwen3()
    quant = IntFormat(type="int", bits=4, symmetric=False, granularity="per_token", residual=8)
    attachment = attach(model, "attn.k", quant)
    try:
        with pytest.raises(RuntimeError, match=r"residual.*runtime\.cuda_graphs: false"):
            StaticCachedModel(model, SMALL)
    finally:
        attachment.remove()


def test_attn_scores_site_is_rejected():
    model = tiny_qwen3()
    attachment = attach(
        model, "attn.scores", IntFormat(type="int", bits=8, symmetric=True, granularity="per_tensor")
    )
    try:
        with pytest.raises(RuntimeError, match=r"attn.scores.*runtime\.cuda_graphs: false"):
            StaticCachedModel(model, SMALL)
    finally:
        attachment.remove()


def test_float_zero_point_key_axis_quantizer_is_rejected():
    model = tiny_qwen3()
    quant = IntFormat(type="int", bits=4, symmetric=False, granularity="per_channel", zero_point="float")
    attachment = attach(model, "attn.k", quant)
    try:
        with pytest.raises(RuntimeError, match="zero_point float"):
            StaticCachedModel(model, SMALL)
    finally:
        attachment.remove()


def test_mx_e4m3_attention_site_is_rejected_with_graphs_only():
    model = tiny_qwen3()
    attachment = attach(model, "attn.k", KEY_AXIS_FORMATS["mx_e4m3_scale"])
    try:
        with pytest.raises(RuntimeError, match=r"host.*runtime\.cuda_graphs: false"):
            StaticCachedModel(model, GRAPHS)
        StaticCachedModel(model, SMALL)
    finally:
        attachment.remove()


@pytest.mark.parametrize("site", ["linear.mlp.*.input", "linear.attn.o_proj.output"])
def test_mx_e4m3_hooks_on_linear_sites_are_rejected_with_graphs_only(site):
    model = tiny_qwen3()
    attachment = attach(model, site, KEY_AXIS_FORMATS["mx_e4m3_scale"])
    try:
        with pytest.raises(RuntimeError, match=r"host.*runtime\.cuda_graphs: false"):
            StaticCachedModel(model, GRAPHS)
        StaticCachedModel(model, SMALL)
    finally:
        attachment.remove()


def test_mx_e8m0_linear_hooks_pass_the_graph_inspection_up_to_the_device_check():
    model = tiny_qwen3()
    attachment = attach(
        model, "linear.mlp.*.input", MxFormat(type="mx", element="e2m1", block_size=16, scale="e8m0")
    )
    try:
        with pytest.raises(RuntimeError, match="CUDA device"):
            StaticCachedModel(model, GRAPHS)
    finally:
        attachment.remove()


@pytest.mark.parametrize("graphs", [False, True])
def test_a_model_spread_over_several_devices_is_rejected(graphs):
    model = tiny_qwen3()
    model.hf_device_map = {"model.embed_tokens": 0, "model.layers": 0, "lm_head": 1}
    message = r"single-device.*cuda:0.*cuda:1.*runtime\.cuda_graphs: false"
    with pytest.raises(NotImplementedError, match=message):
        StaticCachedModel(model, StaticCacheSettings(capacity=96, cuda_graphs=graphs, min_bucket=16))


def test_parameters_on_several_devices_are_rejected():
    model = tiny_qwen3()
    model.lm_head.to("meta")
    with pytest.raises(NotImplementedError, match=r"single-device.*cpu.*meta"):
        StaticCachedModel(model, SMALL)


def test_a_single_device_map_is_accepted():
    model = tiny_qwen3()
    model.hf_device_map = {"": "cpu"}
    StaticCachedModel(model, SMALL)


def test_unsupported_attention_implementation_is_rejected():
    model = tiny_qwen3()
    model.set_attn_implementation("eager")
    model.config._attn_implementation = "flash_attention_2"
    with pytest.raises(ValueError, match="attention implementation"):
        StaticCachedModel(model, SMALL)


GRAPHS = StaticCacheSettings(capacity=96, cuda_graphs=True, min_bucket=16)


def untagged(_module, _args, _output):
    return None


def tagged(_module, _args, _output):
    return None


tagged._effeval_graph_safe = True


def test_untagged_hook_is_rejected_with_graphs_only():
    model = tiny_qwen3()
    handle = model.model.layers[0].mlp.register_forward_hook(untagged)
    try:
        with pytest.raises(RuntimeError, match=r"not graph-safe.*runtime\.cuda_graphs: false"):
            StaticCachedModel(model, GRAPHS)
        StaticCachedModel(model, SMALL)
    finally:
        handle.remove()


def test_global_hook_is_rejected_with_graphs():
    model = tiny_qwen3()
    handle = torch.nn.modules.module.register_module_forward_hook(untagged)
    try:
        with pytest.raises(RuntimeError, match="global module hooks"):
            StaticCachedModel(model, GRAPHS)
    finally:
        handle.remove()


def test_graphs_on_a_cpu_model_fail_fast():
    model = tiny_qwen3()
    handle = model.model.layers[0].mlp.register_forward_hook(tagged)
    try:
        with pytest.raises(RuntimeError, match=r"CUDA device.*runtime\.cuda_graphs: false"):
            StaticCachedModel(model, GRAPHS)
    finally:
        handle.remove()


def test_effeval_hooks_are_tagged_graph_safe():
    model = tiny_qwen3()
    quant = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token")
    attachments = [
        sites.attach(
            model,
            [
                QuantRule(sites=["linear.mlp.*.input"], quant=quant),
                QuantRule(sites=["linear.attn.o_proj.output"], quant=quant),
            ],
        )
    ]
    op = online.OnlineOp(site="linear.mlp.down_proj.input", layers="all", fn=lambda x: x)
    attachments.append(online.register(model, [op]))
    try:
        hooks = [
            h for m in model.modules() for h in (*m._forward_pre_hooks.values(), *m._forward_hooks.values())
        ]
        assert len(hooks) > 3
        assert all(getattr(h, "_effeval_graph_safe", False) for h in hooks)
    finally:
        for attachment in attachments:
            attachment.remove()


def test_hook_added_after_prepare_is_caught_at_reset():
    model = tiny_qwen3()
    static = StaticCachedModel(model, SMALL)
    static.prepare([(1, 1)])
    static.reset()
    handle = model.model.layers[0].mlp.register_forward_hook(tagged)
    try:
        with pytest.raises(RuntimeError, match=r"changed after prepare.*runtime\.cuda_graphs: false"):
            static.reset()
    finally:
        handle.remove()
    static.reset()


def test_attention_site_attached_after_prepare_is_caught_at_reset():
    model = tiny_qwen3()
    static = StaticCachedModel(model, SMALL)
    static.prepare([(1, 1)])
    attachment = attach(
        model, "attn.q", IntFormat(type="int", bits=8, symmetric=True, granularity="per_token")
    )
    try:
        with pytest.raises(RuntimeError, match="changed after prepare"):
            static.reset()
    finally:
        attachment.remove()


# ----------------------------------------------------------------------------------------- GPU (graphs)
def cuda_model(**kwargs):
    return tiny_qwen3(**kwargs).to("cuda")


@gpu
def test_graph_greedy_decode_across_a_bucket_boundary_matches_eager_and_dynamic():
    model = cuda_model()
    prompt = make_prompt(250)
    settings = dict(capacity=1024, min_bucket=256)
    with torch.inference_mode():
        want = greedy(CachedModel(model), prompt, 40)
    assert (
        greedy(StaticCachedModel(model, StaticCacheSettings(cuda_graphs=False, **settings)), prompt, 40)
        == want
    )
    graphed = StaticCachedModel(model, StaticCacheSettings(cuda_graphs=True, **settings))
    info = graphed.prepare([(1, 1)])
    assert info["graphs"] == 3 and info["buckets"] == [256, 512, 1024] and info["capture_ms"] > 0
    replays = []
    original = {key: g.graph.replay for key, g in graphed._graphs.items()}
    for key, g in graphed._graphs.items():
        g.graph.replay = lambda key=key: (replays.append(key), original[key]())[1]
    assert greedy(graphed, prompt, 40) == want
    assert {key[2] for key in replays} == {256, 512}
    assert len(replays) == 40


@gpu
def test_graph_rollback_schedule_matches_dynamic_and_eager():
    model = cuda_model()
    prompt = make_prompt(14)
    with torch.inference_mode():
        want = drive(CachedModel(model), prompt)
    graphed = StaticCachedModel(model, GRAPHS)
    graphed.prepare(drive_shapes())
    assert_blocks_equal(drive(graphed, prompt), want)
    assert_blocks_equal(drive(StaticCachedModel(model, SMALL), prompt), want)


@gpu
@pytest.mark.parametrize("name", ["int4_per_channel", "fp_e4m3_per_tensor"])
def test_graph_key_axis_quantizers_match_dynamic_through_rollbacks(name):
    model = cuda_model()
    attachment = attach(model, "attn.k", KEY_AXIS_FORMATS[name])
    try:
        prompt = make_prompt(14)
        with torch.inference_mode():
            want = drive(CachedModel(model), prompt)
        graphed = StaticCachedModel(model, GRAPHS)
        graphed.prepare(drive_shapes())
        assert_blocks_equal(drive(graphed, prompt), want, atol=1e-9)
    finally:
        attachment.remove()


@gpu
def test_graph_with_effeval_linear_hooks_matches_eager_static():
    model = cuda_model()
    quant = IntFormat(type="int", bits=4, symmetric=False, granularity="per_token")
    attachment = attach(model, "linear.mlp.*.input", quant)
    try:
        prompt = make_prompt(14)
        want = drive(StaticCachedModel(model, SMALL), prompt)
        graphed = StaticCachedModel(model, GRAPHS)
        graphed.prepare(drive_shapes())
        assert_blocks_equal(drive(graphed, prompt), want)
        plain = drive(StaticCachedModel(cuda_model(), SMALL), prompt)
        assert any(not torch.allclose(a.cpu(), b.cpu(), atol=1e-6) for a, b in zip(want, plain, strict=True))
    finally:
        attachment.remove()


@gpu
def test_graph_untagged_hook_added_after_prepare_is_caught_at_reset():
    model = cuda_model()
    graphed = StaticCachedModel(model, GRAPHS)
    graphed.prepare([(1, 1)])
    graphed.reset()
    handle = model.model.layers[0].mlp.register_forward_hook(tagged)
    try:
        with pytest.raises(RuntimeError, match="changed after prepare"):
            graphed.reset()
    finally:
        handle.remove()


@gpu
def test_graph_untagged_hook_is_rejected_at_init():
    model = cuda_model()
    handle = model.model.layers[0].mlp.register_forward_hook(untagged)
    try:
        with pytest.raises(RuntimeError, match="not graph-safe"):
            StaticCachedModel(model, GRAPHS)
    finally:
        handle.remove()


@gpu
def test_graph_feed_requires_prepare():
    graphed = StaticCachedModel(cuda_model(), GRAPHS)
    with pytest.raises(RuntimeError, match="prepare"):
        graphed.feed(make_prompt(5), None, 1)


@gpu
def test_graph_self_check_detects_a_wrong_graph_output(monkeypatch):
    graphed = StaticCachedModel(cuda_model(), GRAPHS)
    real = graphed._capture

    def capture_then_corrupt(q, keep, bucket):
        real(q, keep, bucket)
        captured = graphed._graphs[(q, keep, bucket)]
        captured.out = torch.zeros_like(captured.out)

    monkeypatch.setattr(graphed, "_capture", capture_then_corrupt)
    with pytest.raises(RuntimeError, match="differs from the eager forward"):
        graphed.prepare([(1, 1)])


@gpu
def test_graph_feed_returns_a_copy_of_the_graph_buffer():
    graphed = StaticCachedModel(cuda_model(), GRAPHS)
    graphed.prepare([(1, 1)])
    seq = make_prompt(10).cuda()
    graphed.feed(seq, None, 1)
    seq = torch.cat([seq, torch.tensor([[3]], device="cuda")], dim=1)
    first = graphed.feed(seq, None, 1)
    snapshot = first.clone()
    seq = torch.cat([seq, torch.tensor([[5]], device="cuda")], dim=1)
    second = graphed.feed(seq, None, 1)
    assert torch.equal(first, snapshot)
    assert not torch.equal(first, second)


@gpu
def test_graph_prepare_counts_and_state_after_prepare():
    graphed = StaticCachedModel(cuda_model(), GRAPHS)
    info = graphed.prepare([(1, 1), (2, 1), (3, 3)])
    assert info["graphs"] == 3 * 4 and info["buckets"] == [16, 32, 64, 96] and info["capacity"] == 96
    assert graphed.cached == 0
    assert all(not layer.keys.any() and not layer.values.any() for layer in graphed.cache.layers)
    assert graphed.prepare([(1, 1)])["graphs"] == 12


@gpu
def test_graph_feed_past_capacity_raises():
    graphed = StaticCachedModel(
        cuda_model(), StaticCacheSettings(capacity=16, cuda_graphs=True, min_bucket=8)
    )
    graphed.prepare([(1, 1)])
    with pytest.raises(ValueError, match="capacity"):
        graphed.feed(make_prompt(17).cuda(), None, 1)


# ------------------------------------------------------------------------------- cache write (determinism)
@pytest.fixture
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def test_cache_write_runs_nondeterministic_and_restores_the_flag(deterministic, monkeypatch):
    seen = []
    real = StaticLayer.update

    def spy(self, *args, **kwargs):
        seen.append(torch.are_deterministic_algorithms_enabled())
        return real(self, *args, **kwargs)

    monkeypatch.setattr(StaticLayer, "update", spy)
    static = StaticCachedModel(tiny_qwen3(), SMALL)
    static.prepare([(1, 1)])
    assert torch.are_deterministic_algorithms_enabled()
    static.feed(make_prompt(7), None, 1)
    assert seen and not any(seen)
    assert torch.are_deterministic_algorithms_enabled()


def test_cache_write_restores_the_flag_when_update_raises(deterministic):
    layer = _BucketedStaticLayer(max_cache_len=8)
    layer.view = 8
    layer.update(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
    with pytest.raises(RuntimeError):
        layer.update(torch.zeros(1, 3, 3, 4), torch.zeros(1, 3, 3, 4))
    assert torch.are_deterministic_algorithms_enabled()


def test_cache_write_keeps_a_warn_only_setting():
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        layer = _BucketedStaticLayer(max_cache_len=8)
        layer.view = 8
        layer.update(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


@gpu
def test_cache_write_is_bitwise_equal_to_the_deterministic_write(deterministic):
    generator = torch.Generator().manual_seed(5)
    plain = StaticLayer(max_cache_len=64)
    bucketed = _BucketedStaticLayer(max_cache_len=64)
    bucketed.view = 64
    length = 0
    for _ in range(20):
        count = int(torch.randint(1, 6, (), generator=generator))
        if length + count > 64:
            length = int(torch.randint(0, 20, (), generator=generator))
            plain.cumulative_length.fill_(length)
            bucketed.cumulative_length.fill_(length)
        keys = torch.randn(1, 2, count, 8, generator=generator).cuda().bfloat16()
        values = torch.randn(1, 2, count, 8, generator=generator).cuda().bfloat16()
        plain.update(keys, values)
        out_keys, out_values = bucketed.update(keys, values)
        length += count
        assert torch.equal(plain.keys, bucketed.keys) and torch.equal(plain.values, bucketed.values)
        assert torch.equal(out_keys, plain.keys) and torch.equal(out_values, plain.values)
    assert torch.are_deterministic_algorithms_enabled()


@gpu
def test_graph_with_deterministic_algorithms_matches_dynamic_and_keeps_the_flag(deterministic):
    model = cuda_model()
    prompt = make_prompt(14)
    with torch.inference_mode():
        want = drive(CachedModel(model), prompt)
    graphed = StaticCachedModel(model, GRAPHS)
    graphed.prepare(drive_shapes() | {(1, 1)})
    assert torch.are_deterministic_algorithms_enabled()
    assert_blocks_equal(drive(graphed, prompt), want)
    assert torch.are_deterministic_algorithms_enabled()
