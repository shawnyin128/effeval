import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from conftest import build_tiny_llama, build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import Calibration  # noqa: E402
from effeval.ops import calibration as calibration_module  # noqa: E402
from effeval.ops import (  # noqa: E402
    online,
    sites,
)
from effeval.ops.artifacts import ArtifactCache, artifact_key  # noqa: E402
from effeval.ops.calibration import (  # noqa: E402
    LayerRunner,
    act_stats,
    attention_operand_absmax,
    calibration_windows,
)
from test_sites import tiny_qwen3  # noqa: E402

pytestmark = pytest.mark.compute


class CharTokenizer:
    """Deterministic stand-in for a HF tokenizer: one id per character."""

    def __call__(self, text, return_tensors=None, max_length=None, truncation=False):
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        return type("Encoding", (), {"input_ids": torch.tensor([ids])})()

    def encode(self, text):
        return [ord(c) % 90 + 1 for c in text]


@pytest.fixture
def corpora(monkeypatch):
    rng = random.Random(0)
    letters = "abcdefghijklmnopqrstuvwxyz "
    pile = datasets.Dataset.from_dict(
        {"text": ["".join(rng.choice(letters) for _ in range(rng.randint(5, 90))) + "  " for _ in range(60)]}
    )
    wikitext = {"text": ["".join(rng.choice(letters) for _ in range(rng.randint(20, 60))) for _ in range(30)]}

    def fake_load_dataset(name, *args, **kwargs):
        return pile if name == calibration_module.PILE_VAL else wikitext

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, use_fast):
            return CharTokenizer()

    monkeypatch.setattr(calibration_module, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(calibration_module, "AutoTokenizer", FakeAutoTokenizer)
    return pile, wikitext


def test_wikitext2_windows_equal_the_official_gptq_loader(corpora):
    _, wikitext = corpora
    cal = Calibration(dataset="wikitext2", n_samples=6, seqlen=32, seed=5)
    got = calibration_windows(cal, "any")
    # official datautils.get_wikitext2: global random.seed(seed), then randint over the token stream
    trainenc = CharTokenizer()("\n\n".join(wikitext["text"]), return_tensors="pt")
    random.seed(5)
    expected = []
    for _ in range(6):
        i = random.randint(0, trainenc.input_ids.shape[1] - 32 - 1)
        expected.append(trainenc.input_ids[:, i : i + 32])
    assert len(got) == 6
    assert all(torch.equal(a, b) for a, b in zip(got, expected, strict=True))
    assert not all(
        torch.equal(a, b)
        for a, b in zip(got, calibration_windows(cal.model_copy(update={"seed": 6}), "a"), strict=True)
    )


def test_pile_val_documents_are_shuffled_with_the_seed_and_truncated_one_per_window(corpora):
    pile, _ = corpora
    cal = Calibration(dataset="pile_val", sampling="documents", n_samples=10, seqlen=40, seed=42)
    got = calibration_windows(cal, "any")
    shuffled = pile.shuffle(seed=42)
    tokenizer = CharTokenizer()
    expected = [
        tokenizer(shuffled[i]["text"], return_tensors="pt", max_length=40, truncation=True).input_ids
        for i in range(10)
    ]
    assert len(got) == 10
    assert all(torch.equal(a, b) for a, b in zip(got, expected, strict=True))
    assert (
        max(w.shape[1] for w in got) == 40 and min(w.shape[1] for w in got) < 40
    )  # truncation and short docs
    unshuffled = [
        tokenizer(pile[i]["text"], return_tensors="pt", max_length=40, truncation=True).input_ids
        for i in range(10)
    ]
    assert not all(torch.equal(a, b) for a, b in zip(got, unshuffled, strict=True))


def test_pile_val_concat_blocks_skip_long_documents_and_cut_equal_blocks(corpora):
    pile, _ = corpora
    cal = Calibration(dataset="pile_val", sampling="concat_blocks", n_samples=12, seqlen=64, seed=42)
    got = calibration_windows(cal, "any")
    tokenizer = CharTokenizer()
    kept = []
    for row in pile.shuffle(seed=42):
        ids = tokenizer.encode(row["text"].strip())
        if len(ids) > 64:
            continue
        kept.append(ids)
        if len(kept) == 12:
            break
    joined = torch.tensor([[t for ids in kept for t in ids]])
    expected = [joined[:, i * 64 : (i + 1) * 64] for i in range(joined.shape[1] // 64)]
    assert len(expected) >= 2
    assert len(got) == len(expected)
    assert all(w.shape == (1, 64) for w in got)
    assert all(torch.equal(a, b) for a, b in zip(got, expected, strict=True))
    assert any(len(tokenizer.encode(r["text"].strip())) > 64 for r in pile.shuffle(seed=42).select(range(12)))


def test_calibration_fragment_sampling_rules():
    assert Calibration(dataset="c4", n_samples=1, seqlen=8, seed=0).sampling == "windows"
    with pytest.raises(ValueError, match="pile_val uses sampling"):
        Calibration(dataset="pile_val", n_samples=1, seqlen=8, seed=0)
    with pytest.raises(ValueError, match="pile_val uses sampling"):
        Calibration(dataset="wikitext2", sampling="documents", n_samples=1, seqlen=8, seed=0)


def _windows(count: int = 3, length: int = 12, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


@torch.no_grad()
def test_layer_runner_yields_the_hidden_states_entering_each_layer():
    model = tiny_qwen3()
    windows = _windows()
    reference = [model(w, output_hidden_states=True).hidden_states for w in windows]
    seen = 0
    with LayerRunner(model, windows) as runner:
        assert model.config.use_cache is False
        for index, _layer in runner:
            seen += 1
            for hidden, states in zip(runner.inputs, reference, strict=True):
                assert torch.allclose(hidden, states[index], atol=1e-6), index
    assert seen == len(model.model.layers)
    assert model.config.use_cache is True


@torch.no_grad()
@pytest.mark.parametrize(("propagate", "edits_visible"), [("current", True), ("original", False)])
def test_layer_runner_propagate_chooses_between_edited_and_original_layer_outputs(propagate, edits_visible):
    model = tiny_qwen3()
    windows = _windows()
    reference = [model(w, output_hidden_states=True).hidden_states for w in windows]
    with LayerRunner(model, windows, propagate=propagate) as runner:
        for index, layer in runner:
            if index == 0:
                layer.mlp.down_proj.weight.zero_()
            if index == 1:
                for hidden, states in zip(runner.inputs, reference, strict=True):
                    assert torch.allclose(hidden, states[1], atol=1e-6) is not edits_visible


def test_layer_runner_rejects_unknown_propagate_mode():
    with pytest.raises(ValueError, match="propagate"):
        LayerRunner(tiny_qwen3(), _windows(), propagate="later")


def _recorded_inputs(model, windows, module):
    seen: list[torch.Tensor] = []
    handle = module.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
    with torch.no_grad():
        for w in windows:
            model(w)
    handle.remove()
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in seen]).abs()


def test_act_stats_absmax_and_absmean_equal_a_hand_loop_over_the_linear_inputs():
    model = tiny_qwen3()
    windows = _windows(count=4, length=10)
    patterns = ["linear.mlp.down_proj.input", "linear.attn.q_proj.input"]
    for stat, reduce in (("absmax", lambda t: t.amax(0)), ("absmean", lambda t: t.mean(0))):
        stats = act_stats(model, windows, patterns, stat)
        assert set(stats) == {(site, layer) for site in patterns for layer in range(3)}
        for (site, layer), value in stats.items():
            name = "mlp.down_proj" if "down_proj" in site else "self_attn.q_proj"
            expected = reduce(_recorded_inputs(model, windows, model.model.layers[layer].get_submodule(name)))
            assert value.dtype == torch.float32 and value.shape == expected.shape
            assert torch.allclose(value, expected, rtol=1e-6, atol=1e-7), (stat, site, layer)
    absmax = act_stats(model, windows, patterns, "absmax")
    absmean = act_stats(model, windows, patterns, "absmean")
    assert not torch.equal(absmax[patterns[0], 0], absmean[patterns[0], 0])


def test_act_stats_sees_the_input_after_earlier_forward_pre_hooks_and_removes_its_own():
    model = tiny_qwen3()
    windows = _windows()
    q_proj = model.model.layers[0].self_attn.q_proj
    plain = act_stats(model, windows, ["linear.attn.q_proj.input"], "absmax")[("linear.attn.q_proj.input", 0)]
    handle = q_proj.register_forward_pre_hook(lambda _m, args: (args[0] * 3.0,))
    scaled = act_stats(model, windows, ["linear.attn.q_proj.input"], "absmax")[
        ("linear.attn.q_proj.input", 0)
    ]
    handle.remove()
    assert torch.allclose(scaled, plain * 3.0, rtol=1e-6)
    assert not model.model.layers[0].self_attn.q_proj._forward_pre_hooks
    assert model.config.use_cache is True


def test_act_stats_output_sites_and_errors():
    model = tiny_qwen3()
    windows = _windows()
    out = act_stats(model, windows, ["linear.attn.q_proj.output"], "absmax")
    assert out[("linear.attn.q_proj.output", 0)].shape == (64,)  # 4 heads x head_dim 16
    with pytest.raises(ValueError, match="input or output"):
        act_stats(model, windows, ["linear.*.weight"], "absmax")
    with pytest.raises(sites.SiteError):
        act_stats(model, windows, ["linear.nothing.input"], "absmax")


def _key(**overrides):
    parts = {
        "model": {"name": "m", "revision": "r1"},
        "calibration": Calibration(dataset="wikitext2", n_samples=128, seqlen=2048, seed=0),
        "statistic": "absmax",
        "upstream": [{"kind": "scaling"}],
        "downstream": None,
    }
    parts.update(overrides)
    return artifact_key(**parts)


def test_artifact_key_changes_with_every_input():
    base = _key()
    assert _key() == base
    cal = Calibration(dataset="wikitext2", n_samples=128, seqlen=2048, seed=0)
    variants = [
        _key(calibration=cal.model_copy(update={"seed": 1})),
        _key(calibration=cal.model_copy(update={"n_samples": 64})),
        _key(calibration=cal.model_copy(update={"seqlen": 1024})),
        _key(calibration=None),
        _key(statistic="absmean"),
        _key(upstream=[{"kind": "scaling"}, {"kind": "rotation"}]),
        _key(upstream=[]),
        _key(downstream=[{"bits": 4}]),
        _key(model={"name": "m", "revision": "r2"}),
    ]
    assert len({base, *variants}) == len(variants) + 1


def test_artifact_cache_computes_once_per_key_in_memory():
    cache = ArtifactCache()
    calls = []

    def compute():
        calls.append(1)
        return {"scale": torch.arange(3.0)}

    first = cache.get_or_compute("k", compute)
    second = cache.get_or_compute("k", compute)
    assert len(calls) == 1 and cache.computed == 1 and cache.memory_hits == 1
    assert torch.equal(first["scale"], second["scale"])
    cache.get_or_compute("other", compute)
    assert len(calls) == 2


def test_artifact_cache_disk_roundtrip_across_instances_and_env_switch(tmp_path, monkeypatch):
    key = _key()
    tensors = {"R": torch.randn(4, 4), "perm": torch.tensor([2, 0, 1])}
    ArtifactCache(tmp_path / "c").get_or_compute(key, lambda: tensors)
    assert (tmp_path / "c" / f"{key}.safetensors").exists()

    fresh = ArtifactCache(tmp_path / "c")
    loaded = fresh.get_or_compute(key, lambda: pytest.fail("must come from disk"))
    assert fresh.disk_hits == 1 and fresh.computed == 0
    assert torch.equal(loaded["R"], tensors["R"]) and torch.equal(loaded["perm"], tensors["perm"])

    monkeypatch.delenv("EFFEVAL_CACHE", raising=False)
    assert ArtifactCache.from_env().directory is None
    monkeypatch.setenv("EFFEVAL_CACHE", str(tmp_path / "env"))
    cache = ArtifactCache.from_env()
    assert cache.directory == tmp_path / "env"
    cache.get_or_compute("x", lambda: {"a": torch.ones(2)})
    assert (tmp_path / "env" / "x.safetensors").exists()
    memory_only = ArtifactCache()
    memory_only.get_or_compute("y", lambda: {"a": torch.ones(2)})
    assert not (tmp_path / "y.safetensors").exists()


def _random_crop_calibration(n_samples, seqlen, seed=42):
    # calibration_windows reads only these fields; `Calibration` takes this sampling value once WP1's schema
    # lands.
    return SimpleNamespace(
        dataset="pile_val", sampling="random_crop", n_samples=n_samples, seqlen=seqlen, seed=seed
    )


def test_random_crop_follows_deepcompressor_document_cropping_and_block_cutting(corpora):
    pile, _ = corpora
    got = calibration_windows(_random_crop_calibration(6, 40), "any")
    # deepcompressor LlmCalibDataset: shuffle(seed), random.Random(seed) crops, concatenate, split, drop last
    tokenizer = CharTokenizer()
    rng = random.Random(42)
    pieces, total = [], 0
    for row in pile.shuffle(seed=42):
        line = tokenizer.encode(row["text"].strip())
        if not line:
            continue
        if len(line) > 40:
            start = rng.randint(0, len(line) - 40)
            line = line[start : start + 40]
        pieces.append(line)
        total += len(line)
        if len(pieces) >= 6 and total >= 240:
            break
    blocks = list(torch.tensor([t for p in pieces for t in p]).split(40))
    if total > 240:
        blocks = blocks[:-1]
    expected = blocks[:6]
    assert len(got) == 6
    assert all(w.shape == (1, 40) for w in got)
    assert all(torch.equal(a, b[None]) for a, b in zip(got, expected, strict=True))
    assert any(len(tokenizer.encode(r["text"].strip())) > 40 for r in pile.shuffle(seed=42).select(range(8)))


def test_random_crop_is_deterministic_for_a_seed_and_follows_the_seed(corpora):
    first = calibration_windows(_random_crop_calibration(6, 40, seed=7), "any")
    again = calibration_windows(_random_crop_calibration(6, 40, seed=7), "any")
    other = calibration_windows(_random_crop_calibration(6, 40, seed=8), "any")
    assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_random_crop_keeps_documents_whole_when_seqlen_exceeds_every_document(corpora):
    pile, _ = corpora
    got = calibration_windows(_random_crop_calibration(3, 200), "any")
    tokenizer = CharTokenizer()
    lines, total = [], 0
    for row in pile.shuffle(seed=42):
        line = tokenizer.encode(row["text"].strip())
        if not line:
            continue
        lines.append(line)
        total += len(line)
        if len(lines) >= 3 and total >= 600:
            break
    assert max(len(line) for line in lines) < 200
    joined = torch.tensor([t for line in lines for t in line])
    assert total > 600 and len(got) == 3
    assert torch.equal(torch.cat(got, dim=1)[0], joined[:600])


@torch.no_grad()
def _reference_key_operand(model, windows, layer):
    """Max |K| per KV head and channel of `layer`, recomputed from the attention module's own inputs."""
    attention = model.model.layers[layer].self_attn
    attention_module = __import__(type(attention).__module__, fromlist=["apply_rotary_pos_emb"])
    apply_rope = attention_module.apply_rotary_pos_emb
    captured = []
    handle = attention.register_forward_pre_hook(
        lambda _m, args, kwargs: captured.append((kwargs["hidden_states"], kwargs["position_embeddings"])),
        with_kwargs=True,
    )
    for window in windows:
        model(window, use_cache=False)
    handle.remove()
    best = None
    for hidden, (cos, sin) in captured:
        key = attention.k_proj(hidden).view(*hidden.shape[:-1], -1, attention.head_dim)
        if hasattr(attention, "k_norm"):
            key = attention.k_norm(key)
        key = key.transpose(1, 2)
        _, key = apply_rope(key, key, cos, sin)
        current = key.abs().amax(dim=(0, 2))
        best = current if best is None else torch.maximum(best, current)
    return best


@pytest.mark.parametrize("build", [build_tiny_qwen3, build_tiny_llama])
def test_attention_operand_absmax_equals_a_reference_from_the_attention_inputs(build, register_architecture):
    model = build()
    register_architecture(model)
    windows = _windows()
    implementation = model.config._attn_implementation
    got = attention_operand_absmax(model, windows, "attn.k", "all", [])
    assert sorted(got) == [0, 1]
    for layer in (0, 1):
        reference = _reference_key_operand(model, windows, layer)
        assert got[layer].shape == reference.shape == (2, model.model.layers[layer].self_attn.head_dim)
        assert torch.allclose(got[layer], reference, atol=1e-6)
    assert model.config._attn_implementation == implementation
    assert model.config.use_cache is True
    assert not hasattr(model.model.layers[0].self_attn, sites.ATTN_OPS_ATTR)


def test_attention_operand_absmax_sees_extra_ops_and_selected_layers_only(register_architecture):
    model = build_tiny_qwen3()
    register_architecture(model)
    windows = _windows()
    plain = attention_operand_absmax(model, windows, "attn.k", [1], [])
    scaled = attention_operand_absmax(
        model, windows, "attn.k", [1], [online.OnlineOp("attn.k", lambda x: x * 2.0, [1])]
    )
    assert list(plain) == [1]
    assert torch.allclose(scaled[1], plain[1] * 2.0, rtol=1e-6)
    queries = attention_operand_absmax(model, windows, "attn.q", [0], [])
    assert queries[0].shape == (4, 32)
    with pytest.raises(ValueError, match="attention operand"):
        attention_operand_absmax(model, windows, "linear.attn.q_proj.input", "all", [])
