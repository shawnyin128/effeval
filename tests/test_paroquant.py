import copy
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

import reference.paroquant_official as ref  # noqa: E402
from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.methods.quantization import paroquant as pq  # noqa: E402
from effeval.ops import calibration as calibration_module  # noqa: E402
from effeval.ops import givens  # noqa: E402
from effeval.ops import layerwise as layerwise_module  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402

pytestmark = pytest.mark.compute


def random_layout(columns: int, group_size: int, rotations: int, seed: int) -> torch.Tensor:
    """Pair layout [R, C]: every rotation pairs up the channels of each group with a random permutation."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(rotations):
        rows.append(
            torch.cat([torch.randperm(group_size, generator=gen) for _ in range(columns // group_size)])
        )
    return torch.stack(rows).to(torch.int16)


def test_rotation_of_one_pair_follows_the_documented_direction():
    x = torch.tensor([[1.0, 2.0]])
    out = givens.pairwise_rotate(
        x, torch.tensor([[0, 1]], dtype=torch.int16), torch.tensor([[torch.pi / 2]]), 2
    )
    assert torch.allclose(out, torch.tensor([[2.0, -1.0]]), atol=1e-6)


@pytest.mark.parametrize(("columns", "group_size", "rotations", "batch"), [(32, 16, 3, 5), (48, 8, 4, 2)])
def test_pairwise_rotate_matches_the_naive_reference_and_its_backward(columns, group_size, rotations, batch):
    pairs = random_layout(columns, group_size, rotations, seed=1)
    gen = torch.Generator().manual_seed(2)
    angles = torch.randn(rotations, columns // 2, generator=gen)
    x = torch.randn(batch, columns, generator=gen)

    x_ours, a_ours = x.clone().requires_grad_(), angles.clone().requires_grad_()
    x_ref, a_ref = x.clone().requires_grad_(), angles.clone().requires_grad_()
    ours = givens.pairwise_rotate(x_ours, pairs, a_ours, group_size)
    expected = ref.scaled_pairwise_rotation(x_ref, pairs, a_ref, None, group_size)
    assert (ours - expected).abs().max() < 1e-5

    upstream = torch.randn(batch, columns, generator=gen)
    ours.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x_ours.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a_ours.grad, a_ref.grad, rtol=1e-5, atol=1e-6)


GOLDEN = Path(__file__).parent / "data" / "paroquant_golden.pt"


def test_pairwise_rotate_equals_the_official_cuda_kernel_outputs():
    golden = torch.load(GOLDEN)
    checked = 0
    for name, entries in golden["rotation_after_stage1"].items():
        state = golden["stages"][0]["after"][name]
        assert state["angles_grouped"].abs().max() > 1e-3
        for entry in entries:
            out = givens.pairwise_rotate(entry["x"], state["pairs_grouped"], state["angles_grouped"], 128)
            assert (entry["y"] - entry["x"]).abs().max() > 0.1
            assert (out - entry["y"]).abs().max() < 1e-5
            checked += 1
    assert checked == 21


def test_inverse_after_forward_is_the_identity_and_leading_dimensions_are_kept():
    pairs = random_layout(64, 16, 4, seed=3)
    angles = torch.randn(4, 32, generator=torch.Generator().manual_seed(4))
    x = torch.randn(2, 3, 64, generator=torch.Generator().manual_seed(5))
    rotated = givens.pairwise_rotate(x, pairs, angles, 16)
    assert rotated.shape == x.shape
    assert (rotated - x).abs().max() > 0.1
    restored = givens.pairwise_rotate_inverse(rotated, pairs, angles, 16)
    assert (restored - x).abs().max() < 1e-6


def test_pairwise_rotate_gradients_pass_gradcheck_in_float64():
    pairs = random_layout(8, 4, 2, seed=6)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(3, 8, dtype=torch.float64, generator=gen, requires_grad=True)
    angles = torch.randn(2, 4, dtype=torch.float64, generator=gen, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, t: givens.pairwise_rotate(a, pairs, t, 4), (x, angles))


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_pair_selection_equals_the_official_for_group_128_and_8_rotations(seed):
    ours = givens.random_rotation_pairs(1, 128, 8, 0.5, seed)
    official = ref.get_random_rotation_pairs(torch.zeros(1, 4, 128), 128, 8, 0.5, seed)
    assert ours == official
    assert all(63 <= len(rotation) <= 64 for rotation in ours)  # the greedy search can fall one pair short
    assert ours != givens.random_rotation_pairs(1, 128, 8, 0.5, seed + 1)


def test_pair_selection_over_several_groups_and_kernel_layout_equal_the_official():
    ours = givens.random_rotation_pairs(3, 16, 4, 0.5, 5)
    official = ref.get_random_rotation_pairs(torch.zeros(3, 4, 16), 16, 4, 0.5, 5)
    assert ours == official
    for rotation, official_rotation in zip(ours, official, strict=True):
        flat, mask = givens.align_pairs(rotation, 16)
        official_flat, _, official_mask = ref._align_shape(
            torch.tensor(official_rotation, dtype=torch.int32), torch.zeros(len(official_rotation)), 16, True
        )
        assert torch.equal(flat.to(torch.int64), official_flat.to(torch.int64))
        assert torch.equal(mask, official_mask.bool())


def test_kernel_layout_completes_short_groups_with_masked_dummy_pairs_like_the_official():
    pairs = givens.random_rotation_pairs(2, 16, 2, 0.25, 0)  # 4 pairs per group: 4 dummy pairs complete each
    for rotation in pairs:
        flat, mask = givens.align_pairs(rotation, 16)
        official_flat, _, official_mask = ref._align_shape(
            torch.tensor(rotation, dtype=torch.int32), torch.zeros(len(rotation)), 16, True
        )
        assert torch.equal(flat.to(torch.int64), official_flat.to(torch.int64))
        assert torch.equal(mask, official_mask.bool())
        assert int(mask.sum()) == 2 * 4
        for group in flat.view(2, 16).tolist():
            assert sorted(group) == list(range(16))


def _paired_modules(seed: int, columns: int = 32, rows: int = 24, group_size: int = 16, rotations: int = 3):
    gen = torch.Generator().manual_seed(seed)
    linear = torch.nn.Linear(columns, rows, bias=False)
    linear.weight.data = torch.randn(rows, columns, generator=gen)
    official_pairs = ref.get_random_rotation_pairs(
        torch.zeros(columns // group_size, 1, group_size), group_size, rotations, 0.5, seed
    )
    tensors = [torch.tensor(p, dtype=torch.int32) for p in official_pairs]
    pairs, _, mask = ref.transform_to_kernel_data(tensors, [torch.zeros(len(p)) for p in tensors], group_size)
    angles = torch.randn(rotations, columns // 2, generator=gen) * 0.3
    scales = torch.rand(1, columns, generator=gen) + 0.5
    official = ref.PseudoQuantizedLinear(
        linear,
        [pairs, angles.clone(), mask.clone()],
        scales.clone(),
        group_size=group_size,
        n_bits=4,
        num_rotations=rotations,
    )
    ours = pq.PseudoQuantizedLinear(
        linear, pairs, angles.clone(), mask, scales.clone(), group_size=group_size, n_bits=4
    )
    return official, ours


@pytest.mark.parametrize("learned_quantizer", [False, True])
def test_pseudo_quantized_weight_and_output_are_bit_identical_to_the_official(learned_quantizer):
    official, ours = _paired_modules(seed=0)
    if learned_quantizer:
        official.set_optim_enabled(quantizer=True)
        ours.set_optim_enabled(quantizer=True)
        shift = torch.randn(official.quantizer.scale.shape, generator=torch.Generator().manual_seed(1)) * 1e-3
        for module in (official, ours):
            module.quantizer.scale.data += shift.abs()
            module.quantizer.zero_point_float.data += shift * 100
    x = torch.randn(4, 32, generator=torch.Generator().manual_seed(2))
    assert torch.equal(official.pseudo_weight(), ours.pseudo_weight())
    assert torch.equal(official(x), ours(x))
    w = ours.pseudo_weight()
    assert (w - ours.weight).abs().max() > 1e-3  # quantization changed the weight


def test_pseudo_quantize_function_equals_the_official_static_method():
    x = torch.randn(8, 32, generator=torch.Generator().manual_seed(3))
    assert torch.equal(
        pq.pseudo_quantize(x, 4, 16),
        ref.UniformAffineQuantizer.pseudo_quantize(x, torch.tensor(4), torch.tensor(16)),
    )


# ---- calibration ----


class CharTokenizer:
    def encode(self, text):
        return [ord(c) % 90 + 1 for c in text]


@pytest.fixture
def corpora(monkeypatch):
    rng = random.Random(0)
    letters = "abcdefghijklmnopqrstuvwxyz "

    def make(count: int, longest: int = 70):
        return datasets.Dataset.from_dict(
            {
                "text": [
                    " " + "".join(rng.choice(letters) for _ in range(rng.randint(0, longest))) + " "
                    for _ in range(count)
                ]
            }
        )

    store = {
        "mit-han-lab/pile-val-backup": make(80),
        ("wikitext", "train"): make(80),
        ("wikitext", "validation"): make(80),
        ("wikitext", "test"): make(80),
        ("allenai/c4", "train"): make(80),
        ("allenai/c4", "validation"): make(80),
        "liang2kl/RedPajama-Data-1T-Sample-Backup": make(600),
    }

    def load_dataset(name, *args, **kwargs):
        if name in ("wikitext", "Salesforce/wikitext"):
            return store[("wikitext", kwargs["split"])]
        if name == "allenai/c4":
            return store[(name, kwargs["split"])]
        return store[name]

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(_id, **_kwargs):
            return CharTokenizer()

    monkeypatch.setattr(calibration_module, "load_dataset", load_dataset)
    monkeypatch.setattr(calibration_module, "AutoTokenizer", FakeAutoTokenizer)
    return load_dataset


@pytest.mark.parametrize(
    ("dataset", "split"),
    [
        ("pileval", "validation"),
        ("wikitext2", "train"),
        ("wikitext2", "validation"),
        ("c4", "train"),
        ("c4", "validation"),
        ("redpajama", "train"),
        ("redpajama", "validation"),
        ("redpajama", "test"),
    ],
)
def test_paroquant_blocks_equal_the_official_get_calib_dataset(corpora, dataset, split):
    ours = calibration_module.paroquant_blocks("any", dataset, 5, 32, 7, split)
    official = ref.get_calib_dataset(
        data=dataset,
        tokenizer=CharTokenizer(),
        n_samples=5,
        block_size=32,
        seed=7,
        split=split,
        load_dataset=corpora,
    )
    assert len(ours) == len(official) == 5
    assert all(block.shape == (32,) for block in ours)
    assert all(torch.equal(a, b) for a, b in zip(ours, official, strict=True))
    other_seed = calibration_module.paroquant_blocks("any", dataset, 5, 32, 8, split)
    assert not all(torch.equal(a, b) for a, b in zip(ours, other_seed, strict=True))


def test_paroquant_mixture_equals_the_official_mixed_dataset(corpora):
    names = ["wikitext2", "c4", "redpajama"]
    ours = calibration_module.paroquant_mixture("any", names, 8, 32, 3, "train")
    official = ref.get_mixed_calib_dataset(
        names,
        tokenizer=CharTokenizer(),
        n_samples=8,
        block_size=32,
        seed=3,
        split="train",
        load_dataset=corpora,
    )
    assert len(ours) == 8
    assert all(torch.equal(a, b) for a, b in zip(ours, official, strict=True))
    unshuffled = [
        b
        for n, count in zip(names, [2, 2, 4], strict=True)
        for b in calibration_module.paroquant_blocks("any", n, count, 32, 3, "train")
    ]
    assert not all(torch.equal(a, b) for a, b in zip(ours, unshuffled, strict=True))


# ---- layer-wise training ----


def tiny_model(layers: int = 2):
    return build_tiny_qwen3(head_dim=16, layers=layers, tie=False).float()


def small_config(**overrides) -> pq.ParoQuantConfig:
    values = {
        "n_bits": 4,
        "group_size": 16,
        "num_rotations": 2,
        "num_pairs_factor": 0.5,
        "stages": [
            {"params": {"channel_scales": 0.05, "angles": 0.05}, "epochs": 2},
            {"params": {"weight": 1e-3, "quantizer": 1e-3}, "epochs": 2},
        ],
        "weight_decay": 0.01,
        "betas": (0.9, 0.95),
        "eps": 1e-10,
        "loss": "smooth_l1",
        "calibration": {
            "datasets": ["wikitext2"],
            "train_size": 4,
            "val_dataset": "pileval",
            "validation_size": 2,
            "seqlen": 12,
            "seed": 0,
        },
        "batch_size": 2,
        "cache_shards": 2,
    }
    values.update(overrides)
    return pq.ParoQuantConfig.model_validate(values)


def token_batches(seed: int = 0, train: int = 4, val: int = 2, seqlen: int = 12):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 97, (train, seqlen), generator=gen), torch.randint(
        0, 97, (val, seqlen), generator=gen
    )


def _pseudo_layer(model, cfg: pq.ParoQuantConfig):
    """Layer 0 with every linear replaced by an effeval PseudoQuantizedLinear."""
    layer = model.model.layers[0]
    for name, linear in pq._linears(layer):
        pairs, angles, mask = pq._initial_rotation(linear.in_features, cfg, 0)
        scales = torch.ones(1, linear.in_features)
        pq._replace(
            layer,
            name,
            pq.PseudoQuantizedLinear(
                linear, pairs, angles, mask, scales, group_size=cfg.group_size, n_bits=cfg.n_bits
            ),
        )
    return layer


def _layer_problem(cfg):
    model = tiny_model()
    samples, val_samples = token_batches()
    train_in, kwargs = layerwise_module._capture_first_input(model, samples, cfg.batch_size)
    val_in, _ = layerwise_module._capture_first_input(model, val_samples, cfg.batch_size)
    with torch.no_grad():
        train_out = [_out(model.model.layers[0], b, kwargs) for b in train_in]
        val_out = [_out(model.model.layers[0], b, kwargs) for b in val_in]
    return model, kwargs, train_in, train_out, val_in, val_out


def _out(layer, hidden, kwargs):
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _groups(layer, cfg, params):
    return pq._stage(cfg, params, 1).param_groups(layer)


def test_optimize_layer_follows_the_official_optimize_module_loss_sequence_and_result():
    cfg = small_config()
    model, kwargs, train_in, train_out, val_in, val_out = _layer_problem(cfg)
    layer = _pseudo_layer(model, cfg)
    official_layer = copy.deepcopy(layer)
    params = {"channel_scales": 0.05, "angles": 0.05}

    result = layerwise_module.optimize_layer(
        layer,
        kwargs,
        (train_in, train_out),
        (val_in, val_out),
        _groups(layer, cfg, params),
        loss="smooth_l1",
        epochs=3,
        post_step=pq._reset_masked_angles,
    )

    recorded: list[torch.Tensor] = []
    handle = official_layer.register_forward_hook(
        lambda _m, _a, out: (
            recorded.append((out[0] if isinstance(out, tuple) else out).detach())
            if torch.is_grad_enabled()
            else None
        )
    )
    ref.optimize_module(
        official_layer,
        (train_in, train_out),
        (val_in, val_out),
        kwargs,
        _groups(official_layer, cfg, params),
        loss_fn="smooth_l1",
        n_iter=3,
        early_stop=None,
        post_optim_callback=pq._reset_masked_angles,
    )
    handle.remove()
    criterion = torch.nn.SmoothL1Loss()
    expected = [criterion(train_out[i % len(train_out)], out).item() for i, out in enumerate(recorded)]
    assert len(expected) == len(result.step_losses) == 3 * len(train_in)
    torch.testing.assert_close(result.step_losses, expected, rtol=1e-6, atol=0)
    assert result.step_losses[-1] < result.step_losses[0]
    for key, value in official_layer.state_dict().items():
        torch.testing.assert_close(layer.state_dict()[key], value, rtol=1e-6, atol=1e-9)


def test_optimize_layer_restores_the_state_with_the_lowest_validation_loss():
    cfg = small_config()
    model, kwargs, train_in, train_out, val_in, val_out = _layer_problem(cfg)
    layer = _pseudo_layer(model, cfg)
    # learning rates large enough to wreck the layer: the last epoch is worse than the best state
    groups = _groups(layer, cfg, {"channel_scales": 5.0, "angles": 5.0})
    result = layerwise_module.optimize_layer(
        layer,
        kwargs,
        (train_in, train_out),
        (val_in, val_out),
        groups,
        loss="smooth_l1",
        epochs=3,
        post_step=pq._reset_masked_angles,
    )
    assert result.val_losses[-1] > result.best_val_loss
    criterion = torch.nn.SmoothL1Loss()
    with torch.no_grad():
        final = sum(criterion(t, _out(layer, i, kwargs)) for i, t in zip(val_in, val_out, strict=True)) / len(
            val_in
        )
    assert final.item() == pytest.approx(result.best_val_loss, rel=1e-6)
    assert final.item() <= min([*result.val_losses, result.original_val_loss]) * (1 + 1e-6)


def test_cached_tensor_shards_serve_the_batches_in_order_and_reject_uneven_shards():
    batches = [torch.full((1,), float(i)) for i in range(6)]
    shards = layerwise_module.CachedTensorShards(batches, 3, target_device=torch.device("cpu"))
    assert [float(b) for b in shards] == [0, 1, 2, 3, 4, 5]
    assert len(shards) == 6
    with pytest.raises(ValueError, match="equal shards"):
        layerwise_module.CachedTensorShards(batches, 4, target_device=torch.device("cpu"))


def test_cosine_schedule_matches_the_official_one():
    ours = layerwise_module.CosineAnnealingParam(0.05, 0.0025, 17)
    official = ref.CosineAnnealingParam(0.05, 0.0025, 17)
    assert [ours.step() for _ in range(20)] == [official.step() for _ in range(20)]


# ---- the method ----


def _official_weights(cfg, model, samples, val_samples):
    return ref.run_layerwise(
        model,
        model.model.layers,
        samples,
        val_samples,
        params_to_optimize=[stage.params for stage in cfg.stages],
        epochs=[stage.epochs for stage in cfg.stages],
        group_size=cfg.group_size,
        n_bit=cfg.n_bits,
        num_rotations=cfg.num_rotations,
        batch_size=cfg.batch_size,
        cache_shards=cfg.cache_shards,
        seed=cfg.calibration.seed,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
        eps=cfg.eps,
        loss=cfg.loss,
        dtype=torch.float32,
    )


def test_quantize_model_writes_the_official_pipelines_effective_weights_bit_for_bit():
    cfg = small_config()
    samples, val_samples = token_batches()
    model = tiny_model()
    original = copy.deepcopy(model.state_dict())

    expected = _official_weights(cfg, copy.deepcopy(model), samples, val_samples)
    pq.quantize_model(model, samples, val_samples, cfg, cache=ArtifactCache())

    assert len(expected) == 2 * 7
    for (layer, name), weight in expected.items():
        got = model.model.layers[layer].get_submodule(name).weight
        assert torch.equal(got, weight), f"layer {layer} {name}"
        assert (got - original[f"model.layers.{layer}.{name}.weight"]).abs().max() > 1e-3
    for key, value in model.state_dict().items():
        if "layers" not in key or "proj" not in key:
            assert torch.equal(value, original[key]), key
    assert not any(isinstance(m, pq.PseudoQuantizedLinear) for m in model.modules())


def test_quantize_model_restores_layers_from_the_cache_and_keys_on_the_config():
    cfg = small_config()
    samples, val_samples = token_batches()
    cache = ArtifactCache()
    first = tiny_model()
    pq.quantize_model(first, samples, val_samples, cfg, cache=cache)
    assert (cache.computed, cache.memory_hits) == (2, 0)

    second = tiny_model()
    pq.quantize_model(second, samples, val_samples, cfg, cache=cache)
    assert (cache.computed, cache.memory_hits) == (2, 2)
    for key, value in first.state_dict().items():
        assert torch.equal(second.state_dict()[key], value), key

    changed = small_config(calibration={**cfg.calibration.model_dump(), "seed": 1})
    pq.quantize_model(tiny_model(), samples, val_samples, changed, cache=cache)
    assert cache.computed == 4


def test_the_paroquant_method_draws_the_official_calibration_and_quantizes(corpora):
    cfg = small_config(
        calibration={
            "datasets": ["wikitext2", "c4", "redpajama"],
            "train_size": 4,
            "val_dataset": "pileval",
            "validation_size": 2,
            "seqlen": 12,
            "seed": 0,
        }
    )
    model = tiny_model()
    original = copy.deepcopy(model.state_dict())
    out = pq.paroquant(model, None, cfg)
    assert out is model
    samples = torch.stack(
        calibration_module.paroquant_mixture("", cfg.calibration.datasets, 4, 12, 0, "train")
    )
    val_samples = torch.stack(calibration_module.paroquant_blocks("", "pileval", 2, 12, 0, "validation"))
    reference = tiny_model()
    pq.quantize_model(reference, samples, val_samples, cfg, cache=ArtifactCache())
    changed = 0
    for key, value in model.state_dict().items():
        assert torch.equal(reference.state_dict()[key], value), key
        changed += not torch.equal(value, original[key])
    assert changed == 2 * 7


def test_the_paper_preset_is_a_valid_config_with_the_official_recipe():
    preset = builtin_catalog().presets["paroquant"]["paper_w4g128"]
    values: dict = {}
    for path, entry in preset.values.items():
        if not path.startswith("method.config."):
            continue
        node = values
        *parents, leaf = path.removeprefix("method.config.").split(".")
        for parent in parents:
            node = node.setdefault(parent, {})
        node[leaf] = entry.value
    cfg = pq.ParoQuantConfig.model_validate(values)
    assert (cfg.n_bits, cfg.group_size, cfg.num_rotations, cfg.num_pairs_factor) == (4, 128, 8, 0.5)
    assert [(s.params, s.epochs) for s in cfg.stages] == [
        ({"channel_scales": 0.05, "angles": 0.05}, 10),
        ({"weight": 1e-5, "quantizer": 1e-6}, 10),
    ]
    assert (cfg.weight_decay, cfg.betas, cfg.eps, cfg.loss) == (0.01, (0.9, 0.95), 1e-10, "smooth_l1")
    assert cfg.calibration.model_dump() == {
        "datasets": ["wikitext2", "c4", "redpajama"],
        "train_size": 2048,
        "val_dataset": "pileval",
        "validation_size": 64,
        "seqlen": 2048,
        "seed": 0,
    }
    assert (cfg.batch_size, cfg.cache_shards) == (16, 4)


def test_config_rejects_unknown_parameter_groups_and_uneven_batches():
    with pytest.raises(ValueError, match="unknown parameter group"):
        small_config(stages=[{"params": {"bias": 0.1}, "epochs": 1}])
    with pytest.raises(ValueError, match="multiple of batch_size"):
        small_config(batch_size=3)


@pytest.mark.gpu
@pytest.mark.network
def test_layer_zero_training_on_qwen3_0_6b_follows_the_official_golden_run(tmp_path):
    """Qwen3-0.6B layer 0, the settings of the golden run (official legacy code, 2 stages of 1 epoch, 64 train
    and 8 validation sequences, batch 8, fp16 model): same rotation pairs and the same loss curve."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from transformers import AutoModelForCausalLM

    golden = torch.load(GOLDEN)
    cfg = pq.ParoQuantConfig.model_validate(
        {
            "n_bits": 4,
            "group_size": 128,
            "num_rotations": 8,
            "num_pairs_factor": 0.5,
            "stages": [
                {"params": {"channel_scales": 0.05, "angles": 0.05}, "epochs": 1},
                {"params": {"weight": 1e-5, "quantizer": 1e-6}, "epochs": 1},
            ],
            "weight_decay": 0.01,
            "betas": (0.9, 0.95),
            "eps": 1e-10,
            "loss": "smooth_l1",
            "calibration": {
                "datasets": ["wikitext2", "c4", "redpajama"],
                "train_size": 64,
                "val_dataset": "pileval",
                "validation_size": 8,
                "seqlen": 2048,
                "seed": 0,
            },
            "batch_size": 8,
            "cache_shards": 1,
        }
    )
    model_id = "Qwen/Qwen3-0.6B"
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    model.model.layers = model.model.layers[:1]
    c = cfg.calibration
    samples = torch.stack(
        calibration_module.paroquant_mixture(model_id, c.datasets, c.train_size, c.seqlen, c.seed, "train")
    )
    val = torch.stack(
        calibration_module.paroquant_blocks(
            model_id, c.val_dataset, c.validation_size, c.seqlen, c.seed, "validation"
        )
    )
    history = []
    original = pq.train_layerwise
    pq.train_layerwise = lambda *a, **k: history.append(original(*a, **k)) or history[-1]
    cache = ArtifactCache(tmp_path)
    try:
        pq.quantize_model(model, samples, val, cfg, cache)
    finally:
        pq.train_layerwise = original

    (state,) = cache._memory.values()
    for name, snap in golden["stages"][0]["after"].items():
        assert torch.equal(state[f"{name}::pairs"].cpu(), snap["pairs_grouped"])
        assert torch.equal(state[f"{name}::mask"].cpu(), snap["mask"])
    for stage, result in enumerate(history[0][0]):
        golden_val = [v for s, v, grad in golden["losses"] if s == stage and not grad]
        golden_step = [v for s, v, grad in golden["losses"] if s == stage and grad]
        ours_val = [result.original_val_loss, *result.val_losses]
        torch.testing.assert_close(torch.tensor(ours_val), torch.tensor(golden_val), rtol=0.03, atol=0)
        torch.testing.assert_close(
            torch.tensor(result.step_losses), torch.tensor(golden_step), rtol=0.03, atol=0
        )
        assert ours_val[-1] < ours_val[0]
