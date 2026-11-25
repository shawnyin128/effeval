import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.methods.quantization import paroquant as pq  # noqa: E402
from effeval.ops import layerwise as layerwise_module  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from test_preprocess_quarot import DOWN_PROJ, apply_to, make_preprocess  # noqa: E402

pytestmark = pytest.mark.compute


def tiny_model():
    return build_tiny_qwen3(head_dim=16, layers=2, tie=False).float()


def small_config() -> pq.ParoQuantConfig:
    return pq.ParoQuantConfig.model_validate(
        {
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
    )


def token_batches():
    gen = torch.Generator().manual_seed(0)
    return torch.randint(0, 97, (4, 12), generator=gen), torch.randint(0, 97, (2, 12), generator=gen)


def initial_val_losses(model, monkeypatch) -> list[float]:
    """Stage-0 validation loss of every layer before training, as seen by `train_layerwise`."""
    losses = []
    real = layerwise_module.train_layerwise

    def spy(*args, **kwargs):
        history = real(*args, **kwargs)
        losses.extend(results[0].original_val_loss for results in history)
        return history

    samples, val_samples = token_batches()
    monkeypatch.setattr(pq, "train_layerwise", spy)
    pq.quantize_model(model, samples, val_samples, small_config(), cache=ArtifactCache())
    return losses


def test_the_replacement_runs_the_online_ops_of_the_linear_it_replaces(monkeypatch):
    model = tiny_model()
    applied = apply_to(model, make_preprocess(DOWN_PROJ))
    online_fn = applied.results[0].online[0].fn
    seen = {}

    def inspect(model, samples, val_samples, *, replace_fn, finalize_fn, **_):
        layer = sites.decoder_layers(model)[0]
        original = layer.mlp.down_proj
        x = torch.randn(3, original.in_features)
        assert original._forward_pre_hooks
        replace_fn(0, layer)
        replacement = layer.mlp.down_proj
        assert isinstance(replacement, pq.PseudoQuantizedLinear)
        w_eff = replacement.pseudo_weight()
        with_ops = replacement(x)
        torch.testing.assert_close(with_ops, online_fn(x) @ w_eff.T, atol=1e-5, rtol=1e-5)
        assert (with_ops - x @ w_eff.T).abs().max() > 1e-2, "the online op must change the output"
        seen["during"] = True
        finalize_fn(0, layer, True)
        assert layer.mlp.down_proj is original
        torch.testing.assert_close(original(x), online_fn(x) @ original.weight.T, atol=1e-5, rtol=1e-5)
        seen["after"] = True
        return []

    monkeypatch.setattr(pq, "train_layerwise", inspect)
    samples, val_samples = token_batches()
    pq.quantize_model(model, samples, val_samples, small_config(), cache=ArtifactCache())
    assert seen == {"during": True, "after": True}
    applied.online_ops.remove()
    assert not model.model.layers[0].mlp.down_proj._forward_pre_hooks


def test_online_preprocess_leaves_the_stage_one_validation_loss_at_the_unpreprocessed_level(monkeypatch):
    plain = initial_val_losses(tiny_model(), monkeypatch)
    model = tiny_model()
    apply_to(model, make_preprocess(DOWN_PROJ))
    online = initial_val_losses(model, monkeypatch)
    print("initial validation loss per layer: plain", plain, "with online down_proj Hadamard", online)
    for with_op, without in zip(online, plain, strict=True):
        assert with_op == pytest.approx(without, rel=0.25)


def test_the_layer_cache_key_changes_with_the_code_version(monkeypatch):
    base_key = {"id": "tiny", "dtype": "torch.float32", "weights": "abc"}
    config = small_config().model_dump(mode="json")
    key = pq._layer_key(base_key, 0, config)
    assert key == pq._layer_key(base_key, 0, config)
    monkeypatch.setattr(pq, "CODE_VERSION", pq.CODE_VERSION + 1)
    assert pq._layer_key(base_key, 0, config) != key


def test_the_layer_artifacts_persist_to_disk_and_restore_the_same_weights(tmp_path):
    samples, val_samples = token_batches()
    first, second = tiny_model(), tiny_model()
    trained = ArtifactCache(tmp_path)
    pq.quantize_model(first, samples, val_samples, small_config(), cache=trained)
    assert len(list(tmp_path.glob("*.safetensors"))) == 2
    restored = ArtifactCache(tmp_path)
    pq.quantize_model(second, samples, val_samples, small_config(), cache=restored)
    assert (restored.computed, restored.disk_hits) == (0, 2)
    for name, value in first.state_dict().items():
        assert torch.equal(second.state_dict()[name], value), name
