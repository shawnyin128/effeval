"""Micro-batched validation of `optimize_layer` / `train_layerwise` equals the whole-batch validation."""

import functools

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from torch import nn  # noqa: E402

from effeval.methods.quantization import paroquant as pq  # noqa: E402
from effeval.ops import layerwise as layerwise_module  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from test_paroquant import small_config, tiny_model, token_batches  # noqa: E402

pytestmark = pytest.mark.compute


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5, dtype=torch.float64)
        self.batch_sizes: list[int] = []

    def forward(self, hidden, **kwargs):
        self.batch_sizes.append(hidden.shape[0])
        return self.linear(hidden)


def _val_batches(sizes):
    gen = torch.Generator().manual_seed(1)
    inputs = [torch.randn(n, 3, 5, generator=gen, dtype=torch.float64) for n in sizes]
    targets = [torch.randn(n, 3, 5, generator=gen, dtype=torch.float64) for n in sizes]
    return inputs, targets


def _original_val_loss(layer, val, loss, micro):
    inputs, targets = val
    train = ([inputs[0]], [targets[0]])
    return layerwise_module.optimize_layer(
        layer, {}, train, val, [{"params": [], "lr": 1.0}], loss=loss, epochs=0, val_micro_batch=micro
    ).original_val_loss


@pytest.mark.parametrize("loss", ["mse", "smooth_l1"])
@pytest.mark.parametrize("micro", [1, 2, 4, 6, 100])
def test_micro_batched_validation_is_the_mean_over_batches_of_batch_means(loss, micro):
    val = _val_batches([6, 3, 5])
    layer = _Layer()
    criterion = layerwise_module._loss_fn(loss)
    with torch.no_grad():
        expected = sum(criterion(t, layer(i)) for i, t in zip(*val, strict=True)).item() / 3
    layer.batch_sizes.clear()

    value = _original_val_loss(layer, val, loss, micro)

    assert value == pytest.approx(expected, rel=1e-12)
    assert max(layer.batch_sizes) <= micro


def test_unweighted_mean_of_slices_would_differ_on_uneven_slices():
    val = _val_batches([5])
    layer = _Layer()
    criterion = layerwise_module._loss_fn("mse")
    with torch.no_grad():
        inputs, targets = val[0][0], val[1][0]
        head = criterion(targets[:4], layer(inputs[:4]))
        tail = criterion(targets[4:], layer(inputs[4:]))
        unweighted = (head + tail) / 2
    assert _original_val_loss(layer, val, "mse", 4) != pytest.approx(unweighted.item(), rel=1e-6)


def test_train_layerwise_gives_the_same_losses_and_weights_with_micro_batched_validation(monkeypatch):
    samples, val_samples = token_batches(val=4)
    cfg = small_config(
        calibration={
            "datasets": ["wikitext2"],
            "train_size": 4,
            "val_dataset": "pileval",
            "validation_size": 4,
            "seqlen": 12,
            "seed": 0,
        }
    )

    def run(micro):
        histories = []

        def spy(*args, **kwargs):
            histories.append(layerwise_module.train_layerwise(*args, **{**kwargs, "val_micro_batch": micro}))
            return histories[-1]

        monkeypatch.setattr(pq, "train_layerwise", spy)
        model = tiny_model()
        pq.quantize_model(model, samples, val_samples, cfg, cache=ArtifactCache())
        return model, histories[0]

    reference, reference_history = run(None)
    model, history = run(1)

    for layer_results, reference_results in zip(history, reference_history, strict=True):
        for result, expected in zip(layer_results, reference_results, strict=True):
            assert result.step_losses == expected.step_losses
            assert result.original_val_loss == pytest.approx(expected.original_val_loss, rel=1e-5)
            assert result.val_losses == pytest.approx(expected.val_losses, rel=1e-5)
    for key, value in model.state_dict().items():
        assert torch.equal(reference.state_dict()[key], value), key


def test_default_micro_batch_of_train_layerwise_is_small(monkeypatch):
    seen = []
    real = layerwise_module.optimize_layer
    monkeypatch.setattr(
        layerwise_module,
        "optimize_layer",
        functools.update_wrapper(lambda *a, **k: (seen.append(k["val_micro_batch"]), real(*a, **k))[1], real),
    )
    samples, val_samples = token_batches()
    pq.quantize_model(tiny_model(), samples, val_samples, small_config(), cache=ArtifactCache())
    assert seen and all(m == 2 for m in seen)
