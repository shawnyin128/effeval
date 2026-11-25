import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from effeval.methods.quantization import paroquant as pq  # noqa: E402
from effeval.ops import layerwise as layerwise_module  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from test_paroquant import small_config, tiny_model, token_batches  # noqa: E402

pytestmark = pytest.mark.compute

REAL_OPTIMIZE_LAYER = layerwise_module.optimize_layer


def _run_method(monkeypatch, grad_context):
    """Run the ParoQuant method entry (what `hf._apply` calls) on a fresh tiny model; returns both."""
    samples, val_samples = token_batches()
    monkeypatch.setattr(pq, "paroquant_mixture", lambda *args: list(samples))
    monkeypatch.setattr(pq, "paroquant_blocks", lambda *args: list(val_samples))
    monkeypatch.setattr(pq, "default_cache", ArtifactCache)
    results = []

    def spy(*args, **kwargs):
        results.append(REAL_OPTIMIZE_LAYER(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(layerwise_module, "optimize_layer", spy)
    model = tiny_model()
    with grad_context():
        pq.paroquant(model, None, small_config())
    return model, results


def test_paroquant_trains_under_no_grad_and_matches_the_grad_enabled_run(monkeypatch):
    original = copy.deepcopy(tiny_model().state_dict())
    reference, reference_results = _run_method(monkeypatch, torch.enable_grad)
    model, results = _run_method(monkeypatch, torch.no_grad)

    assert len(results) == len(reference_results) == 4
    for result in results:
        assert len(result.step_losses) > 0
        assert result.step_losses[-1] != result.step_losses[0]
    assert [r.step_losses for r in results] == [r.step_losses for r in reference_results]
    changed = 0
    for key, value in model.state_dict().items():
        assert torch.equal(reference.state_dict()[key], value), key
        changed += not torch.equal(value, original[key])
    assert changed == 2 * 7
