"""ParoQuant layer training on one CUDA device: W_eff from float32 parameters, float16 autocast.

Run with `-m gpu` and CUDA_VISIBLE_DEVICES set to one free GPU index.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")

from effeval.methods.quantization import paroquant as pq  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from test_paroquant_online import small_config, token_batches  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.compute,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def bf16_model():
    from conftest import build_tiny_qwen3

    return build_tiny_qwen3(head_dim=16, layers=2, tie=False).to(torch.bfloat16).cuda()


def test_w_eff_is_computed_in_float32_and_only_the_final_weight_is_cast_to_bf16():
    cfg = small_config()
    samples, val_samples = token_batches()
    model = bf16_model()
    base_key = {
        "id": model.config._name_or_path,
        "dtype": str(torch.bfloat16),
        "weights": pq._weights_digest(model),
    }
    cache = ArtifactCache()
    pq.quantize_model(model, samples, val_samples, cfg, cache=cache)

    checked = 0
    for index, layer in enumerate(model.model.layers):
        stored = cache.get_or_compute(
            pq._layer_key(base_key, index, cfg.model_dump(mode="json")), lambda: pytest.fail("not cached")
        )
        for name, linear in pq._linears(layer):
            prefix = f"{name}::"
            state = {k[len(prefix) :]: v.float() for k, v in stored.items() if k.startswith(prefix)}
            module = pq.PseudoQuantizedLinear.from_state(
                state, group_size=cfg.group_size, n_bits=cfg.n_bits, device=linear.weight.device
            )
            expected = module.pseudo_weight().to(torch.bfloat16)
            assert linear.weight.dtype == torch.bfloat16
            assert torch.equal(linear.weight, expected), f"layer {index} {name}"
            assert stored[f"{prefix}angles"].dtype == torch.float32
            checked += 1
    assert checked == 2 * 7


def test_training_runs_in_float16_autocast_with_an_enabled_grad_scaler(monkeypatch):
    calls = []
    real_forward = pq.PseudoQuantizedLinear.forward

    def record(self, x):
        calls.append(
            (torch.is_grad_enabled(), torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"))
        )
        return real_forward(self, x)

    scalers = []

    class SpyScaler(torch.amp.GradScaler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scaled = 0
            self.stepped = 0
            scalers.append(self)

        def scale(self, outputs):
            self.scaled += 1
            return super().scale(outputs)

        def step(self, optimizer, *args, **kwargs):
            self.stepped += 1
            return super().step(optimizer, *args, **kwargs)

    monkeypatch.setattr(pq.PseudoQuantizedLinear, "forward", record)
    monkeypatch.setattr(torch.amp, "GradScaler", SpyScaler)
    samples, val_samples = token_batches()
    pq.quantize_model(bf16_model(), samples, val_samples, small_config(), cache=ArtifactCache())

    training = {(autocast, dtype) for grad, autocast, dtype in calls if grad}
    evaluation = {autocast for grad, autocast, _ in calls if not grad}
    assert training == {(True, torch.float16)}
    assert evaluation == {False}
    assert scalers and all(s.is_enabled() and s.scaled > 0 and s.stepped == s.scaled for s in scalers)
