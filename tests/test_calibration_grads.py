import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.ops.calibration import act_stats, causal_lm_grads  # noqa: E402


def _windows(count: int = 3, length: int = 12, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 97, (1, length), generator=gen) for _ in range(count)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_window_absmean_sum_is_the_sum_of_per_window_token_means(dtype):
    model = build_tiny_qwen3().to(dtype)
    windows = _windows()
    site = "linear.attn.k_proj.input"
    stats = act_stats(model, windows, [site], "window_absmean_sum")
    for layer, block in enumerate(model.model.layers):
        seen: list[torch.Tensor] = []
        handle = block.self_attn.k_proj.register_forward_pre_hook(
            lambda _m, args, sink=seen: sink.append(args[0].detach())
        )
        with torch.no_grad():
            for w in windows:
                model(w)
        handle.remove()
        expected = sum(x.abs().mean(dim=-2).view(-1).float() for x in seen)
        value = stats[site, layer]
        assert value.dtype == torch.float32
        assert torch.equal(value, expected)


def test_window_absmean_sum_rejects_batched_windows():
    model = build_tiny_qwen3()
    with pytest.raises(ValueError, match="one window"):
        act_stats(
            model, [torch.zeros(2, 8, dtype=torch.long)], ["linear.attn.k_proj.input"], "window_absmean_sum"
        )


def test_causal_lm_grads_equal_autograd_of_shifted_cross_entropy_and_do_not_accumulate():
    model = build_tiny_qwen3()
    windows = _windows(count=3, length=10)
    q = model.model.layers[0].self_attn.q_proj.weight
    seen: list[torch.Tensor] = []
    assert all(p.grad is None for p in model.parameters())

    count = causal_lm_grads(model, windows, [q], lambda: seen.append(q.grad.clone()))

    assert count == 3 and len(seen) == 3
    for w, got in zip(windows, seen, strict=True):
        logits = model(w).logits
        loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), w[:, 1:].flatten())
        (expected,) = torch.autograd.grad(loss, q)
        assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)
    assert not torch.equal(seen[0], seen[1])
    assert all(p.grad is None for p in model.parameters())


def test_causal_lm_grads_freezes_other_parameters_and_restores_flags():
    model = build_tiny_qwen3()
    model.train()
    model.config.use_cache = True
    q = model.model.layers[0].self_attn.q_proj.weight
    k = model.model.layers[0].self_attn.k_proj.weight
    k.requires_grad_(False)
    before = {name: p.requires_grad for name, p in model.named_parameters()}
    during: dict = {}

    def on_batch():
        during["training"] = model.training
        during["use_cache"] = model.config.use_cache
        during["grads"] = {n: p.grad is not None for n, p in model.named_parameters()}

    causal_lm_grads(model, _windows(count=1), [q], on_batch)

    assert during["training"] is False and during["use_cache"] is False
    assert [n for n, has in during["grads"].items() if has] == ["model.layers.0.self_attn.q_proj.weight"]
    assert {name: p.requires_grad for name, p in model.named_parameters()} == before
    assert model.training is True and model.config.use_cache is True


def test_causal_lm_grads_uses_labels_from_a_mapping_batch():
    model = build_tiny_qwen3()
    w = _windows(count=1, length=10)[0]
    labels = w.clone()
    labels[:, :5] = -100
    q = model.model.layers[0].self_attn.q_proj.weight
    seen: list[torch.Tensor] = []
    causal_lm_grads(model, [{"input_ids": w, "labels": labels}], [q], lambda: seen.append(q.grad.clone()))
    logits = model(w).logits
    loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
    (expected,) = torch.autograd.grad(loss, q)
    assert torch.allclose(seen[0], expected, rtol=1e-6, atol=1e-6)


def test_causal_lm_grads_rejects_dropout():
    model = build_tiny_qwen3()
    model.config.attention_dropout = 0.1
    with pytest.raises(ValueError, match="without dropout"):
        causal_lm_grads(
            model, _windows(count=1), [model.model.layers[0].self_attn.q_proj.weight], lambda: None
        )
    model = build_tiny_qwen3()
    model.model.layers[0].mlp.add_module("drop", torch.nn.Dropout(0.2))
    with pytest.raises(ValueError, match="without dropout"):
        causal_lm_grads(
            model, _windows(count=1), [model.model.layers[0].self_attn.q_proj.weight], lambda: None
        )
