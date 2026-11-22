from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.preprocess import Preprocess, resolve_preprocess  # noqa: E402
from effeval.methods.preprocess import engine  # noqa: E402
from effeval.ops import online  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from effeval.ops.scaling import ChannelDivide  # noqa: E402
from test_sites import padded_batch  # noqa: E402

pytestmark = pytest.mark.compute


@torch.no_grad()
def k_operand(model, attachment_ops):
    seen = []

    def observe(x):
        seen.append(x.detach().clone())
        return x

    attachment = online.register(model, [*attachment_ops, online.OnlineOp("attn.k", observe, [0])])
    try:
        input_ids, attention_mask = padded_batch()
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        attachment.remove()
    return seen[0]


def test_online_ops_of_separate_steps_on_one_operand_are_registered_together_in_step_order(monkeypatch):
    engine._discover()
    model = build_tiny_qwen3(head_dim=32)
    generator = torch.Generator().manual_seed(0)
    matrix = torch.linalg.qr(torch.randn(32, 32, generator=generator))[0]
    scale = torch.rand(32, generator=generator) + 0.5
    rotate = online.OnlineOp("attn.k", online.RightMultiply(matrix))
    divide = online.OnlineOp("attn.k", ChannelDivide(scale))
    monkeypatch.setitem(
        engine._SOURCES, ("rotation", "hadamard"), lambda _m, _g, _e, _c: engine.StageResult(online=[rotate])
    )
    monkeypatch.setitem(
        engine._SOURCES,
        ("scaling", "smoothquant"),
        lambda _m, _g, _e, _c: engine.StageResult(online=[divide]),
    )
    calls = []
    real_register = online.register
    monkeypatch.setattr(
        engine.online, "register", lambda m, ops: calls.append(list(ops)) or real_register(m, ops)
    )
    entries = {"position": "attn.qk_rope", "mode": "online"}
    preprocess = resolve_preprocess(
        Preprocess.model_validate(
            {
                "steps": [
                    {"kind": "rotation", "entries": [{**entries, "source": {"type": "hadamard"}}]},
                    {"kind": "scaling", "entries": [{**entries, "source": {"type": "smoothquant"}}]},
                ]
            }
        )
    )
    raw = k_operand(model, [])
    ctx = engine.make_context(
        model, SimpleNamespace(name_or_path="tiny"), {"checkpoint": "tiny"}, artifacts=ArtifactCache()
    )
    calls.clear()
    applied = engine.apply(model, preprocess, ctx)
    applied.online_ops.remove()
    assert calls == [[rotate, divide]]
    assert torch.allclose(
        k_operand(model, [rotate, divide]), (raw @ matrix) / scale[None, None, None, :], atol=1e-5
    )
    assert not torch.allclose(
        k_operand(model, [rotate, divide]), (raw / scale[None, None, None, :]) @ matrix, atol=1e-3
    )
