import copy
import os
import re
import sys
from functools import partial
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import transformers  # noqa: E402

from conftest import build_tiny_qwen3  # noqa: E402
from effeval import validate as validate_module  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.preprocess import (  # noqa: E402
    Preprocess,
    RotationEntry,
    ScalingEntry,
    builtin_preprocess_presets,
    resolve_preprocess,
)
from effeval.methods.preprocess import engine, trained  # noqa: E402
from effeval.methods.preprocess.rotation import prepare_residual  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from effeval.ops.orthogonal import act_quant_mse, cayley_rotation  # noqa: E402
from test_codebook import _official  # noqa: E402
from test_sites import padded_batch  # noqa: E402

pytestmark = pytest.mark.compute

LR = 0.01
BITS = 4
HIDDEN = 64
NORM_PATTERN = re.compile(r".*(input_layernorm|post_attention_layernorm).*")
POSITIONS = ("norm.input_layernorm.out", "norm.post_attention_layernorm.out")
ACT = {"type": "int", "bits": BITS, "symmetric": False, "granularity": "per_token", "zero_point": "float"}
CALIBRATION = {
    "dataset": "tiny",
    "n_samples": 8,
    "batch_size": 2,
    "max_length": 8,
    "epochs": 2,
    "chat_template": False,
}
ROTATION_SOURCE = {"type": "trained", "act": ACT, "lr": LR, "init_seed": 7, "calibration": CALIBRATION}
SCALING_SOURCE = {**ROTATION_SOURCE, "init_seed": 3}


def fake_batches(n: int = 4, seed: int = 0) -> list[transformers.BatchEncoding]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for index in range(n):
        ids = torch.randint(1, 97, (2, 8), generator=generator)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[1, : index % 3] = 0
        batches.append(transformers.BatchEncoding({"input_ids": ids, "attention_mask": mask}))
    return batches


@pytest.fixture
def batches(monkeypatch):
    first = fake_batches(4, 0)
    epochs = [first, [first[i] for i in (2, 0, 3, 1)]]
    monkeypatch.setattr(trained, "text_batches", lambda _cfg, _tokenizer: epochs)
    return [batch for epoch in epochs for batch in epoch]


@pytest.fixture
def model():
    return build_tiny_qwen3(head_dim=16, tie=False)


def make_context(model, cache=None) -> engine.Context:
    return engine.make_context(
        model,
        SimpleNamespace(name_or_path="tiny"),
        {"checkpoint": "tiny"},
        artifacts=cache if cache is not None else ArtifactCache(),
    )


def preprocess_of(*steps: dict):
    return resolve_preprocess(Preprocess.model_validate({"steps": list(steps)}))


def rotation_step(**source) -> dict:
    entry = {
        "position": "residual",
        "mode": "offline",
        "source": {**ROTATION_SOURCE, **source},
        "options": {"untie_embeddings": True},
    }
    return {"kind": "rotation", "entries": [entry]}


def scaling_step(**source) -> dict:
    entries = [
        {"position": position, "mode": "offline", "source": {**SCALING_SOURCE, **source}}
        for position in POSITIONS
    ]
    return {"kind": "scaling", "entries": entries}


@torch.no_grad()
def logits_of(model) -> torch.Tensor:
    input_ids, attention_mask = padded_batch()
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[attention_mask.bool()]


def norm_modules(model) -> dict[str, torch.nn.Module]:
    """Hidden-axis norms of the layers by the official name regex (layer order, input norm first)."""
    return {name: module for name, module in model.named_modules() if NORM_PATTERN.search(name)}


def official(project: str, relative: str):
    """An official CodeQuant module, with the modules it imported removed from sys.modules again."""
    try:
        return _official(project, relative)
    finally:
        root = os.environ.get("EFFEVAL_CODEQUANT_REF", "")
        for name in [
            n
            for n, m in sys.modules.items()
            if root and str(getattr(m, "__file__", None) or "").startswith(root)
        ]:
            del sys.modules[name]


def official_fold_gains(model, rotation_utils) -> None:
    """The official `fuse_weight` of a Qwen layer, with a dense MLP instead of experts."""
    for layer in model.model.layers:
        scale = layer.input_layernorm.weight.detach()
        for name in ("q_proj", "k_proj", "v_proj"):
            rotation_utils._fuse_scale_into_linear(getattr(layer.self_attn, name), scale)
        rotation_utils._set_rms_weight_ones(layer.input_layernorm)
        scale = layer.post_attention_layernorm.weight.detach()
        for name in ("gate_proj", "up_proj"):
            rotation_utils._fuse_scale_into_linear(getattr(layer.mlp, name), scale)
        rotation_utils._set_rms_weight_ones(layer.post_attention_layernorm)
    rotation_utils._fuse_scale_into_linear(model.lm_head, model.model.norm.weight.detach())
    rotation_utils._set_rms_weight_ones(model.model.norm)


def rel_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) / abs(y) for x, y in zip(a, b, strict=True))


# validation


def test_trained_sources_are_implemented_and_versioned():
    from effeval.core.preprocess import RESERVED_SOURCES, SOURCE_VERSIONS

    assert "trained" not in RESERVED_SOURCES
    assert SOURCE_VERSIONS["rotation"]["trained"] == SOURCE_VERSIONS["scaling"]["trained"] == "1"
    engine._discover()
    assert ("rotation", "trained") in engine._SOURCES and ("scaling", "trained") in engine._SOURCES


@pytest.mark.parametrize("preset", ["codequant_aos", "codequant_aos_deepseek", "codequant2_rs"])
def test_presets_validate_on_qwen3_0_6b(preset):
    raw = {
        "field": "quantization",
        "model": {"name": "qwen3-0.6b"},
        "method": {"name": "rtn", "preset": "quarot_w4a4kv4"},
        "preprocess": {"preset": preset},
    }
    res = validate_module.resolve(raw, builtin_catalog())
    assert res.ok, [i.message for i in res.issues]
    kinds = [step.kind for step in res.spec.preprocess.steps]
    assert kinds == (["rotation", "scaling"] if preset == "codequant2_rs" else ["rotation"])
    source = res.spec.preprocess.steps[0].entries[0].source
    assert (source.act.bits, source.init_seed) == (4, 42)


def test_preset_numbers_are_the_released_settings():
    steps = builtin_preprocess_presets()["codequant_aos"].values["preprocess.steps"].value
    source = steps[0]["entries"][0]["source"]
    assert source["lr"] == 1e-4
    assert {k: source["calibration"][k] for k in ("n_samples", "batch_size", "max_length", "epochs")} == {
        "n_samples": 512,
        "batch_size": 16,
        "max_length": 256,
        "epochs": 4,
    }
    deepseek = builtin_preprocess_presets()["codequant_aos_deepseek"].values["preprocess.steps"].value
    calibration = deepseek[0]["entries"][0]["source"]["calibration"]
    assert (calibration["n_samples"], calibration["batch_size"], calibration["epochs"]) == (1024, 32, 16)
    two = builtin_preprocess_presets()["codequant2_rs"].values["preprocess.steps"].value
    assert [s["kind"] for s in two] == ["rotation", "scaling"]
    assert two[0]["entries"][0]["source"]["lr"] == 1e-3
    assert [e["position"] for e in two[1]["entries"]] == list(POSITIONS)


@pytest.mark.parametrize(
    ("position", "mode"),
    [
        ("linear.mlp.down_proj.input", "online"),
        ("linear.mlp.down_proj.input", "offline"),
        ("attn.v_head", "offline"),
        ("residual", "online"),
    ],
)
def test_trained_rotation_is_rejected_off_the_offline_residual(position, mode):
    with pytest.raises(ValidationError, match="residual position, offline, only"):
        RotationEntry.model_validate({"position": position, "mode": mode, "source": ROTATION_SOURCE})
    RotationEntry.model_validate({"position": "residual", "mode": "offline", "source": ROTATION_SOURCE})


@pytest.mark.parametrize(
    ("position", "mode"),
    [
        ("residual", "offline"),
        ("linear.mlp.down_proj.input", "offline"),
        ("norm.input_layernorm.out", "online"),
    ],
)
def test_trained_scaling_is_rejected_off_offline_norm_outputs(position, mode):
    with pytest.raises(ValidationError, match=r"norm\.\*\.out positions, offline, only"):
        ScalingEntry.model_validate({"position": position, "mode": mode, "source": SCALING_SOURCE})
    ScalingEntry.model_validate(
        {"position": "norm.input_layernorm.out", "mode": "offline", "source": SCALING_SOURCE}
    )


def test_trained_rotation_at_a_linear_input_does_not_validate_in_a_run_spec():
    raw = {
        "field": "quantization",
        "model": {"name": "qwen3-0.6b"},
        "method": {"name": "rtn", "preset": "quarot_w4a4kv4"},
        "preprocess": {
            "steps": [
                {
                    "kind": "rotation",
                    "entries": [
                        {
                            "position": "linear.mlp.down_proj.input",
                            "mode": "online",
                            "source": ROTATION_SOURCE,
                        }
                    ],
                }
            ]
        },
    }
    assert not validate_module.resolve(raw, builtin_catalog()).ok


def test_the_trained_grid_must_be_a_per_token_float_zero_point_grid():
    for override in (
        {"symmetric": True},
        {"granularity": "per_channel"},
        {"zero_point": "integer"},
        {"zero_point": None},
    ):
        with pytest.raises(ValidationError, match=r"per-token asymmetric grid|zero_point"):
            RotationEntry.model_validate(
                {
                    "position": "residual",
                    "mode": "offline",
                    "source": {**ROTATION_SOURCE, "act": {**ACT, **override}},
                }
            )


# building blocks


def test_cayley_rotation_equals_the_official_initial_matrix_and_keeps_the_global_rng():
    rotation_utils = official("cq1", "utils/rotation_utils.py")
    torch.manual_seed(5)
    state = torch.get_rng_state()
    ours = cayley_rotation(HIDDEN, 7)
    assert torch.equal(torch.get_rng_state(), state)
    torch.manual_seed(7)
    theirs = rotation_utils.load_or_create_R1(mode="online", device="cpu", dim=HIDDEN)
    assert torch.equal(ours.weight, theirs.weight)
    assert (ours.weight @ ours.weight.T - torch.eye(HIDDEN)).abs().max() < 1e-5
    assert ours.weight.dtype == torch.float32


def test_act_quant_mse_and_its_gradient_equal_the_official_quantizer():
    quantization_utils = official("cq1", "utils/quantization_utils.py")
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(3, 5, 32, generator=generator)
    x[0, 0] = 2.0
    x[1, 2, :] = x[1, 2, 0]
    x = x.requires_grad_()
    ours = act_quant_mse(x, BITS)
    (grad_ours,) = torch.autograd.grad(ours, x)
    y = x.detach().clone().requires_grad_()
    theirs = torch.nn.functional.mse_loss(y, quantization_utils.activation_quantizer((y,), -1, BITS))
    (grad_theirs,) = torch.autograd.grad(theirs, y)
    assert torch.equal(ours, theirs)
    assert torch.allclose(grad_ours, grad_theirs, rtol=1e-6, atol=1e-12)
    assert grad_ours.abs().sum() > 0


# the residual rotation


def test_prepare_residual_folds_the_gains_exactly_as_the_official_fuse_weight(model):
    rotation_utils = official("cq1", "utils/rotation_utils.py")
    expected = copy.deepcopy(model)
    unfolded = copy.deepcopy(model)
    official_fold_gains(expected, rotation_utils)
    ctx = make_context(model)
    entry = RotationEntry.model_validate(rotation_step()["entries"][0])
    prepare_residual(ctx.net, entry, engine.StageResult())
    assert any(not torch.equal(p, q) for p, q in zip(model.parameters(), unfolded.parameters(), strict=True))
    for (name, ours), theirs in zip(model.named_parameters(), expected.parameters(), strict=True):
        assert torch.equal(ours, theirs), name


def test_the_rotation_loss_sequence_equals_the_official_aos_loop(model, batches):
    cq1 = official("cq1", "script/rotation_fine_tune_script.py")
    rotation_utils = official("cq1", "utils/rotation_utils.py")
    reference = copy.deepcopy(model)
    official_fold_gains(reference, rotation_utils)
    ctx = make_context(model)
    prepare_residual(
        ctx.net, RotationEntry.model_validate(rotation_step()["entries"][0]), engine.StageResult()
    )

    torch.manual_seed(7)
    r1 = rotation_utils.load_or_create_R1(mode="online", device="cpu", dim=HIDDEN)
    optimizer = rotation_utils.create_optimizer(r1, lr=LR)
    losses: dict[str, torch.Tensor] = {}
    for name, module in norm_modules(reference).items():
        module.register_forward_hook(cq1.rotation_fine_tune_hook(name, "output", r1, BITS, -1, losses))
    official_history = []
    for batch in batches:
        losses.clear()
        reference(**batch)
        optimizer.zero_grad()
        loss = torch.stack(list(losses.values())).sum()
        loss.backward()
        optimizer.step()
        official_history.append(loss.item())

    rotation = cayley_rotation(HIDDEN, 7)
    norms = list(norm_modules(model).values())
    history = trained.train_rotation(model, norms, batches, rotation, BITS, LR)
    assert len(history) == len(official_history) == 8
    assert rel_diff(history, official_history) < 1e-6
    assert sum(history[4:]) < sum(history[:4])
    assert torch.allclose(rotation.weight, r1.weight, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(rotation.weight, cayley_rotation(HIDDEN, 7).weight, atol=1e-4)


def test_the_trained_rotation_is_folded_and_the_function_is_unchanged(model, batches):
    reference = copy.deepcopy(model)
    embedding_before = model.model.embed_tokens.weight.detach().clone()
    base = logits_of(model)
    ctx = make_context(model)
    applied = engine.apply(model, preprocess_of(rotation_step()), ctx)

    artifact = ctx.artifacts.get_or_compute(
        applied.artifacts["trained_rotation"].key, lambda: pytest.fail("not cached")
    )
    q, history = artifact["R"], artifact["losses"]
    assert history.numel() == 8 and history[4:].sum() < history[:4].sum()
    assert (q @ q.T - torch.eye(HIDDEN)).abs().max() < 1e-5
    assert not torch.allclose(q, cayley_rotation(HIDDEN, 7).weight, atol=1e-4)
    expected = (embedding_before.double() @ q.double()).float()
    assert torch.allclose(model.model.embed_tokens.weight, expected, atol=1e-6)
    assert torch.allclose(logits_of(model), base, atol=1e-4)
    for norm in norm_modules(model).values():
        assert torch.equal(norm.weight, torch.ones_like(norm.weight))
    assert any(
        not torch.allclose(p, q_, atol=1e-3)
        for p, q_ in zip(model.parameters(), reference.parameters(), strict=True)
    )
    assert "trained rotation: 8 Adam steps" in applied.edits[-1]


def test_a_cached_rotation_is_not_trained_again(model, batches, monkeypatch):
    cache = ArtifactCache()
    twin = copy.deepcopy(model)
    engine.apply(model, preprocess_of(rotation_step()), make_context(model, cache))
    monkeypatch.setattr(trained, "text_batches", lambda *_: pytest.fail("trained again"))
    engine.apply(twin, preprocess_of(rotation_step()), make_context(twin, cache))
    assert cache.computed == 1 and cache.memory_hits == 1
    assert torch.equal(model.model.embed_tokens.weight, twin.model.embed_tokens.weight)
    other = copy.deepcopy(model)
    monkeypatch.setattr(trained, "text_batches", lambda *_: [fake_batches(4, 0)])
    engine.apply(other, preprocess_of(rotation_step(lr=LR * 2)), make_context(other, cache))
    assert cache.computed == 2


def test_a_trained_rotation_step_takes_one_entry(model, batches):
    step = rotation_step()
    step["entries"].append({"position": "residual", "mode": "offline", "source": ROTATION_SOURCE})
    with pytest.raises(engine.EngineError, match="exactly one residual entry"):
        engine.apply(model, preprocess_of(step), make_context(model))


# the scaling vectors


def test_the_scaling_loss_sequence_equals_the_official_scaling_loop(model, batches):
    cq2 = official("cq2", "rotation/training.py")
    reference = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(3)
    init = {name: 1 + 0.01 * torch.randn(HIDDEN, generator=generator) for name in norm_modules(reference)}

    vectors = {name: torch.nn.Parameter(init[name].clone()) for name in init}
    optimizers = {name: torch.optim.Adam([v], lr=LR) for name, v in vectors.items()}
    losses: dict[str, torch.Tensor] = {}
    for name, module in norm_modules(reference).items():
        hook = cq2.quantization_loss_hook(name, torch.eye(HIDDEN), vectors[name], BITS, losses)
        module.register_forward_hook(hook)
    official_history = []
    for batch in batches:
        losses.clear()
        reference(**batch)
        for name, loss in losses.items():
            optimizers[name].zero_grad()
            loss.backward()
            optimizers[name].step()
        official_history.append(sum(loss.item() for loss in losses.values()))

    final, history = trained.train_scaling(model, norm_modules(model), batches, init, BITS, LR)
    assert rel_diff(history, official_history) < 1e-6
    assert sum(history[4:]) < sum(history[:4])
    for name, vector in vectors.items():
        assert torch.allclose(final[name], vector.detach(), rtol=1e-6, atol=1e-8), name
        assert not torch.allclose(final[name], init[name], atol=1e-4)


def test_trained_scaling_folds_into_the_gains_and_the_consumers(model, batches):
    base = logits_of(model)
    gains = {name: m.weight.detach().clone() for name, m in norm_modules(model).items()}
    q_proj = model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    ctx = make_context(model)
    applied = engine.apply(model, preprocess_of(scaling_step()), ctx)

    artifact = ctx.artifacts.get_or_compute(
        applied.artifacts["trained_scaling"].key, lambda: pytest.fail("not cached")
    )
    assert artifact["losses"].numel() == 8 and artifact["losses"][4:].sum() < artifact["losses"][:4].sum()
    scale = artifact["norm.input_layernorm.out@0"]
    generator = torch.Generator().manual_seed(3)
    first = 1 + 0.01 * torch.randn(HIDDEN, generator=generator)
    assert not torch.allclose(scale, first, atol=1e-4)
    layer0 = model.model.layers[0]
    assert torch.allclose(
        layer0.input_layernorm.weight, gains["model.layers.0.input_layernorm"] / scale, rtol=1e-6
    )
    assert torch.allclose(layer0.self_attn.q_proj.weight, q_proj * scale[None, :], rtol=1e-6)
    assert torch.allclose(logits_of(model), base, atol=1e-4)
    assert len([k for k in artifact if k.startswith("norm.")]) == 4


def test_scaling_entries_of_one_step_must_share_the_source(model, batches):
    step = scaling_step()
    step["entries"][1]["source"] = {**SCALING_SOURCE, "lr": LR * 2}
    with pytest.raises(engine.EngineError, match="share their source settings"):
        engine.apply(model, preprocess_of(step), make_context(model))


def test_scaling_on_the_rotated_model_matches_the_official_unrotated_loop(model, batches):
    """The official stage rotates inside the hook of the unrotated model; effeval trains the rotated one."""
    cq2 = official("cq2", "rotation/training.py")
    rotation_utils = official("cq1", "utils/rotation_utils.py")
    reference = copy.deepcopy(model)
    official_fold_gains(reference, rotation_utils)
    ctx = make_context(model)
    applied = engine.apply(model, preprocess_of(rotation_step()), ctx)
    q = ctx.artifacts.get_or_compute(applied.artifacts["trained_rotation"].key, lambda: pytest.fail("x"))["R"]

    generator = torch.Generator().manual_seed(3)
    init = {name: 1 + 0.01 * torch.randn(HIDDEN, generator=generator) for name in norm_modules(reference)}
    vectors = {name: torch.nn.Parameter(init[name].clone()) for name in init}
    optimizers = {name: torch.optim.Adam([v], lr=LR) for name, v in vectors.items()}
    losses: dict[str, torch.Tensor] = {}
    for name, module in norm_modules(reference).items():
        module.register_forward_hook(cq2.quantization_loss_hook(name, q, vectors[name], BITS, losses))
    official_history = []
    for batch in batches:
        losses.clear()
        reference(**batch)
        for name, loss in losses.items():
            optimizers[name].zero_grad()
            loss.backward()
            optimizers[name].step()
        official_history.append(sum(loss.item() for loss in losses.values()))

    _, history = trained.train_scaling(model, norm_modules(model), batches, init, BITS, LR)
    assert rel_diff(history, official_history) < 1e-3


def test_rotation_then_scaling_through_the_engine_keeps_the_function(model, batches):
    base = logits_of(model)
    applied = engine.apply(model, preprocess_of(rotation_step(), scaling_step()), make_context(model))
    assert len(applied.results) == 2 and applied.online_ops is None
    assert torch.allclose(logits_of(model), base, atol=1e-4)
    gains = [m.weight for m in norm_modules(model).values()]
    assert all(not torch.equal(g, torch.ones_like(g)) for g in gains)


def test_the_initial_values_come_from_the_seeds(model, batches):
    ctx = make_context(model)
    applied = engine.apply(model, preprocess_of(rotation_step(lr=1e-9), scaling_step(lr=1e-9)), ctx)
    rotation = ctx.artifacts.get_or_compute(
        applied.results[0].artifacts["trained_rotation"].key, lambda: pytest.fail("x")
    )
    assert torch.allclose(rotation["R"], cayley_rotation(HIDDEN, 7).weight, atol=1e-6)
    scales = ctx.artifacts.get_or_compute(
        applied.results[1].artifacts["trained_scaling"].key, lambda: pytest.fail("x")
    )
    generator = torch.Generator().manual_seed(3)
    expected = [1 + 0.01 * torch.randn(HIDDEN, generator=generator) for _ in range(4)]
    names = ["norm.input_layernorm.out@0", "norm.input_layernorm.out@1"]
    names += ["norm.post_attention_layernorm.out@0", "norm.post_attention_layernorm.out@1"]
    for name, init in zip(names, expected, strict=True):
        assert torch.allclose(scales[name], init, atol=1e-6), name


# micro-batching


class RecordingAdam(torch.optim.Adam):
    """Adam that keeps the gradients it is about to apply."""

    seen: ClassVar[list[list[torch.Tensor]]] = []

    def step(self, *args, **kwargs):
        RecordingAdam.seen.append([p.grad.detach().clone() for g in self.param_groups for p in g["params"]])
        return super().step(*args, **kwargs)


def uneven_batches() -> list[transformers.BatchEncoding]:
    generator = torch.Generator().manual_seed(5)
    ids = torch.randint(1, 97, (3, 8), generator=generator)
    mask = torch.ones(3, 8, dtype=torch.long)
    mask[1, :2] = 0
    mask[2, :5] = 0
    return [transformers.BatchEncoding({"input_ids": ids, "attention_mask": mask})]


def recorded_gradients(monkeypatch, run):
    """(gradients Adam applied in the first step, result of `run()`)."""
    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    RecordingAdam.seen = []
    result = run()
    return RecordingAdam.seen[0], result


def scaling_history(*args):
    return trained.train_scaling(*args)[1]


def grad_rel_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    num = sum(float((x - y).norm()) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5
    return num / sum(float(y.norm()) ** 2 for y in b) ** 0.5


@pytest.mark.parametrize("micro", [1, 2])
def test_micro_batched_rotation_gradient_equals_the_full_batch_gradient(model, monkeypatch, micro):
    batches = uneven_batches()
    results = {}
    for size in (None, micro):
        twin = copy.deepcopy(model).double()
        rotation = cayley_rotation(HIDDEN, 7)
        norms = list(norm_modules(twin).values())
        results[size] = recorded_gradients(
            monkeypatch, partial(trained.train_rotation, twin, norms, batches, rotation, BITS, LR, size)
        )
    assert results[None][0][0].abs().sum() > 0
    assert grad_rel_diff(results[micro][0], results[None][0]) < 1e-6
    assert rel_diff(results[micro][1], results[None][1]) < 1e-6


@pytest.mark.parametrize("micro", [1, 2])
def test_micro_batched_scaling_gradient_equals_the_full_batch_gradient(model, monkeypatch, micro):
    batches = uneven_batches()
    results = {}
    for size in (None, micro):
        twin = copy.deepcopy(model).double()
        norms = norm_modules(twin)
        generator = torch.Generator().manual_seed(3)
        init = {name: 1 + 0.01 * torch.randn(HIDDEN, generator=generator) for name in norms}
        results[size] = recorded_gradients(
            monkeypatch, partial(scaling_history, twin, norms, batches, init, BITS, LR, size)
        )
    assert results[None][0][0].abs().sum() > 0
    assert grad_rel_diff(results[micro][0], results[None][0]) < 1e-6
    assert rel_diff(results[micro][1], results[None][1]) < 1e-6


def test_micro_batch_weights_are_row_fractions():
    batch = transformers.BatchEncoding({"input_ids": torch.zeros(5, 3), "attention_mask": torch.ones(5, 3)})
    chunks = trained.micro_batches(batch, 2)
    assert [(c["input_ids"].shape[0], w) for c, w in chunks] == [(2, 0.4), (2, 0.4), (1, 0.2)]
    assert [w for _c, w in trained.micro_batches(batch, None)] == [1.0]
    assert sum(w for _c, w in trained.micro_batches(batch, 7)) == 1.0


def test_micro_batch_size_is_part_of_the_artifact_key_and_runs_through_the_engine(model, batches):
    base = logits_of(model)
    twin = copy.deepcopy(model)
    cache = ArtifactCache()
    engine.apply(model, preprocess_of(rotation_step(micro_batch_size=1)), make_context(model, cache))
    engine.apply(twin, preprocess_of(rotation_step()), make_context(twin, cache))
    assert cache.computed == 2
    assert torch.allclose(logits_of(model), base, atol=1e-4)


def test_the_presets_bound_memory_and_continue_the_shuffle_stream():
    presets = builtin_preprocess_presets()
    for name in ("codequant_aos", "codequant_aos_deepseek", "codequant2_rs"):
        for step in presets[name].values["preprocess.steps"].value:
            assert all(e["source"]["micro_batch_size"] == 4 for e in step["entries"])
    two = presets["codequant2_rs"].values["preprocess.steps"].value
    rotation_cfg = two[0]["entries"][0]["source"]["calibration"]
    scaling_cfg = two[1]["entries"][0]["source"]["calibration"]
    assert scaling_cfg["skip_epochs"] == rotation_cfg["epochs"] == 8
    assert scaling_cfg["shuffle_seed"] == rotation_cfg["shuffle_seed"]
    assert rotation_cfg["drop_empty"] is True and rotation_cfg.get("skip_epochs", 0) == 0
    assert rotation_cfg["pre_shuffle_seed"] == scaling_cfg["pre_shuffle_seed"] == 42


# the norms and folds of the official pipeline


def official_style_moe(model):
    """A copy whose MoE blocks have the module tree the official code reads (gate nn.Linear, experts list)."""
    reference = copy.deepcopy(model)
    for layer in reference.model.layers:
        block = layer.mlp
        gate = torch.nn.Linear(block.gate.weight.shape[1], block.gate.weight.shape[0], bias=False)
        gate.weight.data.copy_(block.gate.weight)
        shim = torch.nn.Module()
        shim.gate = gate
        shim.experts = torch.nn.ModuleList(block.experts.experts)
        layer.mlp = shim
    return reference


def renamed(model) -> dict[str, torch.Tensor]:
    return {n.replace(".experts.experts.", ".experts."): p.detach() for n, p in model.named_parameters()}


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_the_trained_norms_are_the_official_regex_selection(kind):
    from test_preprocess_moe import prepared_model

    model = build_tiny_qwen3(head_dim=16, tie=False) if kind == "dense" else prepared_model()
    ours = trained.residual_norms(make_context(model).net)
    theirs = list(norm_modules(model).values())
    assert len(ours) == len(theirs) == 2 * len(model.model.layers)
    assert all(a is b for a, b in zip(ours, theirs, strict=True))


def test_the_moe_gain_and_rotation_folds_equal_the_official_fuse_weight_and_fuse_rotation():
    from effeval.methods.preprocess.rotation import fold_residual
    from test_preprocess_moe import prepared_model

    rotation_utils = official("cq1", "utils/rotation_utils.py")
    model = prepared_model()
    reference = official_style_moe(model)
    before = renamed(copy.deepcopy(model))
    q = cayley_rotation(model.config.hidden_size, 7).weight.detach()

    rotation_utils.fuse_weight(reference, "qwen")
    rotation_utils.fuse_rotation(reference, "qwen", {"cpu": q}, None)

    ctx = make_context(model)
    entry = RotationEntry.model_validate(rotation_step()["entries"][0])
    result = engine.StageResult()
    prepare_residual(ctx.net, entry, result)
    fold_residual(ctx.net, entry, q.double(), result)

    ours, theirs = renamed(model), renamed(reference)
    assert ours.keys() == theirs.keys()
    for name in ours:
        assert torch.equal(ours[name], theirs[name]), name
    changed = [n for n in ours if not torch.equal(ours[n], before[n])]
    assert any(".mlp.gate." in n for n in changed)
    assert any(".experts." in n and "down_proj" in n for n in changed)


def test_the_scaling_fold_equals_the_official_apply_scaling_vectors(model, batches):
    pipeline = official("cq2", "rotation/pipeline.py")
    reference = copy.deepcopy(model)
    ctx = make_context(model)
    applied = engine.apply(model, preprocess_of(scaling_step()), ctx)
    artifact = ctx.artifacts.get_or_compute(
        applied.artifacts["trained_scaling"].key, lambda: pytest.fail("not cached")
    )
    vectors = {}
    for layer in range(len(model.model.layers)):
        vectors[f"model.layers.{layer}.input_layernorm"] = artifact[f"norm.input_layernorm.out@{layer}"]
        vectors[f"model.layers.{layer}.post_attention_layernorm"] = artifact[
            f"norm.post_attention_layernorm.out@{layer}"
        ]
    pipeline.apply_scaling_vectors(reference, "Qwen/Qwen3-4B-Thinking-2507", vectors)
    for (name, ours), theirs in zip(model.named_parameters(), reference.parameters(), strict=True):
        assert torch.equal(ours, theirs), name
