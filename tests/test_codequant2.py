import copy
import math
import os
import random
import sys
from collections import defaultdict

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
datasets = pytest.importorskip("datasets")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval import validate as validate_module  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import CodebookFormat, IntFormat, QuantRule  # noqa: E402
from effeval.core.preprocess import builtin_preprocess_presets  # noqa: E402
from effeval.methods.quantization import codequant2 as cq  # noqa: E402
from effeval.ops.artifacts import ArtifactCache  # noqa: E402
from test_codebook import _official  # noqa: E402

pytestmark = pytest.mark.compute

BITS = 4
FMT = CodebookFormat(
    type="codebook",
    centroids=16,
    granularity="per_channel",
    kmeans={"iters": 300, "tol": 1e-6, "seed": 42, "compute_dtype": "float32"},
)
WEIGHT_SITES = [f"linear.attn.{p}_proj.weight" for p in "qkv"] + [
    f"linear.mlp.{p}_proj.weight" for p in ("gate", "up")
]
INPUT_SITES = [site.replace(".weight", ".input") for site in WEIGHT_SITES]
ACT_RULE = QuantRule(
    sites=INPUT_SITES,
    quant=IntFormat(type="int", bits=BITS, symmetric=False, granularity="per_token", zero_point="float"),
)
TARGET_ATTRS = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]
LATENCY = {"8": 0.04, "16": 0.04, "32": 0.04}


def _from_reference(module) -> bool:
    root = os.environ.get("EFFEVAL_CODEQUANT_REF", "")
    attributes = vars(module)
    places = [str(attributes.get("__file__") or ""), *map(str, attributes.get("__path__") or [])]
    return bool(root) and any(place.startswith(root) for place in places)


def _purge_reference() -> None:
    for name in [n for n, m in sys.modules.items() if _from_reference(m)]:
        del sys.modules[name]


def official(relative: str):
    """A module of the official CodeQuant2 sources, loaded without leaving its sibling packages
    (`cluster`, `quantization`, ...) in sys.modules."""
    _purge_reference()
    try:
        return _official("cq2", relative)
    finally:
        _purge_reference()


def padded_batches(count: int, seed: int = 0, rows: int = 2, length: int = 8) -> list[dict]:
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for index in range(count):
        mask = torch.ones(rows, length, dtype=torch.long)
        mask[1, : index % 3] = 0
        batches.append(
            {"input_ids": torch.randint(1, 97, (rows, length), generator=generator), "attention_mask": mask}
        )
    return batches


def cluster_rules() -> list[QuantRule]:
    return [QuantRule(sites=WEIGHT_SITES, quant=FMT)]


def aca_config(**overrides) -> cq.AcaConfig:
    values = {
        "calibration": {"dataset": "fake/wiki", "n_samples": 8, "batch_size": 2, "max_length": 8},
        "layers_per_group": 2,
        "change_ratio": 0.3,
        "micro_batch_size": 1,
    }
    return cq.AcaConfig(**{**values, **overrides})


def rac_config(**overrides) -> cq.RacCalibration:
    values = {"batch_size": 2, "max_length": 8, "epochs": 1}
    return cq.RacCalibration(**{**values, **overrides})


def train_config(**overrides) -> cq.TrainConfig:
    values = {
        "calibration": rac_config(),
        "lora_r": 4,
        "lr_centroids": 1e-2,
        "lr_assignment": 1e-2,
        "start_temp": 1.0,
        "end_temp": 0.3,
    }
    return cq.TrainConfig(**{**values, **overrides})


def scaled_model(scale: float, layers: int = 2):
    """Tiny Qwen3 whose linear weights are multiplied by `scale`, so that gradient norms exceed the clip."""
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=layers)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim == 2:
                parameter.mul_(scale)
    return model


def official_kmeans(matrix, k, iters, tol=1e-6):
    """Stands in for the official k-means (global RNG) so both codes start from the same codebooks."""
    return cq._fit(matrix, FMT, k)


# configuration and presets


def test_config_requires_per_channel_codebook_rules_with_the_base_count():
    codebook = {"sites": WEIGHT_SITES, "quant": FMT.model_dump()}
    act = ACT_RULE.model_dump()
    train = train_config().model_dump()
    aca = aca_config().model_dump()
    cq.CodeQuant2Config.model_validate({"rules": [codebook, act], "aca": aca, "train": train})
    with pytest.raises(ValidationError, match="at least one rule needs a codebook format"):
        cq.CodeQuant2Config.model_validate({"rules": [act], "train": train})
    group = {**codebook, "quant": {**FMT.model_dump(), "granularity": "per_group", "group_size": 16}}
    with pytest.raises(ValidationError, match="per_channel"):
        cq.CodeQuant2Config.model_validate({"rules": [group], "train": train})
    odd = {**codebook, "quant": {**FMT.model_dump(), "centroids": 12}}
    with pytest.raises(ValidationError, match=r"choices\[1\]"):
        cq.CodeQuant2Config.model_validate({"rules": [odd], "aca": aca, "train": train})
    cq.CodeQuant2Config.model_validate({"rules": [odd], "train": train})
    with pytest.raises(ValidationError, match="increase"):
        aca_config(choices=[16, 8, 32])


@pytest.mark.parametrize(("preset", "rank"), [("paper_a4w4_4b", 16), ("paper_a4w4_8b", 8)])
def test_method_presets_are_the_released_settings(preset, rank):
    catalog = builtin_catalog()
    assert catalog.manifests["codequant2"].default_preset == "paper_a4w4_4b"
    values = catalog.presets["codequant2"][preset].values
    config = {path.removeprefix("method.config."): v.value for path, v in values.items() if path != "tasks"}
    cfg = cq.CodeQuant2Config.model_validate(config)
    weights = [r for r in cfg.rules if isinstance(r.quant, CodebookFormat)]
    assert [len(r.sites) for r in weights] == [5]
    assert weights[0].quant.centroids == 16 and weights[0].quant.kmeans.compute_dtype == "float32"
    assert cfg.aca.choices == [8, 16, 32] and cfg.aca.calibration.n_samples == 1024
    assert (cfg.aca.layers_per_group, cfg.aca.change_ratio) == (10, 0.15)
    train = cfg.train
    assert (train.lora_r, train.lr_centroids, train.lr_assignment, train.clip) == (rank, 1e-4, 1e-5, 1.0)
    assert (train.start_temp, train.end_temp, train.hard_ratio, train.loss1_weight) == (0.1, 0.001, 0.98, 1.0)
    rac = train.calibration
    assert (rac.total, rac.wiki_ratio, rac.batch_size, rac.max_length, rac.epochs) == (512, 0.9, 16, 512, 4)
    assert (rac.generation.max_new_tokens, rac.generation.temperature, rac.generation.top_k) == (256, 0.6, 20)


def test_the_rotation_scaling_preset_pairs_with_the_codequant2_presets_and_the_spec_validates():
    pairs = builtin_preprocess_presets()["codequant2_rs"].quantizers
    assert pairs == {"codequant2": ["paper_a4w4_4b", "paper_a4w4_8b"]}
    raw = {
        "field": "quantization",
        "model": {"name": "qwen3-4b-thinking-2507"},
        "method": {"name": "codequant2", "preset": "paper_a4w4_4b"},
        "preprocess": {"preset": "codequant2_rs"},
    }
    res = validate_module.resolve(raw, builtin_catalog())
    assert res.ok, [i.message for i in res.issues]


# allocation


def official_allocation(scores, ratio, utils):
    layer_scores = {name: {"I16_to_8": d, "I32_to_16": u, "dim": 1.0} for name, (d, u) in scores.items()}
    result, info = utils.allocate_cluster_latency(
        layer_scores, LATENCY, LATENCY, choices=(8, 16, 32), change_ratio=ratio
    )
    return result, info


def random_scores(generator: random.Random, size: int) -> dict[str, tuple[float, float]]:
    scale = generator.choice([0.1, 1.0, 10.0])
    return {
        f"m{i}": (
            abs(generator.gauss(0, 1)) * scale,
            -abs(generator.gauss(0, 1)) * generator.choice([0.3, 1, 3]) * scale,
        )
        for i in range(size)
    }


def test_allocation_equals_the_official_one_on_random_scores():
    utils = official("utils/adaptive_cluster_utils.py")
    generator = random.Random(0)
    seen_changes = set()
    for trial in range(60):
        size = generator.randint(2, 70)
        ratio = generator.choice([0.15, 0.3, 0.5, 1.0])
        scores = random_scores(generator, size)
        expected, info = official_allocation(scores, ratio, utils)
        got = cq.allocate_cluster_counts(
            {n: d for n, (d, _u) in scores.items()},
            {n: u for n, (_d, u) in scores.items()},
            [8, 16, 32],
            ratio,
        )
        assert got == expected, f"trial {trial}"
        seen_changes.add(info["changed"])
    assert len(seen_changes) > 3 and max(seen_changes) >= 4


def test_allocation_changes_pairs_within_the_ratio_and_keeps_the_mean_index_width():
    generator = random.Random(1)
    for _ in range(40):
        size = generator.randint(10, 80)
        ratio = generator.choice([0.15, 0.3])
        scores = random_scores(generator, size)
        got = cq.allocate_cluster_counts(
            {n: d for n, (d, _u) in scores.items()},
            {n: u for n, (_d, u) in scores.items()},
            [8, 16, 32],
            ratio,
        )
        changed = [n for n, k in got.items() if k != 16]
        assert len(changed) <= int(size * ratio) + 1
        assert sum(k == 32 for k in got.values()) == sum(k == 8 for k in got.values())
        assert sum(math.log2(k) for k in got.values()) == 4 * size


def test_allocation_only_accepts_a_pair_that_lowers_the_summed_score():
    down = {"a": 5.0, "b": 6.0, "c": 7.0, "d": 8.0}
    up = {"a": -1.0, "b": -2.0, "c": -1.5, "d": -0.5}
    assert cq.allocate_cluster_counts(down, up, [8, 16, 32], 1.0) == dict.fromkeys(down, 16)
    cheap = cq.allocate_cluster_counts({**down, "e": 0.5}, {**up, "e": -9.0}, [8, 16, 32], 1.0)
    assert cheap == {"a": 8, "b": 16, "c": 16, "d": 16, "e": 32}


# adaptive cluster allocation against the official scoring loop


def official_aca_run(utils, model, group_batches, layers_per_group, ratio):
    """The loop of the official `adap_clus.main` on a model that is already loaded and rotated."""
    wrappers = utils.replace_linear_with_prelinear(
        model, k_list=(8, 16, 32), kmeans_iters=300, activation_bit=BITS
    )
    for wrapper in wrappers.values():
        wrapper.build_Wk_cache()
    groups = defaultdict(list)
    for name in wrappers:
        groups[max(0, utils.get_layer_idx(name)) // layers_per_group].append(name)
    final = dict.fromkeys(wrappers, 16)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    scores = {}
    for group_id, batches in zip(sorted(groups), group_batches, strict=True):
        members = set(groups[group_id])
        for name in members:
            wrappers[name].reset_scores()
        for batch in batches:
            with torch.no_grad():
                for wrapper in wrappers.values():
                    wrapper.pinned_k = -1
                reference = model(**batch).logits.detach()
            for name, wrapper in wrappers.items():
                wrapper.pinned_k = 16 if name in members else final[name]
            model.zero_grad(set_to_none=True)
            utils.reconstruction_loss(label=reference, output=model(**batch).logits).backward()
            for name in members:
                wrappers[name].accumulate_importance()
            for name, wrapper in wrappers.items():
                if name not in members:
                    wrapper._last_x = wrapper._last_gy = None
        group_scores = {name: wrappers[name].get_scores() for name in wrappers if name in members}
        scores.update(group_scores)
        allocation, _ = utils.allocate_cluster_latency(
            group_scores, LATENCY, LATENCY, choices=(8, 16, 32), change_ratio=ratio
        )
        for name, k in allocation.items():
            final[name] = k
            wrappers[name].pinned_k = k
    return final, scores


@pytest.fixture
def aca_run(monkeypatch):
    """Runs ACA of effeval and of the official code on copies of one tiny model; returns both results."""
    utils = official("utils/adaptive_cluster_utils.py")
    monkeypatch.setattr(utils, "kmeans", official_kmeans)
    captured = []
    real = cq.allocate_cluster_counts

    def spy(downgrade, upgrade, choices, change_ratio):
        captured.append((dict(downgrade), dict(upgrade)))
        return real(downgrade, upgrade, choices, change_ratio)

    monkeypatch.setattr(cq, "allocate_cluster_counts", spy)
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=6)
    reference = copy.deepcopy(model)
    batches = [padded_batches(3, seed=group) for group in range(2)]
    cfg = aca_config(layers_per_group=3)
    targets = cq._targets(model, cluster_rules())
    counts = cq.choose_cluster_counts(model, targets, batches, cfg, BITS)
    expected, scores = official_aca_run(utils, reference, batches, 3, cfg.change_ratio)
    return counts, captured, expected, scores, model, reference


def test_aca_scores_equal_the_official_prelinear_scores(aca_run):
    _counts, captured, _expected, scores, _model, _reference = aca_run
    assert len(captured) == 2
    for downgrade, upgrade in captured:
        assert len(downgrade) == 15
        for name in downgrade:
            torch.testing.assert_close(downgrade[name], scores[name]["I16_to_8"], rtol=1e-6, atol=0)
            torch.testing.assert_close(upgrade[name], scores[name]["I32_to_16"], rtol=1e-6, atol=0)
    assert all(d > 0 for downgrade, _ in captured for d in downgrade.values())


def test_aca_allocation_equals_the_official_one_and_changes_matrices(aca_run):
    counts, _captured, expected, _scores, model, reference = aca_run
    assert counts == expected
    assert 0 < sum(k != 16 for k in counts.values()) <= 2 * (int(15 * 0.3) + 1)
    assert sum(k == 32 for k in counts.values()) == sum(k == 8 for k in counts.values())
    for name, module in model.named_modules():
        assert type(module).__name__ != "_ScoredLinear", name
    assert torch.equal(
        model.model.layers[0].self_attn.q_proj.weight,
        reference.model.layers[0].self_attn.q_proj.linear.weight,
    )


def test_aca_scores_do_not_depend_on_the_micro_batch_size(monkeypatch):
    captured = []
    real = cq.allocate_cluster_counts
    monkeypatch.setattr(
        cq,
        "allocate_cluster_counts",
        lambda d, u, c, r: captured.append((dict(d), dict(u))) or real(d, u, c, r),
    )
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    targets = cq._targets(model, cluster_rules())
    batches = [padded_batches(2, rows=4)]
    for micro in (None, 3, 1):
        cq.choose_cluster_counts(
            model, targets, batches, aca_config(layers_per_group=2, micro_batch_size=micro), BITS
        )
    whole, uneven, rows = captured
    for name in whole[0]:
        for other in (uneven, rows):
            torch.testing.assert_close(other[0][name], whole[0][name], rtol=1e-4, atol=0)
            torch.testing.assert_close(other[1][name], whole[1][name], rtol=1e-4, atol=0)


def test_aca_needs_the_batches_of_every_group_pass():
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    targets = cq._targets(model, cluster_rules())
    with pytest.raises(ValueError, match="group passes"):
        cq.choose_cluster_counts(model, targets, [padded_batches(1)], aca_config(layers_per_group=1), BITS)


# learnable clustering against the official loop


@pytest.fixture
def spied_cluster_linears(monkeypatch):
    """Replaces `_ClusterLinear` by a recording subclass; the list fills as `train_clusters` builds them."""
    created = []
    base = cq._ClusterLinear

    class Recording(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_a = self.lora_a.detach().clone()
            self.soft_trace = []
            self.snapshots = []
            created.append(self)

        def set_progress(self, ratio):
            self.snapshots.append((self.lora_a.detach().clone(), self.lora_b.detach().clone()))
            super().set_progress(ratio)
            self.soft_trace.append(self.soft)

    monkeypatch.setattr(cq, "_ClusterLinear", Recording)
    return created


def official_training(cl, model, counts, batches, cfg, initial_a):
    """`replace_linear` and the loop of the official `finetune_local.train_assignment` for `batches`."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    args = {
        "activation_quantization_bit": BITS,
        "cluster_num": 16,
        "cluster_assignment": counts,
        "start_temperature": cfg.start_temp,
        "end_temperature": cfg.end_temp,
        "hard_ratio": cfg.hard_ratio,
        "use_assignment_lora": True,
        "a_lora_r": cfg.lora_r,
        "soft": True,
        "loss1_weight": cfg.loss1_weight,
        "centroids_lr": cfg.lr_centroids,
    }
    cl.replace_linear(model, TARGET_ATTRS, args, torch.float32)
    linears = {name: m for name, m in model.named_modules() if isinstance(m, cl.ClusterLinear)}
    for name, module in linears.items():
        module.A_assignment.data.copy_(initial_a[name])
    params = [p for n, p in model.named_parameters() if p.requires_grad and "assignment" in n]
    optimizer = torch.optim.SGD(
        [{"params": params, "lr": cfg.lr_assignment, "weight_decay": 0.0}], momentum=cfg.momentum
    )
    model.eval()
    history = {"temperature": [], "grad_norm": [], "nmse": defaultdict(list)}
    for step, batch in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        temperature = float("nan")
        for module in model.modules():
            if hasattr(module, "set_temperature"):
                module.set_temperature(step / len(batches))
                temperature = module.temperature
        model(**batch)
        history["grad_norm"].append(torch.nn.utils.clip_grad_norm_(params, max_norm=1.0).item())
        optimizer.step()
        history["temperature"].append(temperature)
        for name, module in linears.items():
            history["nmse"][name].append(module._step_nmse)
    cl.cluster_linear_post_process(model, TARGET_ATTRS)
    return history, linears


@pytest.mark.parametrize("loss1_weight", [1.0, 0.5])
def test_training_equals_the_official_loop(monkeypatch, spied_cluster_linears, loss1_weight):
    cl = official("cluster/ClusterLinear.py")
    monkeypatch.setattr(cl, "kmeans", official_kmeans)
    model = scaled_model(60.0)
    reference = copy.deepcopy(model)
    targets = cq._targets(model, cluster_rules())
    counts = {t.name: [4, 8, 16][i % 3] for i, t in enumerate(targets)}
    batches = padded_batches(6)
    cfg = train_config(loss1_weight=loss1_weight)
    history = cq.train_clusters(model, targets, counts, batches, cfg, BITS)
    initial_a = {t.name: w.initial_a for t, w in zip(targets, spied_cluster_linears, strict=True)}
    expected, linears = official_training(cl, reference, counts, batches, cfg, initial_a)

    assert max(expected["grad_norm"]) > 1.0, "the global clip must be active for this test to mean anything"
    assert history["temperature"] == pytest.approx(expected["temperature"], rel=1e-12)
    torch.testing.assert_close(history["grad_norm"], expected["grad_norm"], rtol=1e-6, atol=0)
    for target, mine in zip(targets, spied_cluster_linears, strict=True):
        name = target.name
        torch.testing.assert_close(history["nmse"][name], expected["nmse"][name], rtol=1e-6, atol=0)
        official_linear = linears[name]
        torch.testing.assert_close(mine.centroids, official_linear.centroids, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(mine.lora_a, official_linear.A_assignment, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(mine.lora_b, official_linear.B_assignment, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(
            model.get_submodule(name).weight, reference.get_submodule(name).weight, rtol=1e-6, atol=1e-8
        )
        assert mine.centroids.dtype == torch.float32 and mine.assignment.dtype == torch.uint8


def test_centroids_start_from_bfloat16_rounded_k_means_and_the_lora_update_starts_at_zero(
    spied_cluster_linears,
):
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    targets = cq._targets(model, cluster_rules())
    counts = {t.name: 8 for t in targets}
    weights = {t.name: t.linear.weight.detach().clone() for t in targets}
    cfg = train_config(lr_centroids=1e-12, lr_assignment=1e-12)
    cq.train_clusters(model, targets, counts, padded_batches(1), cfg, BITS)
    for target, mine in zip(targets, spied_cluster_linears, strict=True):
        centroids, _ = cq._fit(weights[target.name], FMT, 8)
        torch.testing.assert_close(mine.centroids, centroids.to(torch.bfloat16).float(), rtol=0, atol=1e-8)
        assert not torch.equal(centroids, centroids.to(torch.bfloat16).float())
        assert mine.snapshots[0][1].count_nonzero() == 0
        assert mine.initial_a.abs().max() <= 1 / math.sqrt(4) + 1e-6


def test_assignments_turn_hard_at_the_configured_progress_and_the_lora_factors_stop(spied_cluster_linears):
    model = scaled_model(20.0)
    targets = cq._targets(model, cluster_rules())
    counts = {t.name: 4 for t in targets}
    batches = [b for _ in range(25) for b in padded_batches(4)]
    assert len(batches) == 100
    history = cq.train_clusters(model, targets, counts, batches, train_config(hard_ratio=0.98), BITS)
    ratio = [step / 100 for step in range(100)]
    soft = [r < 0.98 for r in ratio]
    assert sum(soft) == 98 and ratio[98] == 0.98
    for mine in spied_cluster_linears:
        assert mine.soft_trace == soft
        a_before, b_before = mine.snapshots[98]
        torch.testing.assert_close(mine.lora_a, a_before, rtol=0, atol=0)
        torch.testing.assert_close(mine.lora_b, b_before, rtol=0, atol=0)
        assert b_before.abs().max() > 0
        assert not torch.equal(mine.snapshots[97][1], b_before)
    expected = [1.0 + (r / 0.98) * (0.3 - 1.0) for r in ratio]
    assert history["temperature"] == pytest.approx(expected, rel=1e-12)


def test_training_reduces_the_output_error_of_the_clustered_weights():
    model = scaled_model(5.0)
    targets = cq._targets(model, cluster_rules())
    counts = {t.name: 8 for t in targets}
    history = cq.train_clusters(model, targets, counts, padded_batches(8), train_config(), BITS)
    first = sum(series[0] for series in history["nmse"].values())
    last = sum(series[-1] for series in history["nmse"].values())
    assert last < 0.6 * first


def test_training_restores_the_model_and_writes_k_distinct_values_per_row():
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    model.train()
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    others = {n: p.detach().clone() for n, p in model.named_parameters() if "o_proj" in n or "down_proj" in n}
    targets = cq._targets(model, cluster_rules())
    counts = {t.name: [4, 8][i % 2] for i, t in enumerate(targets)}
    cq.train_clusters(model, targets, counts, padded_batches(2), train_config(), BITS)
    assert model.training and {n: p.requires_grad for n, p in model.named_parameters()} == flags
    for target in targets:
        assert isinstance(model.get_submodule(target.name), torch.nn.Linear)
        rows = target.linear.weight.detach()
        assert max(len(row.unique()) for row in rows) <= counts[target.name]
    for name, tensor in others.items():
        assert torch.equal(dict(model.named_parameters())[name], tensor)


# reasoning-aware calibration


class FakeTokenizer:
    """Chat template as `<role>content` turns, one id per character, padding on `padding_side`."""

    pad_token_id = 0
    padding_side = "right"

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=None
    ):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        suffix = "" if enable_thinking is None else f"<think={enable_thinking}>"
        return text + (f"<assistant>{suffix}" if add_generation_prompt else "")

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=False, max_length=None):
        ids = [[ord(c) % 90 + 1 for c in text] for text in texts]
        if truncation:
            ids = [row[:max_length] for row in ids]
        width = max_length if padding == "max_length" else max(map(len, ids))
        pad = [width - len(row) for row in ids]
        left = self.padding_side == "left"
        input_ids = [([0] * n + row) if left else (row + [0] * n) for row, n in zip(ids, pad, strict=True)]
        mask = [
            ([0] * n + [1] * len(row)) if left else ([1] * len(row) + [0] * n)
            for row, n in zip(ids, pad, strict=True)
        ]
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(mask)}

    def batch_decode(self, tokens, skip_special_tokens=True):
        return ["".join(chr(97 + (i - 1) % 26) for i in row.tolist() if i != 0) for row in tokens]


PROMPTS = [
    "<user>one plus one<assistant>",
    "<user>what is two times seven<assistant>",
    "<user>a<assistant>",
    "<user>bc d<assistant>",
]


def test_generated_traces_are_seeded_and_continue_the_prompt_only():
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    tokenizer = FakeTokenizer()
    generation = cq.GenerationConfig(max_new_tokens=16, batch_size=3)
    state = torch.get_rng_state()
    first = cq.generate_traces(model, tokenizer, PROMPTS, generation, 7)
    assert torch.equal(torch.get_rng_state(), state) and tokenizer.padding_side == "right"
    assert cq.generate_traces(model, tokenizer, PROMPTS, generation, 7) == first
    assert cq.generate_traces(model, tokenizer, PROMPTS, generation, 8) != first
    assert len(first) == 4 and all(0 < len(text) <= 16 for text in first)


@pytest.fixture
def fake_datasets(monkeypatch):
    questions = [f"question number {i} about sums" for i in range(10)]
    wiki = ["", *[f"wiki row {i} with some words" for i in range(12)]]

    def load(path, name=None, split=None):
        assert split == "train"
        return datasets.Dataset.from_dict({"question": questions} if "gsm" in path else {"text": wiki})

    monkeypatch.setattr(cq, "load_dataset", load)
    return questions, wiki


def test_reasoning_rows_mix_sampled_traces_with_text_and_cache_the_traces(fake_datasets):
    questions, _wiki = fake_datasets
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    tokenizer = FakeTokenizer()
    cfg = rac_config(
        problem_dataset="fake/gsm",
        wiki_dataset="fake/wiki",
        pool=8,
        total=8,
        wiki_ratio=0.5,
        generation=cq.GenerationConfig(max_new_tokens=6, batch_size=2),
    )
    cache = ArtifactCache()
    rows = cq.reasoning_rows(model, tokenizer, cfg, cache)
    assert len(rows) == 8 and cache.computed == 1
    chats = [r for r in rows if r.startswith("<user>")]
    texts = [r for r in rows if r.startswith("wiki row")]
    assert len(chats) == 4 and len(texts) == 4
    assert all(any(f"<user>{q}<assistant>" in c for q in questions[:8]) for c in chats)
    assert cq.reasoning_rows(model, tokenizer, cfg, cache) == rows and cache.computed == 1
    changed = cfg.model_copy(update={"seed": 5})
    assert cq.reasoning_rows(model, tokenizer, changed, cache) != rows
    assert cache.computed == 2
    only_text = cfg.model_copy(update={"wiki_ratio": 1.0})
    assert all(r.startswith("wiki row") for r in cq.reasoning_rows(model, tokenizer, only_text, cache))
    assert cache.computed == 2
    for name in ("embed_tokens",):
        model.get_submodule(f"model.{name}").weight.data.mul_(1.5)
    cq.reasoning_rows(model, tokenizer, cfg, cache)
    assert cache.computed == 3


# the whole method


def test_the_method_clusters_targets_only_and_attaches_the_activation_rules(monkeypatch, fake_datasets):
    model = build_tiny_qwen3(head_dim=16, tie=False, layers=2)
    untouched = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if n.endswith(("o_proj.weight", "down_proj.weight", "lm_head.weight"))
    }
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    cfg = cq.CodeQuant2Config(
        rules=[QuantRule(sites=WEIGHT_SITES, quant=FMT.model_copy(update={"centroids": 8})), ACT_RULE],
        aca=aca_config(choices=[4, 8, 16], layers_per_group=1, micro_batch_size=2),
        train=train_config(
            calibration=rac_config(
                problem_dataset="fake/gsm",
                wiki_dataset="fake/wiki",
                pool=8,
                total=6,
                wiki_ratio=0.5,
                epochs=2,
                generation=cq.GenerationConfig(max_new_tokens=4, batch_size=3),
            )
        ),
    )
    monkeypatch.setattr(
        cq, "aca_batches", lambda aca, tokenizer, passes: [padded_batches(2, seed=g) for g in range(passes)]
    )
    allocations, trained = [], []
    choose, train = cq.choose_cluster_counts, cq.train_clusters
    monkeypatch.setattr(
        cq, "choose_cluster_counts", lambda *a: allocations.append(choose(*a)) or allocations[-1]
    )
    monkeypatch.setattr(
        cq, "train_clusters", lambda m, t, counts, *a: trained.append(counts) or train(m, t, counts, *a)
    )
    result = cq.codequant2(model, FakeTokenizer(), cfg)
    assert trained == allocations and len(trained) == 1
    assert set(trained[0].values()) <= {4, 8, 16} and len(set(trained[0].values())) > 1
    assert result is model
    names = dict(model.named_parameters())
    for name, tensor in untouched.items():
        assert torch.equal(names[name], tensor)
    changed = [
        n
        for n in names
        if n.endswith("proj.weight")
        and not torch.equal(names[n], original[n])
        and "o_proj" not in n
        and "down_proj" not in n
    ]
    assert len(changed) == 10
    for name in changed:
        assert max(len(row.unique()) for row in names[name].detach()) <= 16
    hooked = [
        m
        for n, m in model.named_modules()
        if any(n.endswith(a) for a in TARGET_ATTRS) and m._forward_pre_hooks
    ]
    assert len(hooked) == 10
    assert not model.model.layers[0].self_attn.o_proj._forward_pre_hooks
    input_ids, mask = torch.randint(1, 97, (2, 8)), torch.ones(2, 8, dtype=torch.long)
    with torch.no_grad():
        assert torch.isfinite(model(input_ids=input_ids, attention_mask=mask).logits).all()


def test_aca_batches_follow_the_shuffled_subset_and_reshuffle_each_pass(monkeypatch):
    rows = [f"row {i} " + "x" * (i % 5) for i in range(20)]
    monkeypatch.setattr(cq, "load_dataset", lambda path, split: datasets.Dataset.from_dict({"text": rows}))
    cfg = aca_config(
        calibration={
            "dataset": "fake/wiki",
            "n_samples": 12,
            "dataset_seed": 3,
            "batch_size": 5,
            "max_length": 40,
            "shuffle_seed": 9,
        }
    )
    tokenizer = FakeTokenizer()
    passes = cq.aca_batches(cfg, tokenizer, 3)
    subset = datasets.Dataset.from_dict({"text": rows}).shuffle(seed=3).select(range(12))["text"]
    generator = torch.Generator().manual_seed(9)
    loader = torch.utils.data.DataLoader(list(range(12)), batch_size=5, shuffle=True, generator=generator)
    orders = [[batch.tolist() for batch in loader] for _ in range(3)]
    assert orders[0] != orders[1] != orders[2]
    for epoch, order in zip(passes, orders, strict=True):
        assert [b["input_ids"].shape[0] for b in epoch] == [5, 5, 2]
        for batch, indices in zip(epoch, order, strict=True):
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": subset[i]}], False, True, enable_thinking=False
                )
                for i in indices
            ]
            reference = tokenizer(
                texts, return_tensors="pt", padding="max_length", truncation=True, max_length=40
            )
            assert torch.equal(batch["input_ids"], reference["input_ids"])
            assert torch.equal(batch["attention_mask"], reference["attention_mask"])
