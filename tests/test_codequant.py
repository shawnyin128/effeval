"""CodeQuant ACCF: the layer-major runner equals a batch-major port of the official loop and its modules."""

from __future__ import annotations

import copy
import math

import pytest
from pydantic import ValidationError

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from transformers import BatchEncoding  # noqa: E402

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.fragments import CodebookFormat, IntFormat, KMeansConfig, QuantRule, TextBatches  # noqa: E402
from effeval.methods.quantization import codequant  # noqa: E402
from effeval.methods.quantization.codequant import ACCFConfig, CodeQuantConfig, quantize_model  # noqa: E402
from effeval.ops import codebook  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402
from effeval.runtime import moe  # noqa: E402
from test_codebook import _official  # noqa: E402
from test_moe_experts import EXPERTS, HIDDEN, LAYERS, build  # noqa: E402

pytestmark = pytest.mark.compute

CENTROIDS = 16
LR = 50.0
ATTENTION_SITES = [f"linear.attn.{p}_proj.weight" for p in ("q", "k", "v")]
MOE_SITES = ["moe.router.weight", "linear.mlp.experts.gate_proj.weight", "linear.mlp.experts.up_proj.weight"]
CALIBRATION = TextBatches(dataset="unused", n_samples=1, batch_size=1, max_length=1, epochs=1)
A4 = IntFormat(type="int", bits=4, symmetric=False, granularity="per_token", zero_point="float")


def codebook_format(compute_dtype: str = "weight") -> CodebookFormat:
    kmeans = KMeansConfig(compute_dtype=compute_dtype)
    return CodebookFormat(type="codebook", centroids=CENTROIDS, granularity="per_channel", kmeans=kmeans)


def make_config(
    sites=(*ATTENTION_SITES, *MOE_SITES),
    *,
    accf: bool = True,
    lr: float = LR,
    kl_weight: float = 1.0,
    route_gradients: bool = True,
    layers="all",
    extra_rules: tuple[QuantRule, ...] = (),
    compute_dtype: str = "weight",
    train_dtype: str = "model",
) -> CodeQuantConfig:
    rules = [QuantRule(sites=list(sites), layers=layers, quant=codebook_format(compute_dtype)), *extra_rules]
    settings = ACCFConfig(
        calibration=CALIBRATION,
        lr=lr,
        kl_weight=kl_weight,
        route_gradients=route_gradients,
        train_dtype=train_dtype,
    )
    return CodeQuantConfig(rules=rules, accf=settings if accf else None)


def moe_model(dtype: torch.dtype = torch.float32) -> nn.Module:
    model = build("qwen3_moe", dtype)
    moe.unfuse_experts(model)
    return model


def make_batches(steps: int = 4, vocab: int = 128, rows: int = 3, length: int = 10) -> list[dict]:
    """Batches whose second row is right-padded by a different amount each step."""
    generator = torch.Generator().manual_seed(7)
    batches = []
    for step in range(steps):
        mask = torch.ones(rows, length, dtype=torch.long)
        mask[1, length - 2 - step :] = 0
        ids = torch.randint(0, vocab, (rows, length), generator=generator)
        batches.append({"input_ids": ids, "attention_mask": mask})
    return batches


def run(model: nn.Module, batches: list[dict] | None, cfg: CodeQuantConfig):
    """Returns (report, {key: (centroids, assignments)}) of `quantize_model`."""
    books = {}
    report = quantize_model(
        model, batches, cfg, lambda key, c, a: books.__setitem__(key, (c.clone(), a.clone()))
    )
    return report, books


def weight_of(model: nn.Module, key: tuple) -> torch.Tensor:
    site, layer = key[0], key[1]
    block = model.model.layers[layer]
    name = site.removesuffix(".weight")
    if name.startswith("linear.attn."):
        return getattr(block.self_attn, name.split(".")[-1]).weight
    if name == "moe.router":
        return block.mlp.gate.weight
    if len(key) == 2:
        return getattr(block.mlp, name.split(".")[-1]).weight
    return getattr(block.mlp.experts.experts[key[2]], name.split(".")[-1]).weight


# batch-major reference: the structure of the official ClusterLinear / ClusterMoE loop, on the same model


class RefLinear(nn.Module):
    """Official `ClusterLinear`: calibration mode caches full-precision outputs, training mode steps."""

    def __init__(self, module: nn.Module, kind: str, lr: float, key: tuple):
        super().__init__()
        self.module, self.kind, self.key = module, kind, key
        self.original = module.weight
        self.weight = module.weight.detach().clone()
        shape = self.weight.shape
        mask = torch.ones_like(self.weight, dtype=torch.bool)
        centroids, assignments = codebook.kmeans(self.weight, mask, CENTROIDS, 300, 1e-6, 42)
        self.centroids = nn.Parameter(centroids)
        self.assignments = assignments.long()
        self.cluster_weight = nn.Parameter(centroids.gather(1, self.assignments).reshape(shape).clone())
        self.optimizer = torch.optim.SGD([self.centroids], lr=lr)
        self.calibration_mode = True
        self.calibration_output = None
        self.cached_input = None

    def forward(self, x):
        if self.calibration_mode:
            out = self.module(x)
            if self.kind == "sa":
                self.calibration_output = out.detach()
            return out
        self.cached_input = x.detach()
        self.module.weight = self.cluster_weight
        out = self.module(x)
        self.module.weight = self.original
        if self.kind == "sa":
            if self.cluster_weight.grad is not None:
                self.cluster_weight.grad.zero_()
            F.mse_loss(out, self.calibration_output).backward()
            return out.detach()
        return out

    def fine_tune(self):
        if self.cluster_weight.grad is None:
            return
        rows, columns = self.weight.shape
        grad = self.cluster_weight.grad
        index = torch.arange(rows).unsqueeze(1) * CENTROIDS + self.assignments
        sums = torch.zeros(rows * CENTROIDS).scatter_add_(0, index.reshape(-1), grad.reshape(-1))
        counts = torch.zeros(rows * CENTROIDS).scatter_add_(0, index.reshape(-1), torch.ones(rows * columns))
        counts[counts == 0] = 1.0
        self.centroids.grad = (sums / counts).view(rows, CENTROIDS)
        self.optimizer.step()
        self.optimizer.zero_grad()
        x = self.cached_input.reshape(-1, columns).float()
        a = (x * x).mean(dim=0).clamp_min(1e-6)
        score = (self.weight * a).unsqueeze(-1) - a.unsqueeze(0).unsqueeze(
            -1
        ) * self.centroids.detach().unsqueeze(1)
        self.assignments = score.pow(2).argmin(dim=2)
        self.cluster_weight.data.copy_(self.centroids.detach().gather(1, self.assignments))


class RefMoE(nn.Module):
    """Official `ClusterMoE`: MSE of the block output plus KL of the router logits, output detached."""

    def __init__(self, block: nn.Module, router: RefLinear, members: list[RefLinear]):
        super().__init__()
        self.block, self.router, self.members = block, router, members
        self.calibration_mode = True

    def forward(self, x):
        out = self.block(x)
        logits = self.router.last_logits
        if self.calibration_mode:
            self.target = (out.detach(), logits.detach())
            return out
        hidden_target, router_target = self.target
        loss = F.mse_loss(out, hidden_target) + F.kl_div(
            F.log_softmax(logits, dim=-1),
            F.log_softmax(router_target, dim=-1),
            reduction="batchmean",
            log_target=True,
        )
        for member in self.members:
            if member.cluster_weight.grad is not None:
                member.cluster_weight.grad.zero_()
        loss.backward()
        return out.detach()


def reference_run(model: nn.Module, batches: list[dict], lr: float = LR):
    """Whole-model passes per step, centroids stepped after each; returns ({key: RefLinear}, model)."""
    for p in model.parameters():
        p.requires_grad_(False)
    linears: dict[tuple, RefLinear] = {}
    for index, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj"):
            key = (f"linear.attn.{name}.weight", index)
            ref = linears[key] = RefLinear(getattr(layer.self_attn, name), "sa", lr, key)
            setattr(layer.self_attn, name, ref)
        mlp = layer.mlp
        if hasattr(mlp, "experts"):
            members = []
            key = ("moe.router.weight", index)
            router = linears[key] = RefLinear(mlp.gate, "ffn", lr, key)
            members.append(router)
            router.register_forward_hook(lambda m, _a, out: setattr(m, "last_logits", out[0]))
            mlp.gate = router
            for e, expert in enumerate(mlp.experts.experts):
                for name in ("gate_proj", "up_proj"):
                    key = (f"linear.mlp.experts.{name}.weight", index, e)
                    ref = linears[key] = RefLinear(getattr(expert, name), "ffn", lr, key)
                    members.append(ref)
                    setattr(expert, name, ref)
            layer.mlp = RefMoE(mlp, router, members)
        else:
            for name in ("gate_proj", "up_proj"):
                key = (f"linear.mlp.{name}.weight", index)
                ref = linears[key] = RefLinear(getattr(mlp, name), "sa", lr, key)
                setattr(mlp, name, ref)
    modules = [m for m in model.modules() if isinstance(m, (RefLinear, RefMoE))]
    for batch in batches:
        for m in modules:
            m.calibration_mode = True
        with torch.no_grad():
            model(**batch, use_cache=False)
        for m in modules:
            m.calibration_mode = False
        model(**batch, use_cache=False)
        for ref in linears.values():
            ref.fine_tune()
    return linears


def assert_same_state(books, model, linears):
    assert set(books) == set(linears)
    for key, ref in linears.items():
        centroids, assignments = books[key]
        assert torch.equal(centroids, ref.centroids.detach()), f"centroids {key}"
        assert torch.equal(assignments.long(), ref.assignments), f"assignments {key}"
        assert torch.equal(weight_of(model, key).data, ref.cluster_weight.detach()), f"weight {key}"


def test_layer_major_accf_equals_the_batch_major_loop_after_every_step():
    batches = make_batches()
    for steps in range(1, len(batches) + 1):
        model = moe_model()
        reference = copy.deepcopy(model)
        _, books = run(model, batches[:steps], make_config())
        linears = reference_run(reference, batches[:steps])
        assert_same_state(books, model, linears)


def test_accf_moves_the_centroids_and_assignments_of_every_kind_of_weight():
    batches = make_batches()
    kmeans_model, trained = moe_model(), moe_model()
    _, initial = run(kmeans_model, None, make_config(accf=False))
    _, final = run(trained, batches, make_config())
    moved = {key[0] for key in final if not torch.equal(final[key][0], initial[key][0])}
    reassigned = {key[0] for key in final if not torch.equal(final[key][1], initial[key][1])}
    assert moved == set(ATTENTION_SITES + MOE_SITES)
    assert reassigned == set(ATTENTION_SITES + MOE_SITES)


def test_kl_term_changes_the_router_and_experts():
    batches = make_batches()
    with_kl, without = moe_model(), moe_model()
    _, books_kl = run(with_kl, batches, make_config())
    _, books_no = run(without, batches, make_config(kl_weight=0.0))
    router = [key for key in books_kl if key[0] == "moe.router.weight"]
    assert all(not torch.equal(books_kl[key][0], books_no[key][0]) for key in router)
    attention = [key for key in books_kl if key[0] in ATTENTION_SITES]
    assert all(torch.equal(books_kl[key][0], books_no[key][0]) for key in attention if key[1] == 0)


def test_the_first_centroid_step_scales_linearly_with_the_learning_rate():
    batches = make_batches(steps=1)
    model = moe_model()
    _, books = run(model, batches, make_config(sites=["linear.attn.q_proj.weight"], layers=0, lr=LR))
    half = moe_model()
    _, books_half = run(half, batches, make_config(sites=["linear.attn.q_proj.weight"], layers=0, lr=LR / 2))
    key = ("linear.attn.q_proj.weight", 0)
    start = codebook.kmeans(
        *codebook.to_units(moe_model().model.layers[0].self_attn.q_proj.weight, None), CENTROIDS
    )[0]
    full_step = books[key][0] - start
    assert full_step.abs().max() > 0
    assert torch.allclose((books_half[key][0] - start) * 2, full_step, atol=1e-5)


def test_dense_model_trains_every_clustered_linear_with_the_local_loss():
    batches = make_batches(vocab=97)
    sites = [*ATTENTION_SITES, "linear.mlp.gate_proj.weight", "linear.mlp.up_proj.weight"]
    model = build_tiny_qwen3(head_dim=16, tie=False, seed=3)
    reference = copy.deepcopy(model)
    _, books = run(model, batches, make_config(sites=sites))
    linears = reference_run(reference, batches)
    assert_same_state(books, model, linears)


def test_route_gradients_false_keeps_k_means_for_the_router_and_experts_only():
    batches = make_batches()
    model, base = moe_model(), moe_model()
    _, books = run(model, batches, make_config(route_gradients=False))
    _, initial = run(base, None, make_config(accf=False))
    for key in books:
        same = torch.equal(books[key][0], initial[key][0]) and torch.equal(books[key][1], initial[key][1])
        assert same == (key[0] in MOE_SITES), key


def test_layers_after_the_last_clustered_one_are_left_alone():
    batches = make_batches(steps=2)
    model, plain = moe_model(), moe_model()
    _, books = run(model, batches, make_config(layers=0))
    assert {key[1] for key in books} == {0}
    assert torch.equal(
        model.model.layers[1].self_attn.q_proj.weight, plain.model.layers[1].self_attn.q_proj.weight
    )
    assert not torch.equal(
        model.model.layers[0].self_attn.q_proj.weight, plain.model.layers[0].self_attn.q_proj.weight
    )


def test_kmeans_only_equals_fake_quantize_and_attaches_the_activation_rules():
    model, plain = moe_model(), moe_model()
    activation = QuantRule(sites=["linear.attn.q_proj.input", "moe.router.input"], quant=A4)
    report, books = run(model, None, make_config(accf=False, extra_rules=(activation,)))
    fmt = codebook_format()
    for key in books:
        expected = fake_quantize(weight_of(plain, key).data, fmt, kind="weight")
        assert torch.equal(weight_of(model, key).data, expected), key
    hooks = sum(len(m._forward_pre_hooks) for m in model.modules())
    assert hooks == 2 * LAYERS
    assert report["layers"] == []


def test_the_report_gives_effective_bits_clustered_fraction_and_the_drift_per_layer():
    batches = make_batches(steps=2)
    model = moe_model()
    total = sum(p.numel() for p in model.parameters())
    report, books = run(model, batches, make_config())
    assert report["total_parameters"] == total
    clustered = sum(math.prod(weight_of(model, key).shape) for key in books)
    assert report["clustered_parameters"] == clustered
    assert report["clustered_fraction"] == pytest.approx(clustered / total)
    assert report["effective_bits"] == pytest.approx(math.log2(CENTROIDS) + 16 * CENTROIDS / HIDDEN)
    assert set(report["effective_bits_by_site"]) == set(ATTENTION_SITES + MOE_SITES)
    assert [layer["layer"] for layer in report["layers"]] == list(range(LAYERS))
    for layer in report["layers"]:
        assert layer["max_relative_centroid_change"] > 0
        assert 0 < layer["changed_centroid_fraction"] <= 1
        assert 0 < layer["reassigned_fraction"] <= 1
    assert len(books) == LAYERS * (3 + 1 + 2 * EXPERTS)


def test_config_accepts_codebook_weight_rules_with_activation_rules_and_rejects_other_weight_formats():
    activation = QuantRule(sites=["linear.attn.q_proj.input"], quant=A4)
    config = make_config(extra_rules=(activation,))
    assert [rule.sites for rule in config.other_rules] == [["linear.attn.q_proj.input"]]
    integer = IntFormat(type="int", bits=4, symmetric=True, granularity="per_channel")
    with pytest.raises(ValidationError, match="use a codebook format"):
        CodeQuantConfig(rules=[QuantRule(sites=["linear.attn.q_proj.weight"], quant=integer)])
    with pytest.raises(ValidationError, match="at least one rule"):
        CodeQuantConfig(rules=[activation])
    with pytest.raises(ValidationError, match="all weights or all non-weights"):
        CodeQuantConfig(
            rules=[QuantRule(sites=["linear.attn.q_proj.weight", "linear.attn.q_proj.input"], quant=A4)]
        )


def test_the_method_builds_its_batches_from_text_batches_and_returns_the_model(monkeypatch):
    batches = make_batches(steps=2)
    epochs = [[BatchEncoding(batch) for batch in batches]]
    seen = {}

    def fake_text_batches(cfg, tokenizer):
        seen["args"] = (cfg, tokenizer)
        return epochs

    monkeypatch.setattr(codequant, "text_batches", fake_text_batches)
    model, reference, untouched = moe_model(), moe_model(), moe_model()
    cfg = make_config()
    assert codequant.codequant(model, "tokenizer", cfg) is model
    assert seen["args"] == (cfg.accf.calibration, "tokenizer")
    run(reference, batches, cfg)
    router = [
        m.mlp.gate.weight
        for m in (model.model.layers[1], reference.model.layers[1], untouched.model.layers[1])
    ]
    assert torch.equal(router[0], router[1])
    assert not torch.equal(router[0], router[2])


# the official modules on a transformers 4.57-style tree


class OfficialBlock(nn.Module):
    """`Qwen3MoeSparseMoeBlock` of transformers 4.57: nn.Linear router, returns (hidden, router_logits)."""

    def __init__(self, block: nn.Module):
        super().__init__()
        router = block.gate
        self.top_k, self.num_experts, self.norm_topk_prob = (
            router.top_k,
            router.num_experts,
            router.norm_topk_prob,
        )
        self.gate = nn.Linear(router.hidden_dim, router.num_experts, bias=False)
        self.gate.weight = nn.Parameter(router.weight.detach().clone())
        self.experts = nn.ModuleList(block.experts.experts)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


class FirstOutput(nn.Module):
    """Lets a 5.x decoder layer call a module that returns (hidden, router_logits)."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)[0]


def official_run(model: nn.Module, batches: list[dict], lr: float = LR):
    """The loop of the official `clustering` on `model` with the official ClusterLinear and ClusterMoE."""
    linear_module = _official("cq1", "cluster/ClusterLinear.py")
    moe_module = _official("cq1", "cluster/ClusterMoE.py")
    for p in model.parameters():
        p.requires_grad_(False)
    clusters: dict[tuple, nn.Module] = {}

    def cluster(module: nn.Module, kind: str, key: tuple):
        new = linear_module.ClusterLinear(
            weight=module.weight.detach(), bias=None, group_size=-1, cluster_num=CENTROIDS,
            calibration_mode=True, module_type=kind, layer_name=str(key), lr=lr,
        ).to(module.weight.device).to(module.weight.dtype)  # fmt: skip
        clusters[key] = new
        return new

    wrapped = []
    for index, layer in enumerate(model.model.layers):
        for name in ("q_proj", "k_proj", "v_proj"):
            setattr(
                layer.self_attn,
                name,
                cluster(getattr(layer.self_attn, name), "sa", (f"linear.attn.{name}.weight", index)),
            )
        block = OfficialBlock(layer.mlp)
        block.gate = cluster(block.gate, "ffn", ("moe.router.weight", index))
        for e, expert in enumerate(block.experts):
            for name in ("gate_proj", "up_proj"):
                key = (f"linear.mlp.experts.{name}.weight", index, e)
                setattr(expert, name, cluster(getattr(expert, name), "ffn", key))
        wrapper = moe_module.ClusterMoE(block, calibration_mode=True, layer_name=f"layer{index}")
        wrapped.append(wrapper)
        layer.mlp = FirstOutput(wrapper)
    for name, param in model.named_parameters():
        param.requires_grad_(name.endswith(("cluster_weight", "centroids")))
    for batch in batches:
        for m in [*clusters.values(), *wrapped]:
            m.set_calibration_mode(True)
        with torch.no_grad():
            model(**batch, use_cache=False)
        for m in [*clusters.values(), *wrapped]:
            m.set_calibration_mode(False)
        model(**batch, use_cache=False)
        for lin in clusters.values():
            lin.fine_tune_centroids()
    return clusters


def test_accf_equals_the_official_cluster_modules_after_every_step():
    batches = make_batches()
    for steps in range(1, len(batches) + 1):
        model = moe_model()
        reference = copy.deepcopy(model)
        _, books = run(model, batches[:steps], make_config())
        clusters = official_run(reference, batches[:steps])
        assert set(books) == set(clusters)
        for key, official in clusters.items():
            centroids, assignments = books[key]
            assert torch.equal(centroids, official.centroids.detach()), f"centroids {key} after {steps} steps"
            assert torch.equal(assignments, official.assignments), f"assignments {key} after {steps} steps"
            assert torch.equal(weight_of(model, key).data, official.cluster_weight.detach()), f"weight {key}"


def test_accf_equals_the_official_cluster_modules_in_bfloat16():
    """The released script builds each ClusterLinear in float32, so k-means runs there, then casts to bf16."""
    batches = make_batches()
    for steps in (1, 2, 4):
        model = moe_model(torch.bfloat16)
        reference = copy.deepcopy(model)
        _, books = run(model, batches[:steps], make_config(compute_dtype="float32"))
        clusters = official_run(reference, batches[:steps])
        assert set(books) == set(clusters)
        for key, official in clusters.items():
            centroids, assignments = books[key]
            assert centroids.dtype == official.centroids.dtype == torch.bfloat16
            assert torch.equal(centroids, official.centroids.detach()), f"centroids {key} after {steps} steps"
            assert torch.equal(assignments, official.assignments), f"assignments {key} after {steps} steps"
            assert torch.equal(weight_of(model, key).data, official.cluster_weight.detach()), f"weight {key}"


def test_bfloat16_model_trains_in_the_model_dtype_or_with_float32_master_centroids():
    batches = make_batches(steps=2)
    model, master = moe_model(torch.bfloat16), moe_model(torch.bfloat16)
    _, books = run(model, batches, make_config(compute_dtype="float32"))
    _, books_master = run(master, batches, make_config(compute_dtype="float32", train_dtype="float32"))
    assert {c.dtype for c, _ in books.values()} == {torch.bfloat16}
    assert {c.dtype for c, _ in books_master.values()} == {torch.float32}
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert master.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    assert not all(torch.equal(books[k][0].float(), books_master[k][0]) for k in books)
    assert all(torch.isfinite(c).all() for c, _ in books_master.values())
