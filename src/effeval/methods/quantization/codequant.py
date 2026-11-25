"""CodeQuant (arXiv 2604.10496): per-row k-means codebooks for weights, then ACCF centroid fine-tuning.

Ported from github.com/SAI-Lab-NYU/CodeQuant@c149bc94: cluster/ClusterLinear.py (local loss, centroid step,
recluster), cluster/ClusterMoE.py (MoE loss), script/cluster_fine_tune_script.py (training loop),
utils/cluster_utils.py (k-means).

ACCF runs layer by layer over `ops.stream`, which reproduces the official batch-major schedule: every step
sees the layer inputs produced by the already fine-tuned earlier layers (quantized path) next to the
full-precision path whose outputs are the targets. Attention linears (and any linear outside a MoE block)
minimize `mse(x_q W_c, x_fp W)` locally and return their output detached; a MoE block minimizes
`mse(hidden) + kl_weight * KL(router logits)` against its full-precision output, with gradients into its
cluster weights and router. Each step then moves every centroid by SGD along the mean gradient of its
members and re-assigns every weight by the official diagonal rule. Activation rules are attached afterwards.

Shape symbols: O/D rows/columns of a weight, G codebook units, S values per unit, K centroids per unit.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from pydantic import Field, model_validator
from torch import nn

from effeval.core.fragments import CodebookFormat, QuantRule, TextBatches
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.ops import codebook, sites
from effeval.ops.calibration import text_batches
from effeval.ops.layerwise import _layer_output
from effeval.ops.stream import move_to, run_streams

logger = logging.getLogger(__name__)


class ACCFConfig(StrictModel):
    calibration: TextBatches
    lr: float = Field(gt=0, description="Learning rate of the plain SGD step on the centroids.")
    kl_weight: float = Field(
        default=1.0, ge=0, description="Weight of the router-logit KL term next to the MoE output MSE."
    )
    train_dtype: Literal["model", "float32"] = Field(
        default="model",
        description="Dtype of the centroids and of the re-assignment after the k-means fit: the model's own "
        "dtype as released, or float32 master centroids.",
    )
    train_dense_ffn: bool = Field(
        default=False,
        description="Also fine-tune clustered linears that are neither attention nor inside a MoE block "
        "(dense MLP gate/up) with the local output MSE; the released code keeps them at k-means values.",
    )
    route_gradients: bool = Field(
        default=True,
        description="When false the router and the routed experts keep their k-means centroids (the released "
        "DeepSeek path gives them no gradient); everything else is still fine-tuned.",
    )


class CodeQuantConfig(StrictModel):
    rules: list[QuantRule] = Field(
        min_length=1,
        description="Weight rules (codebook format) are clustered; rules on other sites are attached after.",
    )
    accf: ACCFConfig | None = Field(
        default=None, description="Centroid fine-tuning; null keeps the k-means centroids."
    )

    @model_validator(mode="after")
    def _weights_are_codebooks(self) -> CodeQuantConfig:
        for index, rule in enumerate(self.rules):
            on_weights = [pattern.endswith(".weight") for pattern in rule.sites]
            if any(on_weights) and not all(on_weights):
                raise ValueError(f"rules[{index}]: a rule's sites must be all weights or all non-weights")
            if all(on_weights) and not isinstance(rule.quant, CodebookFormat):
                raise ValueError(f"rules[{index}]: CodeQuant clusters weights; use a codebook format")
        if not self.weight_rules:
            raise ValueError("CodeQuant needs at least one rule on '.weight' sites")
        return self

    @property
    def weight_rules(self) -> list[QuantRule]:
        return [rule for rule in self.rules if rule.sites[0].endswith(".weight")]

    @property
    def other_rules(self) -> list[QuantRule]:
        return [rule for rule in self.rules if not rule.sites[0].endswith(".weight")]


class _Cluster:
    """Codebook state of one weight: k-means init, and during ACCF the trainable weight W_c and its step."""

    def __init__(self, key: tuple, module: nn.Module, fmt: CodebookFormat, train_dtype: str = "model"):
        self.key = key
        self.site = key[0]
        self.layer = key[1]
        self.module = module
        self.fmt = fmt
        self.original: nn.Parameter = module.weight
        self.shape = tuple(self.original.shape)
        self.group_size = fmt.group_size if fmt.granularity == "per_group" else None
        weight = self.original.data
        work = weight.float() if fmt.kmeans.compute_dtype == "float32" else weight
        units, self.mask = codebook.to_units(work, self.group_size)
        centroids, self.assignments = codebook.kmeans(
            units, self.mask, fmt.centroids, fmt.kmeans.iters, fmt.kmeans.tol, fmt.kmeans.seed
        )
        dtype = weight.dtype if train_dtype == "model" else torch.float32
        self.units = units.to(dtype)
        self.centroids = nn.Parameter(centroids.to(dtype))
        self.initial_centroids = self.centroids.detach().clone()
        self.initial_assignments = self.assignments.clone()
        self.parameter: nn.Parameter | None = None
        self.trainable = False
        self.in_moe = False
        self.stats: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def unit_size(self) -> int:
        columns = self.shape[1]
        return columns if self.group_size is None or self.group_size >= columns else self.group_size

    def dequantized(self) -> torch.Tensor:
        """The [O, D] weight of every element's centroid, in the weight's dtype."""
        values = codebook.gather(self.centroids.detach(), self.assignments)
        return codebook.from_units(values, self.group_size, self.shape).to(self.original.dtype)

    def start_training(self, lr: float, trainable: bool) -> None:
        """Create the weight W_c used during ACCF; a trainable one also gets its SGD optimizer."""
        self.parameter = nn.Parameter(self.dequantized(), requires_grad=trainable)
        self.trainable = trainable
        if trainable:
            self.optimizer = torch.optim.SGD([self.centroids], lr=lr)

    def update(self) -> None:
        """One ACCF step: SGD on the centroids with the mean member gradient, then re-assignment."""
        gradient = self.parameter.grad
        if gradient is None:
            return
        gradient_units, _ = codebook.to_units(gradient.to(self.centroids.dtype), self.group_size)
        mean = codebook.scatter_mean_grad(gradient_units, self.mask, self.assignments, self.fmt.centroids)
        self.centroids.grad = mean.to(self.centroids.dtype)
        self.optimizer.step()
        self.optimizer.zero_grad()
        a, b = self.stats
        self.assignments = codebook.recluster_diag(
            self.units, self.mask, self.centroids.detach(), a, b, self.group_size, self.shape
        )
        self.parameter.data.copy_(self.dequantized())
        self.parameter.grad = None

    def drift(self) -> dict[str, float]:
        """Change of the centroids and assignments since k-means: max relative change, changed fractions."""
        before = self.initial_centroids.float()
        delta = (self.centroids.detach().float() - before).abs()
        relative = delta / before.abs().clamp_min(torch.finfo(torch.float32).tiny)
        return {
            "max_relative_centroid_change": relative.max().item(),
            "changed_centroids": (delta > 0).sum().item(),
            "centroids": delta.numel(),
            "reassigned": (self.assignments != self.initial_assignments).sum().item(),
            "assignments": self.assignments.numel(),
        }


ROUTER_CLASSES = ("Qwen3MoeTopKRouter", "DeepseekV2TopkRouter")
"""Router classes whose forward returns `(raw logits, weights, indices)` (transformers 5.23)."""


class _LayerTrainer:
    """Hooks and loop of ACCF for the clustered weights of one decoder layer."""

    def __init__(self, layer: nn.Module, clusters: list[_Cluster], accf: ACCFConfig, moe) -> None:
        self.layer = layer
        self.clusters = clusters
        self.accf = accf
        self.mode = "fp"
        self.targets: dict[object, torch.Tensor] = {}
        self.router_logits: torch.Tensor | None = None
        self.handles: list = []
        local = [c for c in clusters if c.trainable and not c.in_moe]
        in_block = [c for c in clusters if c.trainable and c.in_moe]
        for cluster in clusters:
            if cluster.trainable:
                self.handles.append(cluster.module.register_forward_pre_hook(self._stats_hook(cluster)))
        for cluster in local:
            self.handles.append(cluster.module.register_forward_hook(self._local_hook(cluster)))
        if in_block:
            router = layer.get_submodule(moe.router)
            if type(router).__name__ not in ROUTER_CLASSES:
                raise ValueError(
                    f"router class {type(router).__name__} is not known to return raw logits first; "
                    f"verified: {ROUTER_CLASSES}"
                )
            self.handles.append(router.register_forward_hook(self._router_hook))
            self.handles.append(
                layer.get_submodule(moe.block).register_forward_hook(self._block_hook(in_block))
            )

    def _stats_hook(self, cluster: _Cluster):
        def hook(_module: nn.Module, args: tuple) -> None:
            if self.mode == "q":
                x = args[0].detach()
                cluster.stats = codebook.input_diag(x, x)

        return hook

    def _local_hook(self, cluster: _Cluster):
        def hook(_module: nn.Module, _args: tuple, output: torch.Tensor):
            if self.mode == "fp":
                self.targets[cluster] = output.detach()
                return None
            cluster.parameter.grad = None
            F.mse_loss(output, self.targets.pop(cluster)).backward()
            return output.detach()

        return hook

    def _router_hook(self, _module: nn.Module, _args: tuple, output: tuple) -> None:
        self.router_logits = output[0]

    def _block_hook(self, in_block: list[_Cluster]):
        def hook(_module: nn.Module, _args: tuple, output: torch.Tensor):
            if self.mode == "fp":
                self.targets["hidden"] = output.detach()
                self.targets["router"] = self.router_logits.detach()
                return None
            loss_hidden = F.mse_loss(output, self.targets.pop("hidden"))
            loss_router = F.kl_div(
                F.log_softmax(self.router_logits, dim=-1),
                F.log_softmax(self.targets.pop("router"), dim=-1),
                reduction="batchmean",
                log_target=True,
            )
            for cluster in in_block:
                cluster.parameter.grad = None
            (loss_hidden + self.accf.kl_weight * loss_router).backward()
            return output.detach()

        return hook

    def _use(self, mode: str) -> None:
        for cluster in self.clusters:
            cluster.module.weight = cluster.original if mode == "fp" else cluster.parameter

    def run(
        self,
        fp_in: list[torch.Tensor],
        q_in: list[torch.Tensor],
        kwargs: list[dict],
        step_order: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Full-precision (per unique batch) and quantized-path (per step) outputs; fine-tunes the clusters.

        Full-precision targets of a batch are computed at its first step and reused by later steps.
        """
        device = next(self.layer.parameters()).device
        frozen = [p for p in self.layer.parameters() if p.requires_grad]
        for p in frozen:
            p.requires_grad_(False)
        fp_out: list[torch.Tensor | None] = [None] * len(fp_in)
        q_out = []
        cache: dict[int, dict] = {}
        try:
            for step, batch in enumerate(step_order):
                step_kwargs = move_to(kwargs[step], device)
                if batch not in cache:
                    self.mode = "fp"
                    self._use("fp")
                    with torch.no_grad():
                        out = _layer_output(self.layer, fp_in[batch].to(device), step_kwargs)
                    fp_out[batch] = move_to(out, "cpu")
                    cache[batch] = move_to(self.targets, "cpu")
                self.targets = move_to(cache[batch], device)
                self.mode = "q"
                self._use("q")
                with torch.enable_grad():
                    out = _layer_output(self.layer, q_in[step].to(device), step_kwargs)
                q_out.append(move_to(out.detach(), "cpu"))
                for cluster in self.clusters:
                    if cluster.trainable:
                        cluster.update()
                self.targets = {}
        finally:
            for handle in self.handles:
                handle.remove()
            for p in frozen:
                p.requires_grad_(True)
            self.targets = {}
        return fp_out, q_out


Entry = tuple[tuple, nn.Module, CodebookFormat]
"""A weight to cluster: its site key `(site, layer[, expert])`, module and format."""

Sink = Callable[[tuple, torch.Tensor, torch.Tensor], None]
"""Receives `(key, centroids [G, K], assignments [G, S] uint8)` of a finished weight."""


def _entries_by_layer(model: nn.Module, rules: list[QuantRule]) -> dict[int, list[Entry]]:
    assignment, _ = sites.resolve(model, rules)
    by_layer: dict[int, list[Entry]] = defaultdict(list)
    for key, (_rule, location, fmt) in assignment.items():
        if key[1] is None:
            raise ValueError("CodeQuant clusters decoder-layer weights; lm_head is not supported")
        by_layer[key[1]].append((key, location.module, fmt))
    return dict(sorted(by_layer.items()))


def _layer_clusters(model: nn.Module, index: int, entries: list[Entry], accf: ACCFConfig | None):
    """K-means state of one layer's weights; with `accf`, also their `in_moe` flag and training state."""
    clusters = [
        _Cluster(key, module, fmt, "model" if accf is None else accf.train_dtype)
        for key, module, fmt in entries
    ]
    if accf is None:
        return clusters
    moe = sites.architecture(model).moe
    in_block: set[int] = set()
    routed: set[int] = set()
    if moe is not None and index in moe.layers:
        layer = sites.decoder_layers(model)[index]
        in_block = {id(m) for m in layer.get_submodule(moe.block).modules()}
        routed = {id(m) for m in layer.get_submodule(moe.experts).modules()}
        routed.add(id(layer.get_submodule(moe.router)))
    for cluster in clusters:
        cluster.in_moe = id(cluster.module) in in_block
        if cluster.in_moe:
            trainable = accf.route_gradients or id(cluster.module) not in routed
        else:
            trainable = cluster.site.startswith("linear.attn.") or accf.train_dense_ffn
        cluster.start_training(accf.lr, trainable)
    return clusters


def _finish_layer(clusters: list[_Cluster], sink: Sink | None) -> dict:
    """Write each weight's cluster weight into its module, hand the codebooks to `sink`, summarize."""
    summary: dict = {
        "weights": [
            {
                "site": c.site,
                "parameters": math.prod(c.shape),
                "bits": math.log2(c.fmt.centroids) + 16 * c.fmt.centroids / c.unit_size,
            }
            for c in clusters
        ]
    }
    drift = [c.drift() for c in clusters if c.trainable]
    if drift:
        summary["max_relative_centroid_change"] = max(d["max_relative_centroid_change"] for d in drift)
        summary["changed_centroid_fraction"] = sum(d["changed_centroids"] for d in drift) / sum(
            d["centroids"] for d in drift
        )
        summary["reassigned_fraction"] = sum(d["reassigned"] for d in drift) / sum(
            d["assignments"] for d in drift
        )
    for cluster in clusters:
        final = cluster.dequantized() if cluster.parameter is None else cluster.parameter
        cluster.original.copy_(final)
        cluster.module.weight = cluster.original
        if sink is not None:
            sink(cluster.key, cluster.centroids.detach(), cluster.assignments)
    return summary


def _report(layers: dict[int, dict], total_parameters: int) -> dict:
    weights = [w for summary in layers.values() for w in summary["weights"]]
    clustered = sum(w["parameters"] for w in weights)
    return {
        "clustered_parameters": clustered,
        "total_parameters": total_parameters,
        "clustered_fraction": clustered / total_parameters,
        "effective_bits": sum(w["bits"] * w["parameters"] for w in weights) / clustered,
        "effective_bits_by_site": {w["site"]: w["bits"] for w in weights},
        "layers": [
            {"layer": index, **{k: v for k, v in summary.items() if k != "weights"}}
            for index, summary in layers.items()
            if "reassigned_fraction" in summary
        ],
    }


def _unique_batches(batches: list[dict]) -> tuple[list[dict], list[int]]:
    """The distinct batches (by content) in first-seen order and, per step, the index of its batch."""
    unique: list[dict] = []
    index: dict[tuple, int] = {}
    order = []
    for batch in batches:
        key = tuple((name, tuple(t.shape), t.cpu().numpy().tobytes()) for name, t in sorted(batch.items()))
        if key not in index:
            index[key] = len(unique)
            unique.append(batch)
        order.append(index[key])
    return unique, order


@torch.no_grad()
def quantize_model(
    model: nn.Module, batches: list[dict] | None, cfg: CodeQuantConfig, sink: Sink | None = None
) -> dict:
    """Cluster the weights of `cfg.rules`, fine-tune the centroids on `batches`, attach the other rules.

    `batches` are the model inputs of every ACCF step in order (all epochs flattened); None when
    `cfg.accf` is null. Weights are replaced by their cluster weights in place, and `sink` receives every
    weight's final codebook. Returns the report: effective bits per weight, the clustered fraction of the
    model's parameters and, per layer, how far ACCF moved the centroids and assignments away from k-means.
    """
    total_parameters = sum(p.numel() for p in model.parameters())
    entries = _entries_by_layer(model, cfg.weight_rules)
    summaries: dict[int, dict] = {}
    if cfg.accf is None:
        for index, layer_entries in entries.items():
            summaries[index] = _finish_layer(_layer_clusters(model, index, layer_entries, None), sink)
    else:
        if not batches:
            raise ValueError("ACCF needs calibration batches")
        moe = sites.architecture(model).moe
        last = max(entries)
        unique, order = _unique_batches(batches)

        def layer_fn(index, layer, fp_in, q_in, kwargs):
            if index > last:
                return fp_in, q_in
            clusters = _layer_clusters(model, index, entries.get(index, []), cfg.accf)
            fp_out, q_out = _LayerTrainer(layer, clusters, cfg.accf, moe).run(fp_in, q_in, kwargs, order)
            if clusters:
                summaries[index] = _finish_layer(clusters, sink)
            return fp_out, q_out

        run_streams(model, unique, order, layer_fn)
    report = _report(summaries, total_parameters)
    logger.info(
        "codequant: %s", json.dumps({k: v for k, v in report.items() if k != "effective_bits_by_site"})
    )
    if cfg.other_rules:
        sites.attach(model, cfg.other_rules)
    return report


@method(
    "codequant",
    title="CodeQuant",
    version="1",
    field="quantization",
    config=CodeQuantConfig,
    uses=("quant_rules",),
    display=("rules", "accf"),
    default_preset="paper_a4w4_qwen3moe",
    description="Per-row 16-centroid k-means weights with ACCF centroid fine-tuning (ported from the "
    "released CodeQuant code).",
)
def codequant(model: nn.Module, tokenizer, cfg: CodeQuantConfig) -> nn.Module:
    batches = None
    if cfg.accf is not None:
        batches = [dict(batch) for epoch in text_batches(cfg.accf.calibration, tokenizer) for batch in epoch]
    quantize_model(model, batches, cfg)
    return model
