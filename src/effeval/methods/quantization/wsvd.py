"""WSVD: per-head whitened SVD of q/k/v, Fisher-weighted fine-tuning, QAT with a learned rotation, then RTN.

Ported from the released code (SAI-Lab-NYU/WSVD, `fake_quant/`): input-whitened per-head SVD of q, k and v,
gradient-ranked rank allocation, data-free Fisher-weighted fine-tuning of the factors, quantization-aware
fine-tuning in the rotated basis with a per-head Stiefel rotation of the rank dimension that is folded into
the factors, and RTN of the remaining weights. Text-only: the calibration loss is the causal-LM cross-entropy
of wikitext2 windows.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from effeval.core.fragments import Calibration, QuantRule
from effeval.core.lowrank import QKV_CANONS
from effeval.core.preprocess import Hadamard, RandomHadamard
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.methods.preprocess.rotation import rotation_matrix
from effeval.methods.quantization.rtn import check_clip_search
from effeval.ops import rotation as rot
from effeval.ops import sites
from effeval.ops.artifacts import artifact_key, default_cache, weights_fingerprint
from effeval.ops.calibration import act_stats, calibration_windows, causal_lm_grads
from effeval.ops.lowrank import (
    install_per_head_qkv,
    per_head_allocation,
    per_head_projected_gradient,
    per_head_whitened_svd,
)
from effeval.ops.wsvd import (
    TRAIN_DTYPE,
    FTSettings,
    QATSettings,
    cholesky_lower_f32,
    fisher_weight,
    init_factors,
    no_tf32,
    prequantize_dense,
    released_head_budget,
    train_projection,
)

logger = logging.getLogger(__name__)

INPUT_SITE = "linear.attn.k_proj.input"
INPUT_GRAM_NAME = "wsvd.input_gram:v1"
GRAD_NAME = "wsvd.grad_stats:v1"
FACTORS_NAME = "wsvd.factors:v1"
PROJECTIONS = [canon.split(".")[1] for canon in QKV_CANONS]
DENSE_CANONS = ("attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


class WSVDDecomposition(StrictModel):
    rank_ratio: float = Field(
        default=0.5,
        gt=0,
        description="Kept parameter ratio of each of q, k and v: per head slot "
        "int(rank_ratio * num_layers * numel / (hidden + out / heads) / heads) singular values, times the "
        "head count, ranked by importance across layers and heads.",
    )
    grad_alpha: float = Field(
        default=1.0,
        ge=0,
        description="Exponent of the singular value's gradient magnitude in its importance score.",
    )
    score_dtype: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="dtype of the gradient projection, its accumulation and the importance scores; the "
        "released code uses bfloat16.",
    )
    k_grad_normalisation: Literal["once", "twice"] = Field(
        default="twice",
        description="How often the k_proj gradient score is divided by the window count and square-rooted. "
        "twice reproduces the released code, which normalises k_proj in a separate step and again in the "
        "q/k/v loop; once is the single normalisation q and v get.",
    )
    empty_head_index: int = Field(
        default=1,
        ge=0,
        description="Singular-value index kept by a head that the allocation leaves empty; the released "
        "code keeps the second one (index 1).",
    )


class WSVDLocalFT(StrictModel):
    iters: int = Field(
        default=150, ge=0, description="Iterations of the data-free fine-tuning of each projection (0: none)."
    )
    lr: float = Field(default=1e-4, gt=0, description="Adam learning rate of the factors.")
    weighted: bool = Field(
        default=True, description="Weight the reconstruction loss elementwise by the Fisher weight."
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Calibration windows per fine-tuning batch; an iteration runs "
        "ceil(calibration.n_samples / batch_size) identical data-free steps.",
    )
    seed: int = Field(default=0, description="Seed of the rotation initialisation and the retraction draws.")


class WSVDQAT(StrictModel):
    enabled: bool = Field(
        default=False,
        description="Quantization-aware fine-tuning with STE fake quantization of the factors and, with "
        "dense_prequant, a first RTN of o_proj, gate, up and down.",
    )
    start_fraction: float = Field(
        default=0.66666667,
        ge=0,
        le=1,
        description="Fraction of local_ft.iters after which fine-tuning turns into QAT.",
    )
    lr_factors: float = Field(default=1e-5, gt=0, description="Adam learning rate of the factors during QAT.")
    lr_rotation: float = Field(
        default=1.0, gt=0, description="Stiefel SGD learning rate of the per-head rotation."
    )
    scale_update_every: int = Field(
        default=1, ge=1, description="Iterations between refreshes of the quantizer scales."
    )
    w_bits: int = Field(default=8, ge=2, le=8, description="Weight bit width of the fake quantization.")
    latent_rotation: Literal["learned", "none"] = Field(
        default="learned",
        description="learned: a per-head rotation of the rank dimension is learned and folded into the "
        "factors; none: no rotation.",
    )
    residual_rotation: Literal["random_hadamard", "hadamard", "none"] = Field(
        default="random_hadamard",
        description="Residual-stream rotation the QAT loss simulates; it must match the paired preprocess "
        "(random_hadamard: the preprocess's random Hadamard of residual_rotation_seed, hadamard: the "
        "fixed Hadamard, none: unrotated).",
    )
    residual_rotation_seed: int = Field(
        default=0, description="Seed of the random Hadamard that residual_rotation simulates."
    )
    value_hadamard: bool = Field(
        default=True, description="Apply the per-head Hadamard of the value path in the v_proj QAT loss."
    )
    dense_prequant: bool = Field(
        default=True,
        description="Quantize o_proj, gate, up and down once to the QAT weight bit width before the "
        "preprocess, as the released code does.",
    )


class WSVDConfig(StrictModel):
    rules: list[QuantRule] = Field(
        default_factory=list, description="Quantization rules applied to the factored model."
    )
    calibration: Calibration
    decomposition: WSVDDecomposition = Field(
        default_factory=WSVDDecomposition, description="Per-head SVD and rank allocation."
    )
    local_ft: WSVDLocalFT = Field(
        default_factory=WSVDLocalFT, description="Data-free fine-tuning of the factors."
    )
    qat: WSVDQAT = Field(default_factory=WSVDQAT, description="Quantization-aware fine-tuning.")

    @model_validator(mode="after")
    def _clip_search_on_weights(self) -> WSVDConfig:
        check_clip_search(self.rules)
        return self


def _device_name(device: torch.device) -> str:
    return "cpu" if device.type == "cpu" else torch.cuda.get_device_name(device)


def _linear_paths(layout) -> dict[str, str]:
    return {canon: path for path, canon in layout.linears.items()}


@torch.no_grad()
def decompose(model: nn.Module, tokenizer, cfg: WSVDConfig) -> None:
    """Factor every layer's q/k/v per head, train the factors (FT, then QAT), pre-quantize the dense weights.

    The input Gram, the gradient statistics and the trained factors come from the artifact cache or are
    computed on the calibration windows. With `qat.enabled and qat.dense_prequant`, o_proj, gate, up and down
    are quantized once per row to `qat.w_bits`. Raises ValueError for a Gram that is not positive definite,
    a projection that keeps no singular value in some layer, or an `empty_head_index` not below the head
    dimension.
    """
    d, qat = cfg.decomposition, cfg.qat
    head_dim = model.config.head_dim
    if d.empty_head_index >= head_dim:
        raise ValueError(f"empty_head_index {d.empty_head_index} is not below the head dimension {head_dim}")
    with no_tf32():
        _decompose(model, cfg)
    if qat.enabled and qat.dense_prequant:
        layout = sites.architecture(model)
        paths = _linear_paths(layout)
        for layer in sites.decoder_layers(model):
            for canon in DENSE_CANONS:
                linear = layer.get_submodule(paths[canon])
                linear.weight.data = prequantize_dense(linear.weight, qat.w_bits)


def _decompose(model: nn.Module, cfg: WSVDConfig) -> None:
    d, ft, qat = cfg.decomposition, cfg.local_ft, cfg.qat
    windows = calibration_windows(cfg.calibration, model.config._name_or_path)
    model_key = {
        "id": model.config._name_or_path,
        "dtype": str(next(model.parameters()).dtype),
        "weights": weights_fingerprint(model),
    }
    cache = default_cache()
    layout = sites.architecture(model)
    layers = sites.decoder_layers(model)
    layer_count = len(layers)
    attentions = [layer.get_submodule(layout.attention_path) for layer in layers]
    dense = [[getattr(attention, name) for name in PROJECTIONS] for attention in attentions]
    config = model.config
    head_counts = dict(
        zip(
            PROJECTIONS,
            [config.num_attention_heads, config.num_key_value_heads, config.num_key_value_heads],
            strict=True,
        )
    )
    head_dim = config.head_dim
    hidden = dense[0][0].in_features
    dtype = dense[0][0].weight.dtype
    devices = [projections[0].weight.device for projections in dense]
    steps = len(range(0, cfg.calibration.n_samples, ft.batch_size))
    score_dtype = getattr(torch, d.score_dtype)
    device_names = sorted({_device_name(device) for device in devices})
    report: dict[str, object] = {}

    def compute_factors() -> dict[str, torch.Tensor]:
        def compute_input_gram() -> dict[str, torch.Tensor]:
            grams = act_stats(model, windows, [INPUT_SITE], "gram")
            return {str(layer): value for (_site, layer), value in grams.items()}

        input_grams = cache.get_or_compute(
            artifact_key(
                model=model_key,
                calibration=cfg.calibration,
                statistic=INPUT_GRAM_NAME,
                upstream=[],
                downstream={"device": device_names},
            ),
            compute_input_gram,
        )
        chol = [
            cholesky_lower_f32(
                input_grams[str(layer)].to(devices[layer]),
                shift_fallback=True,
                label=f"wsvd layer {layer} qkv input",
            )
            for layer in range(layer_count)
        ]
        identity = {
            name: torch.eye(head_dim, device=devices[0]).expand(heads, head_dim, head_dim)
            for name, heads in head_counts.items()
        }
        svds: list[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
        for layer, projections in enumerate(dense):
            svds.append(
                {
                    name: per_head_whitened_svd(
                        proj.weight,
                        head_counts[name],
                        chol[layer][0],
                        identity[name].to(devices[layer]),
                    )
                    for name, proj in zip(PROJECTIONS, projections, strict=True)
                }
            )

        def compute_grad_stats() -> dict[str, torch.Tensor]:
            s_sum: dict[str, torch.Tensor] = {}
            f_sum: dict[str, torch.Tensor] = {}

            def accumulate(store: dict[str, torch.Tensor], key: str, value: torch.Tensor) -> None:
                if key in store:
                    store[key] += value
                else:
                    store[key] = value

            def on_batch() -> None:
                for layer, projections in enumerate(dense):
                    for name, proj in zip(PROJECTIONS, projections, strict=True):
                        U, _S, V = svds[layer][name]
                        heads = head_counts[name]
                        key = f"{layer}.{name}"
                        squared = (
                            per_head_projected_gradient(
                                U,
                                V,
                                proj.weight.grad,
                                chol[layer][1].T,
                                identity[name].to(devices[layer]),
                                score_dtype,
                            )
                            ** 2
                        )
                        accumulate(s_sum, key, squared)
                        grad = proj.weight.grad.detach().to(TRAIN_DTYPE).view(heads, -1, hidden)
                        accumulate(f_sum, key, grad.pow(2))

            params = [proj.weight for projections in dense for proj in projections]
            count = causal_lm_grads(model, windows, params, on_batch)
            out = {f"{key}.s_grad": fisher_weight(value, count) for key, value in s_sum.items()}
            out.update({f"{key}.fisher": fisher_weight(value, count) for key, value in f_sum.items()})
            return out

        stats = cache.get_or_compute(
            artifact_key(
                model=model_key,
                calibration=cfg.calibration,
                statistic=GRAD_NAME,
                upstream=[],
                downstream={"score_dtype": d.score_dtype, "device": device_names},
            ),
            compute_grad_stats,
        )
        s_grad = {
            f"{layer}.{name}": stats[f"{layer}.{name}.s_grad"]
            for layer in range(layer_count)
            for name in PROJECTIONS
        }
        if d.k_grad_normalisation == "twice":
            for layer in range(layer_count):
                key = f"{layer}.k_proj"
                s_grad[key] = s_grad[key].div(len(windows)).abs().sqrt()

        pools = [
            [
                svds[layer][name][1].to(score_dtype).abs()
                * s_grad[f"{layer}.{name}"].to(devices[layer]).to(score_dtype).abs() ** d.grad_alpha
                for layer in range(layer_count)
            ]
            for name in PROJECTIONS
        ]
        budgets = [
            released_head_budget(
                dense[0][p].weight.numel(),
                hidden,
                dense[0][p].out_features,
                head_counts[name],
                layer_count,
                d.rank_ratio,
            )
            for p, name in enumerate(PROJECTIONS)
        ]
        indices, floored = per_head_allocation(pools, budgets, False, floor_index=d.empty_head_index)

        residual = None
        if qat.enabled and qat.residual_rotation != "none":
            source = (
                RandomHadamard(seed=qat.residual_rotation_seed)
                if qat.residual_rotation == "random_hadamard"
                else Hadamard()
            )
            residual = rotation_matrix(source, hidden, "residual")
        value = rot.hadamard(head_dim).float() if qat.enabled and qat.value_hadamard else None
        settings = FTSettings(
            iters=ft.iters,
            lr=ft.lr,
            steps_per_iteration=steps,
            seed=ft.seed,
            qat=QATSettings(
                start_fraction=qat.start_fraction,
                lr_factors=qat.lr_factors,
                lr_rotation=qat.lr_rotation,
                scale_update_every=qat.scale_update_every,
                bits=qat.w_bits,
                learned_rotation=qat.latent_rotation == "learned",
            )
            if qat.enabled
            else None,
        )

        out: dict[str, torch.Tensor] = {}
        ft_losses: list[list[float]] = []
        qat_losses: list[list[float]] = []
        for layer, projections in enumerate(dense):
            device = devices[layer]
            for p, (name, proj) in enumerate(zip(PROJECTIONS, projections, strict=True)):
                heads = head_counts[name]
                U, S, V = svds[layer][name]
                A, B = init_factors(U, S, V, indices[p][layer], chol[layer][1])
                result = train_projection(
                    A,
                    B,
                    proj.weight.view(heads, -1, hidden).to(TRAIN_DTYPE),
                    stats[f"{layer}.{name}.fisher"].to(device) if ft.weighted else None,
                    settings,
                    None if residual is None else residual.to(device),
                    value.to(device) if value is not None and name == "v_proj" else None,
                )
                out[f"{layer}.{name}.down"] = torch.cat(result.B, 0).cpu()
                for h, a in enumerate(result.A):
                    out[f"{layer}.{name}.up.{h}"] = a.cpu()
                ft_losses.append(result.ft_losses)
                qat_losses.append(result.qat_losses)
            svds[layer].clear()
        report["floored"] = floored
        report["ft_losses"] = ft_losses
        report["qat_losses"] = qat_losses
        return out

    qat_key = qat.model_dump(mode="json")
    qat_key.pop("dense_prequant")
    factors = cache.get_or_compute(
        artifact_key(
            model=model_key,
            calibration=cfg.calibration,
            statistic=FACTORS_NAME,
            upstream=[],
            downstream={
                "decomposition": d.model_dump(mode="json"),
                "local_ft": ft.model_dump(mode="json"),
                "qat": qat_key,
                "steps": steps,
                "device": device_names,
                "torch": torch.__version__,
            },
        ),
        compute_factors,
    )
    install = [
        {
            name: (
                factors[f"{layer}.{name}.down"].to(dtype),
                [factors[f"{layer}.{name}.up.{h}"] for h in range(head_counts[name])],
            )
            for name in PROJECTIONS
        }
        for layer in range(layer_count)
    ]
    install_per_head_qkv(model, install)
    _log(install, head_counts, head_dim, hidden, dense, report)


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.4g}" if values else "n/a"


def _log(install, head_counts, head_dim, hidden, dense, report) -> None:
    parts = []
    for name in PROJECTIONS:
        ranks = [up.shape[1] for layer in install for up in layer[name][1]]
        numel = dense[0][PROJECTIONS.index(name)].out_features * hidden * len(install)
        ratio = sum(ranks) * (hidden + head_dim) / numel
        parts.append(
            f"{name} head rank min/mean/max {min(ranks)}/{sum(ranks) / len(ranks):.1f}/{max(ranks)} "
            f"param ratio {ratio:.4f}"
        )
    if "floored" in report:
        first = [losses[0] for losses in report["ft_losses"] if losses]
        last = [losses[-1] for losses in report["ft_losses"] if losses]
        q_first = [losses[0] for losses in report["qat_losses"] if losses]
        q_last = [losses[-1] for losses in report["qat_losses"] if losses]
        training = (
            f"floored heads {report['floored']}, FT loss first/last {_mean(first)}/{_mean(last)}, "
            f"QAT loss first/last {_mean(q_first)}/{_mean(q_last)}"
        )
    else:
        training = "factors from cache"
    logger.info("wsvd %s; %s; device %s", "; ".join(parts), training, _device_name(dense[0][0].weight.device))


@method(
    "wsvd",
    title="WSVD",
    version="1",
    field="quantization",
    config=WSVDConfig,
    uses=("quant_rules", "calibration"),
    display=("rules", "decomposition", "local_ft", "qat"),
    default_preset="paper_w8a8",
    prepare=decompose,
    architecture_transform="qkv_per_head",
    description="WSVD: weighted per-head whitened SVD of q/k/v with gradient-ranked rank allocation, "
    "Fisher-weighted local fine-tuning of the factors, quantization-aware fine-tuning with a learned "
    "per-head rotation folded into the factors, then RTN weights and per-token activations (ported from "
    "the official code; text-only).",
)
def wsvd(model: nn.Module, tokenizer, cfg: WSVDConfig) -> nn.Module:
    if cfg.rules:
        sites.attach(model, cfg.rules)
    return model
