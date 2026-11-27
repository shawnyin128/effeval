"""QSVD: joint low-rank factorisation of q/k/v with a shared down factor, then RTN quantizers.

Ported from the released code (SAI-Lab-NYU/QSVD, `fake_quant/`): activation-whitened SVD of `[Wq; Wk; Wv]`,
gradient-weighted global rank allocation, a random orthogonal rotation of the latent, then rules on the
factors. Text-only: the calibration loss is the causal-LM cross-entropy of wikitext2 windows.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from effeval.core.fragments import Calibration, QuantRule
from effeval.core.lowrank import DOWN_NAME, QKV_CANONS
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.methods.quantization.rtn import check_clip_search
from effeval.ops import sites
from effeval.ops.artifacts import artifact_key, default_cache, weights_fingerprint
from effeval.ops.calibration import act_stats, calibration_windows, causal_lm_grads
from effeval.ops.lowrank import (
    factor_weights,
    global_top_k,
    install_shared_qkv,
    joint_whitened_svd,
    projected_gradient,
    rotate_latent,
)
from effeval.ops.rotation import random_orthogonal

logger = logging.getLogger(__name__)

STATISTIC_SITE = "linear.attn.k_proj.input"
STAT_NAME = "qsvd.abs_mean:v1"
GRAD_NAME = "qsvd.s_grad:v1"
PROJECTIONS = [canon.split(".")[1] for canon in QKV_CANONS]


class QSVDDecomposition(StrictModel):
    rank_ratio: float = Field(
        gt=0,
        le=2,
        description="Singular values kept over all layers: int(rank_ratio / 2 * num_layers * hidden), ranked "
        "by importance (QSVD --rank_ratio; 2 keeps every one).",
    )
    beta: float = Field(
        ge=0,
        le=1,
        description="Singular-value split: up factor U * S^(1 - beta), down factor S^beta * V^T "
        "(QSVD --svd_mode; 0.5 is 'UV').",
    )
    act_alpha: float = Field(
        default=0.5,
        gt=0,
        description="Exponent of the mean absolute input activation that whitens the weights before the SVD.",
    )
    grad_alpha: float = Field(
        default=1.0,
        description="Exponent of the singular value's gradient magnitude in its importance score.",
    )
    score_dtype: Literal["float16", "float32"] = Field(
        default="float16",
        description="dtype the importance scores are ranked in; the released code uses float16.",
    )
    latent_rotation: Literal["random_orthogonal", "none"] = Field(
        default="random_orthogonal",
        description="Rotation applied to the latent space of every layer after the factorisation.",
    )
    latent_seed: int = Field(default=0, description="Seed of the latent rotation, the same for every layer.")


class QSVDConfig(StrictModel):
    rules: list[QuantRule] = Field(
        default_factory=list, description="Quantization rules applied to the factored model."
    )
    calibration: Calibration
    decomposition: QSVDDecomposition

    @model_validator(mode="after")
    def _clip_search_on_weights(self) -> QSVDConfig:
        check_clip_search(self.rules)
        return self


@torch.no_grad()
def decompose(model: nn.Module, tokenizer, cfg: QSVDConfig) -> None:
    """Factor every layer's q/k/v into a shared down factor and per-projection up factors, in place.

    Statistics (activation absolute means, singular-value gradients) come from the artifact cache or are
    computed on the calibration windows. Raises ValueError when a layer keeps no singular value.
    """
    d = cfg.decomposition
    windows = calibration_windows(cfg.calibration, model.config._name_or_path)
    model_key = {
        "id": model.config._name_or_path,
        "dtype": str(next(model.parameters()).dtype),
        "weights": weights_fingerprint(model),
    }
    cache = default_cache()
    arch = sites.architecture(model)
    attentions = [layer.get_submodule(arch.attention_path) for layer in sites.decoder_layers(model)]
    dense = [[getattr(attention, name) for name in PROJECTIONS] for attention in attentions]
    layer_count = len(dense)

    def compute_stats() -> dict[str, torch.Tensor]:
        stats = act_stats(model, windows, [STATISTIC_SITE], "window_absmean_sum")
        return {str(layer): value for (_site, layer), value in stats.items()}

    stats = cache.get_or_compute(
        artifact_key(model=model_key, calibration=cfg.calibration, statistic=STAT_NAME, upstream=[]),
        compute_stats,
    )

    scales, scale_invs, svds = [], [], []
    for layer, projections in enumerate(dense):
        device = projections[0].weight.device
        scale = (
            torch.ones(projections[0].in_features, device=device)
            * stats[str(layer)].to(device).float() ** d.act_alpha
        )
        scales.append(scale)
        scale_invs.append(scale ** (-1))
        svds.append(joint_whitened_svd([p.weight for p in projections], scale))

    def compute_s_grad() -> dict[str, torch.Tensor]:
        accumulated: list[torch.Tensor | None] = [None] * layer_count

        def on_batch() -> None:
            for layer, projections in enumerate(dense):
                grad = torch.cat([p.weight.grad for p in projections], 0)
                squared = projected_gradient(svds[layer][0], svds[layer][2], grad, scale_invs[layer]) ** 2
                if accumulated[layer] is None:
                    accumulated[layer] = squared
                else:
                    accumulated[layer] += squared

        params = [p.weight for projections in dense for p in projections]
        batches = causal_lm_grads(model, windows, params, on_batch)
        return {str(layer): acc.div(batches).sqrt() for layer, acc in enumerate(accumulated)}

    s_grad = cache.get_or_compute(
        artifact_key(
            model=model_key,
            calibration=cfg.calibration,
            statistic=GRAD_NAME,
            upstream=[],
            downstream={"act_alpha": d.act_alpha},
        ),
        compute_s_grad,
    )

    score_dtype = getattr(torch, d.score_dtype)
    scores = [
        svds[layer][1].to(score_dtype).abs()
        * s_grad[str(layer)].to(dense[layer][0].weight.device).to(score_dtype).abs() ** d.grad_alpha
        for layer in range(layer_count)
    ]
    kept = int(d.rank_ratio / 2 * (layer_count * dense[0][0].in_features))
    indices = global_top_k(scores, kept)
    for layer, chosen in enumerate(indices):
        if not chosen:
            raise ValueError(f"layer {layer} keeps no singular value at rank_ratio {d.rank_ratio}")

    factors = []
    for layer, projections in enumerate(dense):
        U, S, V = svds[layer]
        factors.append(
            factor_weights(
                U,
                S,
                V,
                indices[layer],
                scales[layer],
                d.beta,
                [p.out_features for p in projections],
                projections[0].weight.dtype,
            )
        )
    svds.clear()
    install_shared_qkv(model, factors)

    ranks = [len(chosen) for chosen in indices]
    if d.latent_rotation == "random_orthogonal":
        for attention, rank in zip(attentions, ranks, strict=True):
            rotate_latent(
                getattr(attention, DOWN_NAME),
                [getattr(attention, name).up for name in PROJECTIONS],
                random_orthogonal(rank, d.latent_seed),
            )

    in_features = dense[0][0].in_features
    out_sizes = [p.out_features for p in dense[0]]
    total = sum(out_sizes)
    logger.info(
        "qsvd rank min/mean/max %d/%.1f/%.1f, kept %d, R1 %.4f, R2 %.4f",
        min(ranks),
        kept / layer_count,
        max(ranks),
        kept,
        sum(r * (in_features + total) for r in ranks) / (layer_count * in_features * total),
        sum(ranks) / (layer_count * (out_sizes[1] + out_sizes[2])),
    )


@method(
    "qsvd",
    title="QSVD",
    version="1",
    field="quantization",
    config=QSVDConfig,
    uses=("quant_rules", "calibration"),
    display=("rules", "decomposition"),
    default_preset="paper_w4a4",
    prepare=decompose,
    architecture_transform="qkv_shared",
    description="QSVD: joint low-rank factorisation of q/k/v with a shared down factor, gradient-ranked "
    "global rank allocation and a latent rotation, then RTN weights and per-token activations (ported from "
    "the official code; text-only).",
)
def qsvd(model: nn.Module, tokenizer, cfg: QSVDConfig) -> nn.Module:
    if cfg.rules:
        sites.attach(model, cfg.rules)
    return model
