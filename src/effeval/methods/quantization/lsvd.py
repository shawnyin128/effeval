"""LSVD: loss-aware per-head q/k/v factorisation and a hybrid SVD / RTN feed-forward block, then RTN.

Ported from the released code (SAI-Lab-NYU/LSVD, `fake_quant/`): input-whitened (Cholesky) and output-gradient
whitened SVD of every attention head, gradient-weighted global rank allocation, and, optionally, gate/up
projections split into a whitened-SVD branch and an RTN branch of dense channels. Text-only: the calibration
loss is the causal-LM cross-entropy of wikitext2 windows.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from effeval.core.fragments import Calibration, IntFormat, QuantRule, select_layers
from effeval.core.lowrank import FFN_CANONS, QKV_CANONS, hybrid_ffn_sizes, per_head_budget
from effeval.core.preprocess import RandomHadamard
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.methods.preprocess.rotation import rotation_matrix
from effeval.methods.quantization.rtn import check_clip_search
from effeval.ops import rotation as rot
from effeval.ops import sites
from effeval.ops.artifacts import artifact_key, default_cache, weights_fingerprint
from effeval.ops.calibration import (
    act_stats,
    calibration_windows,
    causal_lm_grads,
    collect_input_rows,
    output_grad_grams,
)
from effeval.ops.hybrid_ffn import (
    DOWN_CANON,
    MAX_SELECTION_ROWS,
    HybridLayer,
    install_hybrid_ffn,
    partition_channels,
    rotated_selection_inputs,
    selection_is_reused,
    strided_rows,
    svd_aware_order,
    svd_branch,
    whitening_factor,
)
from effeval.ops.lowrank import (
    cholesky_lower,
    cholesky_upper_heads,
    install_per_head_qkv,
    per_head_allocation,
    per_head_factors,
    per_head_projected_gradient,
    per_head_whitened_svd,
)

logger = logging.getLogger(__name__)

INPUT_SITE = "linear.attn.k_proj.input"
MLP_SITE = "linear.mlp.gate_proj.input"
INPUT_GRAM_NAME = "lsvd.input_gram:v1"
OUTPUT_GRAM_NAME = "lsvd.output_grad_gram:v1"
GRAD_NAME = "lsvd.s_grad:v1"
PROJECTIONS = [canon.split(".")[1] for canon in QKV_CANONS]
NORM_PATH = "post_attention_layernorm"


class LSVDDecomposition(StrictModel):
    rank_ratio: float = Field(
        gt=0,
        description="Kept parameter ratio of each of q, k and v: per head-slot int(rank_ratio * num_layers * "
        "head_dim * hidden / (hidden + head_dim)) singular values, times the head count, ranked by "
        "importance; values at or above (hidden + head_dim) / hidden keep every one.",
    )
    beta: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Singular-value split: up factor U * S^(1 - beta), down factor S^beta * V^T. 0 puts "
        "every singular value in the up factor (released --svd_mode U); 0.5 splits the square root "
        "between the factors (paper). The hybrid FFN's SVD branch supports only 0 and 0.5.",
    )
    grad_alpha: float = Field(
        default=1.0,
        description="Exponent of the singular value's gradient magnitude in its importance score.",
    )
    score_dtype: Literal["bfloat16", "float32"] = Field(
        default="bfloat16",
        description="dtype of the gradient projection, its accumulation and the importance scores; the "
        "released code uses bfloat16.",
    )
    rank_pools: Literal["per_projection", "joint"] = Field(
        default="per_projection",
        description="per_projection ranks q, k and v in separate pools with their own budgets; joint ranks "
        "them in one pool (released --truncate_mixed_qkv_budget).",
    )
    k_grad_normalisation: Literal["once", "twice"] = Field(
        default="once",
        description="How often the k_proj gradient score is divided by the window count and square-rooted. "
        "twice reproduces the released code, which normalises k_proj in a separate step and again in the "
        "q/k/v loop; once is the single normalisation q and v get.",
    )
    qaw: bool = Field(
        default=False,
        description="Quantization-aware whitening: the input Gram is accumulated from fake-quantized "
        "activations (per-token symmetric).",
    )
    qaw_bits: int = Field(
        default=8, ge=2, le=8, description="Activation bit width of quantization-aware whitening."
    )
    qaw_clip_ratio: float = Field(
        default=1.0,
        gt=0,
        le=1,
        description="Clip ratio of the activation quantizer of quantization-aware whitening.",
    )


class LSVDFFN(StrictModel):
    enabled: bool = Field(
        default=False,
        description="Split gate and up into an SVD branch and RTN dense channels in every layer but the last "
        "skip_last_layers.",
    )
    svd_ratio: float = Field(
        default=0.5, gt=0, lt=1, description="Fraction of the intermediate channels on the SVD branch."
    )
    rank_ratio: float = Field(
        default=0.9,
        gt=0,
        le=2,
        description="Rank of the SVD branch: int(rank_ratio * n * hidden / (n + hidden)) for n SVD channels.",
    )
    skip_last_layers: int = Field(
        default=5, ge=0, description="Layers at the end of the network that keep a plain dense FFN."
    )
    skipped_layer_hadamard: Literal["standard", "online_only"] = Field(
        default="standard",
        description="standard: a skipped layer's down_proj carries the offline fold and the online Hadamard "
        "(function-preserving). online_only reproduces the released code, which applies the online Hadamard "
        "there without folding it into the weights, so the layer no longer computes the original function; "
        "it needs the preprocess's online Hadamard on linear.mlp.down_proj.input.",
    )
    latent_rotation: Literal["block_hadamard", "none"] = Field(
        default="block_hadamard", description="Rotation of the SVD branch's latent space."
    )
    whitening_tokens: int = Field(
        default=4096, ge=1, description="Calibration tokens (evenly strided) that whiten the SVD branch."
    )
    damping: float = Field(
        default=0.01,
        ge=0,
        description="Diagonal damping of the whitening Gram, relative to its mean diagonal.",
    )
    selection_ratio: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of channels on the SVD branch in the channel-selection baseline.",
    )
    selection_tokens: int = Field(
        default=MAX_SELECTION_ROWS,
        ge=1,
        le=MAX_SELECTION_ROWS,
        description="First calibration tokens the channel selection is evaluated on.",
    )
    selection_w_bits: int = Field(
        default=8, ge=2, le=8, description="Weight bits of the channel-selection proxy."
    )
    selection_a_bits: int = Field(
        default=8, ge=2, le=8, description="Activation bits of the channel-selection proxy."
    )
    selection_a_clip_ratio: float = Field(
        default=1.0, gt=0, le=1, description="Activation clip ratio of the channel-selection proxy."
    )
    selection_latent_rotation: bool = Field(
        default=False, description="Rotate the latent of the channel-selection baseline."
    )
    selection_rotation: Literal["random_hadamard", "none"] = Field(
        default="random_hadamard",
        description="Basis the channel order is computed in: the residual rotation a QuaRot-style preprocess "
        "applies (the released code selects on the rotated model), or the model as it is.",
    )
    selection_rotation_seed: int = Field(
        default=0, description="Seed of the random Hadamard that selection_rotation simulates."
    )


class LSVDConfig(StrictModel):
    rules: list[QuantRule] = Field(
        default_factory=list, description="Quantization rules applied to the factored model."
    )
    weight_scope: Literal["released", "all"] = Field(
        default="all",
        description="released: weight rules skip the q/k/v factors and o_proj of the layers whose FFN is "
        "hybrid (the released RTN scope). all: weight rules apply wherever they match.",
    )
    calibration: Calibration
    decomposition: LSVDDecomposition
    ffn: LSVDFFN = Field(default_factory=LSVDFFN, description="Hybrid SVD / RTN feed-forward block.")

    @model_validator(mode="after")
    def _clip_search_on_weights(self) -> LSVDConfig:
        check_clip_search(self.rules)
        return self

    @model_validator(mode="after")
    def _ffn_beta(self) -> LSVDConfig:
        if self.ffn.enabled and self.decomposition.beta not in (0.0, 0.5):
            raise ValueError(
                f"decomposition.beta {self.decomposition.beta} is not supported with ffn.enabled: "
                "the hybrid FFN splits the singular values with beta 0 or 0.5"
            )
        return self


def hybrid_layer_ids(cfg: LSVDConfig, num_layers: int) -> list[int]:
    """Layers whose FFN is hybrid: all but the last `ffn.skip_last_layers`, or none when the FFN is off."""
    if not cfg.ffn.enabled:
        return []
    return list(range(num_layers - cfg.ffn.skip_last_layers))


def _linear_paths(layout) -> dict[str, str]:
    return {canon: path for path, canon in layout.linears.items()}


@torch.no_grad()
def decompose(model: nn.Module, tokenizer, cfg: LSVDConfig) -> None:
    """Factor every layer's q/k/v per head and, when `ffn.enabled`, split gate/up into SVD and RTN channels.

    Statistics (input Gram, output-gradient Gram, singular-value gradients) come from the artifact cache or
    are computed on the calibration windows. Raises ValueError for a Gram that is not positive definite, a
    projection that keeps no singular value in some layer, an MLP with a bias, or `skip_last_layers` covering
    every layer.
    """
    d, ffn = cfg.decomposition, cfg.ffn
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
    if ffn.enabled and ffn.skip_last_layers >= layer_count:
        raise ValueError(
            f"skip_last_layers {ffn.skip_last_layers} leaves no hybrid layer in {layer_count} layers"
        )
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
    if ffn.enabled:
        paths = _linear_paths(layout)
        for layer in layers:
            for canon in (*FFN_CANONS, DOWN_CANON):
                if layer.get_submodule(paths[canon]).bias is not None:
                    raise ValueError(f"{canon} has a bias; the hybrid FFN does not support it")

    qaw = (
        IntFormat(
            type="int", bits=d.qaw_bits, symmetric=True, granularity="per_token", clip_ratio=d.qaw_clip_ratio
        )
        if d.qaw
        else None
    )
    qaw_key = (
        {"qaw": True, "qaw_bits": d.qaw_bits, "qaw_clip_ratio": d.qaw_clip_ratio} if d.qaw else {"qaw": False}
    )

    def compute_input_gram() -> dict[str, torch.Tensor]:
        grams = act_stats(model, windows, [INPUT_SITE], "gram", prequant=qaw)
        return {str(layer): value for (_site, layer), value in grams.items()}

    input_grams = cache.get_or_compute(
        artifact_key(
            model=model_key,
            calibration=cfg.calibration,
            statistic=INPUT_GRAM_NAME,
            upstream=[],
            downstream=qaw_key,
        ),
        compute_input_gram,
    )
    chol = [cholesky_lower(input_grams[str(layer)].to(devices[layer])) for layer in range(layer_count)]

    def compute_output_grams() -> dict[str, torch.Tensor]:
        targets = {
            f"{layer}.{name}": (getattr(attention, name), head_counts[name], head_dim)
            for layer, attention in enumerate(attentions)
            for name in PROJECTIONS
        }
        params = [getattr(attentions[0], name).weight for name in PROJECTIONS]
        return output_grad_grams(model, windows, targets, params)

    output_grams = cache.get_or_compute(
        artifact_key(model=model_key, calibration=cfg.calibration, statistic=OUTPUT_GRAM_NAME, upstream=[]),
        compute_output_grams,
    )

    svds: list[dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    output_inv: list[dict[str, torch.Tensor]] = []
    for layer, projections in enumerate(dense):
        svds.append({})
        output_inv.append({})
        for name, proj in zip(PROJECTIONS, projections, strict=True):
            upper, upper_inv = cholesky_upper_heads(output_grams[f"{layer}.{name}"].to(devices[layer]))
            output_inv[layer][name] = upper_inv
            svds[layer][name] = per_head_whitened_svd(proj.weight, head_counts[name], chol[layer], upper)

    score_dtype = getattr(torch, d.score_dtype)
    input_inv_t = [torch.linalg.inv(factor).T for factor in chol]

    def compute_s_grad() -> dict[str, torch.Tensor]:
        accumulated: dict[str, torch.Tensor] = {}

        def on_batch() -> None:
            for layer, projections in enumerate(dense):
                for name, proj in zip(PROJECTIONS, projections, strict=True):
                    U, _S, V = svds[layer][name]
                    squared = (
                        per_head_projected_gradient(
                            U, V, proj.weight.grad, input_inv_t[layer], output_inv[layer][name], score_dtype
                        )
                        ** 2
                    )
                    key = f"{layer}.{name}"
                    if key in accumulated:
                        accumulated[key] += squared
                    else:
                        accumulated[key] = squared

        params = [proj.weight for projections in dense for proj in projections]
        batches = causal_lm_grads(model, windows, params, on_batch)
        return {key: value.div(batches).abs().sqrt() for key, value in accumulated.items()}

    s_grad = cache.get_or_compute(
        artifact_key(
            model=model_key,
            calibration=cfg.calibration,
            statistic=GRAD_NAME,
            upstream=[],
            downstream={"input": qaw_key, "score_dtype": d.score_dtype},
        ),
        compute_s_grad,
    )
    if d.k_grad_normalisation == "twice":
        s_grad = dict(s_grad)
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
    per_head_slots = per_head_budget(layer_count, hidden, head_dim, 1, d.rank_ratio)
    indices, floored = per_head_allocation(
        pools, [per_head_slots] * len(PROJECTIONS), d.rank_pools == "joint"
    )

    factors = [
        {
            name: per_head_factors(
                *svds[layer][name],
                indices[p][layer],
                chol[layer],
                output_inv[layer][name],
                d.beta,
                dtype,
            )
            for p, name in enumerate(PROJECTIONS)
        }
        for layer in range(layer_count)
    ]
    svds.clear()
    install_per_head_qkv(model, factors)
    _log_attention(indices, floored, head_counts, head_dim, hidden, layer_count)

    if ffn.enabled:
        _install_hybrid_ffn(model, windows, ffn, d.beta, hybrid_layer_ids(cfg, layer_count))


def _log_attention(indices, floored, head_counts, head_dim, hidden, layer_count) -> None:
    parts = []
    for p, name in enumerate(PROJECTIONS):
        ranks = [len(head) for layer in indices[p] for head in layer]
        heads = head_counts[name]
        params = sum(ranks) * (hidden + head_dim) / (layer_count * heads * head_dim * hidden)
        parts.append(
            f"{name} head rank min/mean/max {min(ranks)}/{sum(ranks) / len(ranks):.1f}/{max(ranks)} "
            f"param ratio {params:.4f}"
        )
    latent = {
        name: sum(len(head) for layer in indices[p] for head in layer)
        / (layer_count * head_counts[name] * head_dim)
        for p, name in enumerate(PROJECTIONS)
        if name != "q_proj"
    }
    logger.info(
        "lsvd %s, floored heads %d, K latent ratio %.4f, V latent ratio %.4f",
        "; ".join(parts),
        floored,
        latent["k_proj"],
        latent["v_proj"],
    )


def _install_hybrid_ffn(model: nn.Module, windows, ffn: LSVDFFN, beta: float, hybrid_ids: list[int]) -> None:
    layout = sites.architecture(model)
    layers = sites.decoder_layers(model)
    paths = _linear_paths(layout)
    total = sum(window.numel() for window in windows)
    select_rows = torch.arange(min(ffn.selection_tokens, total))
    whiten_rows = strided_rows(total, ffn.whitening_tokens)
    collected = collect_input_rows(model, windows, MLP_SITE, torch.cat([select_rows, whiten_rows]))

    entries: list[HybridLayer | None] = [None] * len(layers)
    for index in hybrid_ids:
        layer = layers[index]
        gate, up, down = (layer.get_submodule(paths[canon]).weight for canon in (*FFN_CANONS, DOWN_CANON))
        dtype, device = gate.dtype, gate.device
        d_ffn, hidden = gate.shape
        gain = layer.get_submodule(NORM_PATH).weight
        x_select = collected[index][: len(select_rows)].to(device)
        x_whiten = collected[index][len(select_rows) :].to(device)
        n_svd, rank = hybrid_ffn_sizes(d_ffn, hidden, ffn.svd_ratio, ffn.rank_ratio)
        selection_n_svd, selection_rank = hybrid_ffn_sizes(d_ffn, hidden, ffn.selection_ratio, ffn.rank_ratio)

        if ffn.selection_rotation == "random_hadamard":
            rotation = rotation_matrix(
                RandomHadamard(seed=ffn.selection_rotation_seed), hidden, "residual"
            ).to(device)
            w_gate, w_up, w_down, x_basis = rotated_selection_inputs(
                gate, up, down, x_select, gain, rotation, dtype
            )
        else:
            w_gate, w_up, w_down, x_basis = gate.float(), up.float(), down.float(), x_select.float()
        selection_chol = whitening_factor(
            x_basis[strided_rows(x_basis.shape[0], ffn.whitening_tokens)], ffn.damping, None
        )
        order = svd_aware_order(
            w_gate,
            w_up,
            w_down,
            x_basis,
            selection_rank,
            ffn.selection_w_bits,
            ffn.selection_a_bits,
            ffn.selection_a_clip_ratio,
            beta,
            selection_chol,
            ffn.selection_latent_rotation,
        )
        perm = partition_channels(order, d_ffn, n_svd, selection_n_svd)
        # the released code reuses the stored selection together with its rank
        if selection_is_reused(len(order), d_ffn, n_svd, selection_n_svd):
            rank = selection_rank

        chol = whitening_factor(x_whiten, ffn.damping, gain)
        rows = torch.tensor(perm[:n_svd], device=device)
        rotate = ffn.latent_rotation == "block_hadamard"
        entries[index] = HybridLayer(
            perm=perm,
            n_svd=n_svd,
            gate=svd_branch(gate[rows], rank, chol, beta, rotate, dtype),
            up=svd_branch(up[rows], rank, chol, beta, rotate, dtype),
        )
        logger.info("lsvd layer %d ffn: %d svd channels of %d, rank %d", index, n_svd, d_ffn, rank)
    install_hybrid_ffn(model, entries)


def scoped_rules(model: nn.Module, rules: list[QuantRule], hybrid_ids: list[int]) -> list[QuantRule]:
    """`rules` with every weight match on an attention linear removed from the hybrid-FFN layers.

    Each rule that matches attention weights is split into the part on all other sites (original layers) and
    the part on the attention weights (original layers minus `hybrid_ids`); patterns that match attention
    weights must end in `.weight`. Locations covered by the result are the original ones minus the removed
    attention weights, with the original override order. Raises ValueError for a pattern that is not a
    `.weight` pattern but matches attention weights.
    """
    layout = sites.architecture(model)
    num_layers = len(sites.decoder_layers(model))
    weight_sites = [f"linear.{canon}.weight" for canon in layout.linears.values()]
    attention_sites = {s for s in weight_sites if s.startswith("linear.attn.")}
    hybrid = set(hybrid_ids)
    out: list[QuantRule] = []
    for rule in rules:
        attention_hits = {
            s for s in attention_sites if any(fnmatch.fnmatchcase(s, pattern) for pattern in rule.sites)
        }
        if not attention_hits or not hybrid:
            out.append(rule)
            continue
        kept: list[str] = []
        for pattern in rule.sites:
            if not any(fnmatch.fnmatchcase(s, pattern) for s in attention_sites):
                kept.append(pattern)
            elif not pattern.endswith(".weight"):
                raise ValueError(
                    f"site pattern {pattern!r} matches attention weights but is not a .weight pattern"
                )
            else:
                kept += [
                    s for s in weight_sites if s not in attention_sites and fnmatch.fnmatchcase(s, pattern)
                ]
        if kept:
            out.append(rule.model_copy(update={"sites": list(dict.fromkeys(kept))}))
        remaining = [i for i in select_layers(rule.layers, num_layers) if i not in hybrid]
        if remaining:
            out.append(rule.model_copy(update={"sites": sorted(attention_hits), "layers": remaining}))
    return out


def _unfold_skipped_down_proj(model: nn.Module, cfg: LSVDConfig) -> None:
    """Undo the offline Hadamard fold on the down_proj of the skipped layers, keeping the online Hadamard.

    Raises ValueError when a skipped layer's down_proj carries no online op (the preprocess did not run).
    """
    layout = sites.architecture(model)
    layers = sites.decoder_layers(model)
    path = _linear_paths(layout)[DOWN_CANON]
    for layer in layers[len(layers) - cfg.ffn.skip_last_layers :]:
        down = layer.get_submodule(path)
        if not down._forward_pre_hooks:
            raise ValueError(
                "skipped_layer_hadamard 'online_only' needs the online Hadamard of the lsvd preprocess on "
                "linear.mlp.down_proj.input"
            )
        down.weight.data = rot.fold_input(down.weight, rot.hadamard(down.in_features).T)


@method(
    "lsvd",
    title="LSVD",
    version="1",
    field="quantization",
    config=LSVDConfig,
    uses=("quant_rules", "calibration"),
    display=("rules", "decomposition", "ffn"),
    default_preset="paper_w8a8",
    prepare=decompose,
    architecture_transform="lsvd",
    description="LSVD: loss-aware per-head low-rank factorisation of q/k/v with importance-ranked rank "
    "allocation and quantization-aware whitening, a hybrid SVD / RTN feed-forward block, then RTN weights "
    "and per-token activations (ported from the official code; text-only).",
)
def lsvd(model: nn.Module, tokenizer, cfg: LSVDConfig) -> nn.Module:
    if cfg.ffn.enabled and cfg.ffn.skipped_layer_hadamard == "online_only" and cfg.ffn.skip_last_layers:
        _unfold_skipped_down_proj(model, cfg)
    if cfg.rules:
        rules = cfg.rules
        if cfg.weight_scope == "released":
            rules = scoped_rules(model, rules, hybrid_layer_ids(cfg, len(sites.decoder_layers(model))))
        sites.attach(model, rules)
    return model
