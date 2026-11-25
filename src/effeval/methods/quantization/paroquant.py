"""ParoQuant weight quantization (arXiv 2511.10645), ported from the official `legacy` branch at
github.com/z-lab/paroquant@8e5ac77 (optimize.py, paroquant/{module,quantizer,optimize,util}.py).

Every linear of a decoder layer becomes W_eff = R^-1(Q(R(W * s))) / s: channel scales s, grouped pairwise
Givens rotations R (ops.givens) and a 4-bit asymmetric group quantizer Q. Layers are trained one by one
against the full-precision layer outputs in two stages (ops.layerwise): rotation angles and channel scales
first, then the weights and the quantizer's scales and zero points. The trained W_eff replaces each weight, so
evaluation needs no activation-side operation. Per-layer results are kept in the artifact cache.

Differences from the official code are structural: the CUDA rotation kernel is replaced by
`ops.givens.pairwise_rotate`, decoder layers are located through ops.sites and the model stays on its device.

Shape symbols: O output features, C input features, G = C / group_size groups, R rotations.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import torch
from pydantic import Field, model_validator
from torch import nn

from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.ops import sites
from effeval.ops.artifacts import ArtifactCache, artifact_key, default_cache
from effeval.ops.calibration import paroquant_blocks, paroquant_mixture
from effeval.ops.givens import align_pairs, pairwise_rotate, pairwise_rotate_inverse, random_rotation_pairs
from effeval.ops.layerwise import Stage, train_layerwise
from effeval.ops.online import share_pre_hooks

PARAMETER_GROUPS = ("weight", "angles", "channel_scales", "quantizer")
CODE_VERSION = 2


class ParoQuantStage(StrictModel):
    params: dict[str, float] = Field(
        min_length=1,
        description="Trainable parameter groups of this stage and their learning rates: any of weight, "
        "angles, channel_scales, quantizer.",
    )
    epochs: int = Field(ge=1, description="Passes over the training set in this stage.")

    @model_validator(mode="after")
    def _known_groups(self) -> ParoQuantStage:
        for name, lr in self.params.items():
            if name not in PARAMETER_GROUPS:
                raise ValueError(f"unknown parameter group {name!r}; expected one of {PARAMETER_GROUPS}")
            if lr <= 0:
                raise ValueError(f"learning rate of {name} must be positive")
        return self


class ParoQuantCalibration(StrictModel):
    datasets: list[Literal["wikitext2", "c4", "redpajama", "pileval"]] = Field(
        min_length=1, description="Training corpora, sampled evenly and shuffled together."
    )
    train_size: int = Field(ge=1, description="Training sequences in total.")
    val_dataset: Literal["wikitext2", "c4", "redpajama", "pileval"] = Field(
        description="Corpus of the validation sequences."
    )
    validation_size: int = Field(ge=1, description="Validation sequences.")
    seqlen: int = Field(ge=8, description="Tokens per sequence.")
    seed: int = Field(
        description="Seed of the data shuffles; layer i draws its rotation pairs with seed + i."
    )


class ParoQuantConfig(StrictModel):
    n_bits: int = Field(ge=1, le=8, description="Weight bit width.")
    group_size: int = Field(ge=2, description="Quantization and rotation group size (input channels).")
    num_rotations: int = Field(ge=1, description="Rotation steps per linear.")
    num_pairs_factor: float = Field(
        gt=0.0, le=0.5, description="Rotated pairs per group and step, as a fraction of group_size."
    )
    stages: list[ParoQuantStage] = Field(min_length=1, description="Optimisation stages, run in order.")
    weight_decay: float = Field(ge=0.0)
    betas: tuple[float, float]
    eps: float = Field(gt=0.0)
    loss: Literal["mse", "smooth_l1"]
    calibration: ParoQuantCalibration
    batch_size: int = Field(ge=1, description="Sequences per optimisation step.")
    cache_shards: int = Field(ge=1, description="Shards of the cached training activations moved to the GPU.")

    @model_validator(mode="after")
    def _sizes_fit(self) -> ParoQuantConfig:
        if self.group_size % 2 != 0:
            raise ValueError("group_size must be even")
        batches, remainder = divmod(self.calibration.train_size, self.batch_size)
        if remainder or batches % self.cache_shards:
            raise ValueError("train_size must be a multiple of batch_size * cache_shards")
        if self.calibration.validation_size % self.batch_size:
            raise ValueError("validation_size must be a multiple of batch_size")
        return self


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    return (x.clamp(min, max) - x).detach() + x


def _scales_and_zero_points(
    weight: torch.Tensor, group_size: int, qmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Min-max scale and float zero point per group; weight [O, C] -> two [O * G, 1] tensors."""
    x = weight.float().reshape(-1, group_size)
    min_val = x.amin(dim=1, keepdim=True)
    max_val = x.amax(dim=1, keepdim=True)
    scale = clamp_ste(max_val - min_val, min=1e-5) / qmax
    return scale, min_val / scale


def pseudo_quantize(
    x: torch.Tensor,
    n_bits: int,
    group_size: int,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
) -> torch.Tensor:
    """Asymmetric group quantize-dequantize of x [O, C] with straight-through rounding and clamping.

    Groups are `group_size` consecutive input channels. Without `scale` / `zero_point` ([O * G, 1]) the
    min-max values of each group are used. Computed in float32, returned in the dtype of x.
    """
    dtype = x.dtype
    x = x.float()
    qmin, qmax = 0, 2**n_bits - 1
    if scale is None or zero_point is None:
        scale, zero_point = _scales_and_zero_points(x, group_size, qmax)
    scale = clamp_ste(scale, min=1e-5, max=1e5)
    round_zero_point = clamp_ste(-round_ste(zero_point), qmin, qmax)
    rows, columns = x.shape
    x_int = round_ste(x.reshape(-1, group_size) / scale)
    x_int = clamp_ste(x_int + round_zero_point, qmin, qmax)
    x_dequant = (x_int - round_zero_point) * scale
    return x_dequant.reshape(rows, columns).to(dtype)


class AffineQuantizer(nn.Module):
    """Learnable per-group scale and zero point, initialised from the min-max values of `weight` [O, C]."""

    def __init__(self, weight: torch.Tensor, n_bits: int, group_size: int):
        super().__init__()
        self.n_bits = n_bits
        self.group_size = group_size
        with torch.no_grad():
            scale, zero_point = _scales_and_zero_points(weight, group_size, 2**n_bits - 1)
        self.scale = nn.Parameter(scale)
        self.zero_point_float = nn.Parameter(zero_point)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pseudo_quantize(x, self.n_bits, self.group_size, self.scale, self.zero_point_float)

    def optim_params(self) -> list[nn.Parameter]:
        return [self.scale, self.zero_point_float]

    def set_optim_enabled(self, enabled: bool) -> None:
        for param in self.optim_params():
            param.requires_grad = enabled


class PseudoQuantizedLinear(nn.Module):
    """A linear whose weight is used as W_eff = R^-1(Q(R(W * s))) / s (see the module docstring).

    State: `weight` [O, C], `pairs` [R, C] and `mask` [R, C / 2] (buffers; mask marks dummy pairs whose angle
    stays 0), `angles` [R, C / 2], `channel_scales` [1, C] and, once enabled, `quantizer`.
    """

    def __init__(
        self,
        linear: nn.Linear,
        pairs: torch.Tensor,
        angles: torch.Tensor,
        mask: torch.Tensor,
        channel_scales: torch.Tensor,
        *,
        group_size: int,
        n_bits: int,
    ):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone())
        device = self.weight.device
        self.group_size = group_size
        self.n_bits = n_bits
        self.register_buffer("pairs", pairs.to(device=device, dtype=torch.int16))
        self.register_buffer("mask", mask.to(device=device, dtype=torch.bool))
        self.angles = nn.Parameter(angles.to(device).clone())
        self.channel_scales = nn.Parameter(channel_scales.to(device))
        self.quantizer: AffineQuantizer | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self._pseudo_quantize(self.weight).T)
        return out if self.bias is None else out + self.bias

    def _pseudo_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        weight = pairwise_rotate(weight * self.channel_scales, self.pairs, self.angles, self.group_size)
        if self.quantizer is not None:
            weight = self.quantizer(weight)
        else:
            weight = pseudo_quantize(weight, self.n_bits, self.group_size)
        weight = pairwise_rotate_inverse(weight, self.pairs, self.angles, self.group_size)
        return weight / self.channel_scales.view(1, -1)

    def set_optim_enabled(
        self,
        *,
        weight: bool = False,
        angles: bool = False,
        channel_scales: bool = False,
        quantizer: bool = False,
    ) -> None:
        """Make exactly the named groups trainable; the quantizer is created on first use from the current
        rotated weight."""
        self.weight.requires_grad = weight
        if self.bias is not None:
            self.bias.requires_grad = False
        self.angles.requires_grad = angles
        self.channel_scales.requires_grad = channel_scales
        if quantizer and self.quantizer is None:
            with torch.no_grad():
                rotated = pairwise_rotate(
                    self.weight * self.channel_scales, self.pairs, self.angles, self.group_size
                )
            self.quantizer = AffineQuantizer(rotated, self.n_bits, self.group_size)
        if self.quantizer is not None:
            self.quantizer.set_optim_enabled(quantizer)

    def get_optim_params(self, name: str) -> list[nn.Parameter]:
        if name == "weight":
            return [self.weight]
        if name == "angles":
            return [self.angles]
        if name == "channel_scales":
            return [self.channel_scales]
        if name == "quantizer":
            return self.quantizer.optim_params()
        raise ValueError(f"unknown parameter group {name!r}")

    @torch.no_grad()
    def reset_angles_by_mask(self) -> None:
        self.angles.data[self.mask] = 0

    @torch.no_grad()
    def pseudo_weight(self) -> torch.Tensor:
        """W_eff [O, C] in the dtype of the weight."""
        return self._pseudo_quantize(self.weight).detach()

    def export_state(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu() for key, value in self.state_dict().items()}

    @classmethod
    def from_state(
        cls, state: dict[str, torch.Tensor], *, group_size: int, n_bits: int, device: torch.device
    ) -> PseudoQuantizedLinear:
        weight = state["weight"].to(device)
        linear = nn.Linear(
            weight.shape[1], weight.shape[0], bias="bias" in state, device=device, dtype=weight.dtype
        )
        module = cls(
            linear,
            state["pairs"],
            state["angles"],
            state["mask"],
            state["channel_scales"],
            group_size=group_size,
            n_bits=n_bits,
        )
        if "quantizer.scale" in state:
            module.set_optim_enabled(quantizer=True)
        module.load_state_dict({key: value.to(device) for key, value in state.items()})
        return module


def _initial_rotation(
    columns: int, cfg: ParoQuantConfig, layer_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random pairs [R, C], zero angles [R, C / 2] and dummy-pair mask [R, C / 2] for `columns` inputs."""
    per_rotation = random_rotation_pairs(
        columns // cfg.group_size,
        cfg.group_size,
        cfg.num_rotations,
        cfg.num_pairs_factor,
        cfg.calibration.seed + layer_index,
    )
    aligned = [align_pairs(pairs, cfg.group_size) for pairs in per_rotation]
    pairs = torch.stack([a[0] for a in aligned])
    mask = torch.stack([a[1] for a in aligned])
    return pairs, torch.zeros(mask.shape, dtype=torch.float32), mask


def _weights_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


class _CacheMiss(Exception):
    pass


def _raise_miss() -> dict[str, torch.Tensor]:
    raise _CacheMiss


def _layer_key(base_key: dict, index: int, config_json: dict) -> str:
    """Artifact key of layer `index`'s trained state; changes with `CODE_VERSION`."""
    return artifact_key(
        model={**base_key, "layer": index},
        calibration=None,
        statistic="paroquant.layer",
        upstream=[],
        downstream={**config_json, "code_version": CODE_VERSION},
    )


def _linears(layer: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, module) for name, module in layer.named_modules() if isinstance(module, nn.Linear)]


def _replace(layer: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    setattr(layer.get_submodule(parent_name), leaf, new)


def _stage(cfg: ParoQuantConfig, params: dict[str, float], epochs: int) -> Stage:
    """Stage training `params` of every pseudo-quantized linear, one AdamW group per linear and parameter."""

    def param_groups(layer: nn.Module) -> list[dict]:
        groups = []
        for module in layer.modules():
            if not isinstance(module, PseudoQuantizedLinear):
                continue
            module.set_optim_enabled(**{name: True for name in params})
            for name, lr in params.items():
                groups.append(
                    {
                        "params": module.get_optim_params(name),
                        "lr": lr,
                        "weight_decay": cfg.weight_decay,
                        "betas": cfg.betas,
                        "eps": cfg.eps,
                    }
                )
        return groups

    return Stage(param_groups, epochs)


def _reset_masked_angles(layer: nn.Module) -> None:
    for module in layer.modules():
        if isinstance(module, PseudoQuantizedLinear):
            module.reset_angles_by_mask()


def quantize_model(
    model: nn.Module,
    samples: torch.Tensor,
    val_samples: torch.Tensor,
    cfg: ParoQuantConfig,
    cache: ArtifactCache | None = None,
) -> None:
    """Train and write W_eff into every decoder-layer linear of `model` (lm_head is untouched).

    `samples` [N, L] and `val_samples` [N_val, L] are the training and validation token ids. A layer already
    in `cache` (default: the process-wide artifact cache) is restored instead of trained.
    """
    cache = default_cache() if cache is None else cache
    dtype = next(sites.decoder_layers(model).parameters()).dtype
    base_key = {
        "id": model.config._name_or_path,
        "dtype": str(dtype),
        "weights": _weights_digest(model),
    }
    config_json = cfg.model_dump(mode="json")
    pending: dict[int, dict[str, nn.Linear]] = {}

    def key(index: int) -> str:
        return _layer_key(base_key, index, config_json)

    def replace_fn(index: int, layer: nn.Module) -> bool:
        try:
            stored = cache.get_or_compute(key(index), _raise_miss)
        except _CacheMiss:
            stored = None
        linears = _linears(layer)
        for name, linear in linears:
            if linear.in_features % cfg.group_size:
                raise ValueError(
                    f"layer {index} {name}: in_features {linear.in_features} is not a multiple of group_size"
                )
        initial: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        pending[index] = {}
        for name, linear in linears:
            if stored is None:
                columns = linear.in_features
                if columns not in initial:
                    initial[columns] = _initial_rotation(columns, cfg, index)
                pairs, angles, mask = initial[columns]
                scales = torch.ones(1, columns, dtype=linear.weight.dtype, device=linear.weight.device)
                new = PseudoQuantizedLinear(
                    linear, pairs, angles, mask, scales, group_size=cfg.group_size, n_bits=cfg.n_bits
                )
            else:
                prefix = f"{name}::"
                state = {k[len(prefix) :]: v for k, v in stored.items() if k.startswith(prefix)}
                new = PseudoQuantizedLinear.from_state(
                    state, group_size=cfg.group_size, n_bits=cfg.n_bits, device=linear.weight.device
                )
            share_pre_hooks(linear, new)
            _replace(layer, name, new)
            pending[index][name] = linear
        return stored is None

    def finalize_fn(index: int, layer: nn.Module, trained: bool) -> None:
        modules = {name: layer.get_submodule(name) for name in pending[index]}
        if trained:
            tensors = {
                f"{name}::{k}": v
                for name, module in modules.items()
                for k, v in module.export_state().items()
            }
            cache.get_or_compute(key(index), lambda: tensors)
        for name, module in modules.items():
            original = pending[index][name]
            original.weight.data.copy_(module.pseudo_weight().to(original.weight.dtype))
            _replace(layer, name, original)
        del pending[index]

    train_layerwise(
        model,
        samples,
        val_samples,
        replace_fn=replace_fn,
        finalize_fn=finalize_fn,
        stages=[_stage(cfg, stage.params, stage.epochs) for stage in cfg.stages],
        loss=cfg.loss,
        batch_size=cfg.batch_size,
        cache_shards=cfg.cache_shards,
        post_step=_reset_masked_angles,
    )


@method(
    "paroquant",
    title="ParoQuant",
    version="1",
    field="quantization",
    config=ParoQuantConfig,
    uses=(),
    display=("n_bits", "group_size", "num_rotations"),
    default_preset="paper_w4g128",
    description="ParoQuant: learned grouped Givens rotations and channel scales before group-wise weight "
    "quantization (ported from the official legacy branch).",
)
def paroquant(model: nn.Module, tokenizer, cfg: ParoQuantConfig) -> nn.Module:
    calibration = cfg.calibration
    tokenizer_id = model.config._name_or_path
    samples = torch.stack(
        paroquant_mixture(
            tokenizer_id,
            calibration.datasets,
            calibration.train_size,
            calibration.seqlen,
            calibration.seed,
            "train",
        )
    )
    val_samples = torch.stack(
        paroquant_blocks(
            tokenizer_id,
            calibration.val_dataset,
            calibration.validation_size,
            calibration.seqlen,
            calibration.seed,
            "validation",
        )
    )
    quantize_model(model, samples, val_samples, cfg)
    return model
