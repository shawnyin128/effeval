"""GPTQ (Frantar et al., arXiv 2210.17323), ported from the official code at
github.com/IST-DASLab/gptq@2d65066 (gptq.py `GPTQ.add_batch`/`fasterquant`, opt.py `opt_sequential`,
datautils.py `get_c4`/`get_wikitext2`).

Differences from the official code are structural only: layers and linears are located through ops.sites,
the quantization grid comes from ops.quant.int_grid (op-for-op identical for the asymmetric grid), and
decoder layers are re-invoked with the keyword arguments captured from the model's own forward pass.
Rules on non-weight sites (activations, attention) are attached after the weights are quantized, as QuaRot
adds its activation quantizers after GPTQ.

Shape symbols: R rows (out features), C columns (in features), L calibration sequence length, Dm hidden size.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
from pydantic import Field, model_validator
from torch import nn

from effeval.core.fragments import Calibration, IntFormat, QuantRule
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.ops import sites
from effeval.ops.calibration import LayerRunner, calibration_windows
from effeval.ops.quant.formats import IntGrid, int_grid


class GPTQConfig(StrictModel):
    rules: list[QuantRule] = Field(
        min_length=1,
        description="Weight rules (integer formats) are solved by GPTQ; rules on other sites follow.",
    )
    act_order: bool
    static_groups: bool
    percdamp: float = Field(gt=0.0)
    blocksize: int = Field(ge=1)
    true_sequential: bool = Field(
        default=False,
        description="Quantize each layer's linears in groups (k/v/q, o, up/gate, down), re-running the layer "
        "between groups so later groups see the quantized earlier ones, as the official `--true-sequential`.",
    )
    calibration: Calibration

    @model_validator(mode="after")
    def _weights_with_integer_grids(self) -> GPTQConfig:
        for index, rule in enumerate(self.rules):
            on_weights = [pattern.endswith(".weight") for pattern in rule.sites]
            if any(on_weights) and not all(on_weights):
                raise ValueError(f"rules[{index}]: a rule's sites must be all weights or all non-weights")
            if not all(on_weights):
                continue
            if not isinstance(rule.quant, IntFormat):
                raise ValueError(f"rules[{index}]: GPTQ needs an integer format (type: int)")
            if rule.quant.granularity not in {"per_channel", "per_group"}:
                raise ValueError(f"rules[{index}]: GPTQ supports granularity per_channel or per_group")
            if rule.quant.clip_ratio != 1.0:
                raise ValueError(f"rules[{index}]: clip_ratio is not part of GPTQ; use clip_search")
        if not self.weight_rules:
            raise ValueError("GPTQ needs at least one rule on '.weight' sites")
        return self

    @property
    def weight_rules(self) -> list[QuantRule]:
        return [rule for rule in self.rules if rule.sites[0].endswith(".weight")]

    @property
    def other_rules(self) -> list[QuantRule]:
        return [rule for rule in self.rules if not rule.sites[0].endswith(".weight")]


class HessianAccumulator:
    """Running H = 2/n * sum(x x^T) over calibration inputs of one linear layer (official `add_batch`)."""

    def __init__(self, columns: int, device: torch.device):
        self.H = torch.zeros((columns, columns), device=device)  # [C, C]
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        batch = inp.shape[0]
        if inp.dim() == 3:
            inp = inp.reshape((-1, inp.shape[-1]))  # [B * L, C]
        inp = inp.t()  # [C, B * L]
        self.H *= self.nsamples / (self.nsamples + batch)
        self.nsamples += batch
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())


def quantize_weight(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    fmt: IntFormat,
    blocksize: int,
    percdamp: float,
    act_order: bool,
    static_groups: bool,
) -> torch.Tensor:
    """GPTQ for one weight [R, C] given its Hessian [C, C]; returns the dequantized weight in float32.

    Follows the official `fasterquant` step by step, including that non-static group grids are fitted on the
    outer W (updated after each block) rather than the in-block working copy.
    """

    def fit(units: torch.Tensor) -> IntGrid:
        return int_grid(units, fmt.bits, fmt.symmetric, fmt.clip_ratio, fmt.clip_search)

    W = weight.clone().float()  # [R, C]
    columns = W.shape[1]
    groupsize = fmt.group_size if fmt.granularity == "per_group" else None
    grid = fit(W) if groupsize is None else None

    H = hessian
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    groups: list[IntGrid] = []
    if static_groups and groupsize is not None:
        groups = [fit(W[:, i : i + groupsize]) for i in range(0, columns, groupsize)]

    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Q = torch.zeros_like(W)  # [R, C]
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(columns, device=W.device)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H  # [C, C], upper Cholesky factor of H^-1

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        W1 = W[:, i1:i2].clone()  # [R, B]
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]  # [R]
            d = Hinv1[i, i]
            if groupsize is not None:
                if not static_groups:
                    if (i1 + i) % groupsize == 0:
                        grid = fit(W[:, (i1 + i) : (i1 + i + groupsize)])
                else:
                    idx = i1 + i
                    if act_order:
                        idx = perm[idx]
                    grid = groups[idx // groupsize]
            q = grid.apply(w.unsqueeze(1)).flatten()  # [R]
            Q1[:, i] = q
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if act_order:
        Q = Q[:, invperm]
    return Q


TRUE_SEQUENTIAL_GROUPS = (
    ("attn.k_proj", "attn.v_proj", "attn.q_proj"),
    ("attn.o_proj",),
    ("mlp.up_proj", "mlp.gate_proj"),
    ("mlp.down_proj",),
)


def _hessian_hook(accumulator: HessianAccumulator):
    def hook(_module: nn.Module, inp: tuple, _out: torch.Tensor) -> None:
        accumulator.add_batch(inp[0].data)

    return hook


def _quantize_group(
    runner: LayerRunner, layer: nn.Module, group: list[tuple[nn.Module, IntFormat]], cfg: GPTQConfig
) -> None:
    """Collect Hessians of `group`'s linears over all windows through `layer`, then quantize them in place."""
    accumulators = {
        id(module): HessianAccumulator(module.weight.shape[1], module.weight.device) for module, _ in group
    }
    hooks = [module.register_forward_hook(_hessian_hook(accumulators[id(module)])) for module, _ in group]
    try:
        for hidden in runner.inputs:
            runner.forward(layer, hidden)
    finally:
        for hook in hooks:
            hook.remove()
    for module, fmt in group:
        quantized = quantize_weight(
            module.weight.data,
            accumulators[id(module)].H,
            fmt,
            blocksize=cfg.blocksize,
            percdamp=cfg.percdamp,
            act_order=cfg.act_order,
            static_groups=cfg.static_groups,
        )
        module.weight.data = quantized.reshape(module.weight.shape).to(module.weight.data.dtype)


@torch.no_grad()
def quantize_model(model: nn.Module, windows: list[torch.Tensor], cfg: GPTQConfig) -> None:
    """Sequential GPTQ over decoder layers; each layer is calibrated on the already-quantized prefix.

    Rules on non-weight sites are attached once every weight is quantized.
    """
    assignment, _ = sites.resolve(model, cfg.weight_rules)
    targets: dict[int, dict[str, tuple[nn.Module, IntFormat]]] = defaultdict(dict)
    for (site, layer), (_rule, location, fmt) in assignment.items():
        if layer is None:
            raise ValueError("GPTQ quantizes decoder-layer linears; lm_head is not supported")
        targets[layer][site.removeprefix("linear.").removesuffix(".weight")] = (location.module, fmt)

    with LayerRunner(model, windows) as runner:
        for index, layer in runner:
            layer_targets = targets.get(index, {})
            if cfg.true_sequential:
                grouped = {name for group in TRUE_SEQUENTIAL_GROUPS for name in group}
                unplaced = sorted(set(layer_targets) - grouped)
                if unplaced:
                    raise ValueError(f"true_sequential has no group for linears {unplaced} in layer {index}")
                steps = [[n for n in group if n in layer_targets] for group in TRUE_SEQUENTIAL_GROUPS]
            else:
                steps = [list(layer_targets)]
            for names in steps:
                if names:
                    _quantize_group(runner, layer, [layer_targets[n] for n in names], cfg)

    if cfg.other_rules:
        sites.attach(model, cfg.other_rules)


@method(
    "gptq",
    title="GPTQ",
    version="1",
    field="quantization",
    config=GPTQConfig,
    uses=("quant_rules", "calibration"),
    display=("rules", "act_order"),
    default_preset="paper",
    description="GPTQ weight quantization (ported from the official implementation).",
)
def gptq(model: nn.Module, tokenizer, cfg: GPTQConfig) -> nn.Module:
    windows = calibration_windows(cfg.calibration, model.config._name_or_path)
    quantize_model(model, windows, cfg)
    return model
