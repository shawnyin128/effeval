"""CodeQuant2 weight clustering (arXiv 2611.09868), ported from github.com/SAI-Lab-NYU/CodeQuant2@eaf112c
(script/adap_clus.py, script/finetune_local.py, cluster/ClusterLinear.py, utils/adaptive_cluster_utils.py,
utils/reasoning_data_utils.py).

The target linears are the weight sites of the rules with a codebook format: each row of a target becomes a
k-means codebook. Stage 1, adaptive cluster allocation (ACA), scores every target for 8 / 16 / 32 centroids by
the gradient-weighted output change of switching its cluster count, on the logits reconstruction loss of the
whole model with 4-bit activations, and swaps pairs of matrices (one 16 -> 32, one 16 -> 8) in each group of
layers. Stage 2 trains, whole model and batch by batch, the centroids of every target on its own
full-precision input and, through a LoRA-parameterised soft assignment annealed to a hard one, the
assignments (one SGD over all layers with a global gradient-norm clip). The hard clustered weights replace
the dense weights; the activation rules are attached afterwards.

Shape symbols: O/D output/input features of a target, K centroids per row, r LoRA rank.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from datasets import load_dataset
from pydantic import Field, field_validator, model_validator
from torch import nn
from torch.utils.data import DataLoader
from transformers import BatchEncoding

from effeval.core.fragments import CodebookFormat, QuantRule
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.ops import codebook, sites
from effeval.ops.artifacts import ArtifactCache, artifact_key, default_cache, weights_fingerprint
from effeval.ops.calibration import reasoning_mixture

logger = logging.getLogger(__name__)

CODE_VERSION = 1


class AcaCalibration(StrictModel):
    dataset: str = Field(description="Hugging Face dataset id with a `text` column.")
    split: str = Field(default="train", description="Dataset split.")
    n_samples: int = Field(ge=1, description="Rows kept after shuffling the split with dataset_seed.")
    dataset_seed: int = Field(default=42, description="Seed of the dataset shuffle that picks the rows.")
    batch_size: int = Field(ge=1, description="Rows per batch.")
    max_length: int = Field(ge=1, description="Tokens per row; rows are padded to this length.")
    shuffle_seed: int = Field(
        default=42, description="Seed of the batch-order generator, shared by the passes of all groups."
    )
    enable_thinking: bool = Field(default=False, description="Passed to the chat template of each row.")


class AcaConfig(StrictModel):
    choices: list[int] = Field(
        default=[8, 16, 32],
        min_length=3,
        max_length=3,
        description="Cluster counts: smallest, base (the rules' centroids) and largest.",
    )
    calibration: AcaCalibration
    layers_per_group: int = Field(
        default=10, ge=1, description="Decoder layers scored per pass; each group runs over all batches."
    )
    change_ratio: float = Field(
        default=0.15, gt=0.0, le=1.0, description="At most this fraction of a group's matrices changes."
    )
    micro_batch_size: int | None = Field(
        default=None,
        ge=1,
        description="Rows per forward/backward chunk of a batch; the result does not depend on it. "
        "None runs the whole batch at once.",
    )

    @field_validator("choices")
    @classmethod
    def _increasing(cls, value: list[int]) -> list[int]:
        if not 1 <= value[0] < value[1] < value[2] <= 256:
            raise ValueError("choices must increase strictly within [1, 256]")
        return value


class GenerationConfig(StrictModel):
    max_new_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.6, gt=0.0)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1)
    batch_size: int = Field(default=16, ge=1, description="Prompts generated together.")


class RacCalibration(StrictModel):
    """Reasoning-aware calibration: model-written traces for math problems mixed with plain text."""

    problem_dataset: str = Field(default="openai/gsm8k")
    problem_config: str | None = Field(default="main")
    problem_split: str = Field(default="train")
    problem_column: str = Field(default="question")
    wiki_dataset: str = Field(default="Self-GRIT/wikitext-2-raw-v1-preprocessed")
    wiki_split: str = Field(default="train")
    pool: int = Field(
        default=512, ge=1, description="Leading problems and non-empty text rows the mixture draws from."
    )
    total: int = Field(default=512, ge=1, description="Rows of the mixture.")
    wiki_ratio: float = Field(default=0.9, ge=0.0, le=1.0, description="Fraction of plain-text rows.")
    seed: int = Field(default=42, description="Seed of the mixture draw and of the sampling of the traces.")
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    batch_size: int = Field(ge=1, description="Rows per training batch.")
    max_length: int = Field(ge=1, description="Tokens per row; rows are padded to this length.")
    epochs: int = Field(ge=1, description="Passes over the rows, each with a fresh shuffle.")
    shuffle_seed: int = Field(default=42, description="Seed of the batch-order generator shared by epochs.")


class TrainConfig(StrictModel):
    calibration: RacCalibration
    lora_r: int = Field(ge=1, description="Rank of the LoRA update of the assignment logits.")
    lr_centroids: float = Field(default=1e-4, gt=0.0)
    lr_assignment: float = Field(default=1e-5, gt=0.0)
    momentum: float = Field(default=0.9, ge=0.0, lt=1.0, description="SGD momentum of both optimizers.")
    clip: float = Field(
        default=1.0,
        gt=0.0,
        description="Gradient-norm clip: per matrix for the centroids, over all assignment parameters "
        "of the model together for the LoRA factors.",
    )
    start_temp: float = Field(default=0.1, gt=0.0, description="Softmax temperature at the first step.")
    end_temp: float = Field(default=0.001, gt=0.0, description="Temperature at progress hard_ratio.")
    hard_ratio: float = Field(
        default=0.98, gt=0.0, le=1.0, description="Training progress from which assignments are hard."
    )
    loss1_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Weight of the loss on 4-bit inputs; the rest goes to the loss on full-precision inputs.",
    )
    init_seed: int = Field(default=42, description="Seed of the LoRA initialisation.")


class CodeQuant2Config(StrictModel):
    rules: list[QuantRule] = Field(
        min_length=1,
        description="Rules with a codebook format name the weights to cluster (per_channel); all other rules "
        "are attached after the clustering, e.g. 4-bit activations on the clustered linears' inputs.",
    )
    act_bits: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Bits of the per-token activation quantizer of the loss and of ACA.",
    )
    aca: AcaConfig | None = Field(default=None, description="Adaptive cluster allocation; off when null.")
    train: TrainConfig

    @model_validator(mode="after")
    def _weight_rules(self) -> CodeQuant2Config:
        formats = [rule.quant for rule in self.rules if isinstance(rule.quant, CodebookFormat)]
        if not formats:
            raise ValueError("at least one rule needs a codebook format")
        if any(fmt.granularity != "per_channel" for fmt in formats):
            raise ValueError("codebooks are per_channel (one per output row)")
        if self.aca is not None and any(fmt.centroids != self.aca.choices[1] for fmt in formats):
            raise ValueError(
                f"with aca, every codebook rule needs centroids = choices[1] = {self.aca.choices[1]}"
            )
        return self


@dataclass
class _Target:
    name: str
    layer: int
    parent: nn.Module
    attr: str
    linear: nn.Linear
    fmt: CodebookFormat


def _targets(model: nn.Module, rules: list[QuantRule]) -> list[_Target]:
    """The dense linears the codebook rules match, in module order. Raises ValueError for any other site."""
    assignment, _ = sites.resolve(model, rules)
    order = {id(module): (index, name) for index, (name, module) in enumerate(model.named_modules())}
    targets = []
    for key, (_rule, loc, fmt) in assignment.items():
        if len(key) != 2 or not key[0].startswith("linear.") or not isinstance(loc.module, nn.Linear):
            raise ValueError(f"codebook rules apply to dense linears, not {key[0]}")
        name = order[id(loc.module)][1]
        parent, _, attr = name.rpartition(".")
        targets.append(_Target(name, loc.layer, model.get_submodule(parent), attr, loc.module, fmt))
    return sorted(targets, key=lambda t: order[id(t.linear)][0])


def _fit(weight: torch.Tensor, fmt: CodebookFormat, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row k-means of a [O, D] weight: centroids [O, K] in its dtype and uint8 assignments [O, D]."""
    work = weight.float() if fmt.kmeans.compute_dtype == "float32" else weight
    units, mask = codebook.to_units(work, None)
    centroids, assignments = codebook.kmeans(
        units, mask, k, fmt.kmeans.iters, fmt.kmeans.tol, fmt.kmeans.seed
    )
    return centroids.to(weight.dtype), assignments


def _activation_quantizer(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token asymmetric min-max fake quantization with a straight-through rounding.

    The range is taken over the last axis in x's dtype, the scale is clamped to at least 1e-5.
    """
    flat = x.reshape(-1, x.shape[-1])
    levels = 2**bits - 1
    with torch.no_grad():
        low = flat.min(dim=-1, keepdim=True)[0]
        scale = ((flat.max(dim=-1, keepdim=True)[0] - low) / levels).clamp(min=1e-5)
    scaled = (flat - low) / scale
    rounded = (torch.round(scaled).clamp(0, levels) - scaled).detach() + scaled
    return (rounded * scale + low).reshape(x.shape)


def _mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((prediction - target) ** 2).mean(-1).mean()


def _to_device(batch, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def _frozen_eval(model: nn.Module) -> Iterator[None]:
    """Run with every parameter of `model` frozen and the model in eval mode, restoring both on exit."""
    flags = [(p, p.requires_grad) for p in model.parameters()]
    training = model.training
    model.eval()
    for parameter, _ in flags:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, flag in flags:
            parameter.requires_grad_(flag)
        model.train(training)


def _swapped(targets: list[_Target], replacements: dict[str, nn.Module]) -> contextlib.AbstractContextManager:
    @contextlib.contextmanager
    def swap() -> Iterator[None]:
        for target in targets:
            setattr(target.parent, target.attr, replacements[target.name])
        try:
            yield
        finally:
            for target in targets:
                setattr(target.parent, target.attr, target.linear)

    return swap()


def _row_batches(
    rows: list[str],
    tokenizer,
    *,
    batch_size: int,
    epochs: int,
    seed: int,
    max_length: int,
    render: Callable[[str], str] | None = None,
) -> list[list[BatchEncoding]]:
    """Batches per epoch (result[epoch][step]) of `DataLoader(shuffle=True)` over `rows` with one generator
    seeded with `seed` for all epochs, tokenized with truncation and padding to `max_length`."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=True, generator=generator, collate_fn=list)
    return [
        [
            tokenizer(
                batch if render is None else [render(text) for text in batch],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            for batch in loader
        ]
        for _ in range(epochs)
    ]


def aca_batches(cfg: AcaConfig, tokenizer, passes: int) -> list[list[BatchEncoding]]:
    """The calibration batches of each of `passes` group passes of ACA.

    Rows are the first n_samples of the split shuffled with dataset_seed; every pass reshuffles the order of
    the batches with the one generator, and each row is a user turn of the chat template.
    """
    calibration = cfg.calibration
    dataset = load_dataset(calibration.dataset, split=calibration.split).shuffle(
        seed=calibration.dataset_seed
    )
    rows = dataset.select(range(min(calibration.n_samples, len(dataset))))["text"]

    def render(text: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=calibration.enable_thinking,
        )

    return _row_batches(
        rows,
        tokenizer,
        batch_size=calibration.batch_size,
        epochs=passes,
        seed=calibration.shuffle_seed,
        max_length=calibration.max_length,
        render=render,
    )


# adaptive cluster allocation


class _ScoredLinear(nn.Module):
    """A target during ACA: the linear with k-means-reconstructed weights for each candidate cluster count.

    Pinned to a cluster count it applies that reconstruction to 4-bit inputs; pinned to -1 it is the
    original linear. It records its input and the gradient of its output, from which `accumulate` adds the
    importance of switching to the smallest and the largest count.
    """

    def __init__(self, linear: nn.Linear, fmt: CodebookFormat, choices: list[int], act_bits: int):
        super().__init__()
        self.linear = linear
        self.fmt = fmt
        self.choices = choices
        self.act_bits = act_bits
        self.pinned = -1
        self.dense: dict[int, torch.Tensor] = {}
        self.x: torch.Tensor | None = None
        self.gy: torch.Tensor | None = None
        self.reset()

    def reset(self) -> None:
        self.down = 0.0
        self.up = 0.0
        self.count = 0

    def reconstruction(self, k: int) -> torch.Tensor:
        """The weight with every row replaced by its nearest of K k-means centroids; fitted once per K."""
        if k not in self.dense:
            with torch.no_grad():
                centroids, assignments = _fit(self.linear.weight.detach(), self.fmt, k)
                self.dense[k] = codebook.gather(centroids, assignments)
        return self.dense[k]

    def keep_only(self, k: int) -> None:
        self.dense = {k: self.reconstruction(k)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pinned == -1:
            return F.linear(x, self.linear.weight, self.linear.bias)
        quantized = _activation_quantizer(x, self.act_bits)
        self.x = quantized.detach()
        weight = self.reconstruction(self.pinned).to(device=quantized.device, dtype=quantized.dtype)
        output = F.linear(quantized, weight, self.linear.bias)
        if output.requires_grad:
            output.register_hook(self._save_grad)
        return output

    def _save_grad(self, grad: torch.Tensor) -> torch.Tensor:
        self.gy = grad.detach()
        return grad

    @torch.no_grad()
    def _pair(
        self, x: torch.Tensor, gy: torch.Tensor, reference: torch.Tensor, target: torch.Tensor
    ) -> float:
        delta = (reference - target).to(device=x.device, dtype=x.dtype)
        inner = (F.linear(x, delta) * gy).sum(dim=-1)
        return float(inner.square().sum().item())

    @torch.no_grad()
    def accumulate(self) -> None:
        """Add the squared inner products of the recorded input and output gradient with the weight change of
        a switch to the smallest and to the largest count; clears the records."""
        if self.x is None or self.gy is None:
            raise RuntimeError("no input or output gradient recorded; the loss does not reach this linear")
        small, _base, big = self.choices
        x = self.x.reshape(-1, self.x.shape[-1])
        gy = self.gy.reshape(-1, self.gy.shape[-1])
        pinned = self.reconstruction(self.pinned)
        self.count += x.shape[0]
        self.down += self._pair(x, gy, pinned, self.reconstruction(small))
        self.up += self._pair(x, gy, pinned, self.reconstruction(big))
        self.x = self.gy = None

    def scores(self) -> tuple[float, float]:
        """(importance of a downgrade, negated importance of an upgrade), means over the recorded tokens."""
        count = max(1, self.count)
        return self.down / count, -self.up / count


def allocate_cluster_counts(
    downgrade: dict[str, float], upgrade: dict[str, float], choices: list[int], change_ratio: float
) -> dict[str, int]:
    """Pair one upgrade to the largest count with one downgrade to the smallest, returning every name's count.

    Matrices are visited by ascending score (upgrade: most negative first, downgrade: least damaging first);
    a pair is accepted when the two scores sum below zero, a rejected pair is dropped, and the loop stops
    once `int(len * change_ratio)` matrices changed (so up to one more than that, since pairs add two).
    """
    small, base, big = choices
    result = dict.fromkeys(downgrade, base)
    by_upgrade = sorted(downgrade, key=lambda name: upgrade[name])
    by_downgrade = sorted(downgrade, key=lambda name: downgrade[name])
    limit = int(len(downgrade) * change_ratio)
    changed: set[str] = set()
    available = set(downgrade)
    up_index = down_index = 0
    while len(changed) < limit and up_index < len(by_upgrade) and down_index < len(by_downgrade):
        up_name = None
        while up_index < len(by_upgrade):
            candidate = by_upgrade[up_index]
            up_index += 1
            if candidate in available:
                up_name = candidate
                break
        if up_name is None:
            break
        down_name = None
        while down_index < len(by_downgrade):
            candidate = by_downgrade[down_index]
            down_index += 1
            if candidate in available and candidate != up_name:
                down_name = candidate
                break
        if down_name is None:
            break
        if upgrade[up_name] + downgrade[down_name] >= 0:
            continue
        available -= {up_name, down_name}
        result[up_name] = big
        result[down_name] = small
        changed |= {up_name, down_name}
    return result


def _require_grad_hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
    """Make the embedding output a gradient source, so the frozen model backpropagates to its activations."""
    if torch.is_grad_enabled():
        output.requires_grad_(True)
    return output


def choose_cluster_counts(
    model: nn.Module,
    targets: list[_Target],
    batches: list[list[BatchEncoding]],
    cfg: AcaConfig,
    act_bits: int,
) -> dict[str, int]:
    """Cluster count of every target by ACA.

    `batches[g]` are the batches of the g-th group pass. Groups of `layers_per_group` consecutive layers are
    scored one after another on the loss `mean((logits_fp - logits)^2)` of the model with the targets on
    4-bit inputs and k-means weights (the group at the base count, earlier groups at their allocation, later
    ones at the base count); the allocation of a group is `allocate_cluster_counts` on its scores.
    """
    base = cfg.choices[1]
    wrappers = {t.name: _ScoredLinear(t.linear, t.fmt, cfg.choices, act_bits) for t in targets}
    groups: dict[int, list[str]] = {}
    for t in targets:
        groups.setdefault(t.layer // cfg.layers_per_group, []).append(t.name)
    if len(batches) != len(groups):
        raise ValueError(f"expected the batches of {len(groups)} group passes, got {len(batches)}")
    device = next(model.parameters()).device
    final = dict.fromkeys(wrappers, base)
    embedding = model.get_input_embeddings().register_forward_hook(_require_grad_hook)
    try:
        with _frozen_eval(model), _swapped(targets, wrappers):
            for (group_id, members), pass_batches in zip(sorted(groups.items()), batches, strict=True):
                for name in members:
                    wrappers[name].reset()
                for batch in pass_batches:
                    batch = _to_device(batch, device)
                    rows = batch["input_ids"].shape[0]
                    step = rows if cfg.micro_batch_size is None else cfg.micro_batch_size
                    for start in range(0, rows, step):
                        chunk = {name: value[start : start + step] for name, value in batch.items()}
                        for wrapper in wrappers.values():
                            wrapper.pinned = -1
                        with torch.no_grad():
                            reference = model(**chunk, use_cache=False).logits
                        for name, wrapper in wrappers.items():
                            wrapper.pinned = base if name in members else final[name]
                        logits = model(**chunk, use_cache=False).logits
                        elements = reference.numel() // chunk["input_ids"].shape[0] * rows
                        ((reference - logits).pow(2).sum() / elements).backward()
                        del reference, logits
                        for name in members:
                            wrappers[name].accumulate()
                        for name, wrapper in wrappers.items():
                            if name not in members:
                                wrapper.x = wrapper.gy = None
                scores = {name: wrappers[name].scores() for name in members}
                allocation = allocate_cluster_counts(
                    {name: down for name, (down, _up) in scores.items()},
                    {name: up for name, (_down, up) in scores.items()},
                    cfg.choices,
                    cfg.change_ratio,
                )
                for name, count in allocation.items():
                    final[name] = count
                    wrappers[name].keep_only(count)
                logger.info(
                    "ACA group %d: %d matrices, %d to %d centroids, %d to %d",
                    group_id,
                    len(members),
                    sum(c == cfg.choices[2] for c in allocation.values()),
                    cfg.choices[2],
                    sum(c == cfg.choices[0] for c in allocation.values()),
                    cfg.choices[0],
                )
    finally:
        embedding.remove()
    del wrappers
    _free_memory()
    return final


# learnable clustering


class _ClusterLinear(nn.Module):
    """A target during training: per-row centroids, hard k-means assignments and a LoRA update of the logits.

    Every forward trains the centroids with one SGD step on the output error of the clustered weight on the
    4-bit input against the original weight on the full-precision input, leaves the gradients of the LoRA
    factors for the caller's optimizer, and returns the original linear's output, so each layer sees
    full-precision inputs. The soft assignment is `softmax((onehot(k-means index) + A B) / temperature)`.
    """

    def __init__(
        self,
        linear: nn.Linear,
        fmt: CodebookFormat,
        k: int,
        cfg: TrainConfig,
        act_bits: int,
        generator: torch.Generator,
    ):
        super().__init__()
        self.linear = linear
        weight = linear.weight
        rows, columns = weight.shape
        self.k = k
        self.act_bits = act_bits
        self.cfg = cfg
        centroids, assignments = _fit(weight.detach(), fmt, k)
        # the released code rounds the centroids through bfloat16, whatever the model dtype
        self.centroids = nn.Parameter(centroids.to(torch.bfloat16).to(weight.dtype))
        self.register_buffer("assignment", assignments)
        factor = torch.empty(rows * k, cfg.lora_r, dtype=torch.float32)
        nn.init.kaiming_uniform_(factor, a=math.sqrt(5), generator=generator)
        self.lora_a = nn.Parameter(factor.to(device=weight.device, dtype=weight.dtype))
        self.lora_b = nn.Parameter(torch.zeros(cfg.lora_r, columns, dtype=weight.dtype, device=weight.device))
        self.optimizer = torch.optim.SGD(
            [self.centroids], lr=cfg.lr_centroids, momentum=cfg.momentum, weight_decay=0
        )
        self.temperature = cfg.start_temp
        self.soft = True
        self.nmse = 0.0

    def set_progress(self, ratio: float) -> None:
        """Anneal the temperature linearly from start_temp (ratio 0) to end_temp (ratio hard_ratio); from
        hard_ratio on the assignments are hard."""
        ratio = max(0.0, min(1.0, ratio))
        cfg = self.cfg
        self.temperature = cfg.start_temp + (ratio / cfg.hard_ratio) * (cfg.end_temp - cfg.start_temp)
        self.soft = ratio < cfg.hard_ratio

    def cluster_weight(self, hard: bool = False) -> torch.Tensor:
        """The [O, D] weight of the centroids under the soft assignment (hard one-hot argmax when `hard` or
        once annealing has finished)."""
        rows, columns = self.linear.weight.shape

        def probabilities() -> torch.Tensor:
            delta = (self.lora_a @ self.lora_b).view(rows, self.k, columns).transpose(1, 2).contiguous()
            onehot = torch.zeros((rows, columns, self.k), device=delta.device, dtype=delta.dtype)
            onehot.scatter_(-1, self.assignment.long().unsqueeze(-1), 1.0)
            return F.softmax((delta + onehot) / max(float(self.temperature), 1e-8), dim=-1)

        if self.soft and not hard:
            weights = probabilities()
        else:
            with torch.no_grad():
                soft = probabilities()
                weights = torch.zeros_like(soft).scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
        return torch.einsum("nmk,nk->nm", weights, self.centroids)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            self._train_step(x.detach())
        return F.linear(x, self.linear.weight.to(x.dtype), self.linear.bias).detach()

    def _train_step(self, x: torch.Tensor) -> None:
        cfg, bias = self.cfg, self.linear.bias
        quantized = _activation_quantizer(x, self.act_bits)
        weight = self.cluster_weight()
        target = F.linear(x, self.linear.weight.to(quantized.dtype), bias)
        loss = cfg.loss1_weight * _mse(F.linear(quantized, weight, bias), target).to(x.dtype)
        if cfg.loss1_weight < 1.0:
            loss = loss + (1.0 - cfg.loss1_weight) * _mse(F.linear(x, weight, bias), target).to(x.dtype)
        self.nmse = loss.item() / target.pow(2).mean().clamp(min=1e-12).item()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([self.centroids], max_norm=cfg.clip)
        self.optimizer.step()


def train_clusters(
    model: nn.Module,
    targets: list[_Target],
    counts: dict[str, int],
    batches: list[BatchEncoding],
    cfg: TrainConfig,
    act_bits: int,
) -> dict[str, list]:
    """Train the targets over `batches` (one optimizer step each) and write the hard clustered weights.

    Returns the per-step `temperature`, the global assignment-gradient norm `grad_norm` and the per-target
    `nmse` of the training loss (normalised by the output power) under `nmse[name]`.
    """
    generator = torch.Generator()
    generator.manual_seed(cfg.init_seed)
    wrappers = {
        t.name: _ClusterLinear(t.linear, t.fmt, counts[t.name], cfg, act_bits, generator) for t in targets
    }
    parameters = [p for w in wrappers.values() for p in (w.lora_a, w.lora_b)]
    optimizer = torch.optim.SGD(parameters, lr=cfg.lr_assignment, momentum=cfg.momentum, weight_decay=0)
    device = next(model.parameters()).device
    history: dict[str, list] = {"temperature": [], "grad_norm": [], "nmse": {name: [] for name in wrappers}}
    with _frozen_eval(model), _swapped(targets, wrappers):
        for step, batch in enumerate(batches):
            optimizer.zero_grad(set_to_none=True)
            for wrapper in wrappers.values():
                wrapper.set_progress(step / len(batches))
            model(**_to_device(batch, device), use_cache=False, logits_to_keep=1)
            history["grad_norm"].append(nn.utils.clip_grad_norm_(parameters, max_norm=cfg.clip).item())
            optimizer.step()
            history["temperature"].append(next(iter(wrappers.values())).temperature)
            for name, wrapper in wrappers.items():
                history["nmse"][name].append(wrapper.nmse)
    with torch.no_grad():
        for target in targets:
            target.linear.weight.copy_(wrappers[target.name].cluster_weight(hard=True))
    return history


# reasoning-aware calibration


@torch.no_grad()
def generate_traces(
    model: nn.Module, tokenizer, prompts: list[str], cfg: GenerationConfig, seed: int
) -> list[str]:
    """One sampled continuation per prompt (new tokens only, special tokens dropped).

    Prompts are generated `batch_size` at a time with left padding; sampling is seeded with `seed` and the
    global random state is restored afterwards.
    """
    device = next(model.parameters()).device
    padding_side, training = tokenizer.padding_side, model.training
    tokenizer.padding_side = "left"
    model.eval()
    completions: list[str] = []
    try:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            for start in range(0, len(prompts), cfg.batch_size):
                inputs = _to_device(
                    tokenizer(prompts[start : start + cfg.batch_size], return_tensors="pt", padding=True),
                    device,
                )
                tokens = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
                completions += tokenizer.batch_decode(
                    tokens[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
                )
    finally:
        tokenizer.padding_side = padding_side
        model.train(training)
    return completions


def reasoning_rows(
    model: nn.Module, tokenizer, cfg: RacCalibration, cache: ArtifactCache | None = None
) -> list[str]:
    """The calibration texts of `cfg`: `reasoning_mixture` of the leading problems and text rows, with the
    traces sampled from `model` and kept in the artifact cache (key: weights, prompts, generation settings).
    """
    cache = default_cache() if cache is None else cache
    problems = load_dataset(cfg.problem_dataset, cfg.problem_config, split=cfg.problem_split)
    problems = problems.select(range(min(cfg.pool, len(problems))))[cfg.problem_column]
    texts = [t for t in load_dataset(cfg.wiki_dataset, split=cfg.wiki_split)["text"] if t.strip()][: cfg.pool]

    def generate(prompts: list[str]) -> list[str]:
        key = artifact_key(
            model={
                "id": model.config._name_or_path,
                "dtype": str(next(model.parameters()).dtype),
                "weights": weights_fingerprint(model),
            },
            calibration=None,
            statistic=f"codequant2.traces:v{CODE_VERSION}",
            upstream=[],
            downstream={"prompts": prompts, "generation": cfg.generation.model_dump(), "seed": cfg.seed},
        )

        def compute() -> dict[str, torch.Tensor]:
            traces = generate_traces(model, tokenizer, prompts, cfg.generation, cfg.seed)
            return {"traces": torch.tensor(list(json.dumps(traces).encode()), dtype=torch.uint8)}

        stored = cache.get_or_compute(key, compute)["traces"]
        return json.loads(bytes(stored.tolist()).decode())

    return reasoning_mixture(problems, texts, generate, tokenizer, cfg.total, cfg.wiki_ratio, cfg.seed)


@method(
    "codequant2",
    title="CodeQuant2",
    version="1",
    field="quantization",
    config=CodeQuant2Config,
    uses=("quant_rules",),
    display=("rules",),
    default_preset="paper_a4w4_4b",
    description="CodeQuant2: per-row k-means weight clustering with adaptive cluster counts and learnable "
    "centroids and soft assignments on reasoning-aware calibration (ported from the official code).",
)
def codequant2(model: nn.Module, tokenizer, cfg: CodeQuant2Config) -> nn.Module:
    targets = _targets(model, [rule for rule in cfg.rules if isinstance(rule.quant, CodebookFormat)])
    counts = {t.name: t.fmt.centroids for t in targets}
    if cfg.aca is not None:
        groups = len({t.layer // cfg.aca.layers_per_group for t in targets})
        counts = choose_cluster_counts(
            model, targets, aca_batches(cfg.aca, tokenizer, groups), cfg.aca, cfg.act_bits
        )
        logger.info("ACA counts: %s", {k: sum(c == k for c in counts.values()) for k in cfg.aca.choices})
    rac = cfg.train.calibration
    rows = reasoning_rows(model, tokenizer, rac)
    epochs = _row_batches(
        rows,
        tokenizer,
        batch_size=rac.batch_size,
        epochs=rac.epochs,
        seed=rac.shuffle_seed,
        max_length=rac.max_length,
    )
    history = train_clusters(
        model, targets, counts, [batch for epoch in epochs for batch in epoch], cfg.train, cfg.act_bits
    )
    first, last = (sum(series[i] for series in history["nmse"].values()) / len(targets) for i in (0, -1))
    logger.info("mean training nmse per target: first step %.4g, last step %.4g", first, last)
    activations = [rule for rule in cfg.rules if not isinstance(rule.quant, CodebookFormat)]
    if activations:
        sites.attach(model, activations)
    return model
