"""Layer-wise training of decoder layers against their full-precision outputs.

Ported from the layer loop of optimize.py and `optimize_module` of paroquant/optimize.py
(github.com/z-lab/paroquant@8e5ac77).

For every decoder layer the full-precision layer provides targets from full-precision inputs, the layer is
rewritten by `replace_fn`, trained in float32 (float16 autocast on CUDA) on inputs produced by the already
processed earlier layers, and the best-validation state is kept.

Shape symbols: N calibration sequences, L sequence length, Dm hidden size, b batch size.
"""

from __future__ import annotations

import contextlib
import gc
import math
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import nn

from effeval.ops import sites

_CPU = torch.device("cpu")


class _StopForward(Exception):
    pass


class CosineAnnealingParam:
    """Cosine schedule from `start_value` to `end_value` over `T_max` calls of `step()`."""

    def __init__(self, start_value: float, end_value: float, T_max: int):
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        """The value for the next step; the first call returns `start_value`."""
        self._step += 1
        if self._step >= self.T_max:
            return self.end_value
        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


class CachedTensorShards:
    """A list of tensors kept on `offload_device` of which one of `num_shards` contiguous shards sits on
    `target_device` at a time. Indexing in order moves each shard once per pass."""

    def __init__(
        self,
        batches: list[torch.Tensor],
        num_shards: int,
        *,
        target_device: torch.device,
        offload_device: torch.device = _CPU,
    ):
        if len(batches) % num_shards != 0:
            raise ValueError(f"{len(batches)} batches do not split into {num_shards} equal shards")
        self.batches = [b.to(offload_device) for b in batches]
        self.num_shards = num_shards
        self.current_shard: int | None = None
        self.cached_shard: list[torch.Tensor] = []
        self.target_device = target_device

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_len = len(self.batches) // self.num_shards
        shard_index = index // shard_len
        if self.current_shard != shard_index:
            start = shard_len * shard_index
            end = len(self.batches) if shard_index == self.num_shards - 1 else shard_len * (shard_index + 1)
            self.current_shard = shard_index
            self.cached_shard = [b.to(self.target_device) for b in self.batches[start:end]]
        return self.cached_shard[index % shard_len]

    def __iter__(self) -> Iterator[torch.Tensor]:
        for index in range(len(self.batches)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.batches)


@dataclass(frozen=True)
class Stage:
    """One optimisation stage of `epochs` passes over the training set.

    `param_groups(layer)` enables the trainable parameters of the layer and returns the AdamW parameter groups
    (each with its own `lr`).
    """

    param_groups: Callable[[nn.Module], list[dict]]
    epochs: int


@dataclass
class OptimizeResult:
    step_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    original_val_loss: float = math.nan
    best_val_loss: float = math.nan


def _loss_fn(name: Literal["mse", "smooth_l1"]) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"unsupported loss function: {name}")


def _layer_output(layer: nn.Module, hidden: torch.Tensor, kwargs: dict) -> torch.Tensor:
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def optimize_layer(
    layer: nn.Module,
    kwargs: dict,
    train: tuple[CachedTensorShards | list[torch.Tensor], CachedTensorShards | list[torch.Tensor]],
    val: tuple[list[torch.Tensor], list[torch.Tensor]],
    param_groups: list[dict],
    *,
    loss: Literal["mse", "smooth_l1"],
    epochs: int,
    post_step: Callable[[nn.Module], None] | None = None,
) -> OptimizeResult:
    """Train `layer` to reproduce the target batches from the input batches, keeping the best state.

    `train` and `val` are (input batches, target batches). AdamW over `param_groups` with a cosine decay of
    every group's lr to lr/20 over all steps; float16 autocast with loss scaling on CUDA, float32 on CPU.
    The validation loss is evaluated before training and after every epoch; the layer's state at the lowest
    validation loss (the initial state if none improves) is loaded at the end. `post_step(layer)` runs after
    every optimizer step. Gradients are enabled for the training forward and backward whatever the caller's
    grad mode (a `torch.no_grad` caller is fine; `torch.inference_mode` is not supported).
    """
    train_in, train_out = train
    val_in, val_out = val
    steps = epochs * len(train_in)
    schedulers = [CosineAnnealingParam(g["lr"], g["lr"] / 20, steps) for g in param_groups]
    optimizer = torch.optim.AdamW(param_groups)
    device_type = next(layer.parameters()).device.type
    cuda = device_type == "cuda"
    scaler = torch.amp.GradScaler(device_type, enabled=cuda)
    criterion = _loss_fn(loss)
    result = OptimizeResult()

    @torch.no_grad()
    def val_loss() -> torch.Tensor:
        total = None
        for inp, target in zip(val_in, val_out, strict=True):
            value = criterion(target, _layer_output(layer, inp, kwargs))
            total = value if total is None else total + value
        return total / len(val_in)

    original = val_loss()
    result.original_val_loss = result.best_val_loss = original.item()
    best_val = original
    best_state = deepcopy(layer.state_dict())

    for _ in range(epochs):
        for inp, target in zip(train_in, train_out, strict=True):
            optimizer.zero_grad()
            with torch.enable_grad():
                with torch.autocast(device_type, enabled=cuda):
                    value = criterion(target, _layer_output(layer, inp, kwargs))
                scaler.scale(value).backward()
            scaler.step(optimizer)
            scaler.update()
            result.step_losses.append(value.item())
            for index, scheduler in enumerate(schedulers):
                optimizer.param_groups[index]["lr"] = scheduler.step()
            if post_step is not None:
                post_step(layer)
        current = val_loss()
        result.val_losses.append(current.item())
        if current < best_val:
            best_val = current
            result.best_val_loss = current.item()
            best_state = deepcopy(layer.state_dict())
    layer.load_state_dict(best_state)
    return result


def _empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def _capture_first_input(
    model: nn.Module, samples: torch.Tensor, batch_size: int
) -> tuple[list[torch.Tensor], dict]:
    """Hidden states entering layer 0 for the batches `samples.chunk(N // batch_size)` of `samples` [N, L].

    Also returns the keyword arguments of the first layer call, without the KV cache.
    """
    layers = sites.decoder_layers(model)
    inputs: list[torch.Tensor] = []
    layer_kwargs: dict = {}

    def catch(_module, args, kwargs):
        inputs.append((args[0] if args else kwargs["hidden_states"]).cpu())
        if not layer_kwargs:
            layer_kwargs.update({k: v for k, v in kwargs.items() if k != "hidden_states"})
        raise _StopForward

    device = next(model.parameters()).device
    handle = layers[0].register_forward_pre_hook(catch, with_kwargs=True)
    try:
        for batch in samples.chunk(samples.shape[0] // batch_size):
            with contextlib.suppress(_StopForward):
                model(batch.to(device))
    finally:
        handle.remove()
    layer_kwargs["use_cache"] = False
    layer_kwargs.pop("past_key_value", None)
    layer_kwargs.pop("past_key_values", None)
    return inputs, layer_kwargs


@torch.no_grad()
def _forward_batches(
    layer: nn.Module, batches: list[torch.Tensor], kwargs: dict, dtype: torch.dtype
) -> list[torch.Tensor]:
    device = next(layer.parameters()).device
    outputs = []
    for batch in batches:
        out = _layer_output(layer, batch.to(dtype).to(device), kwargs)
        outputs.append(out.to(dtype).cpu())
    _empty_cache()
    return outputs


def train_layerwise(
    model: nn.Module,
    samples: torch.Tensor,
    val_samples: torch.Tensor,
    *,
    replace_fn: Callable[[int, nn.Module], bool],
    finalize_fn: Callable[[int, nn.Module, bool], None],
    stages: list[Stage],
    loss: Literal["mse", "smooth_l1"],
    batch_size: int,
    cache_shards: int,
    val_batch_size: int | None = None,
    post_step: Callable[[nn.Module], None] | None = None,
) -> list[list[OptimizeResult]]:
    """Process the decoder layers of `model` in order; returns the per-layer, per-stage `OptimizeResult`s.

    `samples` [N, L] and `val_samples` [N_val, L] are token ids. Per layer: targets are the layer's own
    outputs on full-precision inputs; `replace_fn(index, layer)` installs the trainable form and returns True
    when it needs training (False when restored from a saved result); training inputs are the outputs of the
    already finalized earlier layers (the layer's original inputs for layer 0). `finalize_fn(index, layer,
    trained)` turns the layer back into plain linears while a trained layer is still in float32; the layer is
    then cast to the model dtype and the next layer's inputs come from it. The model stays on its device;
    cached activations live on the CPU, `cache_shards` of the training set at a time on the device. The model
    is expected on one device.
    """
    layers = sites.decoder_layers(model)
    device = next(model.parameters()).device
    dtype = next(layers[0].parameters()).dtype
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        fp_train, kwargs = _capture_first_input(model, samples, batch_size)
        fp_val, _ = _capture_first_input(model, val_samples, val_batch_size or batch_size)
        quantized_train, quantized_val = fp_train, fp_val
        history: list[list[OptimizeResult]] = []

        for index, layer in enumerate(layers):
            _empty_cache()
            fp_train_out = _forward_batches(layer, fp_train, kwargs, dtype)
            fp_val_out = _forward_batches(layer, fp_val, kwargs, dtype)

            for param in layer.parameters():
                param.requires_grad = False
            trained = replace_fn(index, layer)
            results: list[OptimizeResult] = []
            if trained:
                layer.to(device).float()
                train_in = CachedTensorShards(quantized_train, cache_shards, target_device=device)
                train_out = CachedTensorShards(fp_train_out, cache_shards, target_device=device)
                val_in = [b.to(device) for b in quantized_val]
                val_out = [b.to(device) for b in fp_val_out]
                for stage in stages:
                    _empty_cache()
                    results.append(
                        optimize_layer(
                            layer,
                            kwargs,
                            (train_in, train_out),
                            (val_in, val_out),
                            stage.param_groups(layer),
                            loss=loss,
                            epochs=stage.epochs,
                            post_step=post_step,
                        )
                    )
                del train_in, train_out, val_in, val_out
                _empty_cache()
            finalize_fn(index, layer, trained)
            layer.to(dtype).to(device)
            history.append(results)

            new_train = _forward_batches(layer, quantized_train, kwargs, dtype)
            new_val = _forward_batches(layer, quantized_val, kwargs, dtype)
            fp_train, fp_val = fp_train_out, fp_val_out
            quantized_train, quantized_val = new_train, new_val
        return history
    finally:
        model.config.use_cache = use_cache
