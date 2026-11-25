"""Layer-major execution of a step schedule that the reference implementation runs batch-major.

A batch-major loop runs the whole model once per step and updates every layer after the pass. Layer l's
update at step s depends only on its inputs at step s and its own state after step s - 1, so the same states
and outputs come out of processing one layer at a time over all steps. `run_streams` supplies the inputs.

Shape symbols: B unique batches, T steps (len(step_order)).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from effeval.ops import sites
from effeval.ops.layerwise import _empty_cache, _StopForward

LayerFn = Callable[[int, nn.Module, list[torch.Tensor], list[torch.Tensor], list[dict]], tuple[list, list]]


def move_to(value: Any, device: torch.device | str) -> Any:
    """`value` with every tensor inside nested tuples, lists and dicts moved to `device`."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(move_to(v, device) for v in value)
    if isinstance(value, list):
        return [move_to(v, device) for v in value]
    if isinstance(value, dict):
        return {k: move_to(v, device) for k, v in value.items()}
    return value


def _to_stream(value: Any) -> Any:
    value = move_to(value, "cpu")
    if isinstance(value, torch.Tensor) and torch.cuda.is_available():
        return value.pin_memory()
    return value


@torch.no_grad()
def capture_inputs(model: nn.Module, batches: Sequence[dict]) -> tuple[list[torch.Tensor], list[dict]]:
    """Hidden states entering decoder layer 0 and the keyword arguments of that call, one entry per batch.

    `batches[b]` holds the model inputs (e.g. `input_ids`, `attention_mask`) of batch b. Results live on the
    CPU (pinned when CUDA is available); the KV cache arguments are dropped and `use_cache` is False.
    """
    first = sites.decoder_layers(model)[0]
    inputs: list[torch.Tensor] = []
    kwargs: list[dict] = []

    def catch(_module, args, layer_kwargs):
        hidden = args[0] if args else layer_kwargs["hidden_states"]
        inputs.append(_to_stream(hidden))
        kept = {k: v for k, v in layer_kwargs.items() if k != "hidden_states"}
        kept.pop("past_key_value", None)
        kept.pop("past_key_values", None)
        kept["use_cache"] = False
        kwargs.append(_to_stream(kept))
        raise _StopForward

    device = next(model.parameters()).device
    handle = first.register_forward_pre_hook(catch, with_kwargs=True)
    try:
        for batch in batches:
            with contextlib.suppress(_StopForward):
                model(**move_to(batch, device), use_cache=False)
    finally:
        handle.remove()
    return inputs, kwargs


def run_streams(
    model: nn.Module, batches: Sequence[dict], step_order: Sequence[int], layer_fn: LayerFn
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Process the decoder layers of `model` in order over the steps `step_order`; returns the last outputs.

    Step s uses batch `step_order[s]`. For layer `index`, `layer_fn(index, layer, fp_in, q_in, kwargs)` gets
    `fp_in` (B tensors: full-precision-path inputs per unique batch), `q_in` (T tensors: quantized-path
    inputs, entry s being what the already processed earlier layers produced at step s) and `kwargs` (T
    dicts: the layer call arguments of step s's batch). It returns `(fp_out, q_out)` with B and T entries,
    which become the next layer's `fp_in` and `q_in`. All streams are on the CPU and shared between entries
    of the same batch, so `layer_fn` must not modify them in place. The model stays where it was loaded.
    """
    if not step_order or not all(0 <= b < len(batches) for b in step_order):
        raise ValueError(f"step_order must be non-empty with indices in [0, {len(batches)})")
    layers = sites.decoder_layers(model)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        fp_in, batch_kwargs = capture_inputs(model, batches)
        q_in = [fp_in[b] for b in step_order]
        step_kwargs = [batch_kwargs[b] for b in step_order]
        for index, layer in enumerate(layers):
            _empty_cache()
            fp_out, q_out = layer_fn(index, layer, fp_in, q_in, step_kwargs)
            if len(fp_out) != len(fp_in) or len(q_out) != len(q_in):
                raise ValueError(
                    f"layer {index}: layer_fn returned {len(fp_out)} fp and {len(q_out)} quantized outputs, "
                    f"expected {len(fp_in)} and {len(q_in)}"
                )
            fp_in, q_in = fp_out, q_out
        return fp_in, q_in
    finally:
        model.config.use_cache = use_cache
