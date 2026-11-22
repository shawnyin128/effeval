"""Calibration data and layer-by-layer execution shared by the methods that calibrate on activations.

Shape symbols: L calibration sequence length, Dm hidden size, C channels of one linear input.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator
from typing import Literal

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer

from effeval.core.fragments import Calibration, IntFormat, QuantRule, select_layers
from effeval.ops import online, sites

PILE_VAL = "mit-han-lab/pile-val-backup"


def calibration_windows(calibration: Calibration, tokenizer_id: str) -> list[torch.Tensor]:
    """Token sequences [1, L] for a calibration fragment.

    `windows`: random windows exactly as the official GPTQ datautils loaders draw them. The official loaders
    seed Python's global `random`; `random.Random(seed)` yields the same draws. The official code passes the
    config name 'allenai--c4', which current `datasets` rejects; the data file is the same.
    `documents` and `concat_blocks` read the Pile validation set (see `Calibration`), as SmoothQuant
    (calibration.py `get_act_scales`) and AWQ (calib_data.py `get_calib_dataset`) do. `random_crop` reads it
    as QServe's deepcompressor does (see `_random_crop`).
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=False)
    seqlen = calibration.seqlen
    if calibration.dataset == "pile_val":
        dataset = load_dataset(PILE_VAL, split="validation").shuffle(seed=calibration.seed)
        if calibration.sampling == "documents":
            return [
                tokenizer(
                    dataset[i]["text"], return_tensors="pt", max_length=seqlen, truncation=True
                ).input_ids
                for i in range(calibration.n_samples)
            ]
        if calibration.sampling == "random_crop":
            return _random_crop(dataset, tokenizer, calibration.n_samples, seqlen, calibration.seed)
        return _concat_blocks(dataset, tokenizer, calibration.n_samples, seqlen)
    rng = random.Random(calibration.seed)
    windows: list[torch.Tensor] = []
    if calibration.dataset == "c4":
        traindata = load_dataset(
            "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train"
        )
        for _ in range(calibration.n_samples):
            while True:
                i = rng.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            windows.append(trainenc.input_ids[:, i : i + seqlen])
    elif calibration.dataset == "wikitext2":
        traindata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        for _ in range(calibration.n_samples):
            i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            windows.append(trainenc.input_ids[:, i : i + seqlen])
    else:
        raise ValueError(f"unsupported calibration dataset {calibration.dataset!r}")
    return windows


def _concat_blocks(dataset, tokenizer, n_samples: int, block_size: int) -> list[torch.Tensor]:
    """AWQ: take documents of at most block_size tokens until n_samples, concatenate, cut into blocks."""
    samples: list[torch.Tensor] = []
    for row in dataset:
        encoded = tokenizer.encode(row["text"].strip())
        if len(encoded) > block_size or len(encoded) == 0:
            continue
        samples.append(torch.tensor([encoded]))
        if len(samples) == n_samples:
            break
    if not samples:
        raise ValueError(f"no document with at most {block_size} tokens in the corpus")
    joined = torch.cat(samples, dim=1)  # [1, total tokens]
    return [joined[:, i * block_size : (i + 1) * block_size] for i in range(joined.shape[1] // block_size)]


def _random_crop(dataset, tokenizer, n_samples: int, seq_length: int, seed: int) -> list[torch.Tensor]:
    """QServe / deepcompressor `LlmCalibDataset`: crop long documents at random, concatenate, cut into blocks.

    Documents are read in the shuffled order of `dataset`; empty ones are skipped and one longer than
    `seq_length` tokens is cut to a random window of `seq_length` tokens (`random.Random(seed)`). Reading
    stops once there are n_samples documents and n_samples * seq_length tokens. The concatenation is cut into
    seq_length blocks, the last block is dropped when the total exceeds n_samples * seq_length, and at most
    n_samples blocks are kept (fewer when the corpus runs out). Returns [1, seq_length] tensors.
    """
    rng = random.Random(seed)
    num_tokens = n_samples * seq_length
    pieces: list[torch.Tensor] = []
    total = 0
    for row in dataset:
        line = tokenizer.encode(row["text"].strip())
        if len(line) == 0:
            continue
        piece = torch.tensor(line)
        if len(line) > seq_length:
            start = rng.randint(0, len(line) - seq_length)
            piece = piece[start : start + seq_length]
        pieces.append(piece)
        total += piece.numel()
        if len(pieces) >= n_samples and total >= num_tokens:
            break
    if not pieces:
        raise ValueError("no non-empty document in the corpus")
    blocks = list(torch.cat(pieces).split(seq_length))
    if total > num_tokens:
        blocks = blocks[:-1]
    return [block[None, :] for block in blocks[:n_samples]]


class _StopForward(Exception):
    pass


class LayerRunner:
    """Runs calibration sequences through the decoder layers one layer at a time.

    The inputs of layer 0 and the keyword arguments of the layer call (position embeddings, mask) are captured
    from the model's own forward pass. Use as a context manager and iterate: each step yields
    `(index, layer)` with `inputs` (one [1, L, Dm] tensor per window) holding the hidden states entering that
    layer. When the loop body finishes, `inputs` is replaced by the layer's outputs, computed after the body
    ran (`propagate="current"`, so edits to the layer take effect, as in GPTQ) or before it
    (`propagate="original"`, the unedited layer, as in AWQ). `use_cache` is disabled inside the context.
    """

    def __init__(
        self,
        model: nn.Module,
        windows: list[torch.Tensor],
        propagate: Literal["current", "original"] = "current",
    ):
        if propagate not in ("current", "original"):
            raise ValueError(f"propagate must be 'current' or 'original', got {propagate!r}")
        self.model = model
        self.windows = windows
        self.propagate = propagate
        self.layers = sites.decoder_layers(model)
        self.inputs: list[torch.Tensor] = []
        self.kwargs: dict = {}
        self._use_cache = model.config.use_cache

    def __enter__(self) -> LayerRunner:
        self.model.config.use_cache = False
        self._capture()
        return self

    def __exit__(self, *exc_info) -> None:
        self.model.config.use_cache = self._use_cache

    @torch.no_grad()
    def _capture(self) -> None:
        def catch(_module, args, kwargs):
            self.inputs.append(args[0] if args else kwargs.pop("hidden_states"))
            self.kwargs.clear()
            self.kwargs.update({k: v for k, v in kwargs.items() if k != "hidden_states"})
            raise _StopForward

        device = next(self.model.parameters()).device
        handle = self.layers[0].register_forward_pre_hook(catch, with_kwargs=True)
        try:
            for window in self.windows:
                with contextlib.suppress(_StopForward):
                    self.model(window.to(device))
        finally:
            handle.remove()

    def forward(self, layer: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        """One layer call on `hidden` [1, L, Dm] with the captured keyword arguments."""
        out = layer(hidden, **self.kwargs)
        if not isinstance(out, torch.Tensor):
            raise TypeError(
                f"decoder layer returned {type(out).__name__}; expected a tensor (transformers>=5)"
            )
        return out

    @torch.no_grad()
    def __iter__(self) -> Iterator[tuple[int, nn.Module]]:
        for index, layer in enumerate(self.layers):
            original = [self.forward(layer, h) for h in self.inputs] if self.propagate == "original" else None
            yield index, layer
            self.inputs = original if original is not None else [self.forward(layer, h) for h in self.inputs]


@torch.no_grad()
def act_stats(
    model: nn.Module,
    windows: list[torch.Tensor],
    site_patterns: list[str],
    stat: Literal["absmax", "absmean"],
) -> dict[tuple[str, int | None], torch.Tensor]:
    """Per-channel statistic of the tensors at activation sites over all calibration tokens.

    `site_patterns` are fnmatch patterns over `linear.<name>.input` / `.output` sites, as in quant rules.
    absmax: max |x| per channel; absmean: mean |x| per channel over every token of every window. Returns
    float32 [C] tensors keyed by (site, layer). Raises sites.SiteError when a pattern matches nothing.
    """
    if stat not in ("absmax", "absmean"):
        raise ValueError(f"unsupported statistic {stat!r}")
    if not all(pattern.endswith((".input", ".output")) for pattern in site_patterns):
        raise ValueError("act_stats observes linear input or output sites")
    # `resolve` only reads the patterns; the format is a placeholder to build a valid rule.
    placeholder = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token")
    assignment, _ = sites.resolve(model, [QuantRule(sites=site_patterns, quant=placeholder)])

    maxima: dict[tuple[str, int | None], torch.Tensor] = {}
    sums: dict[tuple[str, int | None], torch.Tensor] = {}
    counts: dict[tuple[str, int | None], int] = {}

    def observe(key, tensor: torch.Tensor) -> None:
        flat = tensor.reshape(-1, tensor.shape[-1]).abs().float()  # [tokens, C]
        if stat == "absmax":
            current = flat.amax(dim=0)
            maxima[key] = current if key not in maxima else torch.maximum(maxima[key], current)
        else:
            total = flat.sum(dim=0, dtype=torch.float64)
            sums[key] = total if key not in sums else sums[key] + total
            counts[key] = counts.get(key, 0) + flat.shape[0]

    handles = []
    for key, (_rule, location, _fmt) in assignment.items():
        if key[0].endswith(".input"):
            handles.append(
                location.module.register_forward_pre_hook(lambda _m, args, key=key: observe(key, args[0]))
            )
        else:
            handles.append(
                location.module.register_forward_hook(lambda _m, _args, out, key=key: observe(key, out))
            )
    use_cache = model.config.use_cache
    model.config.use_cache = False
    device = next(model.parameters()).device
    try:
        for window in windows:
            model(window.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.config.use_cache = use_cache
    if stat == "absmax":
        return maxima
    return {key: (sums[key] / counts[key]).float() for key in sums}


@torch.no_grad()
def attention_operand_absmax(
    model: nn.Module,
    windows: list[torch.Tensor],
    site: str,
    layers: str | list[int],
    extra_ops: list[online.OnlineOp],
) -> dict[int, torch.Tensor]:
    """Per-head, per-channel max |x| of an attention operand over all calibration tokens.

    `site` is `attn.q`, `attn.k` or `attn.v` (post-RoPE operands, [B, heads, T, D]). `extra_ops` are
    registered first and an observer last, so the observer sees the operand after the extra ops (and any
    earlier attention op on the same operand chained in order). Returns float32 [heads, D] tensors keyed by
    layer index, for the layers selected by `layers`. The attention backend and use_cache are restored on
    exit.
    """
    if site not in ("attn.q", "attn.k", "attn.v"):
        raise ValueError(f"attention operand site expected, got {site!r}")
    maxima: dict[int, torch.Tensor] = {}

    def observer(layer: int):
        def observe(x: torch.Tensor) -> torch.Tensor:
            current = x.detach().abs().float().amax(dim=(0, 2))  # [heads, D]
            maxima[layer] = current if layer not in maxima else torch.maximum(maxima[layer], current)
            return x

        return observe

    selected = select_layers(layers, len(sites.decoder_layers(model)))
    observers = [online.OnlineOp(site, observer(layer), [layer]) for layer in selected]
    attachment = online.register(model, [*extra_ops, *observers])
    use_cache = model.config.use_cache
    model.config.use_cache = False
    device = next(model.parameters()).device
    try:
        for window in windows:
            model(window.to(device), use_cache=False)
    finally:
        model.config.use_cache = use_cache
        attachment.remove()
    return maxima
