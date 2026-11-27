"""Calibration data and layer-by-layer execution shared by the methods that calibrate on activations.

Shape symbols: L calibration sequence length, Dm hidden size, C channels of one linear input.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Literal

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding

from effeval.core.fragments import (
    Calibration,
    IntFormat,
    QuantFormat,
    QuantRule,
    TextBatches,
    select_layers,
)
from effeval.ops import online, sites
from effeval.ops.quant.quant import fake_quantize

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


def text_batches(cfg: TextBatches, tokenizer) -> list[list[BatchEncoding]]:
    """Tokenized batches per epoch: result[epoch][step], on CPU.

    Rows are the `text` column of the first n_samples rows of the split. Each epoch is one pass of a
    `DataLoader(shuffle=True)` whose generator is seeded once with `shuffle_seed`, so epochs differ and the
    sequence of batches equals the official CodeQuant loaders. A row becomes one user turn of the chat
    template when `chat_template` is set; each batch is tokenized with truncation to `max_length`, `padding`.
    `drop_empty` drops blank rows first; `pre_shuffle_seed` reorders the selected rows before the loader;
    `skip_epochs` passes over the loader without tokenizing before the first returned epoch.
    """
    rows = load_dataset(cfg.dataset, split=cfg.split)
    if cfg.drop_empty:
        texts = [text for text in rows["text"] if text.strip()][: cfg.n_samples]
    else:
        texts = rows.select(range(min(cfg.n_samples, len(rows))))["text"]
    if cfg.pre_shuffle_seed is not None:
        rng = random.Random(cfg.pre_shuffle_seed)
        texts = [texts[i] for i in rng.sample(range(len(texts)), len(texts))]
        rng.shuffle(texts)
    generator = torch.Generator()
    generator.manual_seed(cfg.shuffle_seed)
    loader = DataLoader(texts, batch_size=cfg.batch_size, shuffle=True, generator=generator, collate_fn=list)
    template_options = {} if cfg.enable_thinking is None else {"enable_thinking": cfg.enable_thinking}
    for _ in range(cfg.skip_epochs):
        for _ in loader:
            pass
    epochs: list[list[BatchEncoding]] = []
    for _ in range(cfg.epochs):
        batches = []
        for batch in loader:
            if cfg.chat_template:
                batch = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True,
                        **template_options,
                    )
                    for text in batch
                ]
            batches.append(
                tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True if cfg.padding == "longest" else "max_length",
                    truncation=True,
                    max_length=cfg.max_length,
                )
            )
        epochs.append(batches)
    return epochs


def reasoning_mixture(
    problems: list[str],
    wiki_texts: list[str],
    generate: Callable[[list[str]], list[str]],
    tokenizer,
    total: int,
    wiki_ratio: float,
    seed: int = 42,
) -> list[str]:
    """CodeQuant2's reasoning-aware calibration texts: model-written traces mixed with plain text, shuffled.

    `int(total * wiki_ratio)` rows are drawn from the non-empty `wiki_texts` (raw text) and the rest from
    `problems`, drawn with `random.Random(seed)` as the official `MixedDataset` does. Only the drawn problems
    are generated: `generate` receives their chat-templated prompts (user turn, generation prompt, thinking
    enabled) and returns one completion each. A trace becomes a chat with the problem as user turn and the
    completion as assistant turn. Raises ValueError when fewer than `total` rows are available.
    """
    wiki_texts = [text for text in wiki_texts if text.strip()]
    rng = random.Random(seed)
    wiki_count = min(int(total * wiki_ratio), len(wiki_texts))
    generated_count = min(total - wiki_count, len(problems))
    if wiki_count + generated_count < total:
        raise ValueError(
            f"requested {total} rows but only {generated_count} problems and {wiki_count} wiki rows exist"
        )
    chosen_problems = [problems[i] for i in rng.sample(range(len(problems)), generated_count)]
    chosen_wiki = [wiki_texts[i] for i in rng.sample(range(len(wiki_texts)), wiki_count)]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        for problem in chosen_problems
    ]
    completions = generate(prompts) if prompts else []
    if len(completions) != len(prompts):
        raise ValueError(f"generate returned {len(completions)} completions for {len(prompts)} prompts")
    rows = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}, {"role": "assistant", "content": completion}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        for problem, completion in zip(chosen_problems, completions, strict=True)
    ] + chosen_wiki
    rng.shuffle(rows)
    return rows


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
    stat: Literal["absmax", "absmean", "window_absmean_sum", "gram"],
    prequant: QuantFormat | None = None,
) -> dict[tuple[str, int | None], torch.Tensor]:
    """Per-channel statistic of the tensors at activation sites over all calibration tokens.

    `site_patterns` are fnmatch patterns over `linear.<name>.input` / `.output` sites, as in quant rules.
    absmax: max |x| per channel; absmean: mean |x| per channel over every token of every window;
    window_absmean_sum: sum over windows of the per-window token mean of |x| (ASVD / QSVD `abs_mean`), which
    needs every hooked call to hold one window; gram: sum over all calls and tokens of x^T x (float32 [C, C]),
    with every observed tensor first fake-quantized as an activation by `prequant` when given (`prequant` is
    only allowed with gram). Returns float32 [C] tensors ([C, C] for gram) keyed by (site, layer). Raises
    sites.SiteError when a pattern matches nothing.
    """
    if stat not in ("absmax", "absmean", "window_absmean_sum", "gram"):
        raise ValueError(f"unsupported statistic {stat!r}")
    if prequant is not None and stat != "gram":
        raise ValueError("prequant is only supported with the gram statistic")
    if not all(pattern.endswith((".input", ".output")) for pattern in site_patterns):
        raise ValueError("act_stats observes linear input or output sites")
    # `resolve` only reads the patterns; the format is a placeholder to build a valid rule.
    placeholder = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token")
    assignment, _ = sites.resolve(model, [QuantRule(sites=site_patterns, quant=placeholder)])

    maxima: dict[tuple[str, int | None], torch.Tensor] = {}
    sums: dict[tuple[str, int | None], torch.Tensor] = {}
    counts: dict[tuple[str, int | None], int] = {}
    window_sums: dict[tuple[str, int | None], torch.Tensor] = {}
    grams: dict[tuple[str, int | None], torch.Tensor] = {}

    def observe(key, tensor: torch.Tensor) -> None:
        if stat == "gram":
            x = tensor.detach()
            x = (fake_quantize(x, prequant, "activation") if prequant is not None else x).float()
            if x.dim() == 2:
                x = x.unsqueeze(0)
            add = torch.matmul(x.transpose(1, 2), x).sum(dim=0)
            grams[key] = add if key not in grams else grams[key] + add
            return
        if stat == "window_absmean_sum":
            if tensor.shape[0] != 1:
                raise ValueError(f"window_absmean_sum needs one window per call, got batch {tensor.shape[0]}")
            value = tensor.abs().mean(dim=-2).reshape(-1).float()
            window_sums[key] = value if key not in window_sums else window_sums[key] + value
            return
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
    if stat == "window_absmean_sum":
        return window_sums
    if stat == "gram":
        return grams
    return {key: (sums[key] / counts[key]).float() for key in sums}


def causal_lm_grads(
    model: nn.Module,
    batches: Sequence[torch.Tensor | Mapping[str, torch.Tensor]],
    params: Sequence[nn.Parameter],
    on_batch: Callable[[], None],
) -> int:
    """Backpropagates the causal-LM loss per batch and calls `on_batch` while `params` hold its gradient.

    A tensor batch means `{"input_ids": batch}`; the labels are `batch["labels"]` when present, else the
    `input_ids`. Only `params` require gradients during the call; `p.grad` is cleared after every batch. The
    model runs in eval mode with `use_cache=False`; every parameter's `requires_grad`, `model.training` and
    `config.use_cache` are restored on exit. Returns the number of batches. Raises ValueError when the model
    has dropout (`nn.Dropout` with p > 0 or `config.attention_dropout` > 0).
    """
    if (
        any(isinstance(m, nn.Dropout) and m.p > 0 for m in model.modules())
        or (getattr(model.config, "attention_dropout", 0.0) or 0.0) > 0
    ):
        raise ValueError("gradient statistics assume a model without dropout")
    all_params = list(model.parameters())
    requires_grad = [p.requires_grad for p in all_params]
    training = model.training
    use_cache = model.config.use_cache
    trainable = {id(p) for p in params}
    device = next(model.parameters()).device
    count = 0
    try:
        for p in all_params:
            p.requires_grad_(id(p) in trainable)
        model.eval()
        model.config.use_cache = False
        with torch.enable_grad():
            for batch in batches:
                inputs = {"input_ids": batch} if isinstance(batch, torch.Tensor) else dict(batch)
                inputs = {name: value.to(device) for name, value in inputs.items()}
                labels = inputs.pop("labels", inputs["input_ids"])
                loss = model(**inputs, labels=labels).loss
                loss.backward()
                on_batch()
                for p in params:
                    p.grad = None
                count += 1
    finally:
        for p, flag in zip(all_params, requires_grad, strict=True):
            p.requires_grad_(flag)
        model.train(training)
        model.config.use_cache = use_cache
    return count


def output_grad_grams(
    model: nn.Module,
    batches: Sequence[torch.Tensor | Mapping[str, torch.Tensor]],
    targets: Mapping[str, tuple[nn.Module, int, int]],
    params: Sequence[nn.Parameter],
) -> dict[str, torch.Tensor]:
    """Per-head Gram of the loss gradient at module outputs, summed over all batches and tokens.

    `targets` maps a key to (module, heads, head_dim); the module output must be [B, T, heads * head_dim].
    Gradients flow as in `causal_lm_grads(model, batches, params)`, so each `params` entry that lies below a
    target must be listed there. Returns float32 [heads, head_dim, head_dim] tensors keyed like `targets`; the
    hooks are removed on exit.
    """
    grams: dict[str, torch.Tensor] = {}

    def observe(key: str, heads: int, head_dim: int, grad_out: torch.Tensor) -> None:
        g = grad_out.detach()
        g = g.view(g.shape[0], -1, heads, head_dim).permute(0, 2, 1, 3).contiguous().float()
        cov = torch.matmul(g.transpose(-1, -2), g).sum(dim=0)
        grams[key] = cov if key not in grams else grams[key] + cov

    handles = [
        module.register_full_backward_hook(
            lambda _m, _gin, gout, key=key, h=heads, d=head_dim: observe(key, h, d, gout[0])
        )
        for key, (module, heads, head_dim) in targets.items()
    ]
    try:
        causal_lm_grads(model, batches, params, on_batch=lambda: None)
    finally:
        for handle in handles:
            handle.remove()
    return grams


@torch.no_grad()
def collect_input_rows(
    model: nn.Module,
    windows: list[torch.Tensor],
    site_pattern: str,
    rows: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Selected token rows of the input of a linear site, per decoder layer.

    `rows` are 1-D indices into the token stream formed by concatenating the windows in order (token t of
    window w is row `sum of earlier window sizes + t`). Returns, per layer, a CPU tensor [len(rows), C] in the
    activation dtype holding those rows in the order given; no other token is stored. Raises ValueError when
    a row lies outside the stream and sites.SiteError when the pattern matches nothing.
    """
    if not site_pattern.endswith(".input"):
        raise ValueError("collect_input_rows observes a linear input site")
    total = sum(window.numel() for window in windows)
    if rows.dim() != 1 or (rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= total)):
        raise ValueError(f"rows must be a 1-D index into the {total}-token stream")
    placeholder = IntFormat(type="int", bits=8, symmetric=True, granularity="per_token")
    assignment, _ = sites.resolve(model, [QuantRule(sites=[site_pattern], quant=placeholder)])

    out: dict[int, torch.Tensor] = {}
    offsets: dict[int, int] = {}

    def observe(layer: int, tensor: torch.Tensor) -> None:
        flat = tensor.reshape(-1, tensor.shape[-1])
        start = offsets.get(layer, 0)
        offsets[layer] = start + flat.shape[0]
        hit = ((rows >= start) & (rows < start + flat.shape[0])).nonzero().squeeze(-1)
        if layer not in out:
            out[layer] = torch.empty(rows.numel(), flat.shape[-1], dtype=flat.dtype)
        out[layer][hit] = flat[(rows[hit] - start).to(flat.device)].cpu()

    handles = [
        location.module.register_forward_pre_hook(lambda _m, args, layer=key[1]: observe(layer, args[0]))
        for key, (_rule, location, _fmt) in assignment.items()
    ]
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
    return out


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


def paroquant_blocks(
    tokenizer_id: str, dataset: str, n_samples: int, seqlen: int, seed: int, split: str
) -> list[torch.Tensor]:
    """ParoQuant calibration blocks (paroquant/util.py `get_calib_dataset`): n_samples tensors [seqlen].

    The corpus (`pileval`, `wikitext2`, `c4` or `redpajama`) is shuffled with `seed`; stripped documents
    longer than seqlen tokens or empty are skipped, the rest concatenated until n_samples * seqlen tokens and
    cut into seqlen blocks (fewer blocks when the corpus is exhausted). `split` is ignored for `pileval`. c4
    reads the first training file (`train`) or validation file 1 (`validation`); redpajama is split 70/10/20
    into train/validation/test after shuffling.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if dataset == "pileval":
        corpus = load_dataset(PILE_VAL, split="validation").shuffle(seed=seed)
    elif dataset == "wikitext2":
        corpus = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split).shuffle(seed=seed)
    elif dataset == "c4":
        files = {
            "train": "en/c4-train.00000-of-01024.json.gz",
            "validation": "en/c4-validation.00001-of-00008.json.gz",
        }
        if split not in files:
            raise ValueError(f"c4 has the splits train and validation, got {split!r}")
        corpus = load_dataset("allenai/c4", data_files={split: files[split]}, split=split).shuffle(seed=seed)
    elif dataset == "redpajama":
        corpus = load_dataset(
            "liang2kl/RedPajama-Data-1T-Sample-Backup", split="train", trust_remote_code=True
        ).shuffle(seed=seed)
        test_size = int(len(corpus) * 0.2)
        val_size = int(len(corpus) * 0.1)
        train_size = len(corpus) - test_size - val_size
        if split == "test":
            corpus = corpus.select(range(len(corpus) - test_size, len(corpus)))
        elif split == "validation":
            corpus = corpus.select(range(len(corpus) - test_size - val_size, len(corpus) - test_size))
        elif split == "train":
            corpus = corpus.select(range(0, train_size))
        else:
            raise ValueError(f"invalid split {split!r}")
    else:
        raise ValueError(f"unsupported calibration dataset {dataset!r}")

    samples: list[torch.Tensor] = []
    total_len = 0
    for row in corpus:
        encoded = tokenizer.encode(row["text"].strip())
        if len(encoded) > seqlen or len(encoded) == 0:
            continue
        samples.append(torch.tensor([encoded]))
        total_len += len(encoded)
        if total_len >= n_samples * seqlen:
            break
    joined = torch.cat(samples, dim=1).squeeze(0)  # [total tokens]
    n_split = min(joined.shape[0] // seqlen, n_samples)
    return [joined[i * seqlen : (i + 1) * seqlen] for i in range(n_split)]


def paroquant_mixture(
    tokenizer_id: str, datasets: list[str], n_samples: int, seqlen: int, seed: int, split: str
) -> list[torch.Tensor]:
    """ParoQuant training mixture (paroquant/util.py `get_mixed_calib_dataset`): n_samples tensors [seqlen].

    Each dataset contributes n_samples // len(datasets) blocks (the last one the remainder); the result is
    shuffled with `random.Random(seed)`. Raises ValueError if a dataset yields fewer blocks than requested.
    """
    per_dataset = n_samples // len(datasets)
    results: list[torch.Tensor] = []
    for position, dataset in enumerate(datasets):
        count = per_dataset if position < len(datasets) - 1 else n_samples - len(results)
        results.extend(paroquant_blocks(tokenizer_id, dataset, count, seqlen, seed, split))
    if len(results) != n_samples:
        raise ValueError(f"expected {n_samples} samples, got {len(results)}")
    random.Random(seed).shuffle(results)
    return results
