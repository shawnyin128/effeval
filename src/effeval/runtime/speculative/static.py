"""A causal LM over a preallocated static KV cache, with optional CUDA graphs per decode shape."""

from __future__ import annotations

import time
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.cache_utils import Cache, StaticLayer

from effeval.core.fragments import MxFormat, sequence_options
from effeval.runtime.speculative.static_attention import (
    EAGER_NAME,
    SDPA_NAME,
    register_static_attention,
)

# Attribute names shared with effeval.ops (the layers contract forbids importing it from here).
SITES_ATTR = "_effeval_attention_sites"
OPS_ATTR = "_effeval_attention_ops"
GRAPH_SAFE_ATTR = "_effeval_graph_safe"
FORMAT_ATTR = "_effeval_format"

HINT = "set runtime.cuda_graphs: false"
SDPA_IMPLS = frozenset({"sdpa", "effeval_sdpa", SDPA_NAME})
EAGER_IMPLS = frozenset({"eager", "effeval_eager", EAGER_NAME})
GROUPED_IMPL = {"sdpa": SDPA_NAME, "eager": EAGER_NAME}
UNSUPPORTED_SITES = frozenset({"attn.scores", "attn.probs"})
KEY_AXIS_GRANULARITIES = frozenset({"per_tensor", "per_channel"})
_HOST_SYNC = "MX scale e4m3 on {where} reads its tensor absmax on the host, which CUDA graphs cannot capture"


@dataclass(frozen=True)
class StaticCacheSettings:
    """Static-cache decoding setup shared by every model of a job.

    `capacity` is the number of cache positions per generation, `cuda_graphs` captures graphs for the
    decode shapes (False runs the same cache logic eagerly), `min_bucket` is the smallest attention span.
    `grouped_attention` replaces the model's plain `sdpa`/`eager` attention by GQA-grouped matmuls; models
    with effeval attention sites keep their attention function.
    """

    capacity: int
    cuda_graphs: bool
    min_bucket: int = 256
    grouped_attention: bool = True

    def __post_init__(self) -> None:
        if self.capacity < 1 or self.min_bucket < 1:
            raise ValueError(f"capacity and min_bucket must be positive, got {self}")


def buckets(capacity: int, min_bucket: int) -> list[int]:
    """Powers of two from `min_bucket` below `capacity`, then `capacity` itself (ascending)."""
    out = []
    size = min_bucket
    while size < capacity:
        out.append(size)
        size *= 2
    out.append(capacity)
    return out


@contextmanager
def _nondeterministic_writes() -> Iterator[None]:
    """Switch deterministic algorithms off inside the block and restore the previous setting."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    if enabled:
        torch.use_deterministic_algorithms(False)
    try:
        yield
    finally:
        if enabled:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)


class _BucketedStaticLayer(StaticLayer):
    """A static layer whose `update` returns only the first `view` cache positions.

    The `index_copy_` cache write runs with deterministic algorithms off: its indices are distinct, so the
    result is the same as the deterministic path, which sorts them first.
    """

    view: int

    def update(self, key_states, value_states, *args, **kwargs):
        with _nondeterministic_writes():
            keys, values = super().update(key_states, value_states, *args, **kwargs)
        return keys[:, :, : self.view], values[:, :, : self.view]


@dataclass
class _Graph:
    graph: torch.cuda.CUDAGraph
    ids: torch.Tensor
    out: torch.Tensor


class StaticCachedModel:
    """Same contract as `CachedModel` (model, cached, device, reset, feed, rollback) over a static cache.

    Positions are a device tensor, so one forward is graph-capturable. Attention spans a power-of-two bucket
    of the cache (the smallest one covering the tokens present). Rollback resets positions; the rejected rows
    stay in the cache, hidden by the causal mask, and are zeroed only when a key-axis quantizer would
    see them.
    Raises NotImplementedError for models with non-full attention layers or spread over several devices,
    ValueError for unsupported attention implementations, RuntimeError for hooks or sites the static path
    cannot reproduce (message ends with the runtime setting that falls back to the dynamic cache).
    """

    def __init__(self, model: nn.Module, settings: StaticCacheSettings):
        config = model.config
        if any(t != "full_attention" for t in getattr(config, "layer_types", None) or ()):
            raise NotImplementedError("StaticCachedModel supports full-attention layers only")
        impl = config._attn_implementation
        if impl not in SDPA_IMPLS | EAGER_IMPLS:
            raise ValueError(f"static cache supports attention implementations sdpa and eager, not {impl!r}")
        self.model = model
        self.settings = settings
        self.cached = 0
        self._buckets = buckets(settings.capacity, settings.min_bucket)
        self._additive = impl in EAGER_IMPLS
        self._graphs: dict[tuple[int, int, int], _Graph] = {}
        self._pool = None
        self._capture_ms = 0.0
        self._prepared = False
        self._fingerprint: tuple = ()
        self._zero_stale = False
        _check_single_device(model)
        self._inspect()
        if settings.cuda_graphs and model.device.type != "cuda":
            raise RuntimeError(f"cuda_graphs needs a model on a CUDA device, got {model.device}; {HINT}")
        self.previous_attn_implementation = impl
        if settings.grouped_attention and impl in GROUPED_IMPL:
            register_static_attention()
            model.set_attn_implementation(GROUPED_IMPL[impl])
        with torch.inference_mode():
            self.cache = Cache(
                layers=[
                    _BucketedStaticLayer(max_cache_len=settings.capacity)
                    for _ in range(config.num_hidden_layers)
                ]
            )
            self._position = torch.zeros((), dtype=torch.long, device=model.device)
            self._kv_index = torch.arange(settings.capacity, device=model.device)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reserve(self, length: int) -> None:
        """Raise ValueError when a generation needing `length` cache positions exceeds the capacity."""
        if length > self.settings.capacity:
            raise ValueError(
                f"generation needs {length} cache positions, capacity is {self.settings.capacity}; "
                f"raise runtime.static_cache_tokens, or {HINT}"
            )

    def prepare(self, shapes: Iterable[tuple[int, int]]) -> dict[str, Any]:
        """Allocate the cache and, with cuda_graphs, capture a graph per (tokens, logits rows kept, bucket).

        Verifies every graph against the eager forward (exact logits equality). Call outside any timed region.
        Returns {"graphs", "capture_ms", "buckets", "capacity", "attention"}, the last being the name of the
        attention function in use. Raises RuntimeError when a replay differs.
        """
        shapes = sorted(set(shapes))
        for q, keep in shapes:
            if not 1 <= keep <= q:
                raise ValueError(f"shape (tokens={q}, keep={keep}) needs 1 <= keep <= tokens")
        with torch.inference_mode():
            self._allocate()
            if self.settings.cuda_graphs:
                started = time.perf_counter()
                for q, keep in shapes:
                    for bucket in self._buckets:
                        if bucket >= q and (q, keep, bucket) not in self._graphs:
                            self._capture(q, keep, bucket)
                self._self_check()
                self._capture_ms += (time.perf_counter() - started) * 1e3
            self._clear()
            self._inspect()
            self._prepared = True
        return {
            "graphs": len(self._graphs),
            "capture_ms": self._capture_ms,
            "buckets": list(self._buckets),
            "capacity": self.settings.capacity,
            "attention": self.model.config._attn_implementation,
        }

    def reset(self) -> None:
        """Drop the cached tokens; raises RuntimeError when hooks or sites changed since `prepare`."""
        self._inspect(compare=self._prepared)
        self.rollback(0)

    def feed(self, sequence: torch.Tensor, extra: torch.Tensor | None, keep: int) -> torch.Tensor:
        """Run `sequence[:, cached:]` followed by `extra` through the model and extend the cache.

        Returns the last `keep` logits rows, shape [keep, V], in float32 (float64 stays float64), as a fresh
        tensor. `extra` tokens are cached too. Raises ValueError when there is nothing new to run, `keep`
        exceeds the run tokens, or the cache capacity would be exceeded; RuntimeError when graphs are enabled
        and `prepare` was not called.
        """
        new = sequence[:, self.cached :]
        if extra is not None:
            new = torch.cat([new, extra.to(new.device)], dim=1)
        if new.shape[0] != 1 or new.shape[1] == 0:
            raise ValueError(f"feed needs a batch of 1 with an uncached token, got shape {tuple(new.shape)}")
        q = new.shape[1]
        if keep > q:
            raise ValueError(f"keep={keep} exceeds the {q} tokens run")
        if self.settings.cuda_graphs and not self._prepared:
            raise RuntimeError("graph mode needs prepare() before feed")
        end = self.cached + q
        self.reserve(end)
        bucket = self._bucket(end)
        with torch.inference_mode():
            captured = self._graphs.get((q, keep, bucket))
            if captured is not None:
                captured.ids.copy_(new)
                captured.graph.replay()
                logits = captured.out[0]
                logits = logits.to(torch.promote_types(logits.dtype, torch.float32), copy=True)
            else:
                logits = self._forward(new, keep, bucket)[0]
                logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        self.cached = end
        return logits

    def rollback(self, length: int) -> None:
        """Keep the first `length` cached tokens."""
        if not 0 <= length <= self.cached:
            raise ValueError(f"cannot roll back to {length} with {self.cached} cached")
        with torch.inference_mode():
            if self._zero_stale and self.cached > length:
                self._zero_rows(length, self.cached)
            self._set_length(length)

    def _bucket(self, length: int) -> int:
        return self._buckets[bisect_left(self._buckets, length)]

    def _set_length(self, length: int) -> None:
        for layer in self.cache.layers:
            layer.cumulative_length.fill_(length)
        self._position.fill_(length)
        self.cached = length

    def _zero_rows(self, start: int, stop: int) -> None:
        for layer in self.cache.layers:
            if layer.is_initialized:
                layer.keys[:, :, start:stop].zero_()
                layer.values[:, :, start:stop].zero_()

    def _forward(self, ids: torch.Tensor, keep: int, bucket: int) -> torch.Tensor:
        """One forward of `ids` [1, q] at the current position over a `bucket`-wide attention span."""
        q = ids.shape[1]
        positions = self._position + torch.arange(q, device=ids.device)  # [q]
        allowed = self._kv_index[:bucket][None, :] <= positions[:, None]  # [q, bucket]
        if self._additive:
            dtype = self.model.dtype
            mask = torch.zeros(allowed.shape, dtype=dtype, device=ids.device)
            mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]
        else:
            mask = allowed[None, None]
        for layer in self.cache.layers:
            layer.view = bucket
        out = self.model(
            input_ids=ids,
            position_ids=positions[None],
            attention_mask=mask,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=keep,
        )
        self._position.add_(q)
        return out.logits

    def _allocate(self) -> None:
        if self.cache.layers[0].is_initialized:
            return
        q = min(4, self.settings.capacity)
        self._set_length(0)
        self._forward(torch.zeros((1, q), dtype=torch.long, device=self.device), 1, self._bucket(q))
        self._set_length(0)

    def _clear(self) -> None:
        for layer in self.cache.layers:
            layer.keys.zero_()
            layer.values.zero_()
        self._set_length(0)

    def _capture(self, q: int, keep: int, bucket: int) -> None:
        ids = torch.zeros((1, q), dtype=torch.long, device=self.device)
        self._set_length(0)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                self._forward(ids, keep, bucket)
                self._set_length(0)
        torch.cuda.current_stream().wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            out = self._forward(ids, keep, bucket)
        self._set_length(0)
        torch.cuda.synchronize()
        self._graphs[(q, keep, bucket)] = _Graph(graph, ids, out)

    def _self_check(self) -> None:
        """Replay every graph and compare its logits with the eager forward on the same cache and inputs."""
        generator = torch.Generator().manual_seed(0)
        vocab = self.model.config.vocab_size
        for (q, keep, bucket), captured in sorted(self._graphs.items()):
            context = min(bucket - q, 8)
            self._set_length(0)
            if context:
                prefix = torch.randint(0, vocab, (1, context), generator=generator).to(self.device)
                self._forward(prefix, 1, self._bucket(context))
            probe = torch.randint(0, vocab, (1, q), generator=generator).to(self.device)
            eager = self._forward(probe, keep, bucket).clone()
            self._set_length(context)
            captured.ids.copy_(probe)
            captured.graph.replay()
            replayed = captured.out.clone()
            torch.cuda.synchronize()
            if not torch.equal(eager, replayed):
                raise RuntimeError(
                    f"CUDA graph replay differs from the eager forward for shape {(q, keep, bucket)}"
                )

    def _inspect(self, compare: bool = False) -> None:
        """Validate hooks and attention sites and refresh the stale-row flag; `compare` checks for changes."""
        fingerprint: list[Any] = [self.model.config._attn_implementation]
        zero_stale = False
        graphs = self.settings.cuda_graphs
        global_hooks = [
            *nn.modules.module._global_forward_pre_hooks.values(),
            *nn.modules.module._global_forward_hooks.values(),
        ]
        if graphs and global_hooks:
            raise RuntimeError(
                f"global module hooks are present; they would be skipped by CUDA graphs; {HINT}"
            )
        fingerprint.append(tuple(id(h) for h in global_hooks))
        for name, module in self.model.named_modules():
            hooks = [*module._forward_pre_hooks.values(), *module._forward_hooks.values()]
            if graphs:
                for hook in hooks:
                    if not getattr(hook, GRAPH_SAFE_ATTR, False):
                        raise RuntimeError(
                            f"module {name!r} has a forward hook that is not graph-safe ({hook!r}); {HINT}"
                        )
                    fmt = getattr(hook, FORMAT_ATTR, None)
                    if _reads_absmax_on_host(fmt):
                        problem = _HOST_SYNC.format(where="a module input or output")
                        raise RuntimeError(f"{name}: {problem}; {HINT}")
            sites = getattr(module, SITES_ATTR, None)
            ops = getattr(module, OPS_ATTR, None)
            if sites:
                zero_stale |= _check_attention_sites(name, sites, graphs)
            if hooks or sites is not None or ops is not None:
                fingerprint.append(
                    (name, tuple(id(h) for h in hooks), id(sites), tuple(sorted(sites or ())), id(ops))
                )
        if compare and tuple(fingerprint) != self._fingerprint:
            raise RuntimeError(
                f"hooks, attention sites or attention implementation changed after prepare; {HINT}"
            )
        self._fingerprint = tuple(fingerprint)
        self._zero_stale = zero_stale


def _reads_absmax_on_host(fmt: Any) -> bool:
    """True for a quantizer format that syncs with the host (MX with scale e4m3)."""
    return isinstance(fmt, MxFormat) and fmt.scale == "e4m3"


def _check_single_device(model: nn.Module) -> None:
    """Raise NotImplementedError when the model's parameters or `hf_device_map` span several devices."""
    devices = {str(p.device) for p in model.parameters()}
    for device in (getattr(model, "hf_device_map", None) or {}).values():
        devices.add(f"cuda:{device}" if isinstance(device, int) else str(device))
    if len(devices) > 1:
        raise NotImplementedError(
            f"the static cache supports single-device models only, this one spans {sorted(devices)}; {HINT}"
        )


def _check_attention_sites(name: str, sites: dict[str, Any], graphs: bool) -> bool:
    """Reject sites the static cache cannot reproduce; True when a quantizer reduces over the key axis."""
    key_axis = False
    for site, fmt in sites.items():
        if graphs and _reads_absmax_on_host(fmt):
            raise RuntimeError(f"{name}: {_HOST_SYNC.format(where=site)}; {HINT}")
        if site in UNSUPPORTED_SITES:
            raise RuntimeError(f"{name}: site {site} is not supported with the static cache; {HINT}")
        if site not in ("attn.k", "attn.v"):
            continue
        if sequence_options(fmt):
            raise RuntimeError(
                f"{name}: {site} format uses {' and '.join(sequence_options(fmt))}, not supported with the "
                f"static cache; {HINT}"
            )
        if isinstance(fmt, MxFormat):
            reduces = fmt.scale == "e4m3"
        else:
            reduces = fmt.granularity in KEY_AXIS_GRANULARITIES
        if reduces and getattr(fmt, "zero_point", None) == "float":
            raise RuntimeError(
                f"{name}: {site} format with zero_point float reduces over the key axis, whose min/max the "
                f"static cache's zero rows would change; {HINT}"
            )
        key_axis |= reduces
    return key_axis
