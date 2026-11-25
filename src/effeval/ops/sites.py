"""Injection sites: canonical names for tensor locations, resolution of rules against a model, and attachment.

Site vocabulary (architecture.md §5.7):
  linear.<group>.<proj>.{weight,input,output}   e.g. linear.attn.q_proj.weight, linear.mlp.down_proj.input
  lm_head.{weight,input,output}
  attn.q / attn.k / attn.v      operands of QK^T and PV as the attention computation sees them
                                (after RoPE and the KV cache update)
  attn.scores                   QK^T * scaling, before the mask is added
  attn.probs                    softmax output
  attn.out                      PV output, before the output projection
  moe.router.{weight,input}     the router of a MoE layer; never matched by `linear.*` (its `output`
                                is a tuple and is rejected)

Routed experts are addressed like any linear, one location per expert: `linear.mlp.experts.gate_proj.input`
hooks every expert and fires only on the tokens routed to it. Per-token and per-group activation formats then
equal quantizing the block input once; a per-tensor dynamic format does not, since each expert sees only
its own tokens.

Shape symbols: B batch, H heads, S query length, T key length, D head dim.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field

import torch
from pydantic import BaseModel
from torch import nn
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import eager_mask, sdpa_mask

from effeval.core.catalog import ArchitectureDef, builtin_catalog
from effeval.core.fragments import (
    ATTENTION_SITES,
    SITE_TENSORS,
    FpFormat,
    IntFormat,
    MxFormat,
    QuantRule,
    select_layers,
)
from effeval.ops.quant.quant import fake_quantize

QuantFormat = IntFormat | FpFormat | MxFormat

ATTN_ATTR = "_effeval_attention_sites"
ATTN_OPS_ATTR = "_effeval_attention_ops"
ATTN_SITES = ATTENTION_SITES
ATTN_EAGER_ONLY = frozenset({"attn.scores", "attn.probs", "attn.out"})
SDPA_IMPL = "effeval_sdpa"
EAGER_IMPL = "effeval_eager"


class SiteHit(BaseModel):
    rule: int
    sites: list[str]
    layers: list[int]
    locations: int


class SiteError(ValueError):
    """Raised when an architecture has no layout or a rule matches nothing on this model."""


@dataclass
class Attachment:
    """Hooks and attention settings installed on a model; `remove()` undoes everything except weight edits."""

    model: nn.Module
    hits: list[SiteHit]
    handles: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    attention_modules: list[nn.Module] = field(default_factory=list)
    previous_attn_implementation: str | None = None
    attention_attr: str = ATTN_ATTR

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for module in self.attention_modules:
            if hasattr(module, self.attention_attr):
                delattr(module, self.attention_attr)
        self.attention_modules.clear()
        if self.previous_attn_implementation is not None:
            self.model.set_attn_implementation(self.previous_attn_implementation)
            self.previous_attn_implementation = None


def architecture(model: nn.Module) -> ArchitectureDef:
    """Layout of the model's HF architecture class, looked up in the architecture table."""
    name = type(model).__name__
    known = builtin_catalog().architectures
    if name not in known:
        raise SiteError(
            f"architecture {name} has no layout in effeval; supported: {sorted(known)}. "
            "Add one with `effeval arch extract <checkpoint>`."
        )
    return known[name]


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    return model.get_submodule(architecture(model).layers_path)


@dataclass(frozen=True)
class _Location:
    site: str
    layer: int | None
    module: nn.Module
    expert: int | None = None

    @property
    def key(self) -> tuple:
        """`(site, layer)`, extended with the expert index for a routed-expert location."""
        return (self.site, self.layer) if self.expert is None else (self.site, self.layer, self.expert)


def _find(root: nn.Module, path: str) -> nn.Module | None:
    module = root
    for part in path.split("."):
        module = module._modules.get(part)
        if module is None:
            return None
    return module


def expand_path(layer: nn.Module, path: str, *, strict: bool = True) -> list[tuple[int | None, nn.Module]]:
    """Modules a layout path names inside one decoder layer, as `(expert index or None, module)`.

    A `*` stands for the routed experts of an unfused experts container (`<container>.*.<rest>`). A path the
    layer does not have yields an empty list when `strict` is False and raises AttributeError otherwise.
    Raises SiteError when the container still holds fused experts.
    """
    if "*" not in path:
        module = layer.get_submodule(path) if strict else _find(layer, path)
        return [] if module is None else [(None, module)]
    prefix, rest = path.split(".*.", 1)
    container = layer.get_submodule(prefix) if strict else _find(layer, prefix)
    if container is None:
        return []
    experts = getattr(container, "experts", None)
    if not isinstance(experts, nn.ModuleList):
        raise SiteError(f"{prefix} holds fused experts; unfuse them first (runtime.moe.unfuse_experts)")
    found = ((e, _find(expert, rest)) for e, expert in enumerate(experts))
    return [(e, module) for e, module in found if module is not None]


def _candidates(model: nn.Module) -> list[_Location]:
    layout = architecture(model)
    layers = decoder_layers(model)
    moe = layout.moe
    out: list[_Location] = []
    for index, layer in enumerate(layers):
        for path, canon in layout.linears.items():
            for expert, module in expand_path(layer, path, strict=moe is None):
                out.extend(_Location(f"linear.{canon}.{t}", index, module, expert) for t in SITE_TENSORS)
        if moe is not None and index in moe.layers:
            router = _find(layer, moe.router)
            if router is not None:
                out.extend(_Location(f"moe.router.{t}", index, router) for t in SITE_TENSORS)
        if layout.attention_path is not None and layout.supports_attention_backend:
            attention = layer.get_submodule(layout.attention_path)
            out.extend(_Location(site, index, attention) for site in ATTN_SITES)
    lm_head = getattr(model, "lm_head", None)
    if isinstance(lm_head, nn.Linear):
        out.extend(_Location(f"lm_head.{t}", None, lm_head) for t in SITE_TENSORS)
    return out


def resolve(
    model: nn.Module, rules: list[QuantRule]
) -> tuple[dict[tuple, tuple[int, _Location, QuantFormat]], list[SiteHit]]:
    """Match rules against the model. Later rules override earlier ones at the same location.

    Keys are `(site, layer)`, or `(site, layer, expert)` for a routed-expert location.

    Raises SiteError when a rule matches no location on this model.
    """
    candidates = _candidates(model)
    num_layers = len(decoder_layers(model))
    assignment: dict[tuple, tuple[int, _Location, QuantFormat]] = {}
    hits: list[SiteHit] = []
    for index, rule in enumerate(rules):
        layer_ids = set(select_layers(rule.layers, num_layers))
        matched = [
            loc
            for loc in candidates
            if any(fnmatch.fnmatchcase(loc.site, pattern) for pattern in rule.sites)
            and (loc.layer in layer_ids if loc.layer is not None else rule.layers == "all")
        ]
        if not matched:
            raise SiteError(
                f"rule {index} (sites {rule.sites}, layers {rule.layers!r}) matches nothing on "
                f"{type(model).__name__}"
            )
        for loc in matched:
            assignment[loc.key] = (index, loc, rule.quant)
        hits.append(
            SiteHit(
                rule=index,
                sites=sorted({loc.site for loc in matched}),
                layers=sorted({loc.layer for loc in matched if loc.layer is not None}),
                locations=len(matched),
            )
        )
    return assignment, hits


def attach(model: nn.Module, rules: list[QuantRule]) -> Attachment:
    """Apply rules: weights are fake-quantized in place, activations and attention get runtime hooks."""
    assignment, hits = resolve(model, rules)
    attachment = Attachment(model=model, hits=hits)
    attention_sites: dict[nn.Module, dict[str, QuantFormat]] = {}
    try:
        for key, (_rule, loc, fmt) in assignment.items():
            site = key[0]
            if site == "moe.router.output":
                raise SiteError("moe.router.output is a tuple (logits, scores, indices), not quantizable")
            if site.endswith(".weight"):
                with torch.no_grad():
                    loc.module.weight.copy_(fake_quantize(loc.module.weight, fmt, kind="weight"))
            elif site.endswith(".input"):
                attachment.handles.append(loc.module.register_forward_pre_hook(_input_hook(fmt)))
            elif site.endswith(".output"):
                attachment.handles.append(loc.module.register_forward_hook(_output_hook(fmt)))
            elif site in ATTN_SITES:
                attention_sites.setdefault(loc.module, {})[site] = fmt
            else:
                raise SiteError(f"unhandled site {site!r}")
        if attention_sites:
            register_attention_functions()
            needs_eager = any(ATTN_EAGER_ONLY & sites.keys() for sites in attention_sites.values())
            attachment.previous_attn_implementation = model.config._attn_implementation
            model.set_attn_implementation(EAGER_IMPL if needs_eager else SDPA_IMPL)
            for module, sites in attention_sites.items():
                setattr(module, ATTN_ATTR, sites)
                attachment.attention_modules.append(module)
    except Exception:
        attachment.remove()
        raise
    return attachment


def _input_hook(fmt: QuantFormat):
    def hook(_module: nn.Module, args: tuple) -> tuple:
        return (fake_quantize(args[0], fmt, kind="activation"), *args[1:])

    hook._effeval_graph_safe = True
    hook._effeval_format = fmt
    return hook


def _output_hook(fmt: QuantFormat):
    def hook(_module: nn.Module, _args: tuple, output: torch.Tensor) -> torch.Tensor:
        return fake_quantize(output, fmt, kind="activation")

    hook._effeval_graph_safe = True
    hook._effeval_format = fmt
    return hook


def _maybe_quantize(sites: dict[str, QuantFormat], site: str, tensor: torch.Tensor) -> torch.Tensor:
    fmt = sites.get(site)
    return tensor if fmt is None else fake_quantize(tensor, fmt, kind="activation")


def _operand(module: nn.Module, site: str, tensor: torch.Tensor) -> torch.Tensor:
    """Apply the online op registered for `site` on this attention module (if any), then its quantizer."""
    op = getattr(module, ATTN_OPS_ATTR, {}).get(site)
    if op is not None:
        tensor = op(tensor)
    return _maybe_quantize(getattr(module, ATTN_ATTR, {}), site, tensor)


def _repeat_kv(hidden: torch.Tensor, groups: int) -> torch.Tensor:
    """[B, Hkv, T, D] -> [B, Hkv * groups, T, D]."""
    if groups == 1:
        return hidden
    batch, kv_heads, length, dim = hidden.shape
    return (
        hidden[:, :, None, :, :]
        .expand(batch, kv_heads, groups, length, dim)
        .reshape(batch, kv_heads * groups, length, dim)
    )


def effeval_sdpa_attention(module, query, key, value, attention_mask, **kwargs):
    """Apply online ops and quantize attention operands (attn.q/k/v), then run transformers' SDPA path."""
    query = _operand(module, "attn.q", query)  # [B, H, S, D]
    key = _operand(module, "attn.k", key)  # [B, Hkv, T, D]
    value = _operand(module, "attn.v", value)  # [B, Hkv, T, D]
    return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)


def effeval_eager_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention with every intermediate exposed; identical math to transformers' eager path."""
    sites = getattr(module, ATTN_ATTR, {})
    query = _operand(module, "attn.q", query)  # [B, H, S, D]
    key = _repeat_kv(_operand(module, "attn.k", key), module.num_key_value_groups)  # [B, H, T, D]
    value = _repeat_kv(_operand(module, "attn.v", value), module.num_key_value_groups)  # [B, H, T, D]
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling  # [B, H, S, T]
    scores = _maybe_quantize(sites, "attn.scores", scores)
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)  # [B, H, S, T]
    probs = _maybe_quantize(sites, "attn.probs", probs)
    probs = nn.functional.dropout(probs, p=dropout, training=module.training)
    out = torch.matmul(probs, value)  # [B, H, S, D]
    out = _maybe_quantize(sites, "attn.out", out)
    return out.transpose(1, 2).contiguous(), probs  # [B, S, H, D]


def register_attention_functions() -> None:
    """Register both attention functions with the mask formatter each one expects.

    transformers skips mask creation (passing attention_mask=None) for attention names without a registered
    mask function, which would silently drop causal and padding masks.
    """
    AttentionInterface.register(SDPA_IMPL, effeval_sdpa_attention)
    AttentionMaskInterface.register(SDPA_IMPL, sdpa_mask)
    AttentionInterface.register(EAGER_IMPL, effeval_eager_attention)
    AttentionMaskInterface.register(EAGER_IMPL, eager_mask)
