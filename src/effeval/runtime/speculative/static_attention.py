"""Decode attention over a static KV cache: GQA-grouped matmuls with a float32 softmax, no `repeat_kv`."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AttentionInterface, AttentionMaskInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import eager_mask, sdpa_mask

SDPA_NAME = "effeval_static_sdpa"
EAGER_NAME = "effeval_static_eager"
MAX_GROUPED_QUERIES = 16


def grouped_attention(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None,
    dropout: float,
) -> tuple[torch.Tensor, None]:
    """Attention of `query` [B, H, S, D] over `key`/`value` [B, Hkv, T, D]; returns ([B, S, H, D], None).

    Each KV head serves its `H // Hkv` query heads in one matmul. `attention_mask` is boolean (True = may
    attend) or additive, broadcastable to [B, 1, S, T], or None for no masking. Scores are rounded to the
    input dtype, masked, and softmaxed in float32 (float64 stays float64), like transformers' eager attention.
    """
    batch, heads, queries, dim = query.shape
    kv_heads, span = key.shape[1], key.shape[2]
    groups = heads // kv_heads
    scaling = dim**-0.5 if scaling is None else scaling
    grouped = query.reshape(batch, kv_heads, groups * queries, dim)
    scores = torch.matmul(grouped, key.transpose(2, 3)) * scaling  # [B, Hkv, G*S, T]
    scores = scores.view(batch, kv_heads, groups, queries, span)
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :span].unsqueeze(2)  # [B, 1, 1, S, T]
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        else:
            scores = scores + mask
    softmax_dtype = torch.promote_types(scores.dtype, torch.float32)
    probs = nn.functional.softmax(scores, dim=-1, dtype=softmax_dtype).to(query.dtype)
    probs = nn.functional.dropout(probs, p=dropout, training=module.training)
    out = torch.matmul(probs.view(batch, kv_heads, groups * queries, span), value)  # [B, Hkv, G*S, D]
    return out.view(batch, heads, queries, dim).transpose(1, 2).contiguous(), None


def static_sdpa_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Grouped attention for at most `MAX_GROUPED_QUERIES` queries; transformers' SDPA path otherwise.

    A mask-less multi-query call needs SDPA's own causal handling, so it also takes the SDPA path.
    """
    if query.shape[2] > MAX_GROUPED_QUERIES or (attention_mask is None and query.shape[2] > 1):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )
    return grouped_attention(module, query, key, value, attention_mask, scaling, dropout)


def static_eager_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
    """Grouped attention with an additive mask, for every query count (eager attention has no fused path)."""
    return grouped_attention(module, query, key, value, attention_mask, scaling, dropout)


def register_static_attention() -> None:
    """Register both attention functions with the mask formatter each one expects."""
    AttentionInterface.register(SDPA_NAME, static_sdpa_attention)
    AttentionMaskInterface.register(SDPA_NAME, sdpa_mask)
    AttentionInterface.register(EAGER_NAME, static_eager_attention)
    AttentionMaskInterface.register(EAGER_NAME, eager_mask)
