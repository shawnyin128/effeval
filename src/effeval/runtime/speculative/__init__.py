"""Batch-size-1 speculative decoding loop for the transformers runtime (torch only)."""

from effeval.runtime.speculative.drafters import Drafter, ModelDrafter
from effeval.runtime.speculative.loop import (
    AutoregressiveDecoder,
    Decoder,
    DecodeResult,
    DecodeStats,
    SpeculativeDecoder,
)
from effeval.runtime.speculative.rules import (
    AcceptanceRule,
    GreedyAcceptance,
    Proposal,
    SpeculativeSampling,
    TokenSampler,
    residual_distribution,
)
from effeval.runtime.speculative.state import CachedModel
from effeval.runtime.speculative.static import StaticCachedModel, StaticCacheSettings

__all__ = [
    "AcceptanceRule",
    "AutoregressiveDecoder",
    "CachedModel",
    "DecodeResult",
    "DecodeStats",
    "Decoder",
    "Drafter",
    "GreedyAcceptance",
    "ModelDrafter",
    "Proposal",
    "SpeculativeDecoder",
    "SpeculativeSampling",
    "StaticCacheSettings",
    "StaticCachedModel",
    "TokenSampler",
    "residual_distribution",
]
