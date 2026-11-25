"""The fused MoE experts classes of Hugging Face transformers that effeval knows how to address and unfuse.

Torch-free so that layout extraction (`ops.introspect`) and expert unfusing (`runtime.moe`) read one table.
"""

from __future__ import annotations

from typing import NamedTuple


class ExpertAdapter(NamedTuple):
    """How one fused experts class is laid out."""

    linears: tuple[str, str, str]
    """Names of the gate, up and down projection of one unfused expert."""
    router: str
    """Attribute of the MoE block holding the router (a module with a `weight`, not an nn.Linear)."""
    fused_params: tuple[str, str]
    """Fused parameters: `gate_up_proj` [E, 2I, H] (gate rows first, then up) and `down_proj` [E, H, I]."""


_GATED = ("gate_proj", "up_proj", "down_proj")
_FUSED = ("gate_up_proj", "down_proj")
EXPERT_ADAPTERS: dict[str, ExpertAdapter] = {
    "Qwen3MoeExperts": ExpertAdapter(_GATED, "gate", _FUSED),
    "DeepseekV2Experts": ExpertAdapter(_GATED, "gate", _FUSED),
}
