"""Split fused MoE experts into one `nn.Linear` triple per expert.

transformers 5.23 stores the routed experts of a layer as two 3D parameters: `gate_up_proj` [E, 2I, H]
(rows 0..I-1 are gate, I..2I-1 are up) and `down_proj` [E, H, I]. Site resolution, GPTQ and rotation only
see `nn.Linear`, so `unfuse_experts` replaces each fused module with `ExpertLinears`, whose forward repeats
the HF eager loop op for op.

Fused classes handled (installed transformers 5.23, eager forward at the given lines):
- `Qwen3MoeExperts`: models/qwen3_moe/modeling_qwen3_moe.py, class 209-244, forward 221-244.
- `DeepseekV2Experts`: models/deepseek_v2/modeling_deepseek_v2.py, class 46-81, forward 58-81.
"""

from __future__ import annotations

import torch
from torch import nn

from effeval.core.moe import EXPERT_ADAPTERS

FUSED_EXPERTS = frozenset(EXPERT_ADAPTERS)


class ExpertMLP(nn.Module):
    """One expert: `down_proj(act_fn(gate_proj(x)) * up_proj(x))`."""

    def __init__(
        self, gate_proj: nn.Linear, up_proj: nn.Linear, down_proj: nn.Linear, act_fn: nn.Module
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ExpertLinears(nn.Module):
    """Routed experts of one MoE block with the call signature of the fused HF experts module."""

    def __init__(self, experts: list[ExpertMLP]) -> None:
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_hidden_states = self.experts[expert_idx](hidden_states[token_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


def _linear(weight: torch.Tensor, requires_grad: bool) -> nn.Linear:
    out_features, in_features = weight.shape
    linear = nn.Linear(in_features, out_features, bias=False, device="meta")
    linear.weight = nn.Parameter(weight.detach().clone(), requires_grad=requires_grad)
    return linear


def _unfuse(fused: nn.Module) -> ExpertLinears:
    gate_up, down = (getattr(fused, name) for name in EXPERT_ADAPTERS[type(fused).__name__].fused_params)
    inter = fused.intermediate_dim
    grad = gate_up.requires_grad
    experts = [
        ExpertMLP(
            _linear(gate_up[e, :inter], grad),
            _linear(gate_up[e, inter:], grad),
            _linear(down[e], grad),
            fused.act_fn,
        )
        for e in range(fused.num_experts)
    ]
    return ExpertLinears(experts)


def _replace(model: nn.Module, name: str) -> None:
    parent_name, _, child = name.rpartition(".")
    setattr(model.get_submodule(parent_name), child, _unfuse(model.get_submodule(name)))


def unfuse_experts(model: nn.Module) -> int:
    """Replace every fused experts module of `model` in place; returns the number of modules replaced."""
    names = [name for name, m in model.named_modules() if type(m).__name__ in FUSED_EXPERTS]
    for name in names:
        _replace(model, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return len(names)
