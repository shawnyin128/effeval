"""Round-to-nearest: apply quantization rules directly (weights in place, activations/attention via hooks)."""

from __future__ import annotations

from pydantic import Field, model_validator
from torch import nn

from effeval.core.fragments import IntFormat, QuantRule
from effeval.core.registry import method
from effeval.core.types import StrictModel
from effeval.ops import sites


def check_clip_search(rules: list[QuantRule]) -> None:
    """Raises ValueError when a rule with `clip_search` has a site that is not a '.weight' site."""
    for index, rule in enumerate(rules):
        if (
            isinstance(rule.quant, IntFormat)
            and rule.quant.clip_search is not None
            and not all(pattern.endswith(".weight") for pattern in rule.sites)
        ):
            raise ValueError(f"rules[{index}]: clip_search applies to '.weight' sites only")


class RTNConfig(StrictModel):
    rules: list[QuantRule] = Field(min_length=1)

    @model_validator(mode="after")
    def _clip_search_on_weights(self) -> RTNConfig:
        check_clip_search(self.rules)
        return self


@method(
    "rtn",
    title="RTN",
    version="1",
    field="quantization",
    config=RTNConfig,
    uses=("quant_rules",),
    display=("rules",),
    default_preset="gptq_paper",
    description="Round-to-nearest on arbitrary sites; the GPTQ paper's RTN baseline is the default preset.",
)
def rtn(model: nn.Module, tokenizer, cfg: RTNConfig) -> nn.Module:
    sites.attach(model, cfg.rules)
    return model
