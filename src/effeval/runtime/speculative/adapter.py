"""Bridge from HF-style `model.generate(...)` calls (lm_eval, EvalScope) to a speculative `Decoder`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from effeval.core.types import metric_str
from effeval.runtime.speculative.loop import Decoder, DecodeStats
from effeval.runtime.speculative.rules import TokenSampler

_IGNORED = frozenset({"pad_token_id", "use_cache", "return_dict_in_generate"})
_MUST_BE_NONE = ("top_p", "top_k")


def _add(total: DecodeStats, part: DecodeStats) -> None:
    for name in ("rounds", "proposed", "accepted", "plain_steps", "new_tokens"):
        setattr(total, name, getattr(total, name) + getattr(part, name))
    for name in ("prefill_ms", "draft_ms", "verify_ms"):
        setattr(total, name, getattr(total, name) + getattr(part, name))
    total.accept_lengths.extend(part.accept_lengths)


def decoding_metrics(stats: DecodeStats) -> dict[str, str]:
    """`acceptance_rate` and `mean_accepted_tokens` (tau) of accumulated `stats`, as metric strings."""
    return {
        "acceptance_rate": metric_str(stats.acceptance_rate),
        "mean_accepted_tokens": metric_str(stats.tau),
    }


def decoding_stats(stats: DecodeStats) -> dict[str, Any]:
    """JSON-ready counters and phase times of accumulated `stats` for a record's protocol."""
    return {
        "rounds": stats.rounds,
        "proposed": stats.proposed,
        "accepted": stats.accepted,
        "plain_steps": stats.plain_steps,
        "new_tokens": stats.new_tokens,
        "prefill_ms": stats.prefill_ms,
        "draft_ms": stats.draft_ms,
        "verify_ms": stats.verify_ms,
    }


class GenerateProxy:
    """Stands in for the target model in code that only calls `generate`, running `decoder` instead.

    Exposes `device`, `dtype`, `config` and `generation_config` of `model`; `stats` accumulates the
    `DecodeStats` of every `generate` call. Batch size must be 1.
    """

    def __init__(self, decoder: Decoder, model: Any, tokenizer: Any):
        self.decoder = decoder
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.dtype = model.dtype
        self.config = model.config
        self.generation_config = model.generation_config
        self.stats = DecodeStats()

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: Any):
        """HF-compatible greedy/temperature `generate` for a [1, L] prompt.

        Returns `[input_ids, generated]` concatenated, or an object with `.sequences` and `.logits = None`
        when `return_dict_in_generate` is true. Raises TypeError for kwargs outside the supported set and
        ValueError for batch size != 1, a non-trivial attention mask or unsupported values.
        """
        max_new_tokens = kwargs.pop("max_new_tokens", None)
        max_length = kwargs.pop("max_length", None)
        stopping_criteria = kwargs.pop("stopping_criteria", None)
        do_sample = kwargs.pop("do_sample", None)
        temperature = kwargs.pop("temperature", None)
        return_dict = kwargs.get("return_dict_in_generate", False)
        if kwargs.pop("num_return_sequences", None) not in (None, 1):
            raise ValueError("num_return_sequences must be 1")
        for name in _MUST_BE_NONE:
            if kwargs.pop(name, None) is not None:
                raise ValueError(f"{name} is not supported by the speculative decoder")
        if kwargs.pop("output_logits", None):
            raise ValueError("output_logits is not supported by the speculative decoder")
        unknown = set(kwargs) - _IGNORED
        if unknown:
            raise TypeError(f"unsupported generate kwargs: {sorted(unknown)}")
        if input_ids.shape[0] != 1:
            raise ValueError(f"speculative decoding needs batch size 1, got {input_ids.shape[0]}")
        if attention_mask is not None and not bool(attention_mask.all()):
            raise ValueError("attention_mask must be all ones (no padding) at batch size 1")
        if (max_new_tokens is None) == (max_length is None):
            raise ValueError("give exactly one of max_new_tokens and max_length")
        if max_new_tokens is None:
            max_new_tokens = max_length - input_ids.shape[1]

        if do_sample is None:
            do_sample = bool(self.generation_config.do_sample)
        if do_sample:
            temperature = self.generation_config.temperature if temperature is None else temperature
            sampler = TokenSampler(1.0 if temperature is None else temperature)
        else:
            sampler = TokenSampler(None)
        eos = self.generation_config.eos_token_id
        eos_ids = frozenset() if eos is None else frozenset([eos] if isinstance(eos, int) else eos)

        stop = None
        if stopping_criteria is not None:
            criteria = (
                list(stopping_criteria)
                if isinstance(stopping_criteria, (list, tuple))
                else [stopping_criteria]
            )

            def stop(sequence: torch.Tensor) -> bool:
                return any(bool(torch.as_tensor(c(sequence, None)).any()) for c in criteria)

        result = self.decoder.generate(
            input_ids, max_new_tokens=max_new_tokens, sampler=sampler, eos_ids=eos_ids, stop=stop
        )
        _add(self.stats, result.stats)
        out = torch.tensor([result.output_ids], dtype=input_ids.dtype, device=input_ids.device)
        sequences = torch.cat([input_ids, out], dim=1)
        return SimpleNamespace(sequences=sequences, logits=None) if return_dict else sequences
