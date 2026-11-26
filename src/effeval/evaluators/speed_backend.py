"""The interface between the speed evaluator and a decoding backend.

The evaluator renders prompts, times calls and computes metrics; a backend only generates. Transformers (our
in-process loop), SGLang, vLLM or an EAGLE-style drafter plug in by implementing `SpeedBackend`. Fields a
backend cannot provide stay None and the metrics that need them are omitted, never estimated. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from effeval.core.spec import JobSpec, SamplingSettings


@dataclass(frozen=True)
class Generation:
    """Result of one `SpeedBackend.generate` call; every field but `output_ids` is optional.

    `seconds` is the backend's own wall time, for backends that run out of process; None lets the evaluator
    time the call. The counters are those of speculative decoding: `rounds` verification rounds, `proposed`
    drafted tokens, `accepted` drafted tokens kept (so mean accepted tokens per round is
    1 + accepted / rounds), `draft_ms` drafting time and `decode_ms` total decoding time (prefill, draft and
    verify). `ttft_ms` is the time to the first token.
    """

    output_ids: list[int]
    seconds: float | None = None
    ttft_ms: float | None = None
    rounds: int | None = None
    proposed: int | None = None
    accepted: int | None = None
    draft_ms: float | None = None
    decode_ms: float | None = None


class SpeedBackend(Protocol):
    tokenizer: Any
    """Tokenizer of the target model, with `apply_chat_template`, `decode` and `chat_template`."""
    sampled: bool
    """Whether generation samples instead of decoding greedily."""

    def describe(self) -> dict[str, Any]:
        """Protocol entries of this backend: `decoder`, `device`, `timing`, `temperature` and
        `sampling` (None when greedy, else `temperature`, `top_p`, `top_k` and `seed`).

        Optional keys: `decode_cache` (copied into the protocol) and `torch_determinism` (bool, default true;
        false when the backend does not run under effeval's torch determinism settings).
        """
        ...

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Generation:
        """Generate at most `max_new_tokens` after `prompt_ids`, ending at the model's end-of-sequence."""
        ...

    def reset(self) -> None:
        """Clear every cache and per-request state that could carry over between calls.

        The evaluator calls it before each generation and after warmup, so a call never reuses work of an
        earlier one. An engine backend must disable prefix caching (vLLM, SGLang workers) or flush it here;
        a backend whose calls are independent implements it as a no-op.
        """
        ...

    def synchronize(self) -> None:
        """Wait until all work started by `generate` has finished (no-op where calls are synchronous)."""
        ...

    def reset_peak_memory(self) -> None:
        """Start tracking peak memory from now."""
        ...

    def peak_memory_gb(self) -> float | None:
        """Peak memory in GB since `reset_peak_memory`, summed over every device the backend uses, or None
        when the backend cannot tell."""
        ...


def sampling_description(settings: SamplingSettings | None, seed: int) -> dict[str, Any] | None:
    """The protocol's `sampling` entry: None for greedy decoding, else the settings and the job's seed."""
    if settings is None:
        return None
    return {
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "top_k": settings.top_k,
        "seed": seed,
    }


def decoder_description(job: JobSpec) -> str | dict[str, Any]:
    """The protocol's `decoder` entry for `job`: "autoregressive" or the speculative decoding settings."""
    decoding = job.decoding
    if decoding is None:
        return "autoregressive"
    return {
        "mode": decoding.mode,
        "draft": decoding.draft.model.name,
        "num_speculative_tokens": decoding.num_speculative_tokens,
        "verification": decoding.verification,
    }
