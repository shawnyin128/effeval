"""`SpeedBackend` over an engine worker (vLLM, SGLang) reached through its client; torch-free.

The engine times each generation itself. TTFT, peak memory and draft time are not measured by the worker,
so they stay absent.
"""

from __future__ import annotations

from typing import Any

from effeval.core.spec import JobSpec, sampling_settings
from effeval.core.types import EngineHandle
from effeval.evaluators.speed_backend import Generation, decoder_description, sampling_description


class EngineBackend:
    """Generation with the job's sampling settings in the handle's engine worker, one request at a time."""

    def __init__(self, handle: EngineHandle, job: JobSpec):
        self.tokenizer = handle.tokenizer
        self._sampling = sampling_settings(job.generation)
        self.sampled = self._sampling is not None
        self._handle = handle
        self._job = job

    def describe(self) -> dict[str, Any]:
        return {
            "decoder": decoder_description(self._job),
            "device": self._handle.device,
            "timing": f"perf_counter around one {self._handle.runtime} generate call inside the engine "
            "worker; excludes tokenization and effeval's IPC",
            "temperature": None if self._sampling is None else self._sampling.temperature,
            "sampling": sampling_description(self._sampling, self._job.determinism.seed),
            "torch_determinism": False,
        }

    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Generation:
        needed = len(prompt_ids) + max_new_tokens - 1
        capacity = self._job.runtime.static_cache_tokens
        if needed > capacity:
            raise ValueError(
                f"this generation needs {needed} cache positions, runtime.static_cache_tokens is {capacity}; "
                "raise runtime.static_cache_tokens"
            )
        reply = self._handle.client.generate(prompt_ids, max_new_tokens)
        return Generation(
            reply.output_ids,
            seconds=reply.seconds,
            rounds=reply.rounds,
            proposed=reply.proposed,
            accepted=reply.accepted,
        )

    def reset(self) -> None:
        """No-op: the worker runs without prefix caching and the client rejects any cache hit."""

    def synchronize(self) -> None:
        """No-op: the worker's generate call returns after the tokens are produced."""

    def reset_peak_memory(self) -> None:
        """No-op: peak memory is not reported."""

    def peak_memory_gb(self) -> float | None:
        return None
