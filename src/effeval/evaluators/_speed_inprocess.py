"""`SpeedBackend` over the in-process transformers decoders (autoregressive and speculative)."""

from __future__ import annotations

from typing import Any

import torch

from effeval.core.catalog import DEFAULT_SAMPLING_TEMPERATURE
from effeval.core.spec import JobSpec
from effeval.core.types import InProcess
from effeval.evaluators.speed_backend import Generation
from effeval.runtime.speculative import AutoregressiveDecoder, TokenSampler


def _eos_ids(handle: InProcess) -> frozenset[int]:
    ids = set()
    eos = handle.tokenizer.eos_token_id
    if eos is not None:
        ids.add(eos)
    config_eos = getattr(handle.model.generation_config, "eos_token_id", None)
    if config_eos is not None:
        ids.update(config_eos if isinstance(config_eos, (list, tuple)) else [config_eos])
    return frozenset(ids)


def _cuda_devices(models: list[Any]) -> list[torch.device]:
    """The CUDA devices `models` live on: those of their parameters and buffers and of `hf_device_map`."""
    found: set[int] = set()

    def add(device: Any) -> None:
        if isinstance(device, int):
            device = torch.device("cuda", device)
        elif isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda":
            found.add(torch.cuda.current_device() if device.index is None else device.index)

    for model in models:
        for tensor in (*model.parameters(), *model.buffers()):
            add(tensor.device)
        for device in (getattr(model, "hf_device_map", None) or {}).values():
            if device not in ("cpu", "disk"):
                add(device)
    return [torch.device("cuda", index) for index in sorted(found)]


class InProcessBackend:
    """Decodes with the handle's speculative decoder, or plain autoregressively when it has none.

    Synchronization and peak memory cover every CUDA device the target and the draft use.
    """

    def __init__(self, handle: InProcess, job: JobSpec):
        self.tokenizer = handle.tokenizer
        self._decoder = handle.decoder or AutoregressiveDecoder(handle.model)
        self._speculative = handle.decoder is not None
        self._device = handle.model.get_input_embeddings().weight.device
        self._cuda_devices = _cuda_devices(self._decoder.models)
        self._eos_ids = _eos_ids(handle)
        self._job = job
        generation = job.generation
        self._temperature = None
        generator = None
        if generation.do_sample:
            self._temperature = generation.temperature or DEFAULT_SAMPLING_TEMPERATURE
            generator = torch.Generator(device=self._device).manual_seed(job.determinism.seed)
        self._sampler = TokenSampler(self._temperature, generator)
        self.sampled = not self._sampler.greedy

    def describe(self) -> dict[str, Any]:
        decoding = self._job.decoding
        decoder = (
            "autoregressive"
            if decoding is None
            else {
                "mode": decoding.mode,
                "draft": decoding.draft.model.name,
                "num_speculative_tokens": decoding.num_speculative_tokens,
                "verification": decoding.verification,
            }
        )
        timing = "perf_counter around each turn's decoder.generate; excludes tokenization"
        if self._cuda_devices:
            timing = "synchronize + " + timing
        names = [torch.cuda.get_device_name(d) for d in self._cuda_devices]
        return {
            "decoder": decoder,
            "device": ", ".join(names) if names else "cpu",
            "timing": timing,
            "temperature": self._temperature,
        }

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], max_new_tokens: int) -> Generation:
        ids = torch.tensor([prompt_ids], device=self._device)
        result = self._decoder.generate(
            ids, max_new_tokens=max_new_tokens, sampler=self._sampler, eos_ids=self._eos_ids
        )
        stats = result.stats
        if not self._speculative:
            return Generation(result.output_ids, ttft_ms=stats.first_token_ms)
        return Generation(
            result.output_ids,
            ttft_ms=stats.first_token_ms,
            rounds=stats.rounds,
            proposed=stats.proposed,
            accepted=stats.accepted,
            draft_ms=stats.draft_ms,
            decode_ms=stats.prefill_ms + stats.draft_ms + stats.verify_ms,
        )

    def reset(self) -> None:
        """No-op: the decoders start every generation from empty caches."""

    def synchronize(self) -> None:
        for device in self._cuda_devices:
            torch.cuda.synchronize(device)

    def reset_peak_memory(self) -> None:
        for device in self._cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)

    def peak_memory_gb(self) -> float | None:
        if not self._cuda_devices:
            return None
        return sum(torch.cuda.max_memory_allocated(d) for d in self._cuda_devices) / 1e9
