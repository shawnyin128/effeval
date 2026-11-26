"""Start an engine worker for a job: stop ids and configuration computed in the main process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

from effeval.core.engines import ENGINES
from effeval.core.spec import JobSpec, sampling_settings
from effeval.core.types import EngineHandle
from effeval.runtime.engines.client import EngineClient
from effeval.runtime.hf import stop_token_ids

GPU_MEMORY_FRACTION = 0.8

COUNTERS = {
    "vllm": "per-request deltas of vllm:spec_decode_num_drafts / num_draft_tokens / num_accepted_tokens "
    "from LLM.get_metrics()",
    "sglang": "meta_info spec_verify_ct / spec_num_proposed_drafts / spec_num_correct_drafts",
}
DRAFT_SAMPLING = {
    "vllm": "probabilistic at the request temperature (vLLM ignores top_k/top_p for drafts); verification "
    "uses the draft's distribution",
    "sglang": "greedy argmax draft, verified as a one-hot proposal",
}
COUNTER_NOTE = (
    "; rounds and proposed are the engine's counts (k drafts per round, fewer where the engine limits "
    "drafts near its context length, and drafts beyond the budget included); accepted is the engine's count "
    "minus the drafts accepted beyond max_new_tokens when the output ends by length, so that 1 + the sum "
    "over rounds of (accepted + 1) equals the output length, and nothing is subtracted when the output ends "
    "in a stop id; the engine's own totals per job are in raw_counters"
)


def worker_config(spec: JobSpec, stop_ids: frozenset[int]) -> dict[str, Any]:
    """The worker's configuration for `spec`; `draft` and `num_speculative_tokens` are None for autoregressive
    decoding. The engine context length is `runtime.static_cache_tokens` plus one."""
    decoding = spec.decoding
    sampling = sampling_settings(spec.generation)
    return {
        "engine": spec.runtime.name,
        "library": ENGINES[spec.runtime.name].library,
        "model": spec.model.checkpoint,
        "revision": spec.model.revision,
        "dtype": spec.model.dtype,
        "seed": spec.determinism.seed,
        "context_length": spec.runtime.static_cache_tokens + 1,
        "cuda_graphs": spec.runtime.cuda_graphs,
        "gpu_memory_fraction": GPU_MEMORY_FRACTION,
        "stop_ids": sorted(stop_ids),
        "draft": None
        if decoding is None
        else {"model": decoding.draft.model.checkpoint, "revision": decoding.draft.model.revision},
        "num_speculative_tokens": None if decoding is None else decoding.num_speculative_tokens,
        "sampling": None if sampling is None else sampling.model_dump(mode="json"),
    }


def _generation_config(checkpoint: str, revision: str | None) -> GenerationConfig:
    """The generation config transformers builds for a loaded model of this checkpoint."""
    try:
        return GenerationConfig.from_pretrained(checkpoint, revision=revision)
    except OSError:
        return GenerationConfig.from_pretrained(
            checkpoint, revision=revision, config_file_name="config.json", _from_model_config=True
        )


def prepare_engine(spec: JobSpec, workdir: Path) -> tuple[EngineHandle, dict[str, Any]]:
    """Start the engine worker for `spec` and return its handle and the protocol entries preparation adds.

    The stop ids are those the transformers loop uses for the same model. Raises RuntimeError when the
    engine's Python executable is not configured, EngineError when the worker fails to start.
    """
    name = spec.runtime.name
    env_var = ENGINES[name].python_env
    python = os.environ.get(env_var, "")
    if not os.path.isfile(python):
        raise RuntimeError(f"the {name} runtime needs {env_var} to name the engine's Python executable")
    model = spec.model
    tokenizer = AutoTokenizer.from_pretrained(model.checkpoint, revision=model.revision)
    stop_ids = stop_token_ids(tokenizer, _generation_config(model.checkpoint, model.revision))
    config = worker_config(spec, stop_ids)
    log = workdir / "engine.log"
    client = EngineClient.start(python, config, workdir, expect_library=config["library"])
    ready = client.ready
    try:
        device = torch.cuda.get_device_name(0)
    except BaseException:
        client.close()
        raise
    protocol = {
        "engine": {
            "name": name,
            "library_installed": ready["library"],
            "engine_args": ready["engine_args"],
            "resolved_config": ready["resolved"],
            "config": config,
            "log": str(log),
            "engine_seed": spec.determinism.seed,
            "torch": ready["torch"],
            "cuda": ready["cuda"],
            "counters": COUNTERS[name] + COUNTER_NOTE,
            "draft_sampling": DRAFT_SAMPLING[name] if config["draft"] and config["sampling"] else None,
            "not_reported": {
                "ttft_ms": "not measured by the engine worker",
                "peak_memory_gb": "the engine preallocates gpu_memory_fraction of the GPU",
                "draft_time_share": "the engines do not report drafting time",
            },
        }
    }
    return EngineHandle(client=client, tokenizer=tokenizer, runtime=name, device=device), protocol
