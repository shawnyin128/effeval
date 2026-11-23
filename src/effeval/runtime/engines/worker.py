"""Engine worker: runs one vLLM or SGLang engine and serves JSON-line requests on stdin/stdout.

Started as `<engine python> -I worker.py --config <file>`. The file is self-contained (standard library plus
the engine, imported lazily): it runs in the engine's virtual environment, where effeval is not installed.
Engine logs and child processes write to stderr; only protocol lines go to the real stdout.
"""

import argparse
import importlib.metadata
import json
import os
import sys
import time

VLLM_COUNTERS = (
    "vllm:spec_decode_num_drafts",
    "vllm:spec_decode_num_draft_tokens",
    "vllm:spec_decode_num_accepted_tokens",
)
VLLM_CACHE_HITS = "vllm:prefix_cache_hits"
# SGLang rejects inputs of context_length - 6 tokens or more and clips max_new_tokens to
# context_length - 2 - input_len; the margin keeps the full budget of every request that fits the config.
SGLANG_CONTEXT_MARGIN = 6


def vllm_kwargs(config):
    """Keyword arguments of `vllm.LLM` for a worker config; `speculative_config` is absent for the AR arm."""
    kwargs = {
        "model": config["model"],
        "revision": config["revision"],
        "dtype": config["dtype"],
        "seed": config["seed"],
        "gpu_memory_utilization": config["gpu_memory_fraction"],
        "max_model_len": config["context_length"],
        "enforce_eager": not config["cuda_graphs"],
        "enable_prefix_caching": False,
        "disable_log_stats": False,
    }
    draft = config.get("draft")
    if draft is not None:
        kwargs["speculative_config"] = {
            "method": "draft_model",
            "model": draft["model"],
            "revision": draft["revision"],
            "num_speculative_tokens": config["num_speculative_tokens"],
            "draft_sample_method": "greedy",
        }
    return kwargs


def sglang_kwargs(config):
    """Keyword arguments of `sglang.Engine` for a worker config; no `speculative_*` keys for the AR arm."""
    kwargs = {
        "model_path": config["model"],
        "revision": config["revision"],
        "dtype": config["dtype"],
        "random_seed": config["seed"],
        "mem_fraction_static": config["gpu_memory_fraction"],
        "context_length": config["context_length"] + SGLANG_CONTEXT_MARGIN,
        "max_total_tokens": 4 * config["context_length"],
        "disable_radix_cache": True,
        "disable_cuda_graph": not config["cuda_graphs"],
        "log_level": "info",
    }
    draft = config.get("draft")
    if draft is not None:
        k = config["num_speculative_tokens"]
        kwargs.update(
            speculative_algorithm="STANDALONE",
            speculative_draft_model_path=draft["model"],
            speculative_draft_model_revision=draft["revision"],
            speculative_num_steps=k,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=k + 1,
        )
    return kwargs


class VllmEngine:
    """vLLM offline `LLM`; spec stats are per-request deltas of its Prometheus counters."""

    def __init__(self, config):
        import vllm

        self.config = config
        self.kwargs = vllm_kwargs(config)
        self.engine_args = self.kwargs
        self.speculative = config.get("draft") is not None
        self.sampling_params = vllm.SamplingParams
        self.llm = vllm.LLM(**self.kwargs)

    def _counters(self):
        totals = {}
        for metric in self.llm.get_metrics():
            if metric.name in VLLM_COUNTERS or metric.name == VLLM_CACHE_HITS:
                totals[metric.name] = totals.get(metric.name, 0) + metric.value
        return totals

    def generate(self, prompt_ids, max_new_tokens, clock):
        """Generate greedily; returns output_ids, seconds, stats (None for AR) and cache_hits."""
        params = self.sampling_params(
            max_tokens=max_new_tokens,
            temperature=0.0,
            stop_token_ids=self.config["stop_ids"],
            detokenize=False,
        )
        before = self._counters()
        t0 = clock()
        outputs = self.llm.generate([{"prompt_token_ids": prompt_ids}], params, use_tqdm=False)
        seconds = clock() - t0
        after = self._counters()
        delta = {name: after[name] - before[name] for name in after if name in before}
        if VLLM_CACHE_HITS not in delta:
            raise RuntimeError(f"vLLM did not report the counter {VLLM_CACHE_HITS}")
        stats = None
        if self.speculative:
            missing = [name for name in VLLM_COUNTERS if name not in delta]
            if missing:
                raise RuntimeError(f"vLLM did not report the speculative counters {missing}")
            drafts, proposed, accepted = (int(delta[name]) for name in VLLM_COUNTERS)
            stats = {"rounds": drafts, "proposed": proposed, "accepted": accepted}
        return {
            "output_ids": list(outputs[0].outputs[0].token_ids),
            "seconds": seconds,
            "stats": stats,
            "cache_hits": int(delta[VLLM_CACHE_HITS]),
        }

    def resolved(self):
        """The engine's resolved configuration as text."""
        return str(self.llm.llm_engine.vllm_config)

    def close(self):
        """Nothing to do: vLLM has no shutdown call and exits with the interpreter."""


class SglangEngine:
    """SGLang offline `Engine`; spec stats come from the `meta_info` of each reply."""

    def __init__(self, config):
        import sglang

        self.config = config
        self.kwargs = sglang_kwargs(config)
        self.engine_args = self.kwargs
        self.speculative = config.get("draft") is not None
        self.engine = sglang.Engine(**self.kwargs)

    def generate(self, prompt_ids, max_new_tokens, clock):
        """Generate greedily; returns output_ids, seconds, stats (None for AR) and cache_hits."""
        sampling = {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "stop_token_ids": self.config["stop_ids"],
        }
        t0 = clock()
        out = self.engine.generate(input_ids=prompt_ids, sampling_params=sampling)
        seconds = clock() - t0
        meta = out["meta_info"]
        stats = None
        if self.speculative:
            stats = {
                "rounds": meta.get("spec_verify_ct", 0),
                "proposed": meta.get("spec_num_proposed_drafts", 0),
                "accepted": meta.get("spec_num_correct_drafts", 0),
            }
        return {
            "output_ids": list(out["output_ids"]),
            "seconds": seconds,
            "stats": stats,
            "cache_hits": meta["cached_tokens"],
        }

    def resolved(self):
        """The engine's resolved server arguments as text."""
        return str(self.engine.server_args)

    def close(self):
        """Shut the engine and its subprocess tree down."""
        self.engine.shutdown()


ADAPTERS = {"vllm": VllmEngine, "sglang": SglangEngine}


def library_mismatch(library, version_of=importlib.metadata.version):
    """Why the installed library differs from the pin `dist==x.y.z` (local version ignored), else None."""
    dist, _, pinned = library.partition("==")
    try:
        installed = version_of(dist)
    except importlib.metadata.PackageNotFoundError:
        return f"{dist} is not installed in the engine environment (pinned {pinned})"
    if installed.split("+")[0] != pinned:
        return f"{dist} {installed} is installed but effeval pins {pinned}"
    return None


def torch_versions():
    """The (torch version, CUDA version) of the torch the engine imported; CUDA is None for a CPU build."""
    torch = sys.modules["torch"]
    return torch.__version__, torch.version.cuda


def serve(engine, lines_in, out, clock=time.perf_counter):
    """Answer one JSON request per input line on `out` until an exit command or EOF; returns the exit code."""
    for line in lines_in:
        if not line.strip():
            continue
        request = None
        try:
            request = json.loads(line)
            if request.get("cmd") == "exit":
                out.write(json.dumps({"bye": True}) + "\n")
                out.flush()
                return 0
            reply = {
                "id": request["id"],
                **engine.generate(request["prompt_ids"], request["max_new_tokens"], clock),
            }
        except Exception as exc:
            rid = request.get("id") if isinstance(request, dict) else None
            reply = {"id": rid, "error": f"{type(exc).__name__}: {exc}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()
    return 0


def main(argv):
    """Entry point: isolate stdout, check the library, start the engine, announce readiness and serve."""
    parser = argparse.ArgumentParser(description="effeval engine worker")
    parser.add_argument("--config", required=True, help="worker config JSON file")
    args = parser.parse_args(argv)

    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    print(f"effeval engine worker: {config['engine']} {config['model']} config {args.config}")

    message = library_mismatch(config["library"])
    if message is not None:
        proto.write(json.dumps({"error": message}) + "\n")
        return 3

    engine = ADAPTERS[config["engine"]](config)
    try:
        library = config["library"].partition("==")[0]
        torch_version, cuda_version = torch_versions()
        ready = {
            "ready": True,
            "library": f"{library}=={importlib.metadata.version(library)}",
            "torch": torch_version,
            "cuda": cuda_version,
            "engine_args": engine.engine_args,
            "resolved": engine.resolved(),
        }
        proto.write(json.dumps(ready) + "\n")
        return serve(engine, sys.stdin, proto)
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
