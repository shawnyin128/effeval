"""Operator benchmarks of low-bit GEMM kernels on a model's own linear shapes and weights.

For every linear weight of one decoder layer (the middle one) and every token count M of the task, the
weight is quantized on the job's integer grid (the same grid RTN uses), packed for the kernel, and timed with
CUDA events against float16 and bfloat16 torch matmul. Each (linear, M) becomes one sample; the record's
metrics aggregate them.

Marlin: github.com/IST-DASLab/marlin, `marlin.Layer`: FP16 activations x INT4 weights, symmetric, group size
128 or per channel, K % 128 == 0 and N % 256 == 0. `Layer.pack` takes a linear whose weight is the dequantized
weight and scales of shape (N, groups); this follows the repository's test.py.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from effeval.core.catalog import ArchitectureDef, BenchmarkDef, TaskDef, builtin_catalog
from effeval.core.fragments import IntFormat, QuantRule, linear_weight_rules
from effeval.core.spec import JobSpec, ResolvedModel
from effeval.core.types import Artifact, InProcess, QualityResult, Sample, metric_str
from effeval.ops.quant.formats import int_grid

Timer = Callable[[Callable[[], object]], list[float]]
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
L2_FLUSH_BYTES = 256 * 2**20


@dataclass(frozen=True)
class Kernel:
    """A low-bit GEMM: `prepare(dequantized[N, K], scales[N, groups], group)` returns `matmul(a[M, K])`."""

    name: str
    version: Callable[[], str]
    unsupported: Callable[[int, int], str | None]
    prepare: Callable[[torch.Tensor, torch.Tensor, int], Callable[[torch.Tensor], torch.Tensor]]


def _marlin_version() -> str:
    try:
        return importlib.metadata.version("marlin")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _marlin_unsupported(n: int, k: int) -> str | None:
    if k % 128 or n % 256:
        return f"Marlin needs K % 128 == 0 and N % 256 == 0, got K={k}, N={n}"
    return None


def _marlin_prepare(
    weight: torch.Tensor, scales: torch.Tensor, group_size: int
) -> Callable[[torch.Tensor], torch.Tensor]:
    import marlin  # compute-node dependency: pip install git+https://github.com/IST-DASLab/marlin

    n, k = weight.shape
    layer = marlin.Layer(k, n, groupsize=-1 if group_size == k else group_size).to(weight.device)
    linear = nn.Linear(k, n, bias=False)
    linear.weight.data = weight.half()
    layer.pack(linear, scales.half())
    return layer


KERNELS: dict[str, Kernel] = {
    "marlin": Kernel("marlin", _marlin_version, _marlin_unsupported, _marlin_prepare),
}


def cuda_timer(warmup: int, repeats: int, device: torch.device) -> Timer:
    """Per-call latencies (ms) from CUDA events after `warmup` calls; L2 is flushed before each call."""
    flush = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.int32, device=device)

    def time(fn: Callable[[], object]) -> list[float]:
        for _ in range(warmup):
            fn()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for start, end in zip(starts, ends, strict=True):
            flush.zero_()
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize(device)
        return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]

    return time


def quantize_for_kernel(weight: torch.Tensor, fmt: IntFormat) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(dequantized [N, K] float32, scales [N, groups] float32, group size) on the job's integer grid."""
    n, k = weight.shape
    group = fmt.group_size if fmt.granularity == "per_group" else k
    units = weight.float().reshape(n * (k // group), group)  # [N * groups, group]
    grid = int_grid(units, fmt.bits, fmt.symmetric, fmt.clip_ratio)
    return grid.apply(units).reshape(n, k), grid.scale.reshape(n, k // group), group


def _rel_err(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value - reference) / torch.linalg.vector_norm(reference))


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


@torch.no_grad()
def benchmark(
    weights: dict[str, torch.Tensor],
    rules: dict[str, QuantRule],
    kernel: Kernel,
    bench: BenchmarkDef,
    token_counts: list[int],
    timer: Timer,
    seed: int,
) -> tuple[list[Sample], list[str]]:
    """Time and check `kernel` on every (linear, token count); returns (samples, skipped-shape notes).

    Sample ids are `<site>:<M>`; scores hold the kernel latency (median, p10, p90 in ms), baseline medians,
    the speedup over the float16 baseline, and the kernel and quantization relative errors.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    act = DTYPES[bench.activation_dtype]
    samples, skipped = [], []
    for site, weight in weights.items():
        n, k = weight.shape
        why = kernel.unsupported(n, k)
        if why is not None:
            skipped.append(f"{site}: {why}")
            continue
        dequantized, scales, group = quantize_for_kernel(weight, rules[site].quant)
        matmul = kernel.prepare(dequantized.to(act), scales, group)
        baselines = {name: weight.to(DTYPES[name]) for name in bench.baselines}
        for m in token_counts:
            a = torch.randn(m, k, generator=generator).to(device=weight.device, dtype=act)
            out = matmul(a).float()
            exact = a.float() @ weight.float().t()
            on_grid = a.float() @ dequantized.t()
            latencies = timer(lambda a=a, matmul=matmul: matmul(a))
            scores = {
                "latency_ms": statistics.median(latencies),
                "latency_p10_ms": _percentile(latencies, 0.1),
                "latency_p90_ms": _percentile(latencies, 0.9),
                "kernel_rel_err": _rel_err(out, on_grid),
                "quant_rel_err": _rel_err(on_grid, exact),
            }
            for name, w in baselines.items():
                x = a.to(DTYPES[name])
                scores[f"{name}_latency_ms"] = statistics.median(timer(lambda x=x, w=w: x @ w.t()))
            scores["speedup"] = scores[f"{bench.baselines[0]}_latency_ms"] / scores["latency_ms"]
            samples.append(Sample(id=f"{site}:{m}", target={"m": m, "n": n, "k": k}, scores=scores))
    return samples, skipped


def _checkpoint_file(model: ResolvedModel, filename: str) -> Path | None:
    local = Path(model.checkpoint)
    if local.is_dir():
        path = local / filename
        return path if path.exists() else None
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return Path(hf_hub_download(model.checkpoint, filename, revision=model.revision))
    except EntryNotFoundError:
        return None


def load_layer_weights(
    model: ResolvedModel, architecture: ArchitectureDef
) -> tuple[int, dict[str, torch.Tensor]]:
    """(layer index, site -> weight [N, K]) of the model's middle decoder layer, from its safetensors."""
    from safetensors import safe_open

    config = json.loads(_checkpoint_file(model, "config.json").read_text(encoding="utf-8"))
    layer = config["num_hidden_layers"] // 2
    names = {
        f"linear.{site}.weight": f"{architecture.layers_path}.{layer}.{path}.weight"
        for path, site in architecture.linears.items()
    }
    index = _checkpoint_file(model, "model.safetensors.index.json")
    files = json.loads(index.read_text(encoding="utf-8"))["weight_map"] if index else {}
    weights = {}
    for site, tensor_name in names.items():
        shard = _checkpoint_file(model, files.get(tensor_name, "model.safetensors"))
        with safe_open(str(shard), framework="pt") as handle:
            weights[site] = handle.get_tensor(tensor_name)
    return layer, weights


class KernelEvaluator:
    name = "kernel"

    def list_tasks(self) -> None:
        return None

    def run(self, handle: InProcess | None, job: JobSpec, task: TaskDef, workdir: Path) -> QualityResult:
        """Benchmark the task's kernel on the job's model; needs a CUDA GPU and the kernel's package."""
        return _run(job, task)


def _run(job: JobSpec, task: TaskDef) -> QualityResult:
    if not torch.cuda.is_available():
        raise RuntimeError("kernel benchmarks need a CUDA GPU on this node")
    if task.benchmark is None:
        raise ValueError(f"task {task.name!r} is not a kernel benchmark")
    kernel = KERNELS.get(task.benchmark.kernel)
    if kernel is None:
        raise ValueError(f"no kernel {task.benchmark.kernel!r} for task {task.name!r}")
    bench = task.benchmark
    settings = task.settings(job.task.params)
    device = torch.device("cuda", 0)
    architecture = builtin_catalog().architectures[job.model.architecture]
    layer, weights = load_layer_weights(job.model, architecture)
    weights = {site: w.to(device) for site, w in weights.items()}
    rules = [QuantRule.model_validate(r) for r in job.method.config["rules"]]
    applied = {
        site: rule for site, rule in linear_weight_rules(rules, list(weights)).items() if rule is not None
    }
    timer = cuda_timer(settings.warmup, settings.repeats, device)
    samples, skipped = benchmark(
        weights, applied, kernel, bench, settings.token_counts, timer, job.determinism.seed
    )
    if not samples:
        raise RuntimeError(f"no shape of {job.model.name} fits {kernel.name}: {skipped}")
    speedups = [s.scores["speedup"] for s in samples]
    metrics = {
        "speedup": metric_str(math.exp(statistics.fmean(math.log(x) for x in speedups))),
        "kernel_rel_err": metric_str(max(s.scores["kernel_rel_err"] for s in samples)),
        "quant_rel_err": metric_str(statistics.fmean(s.scores["quant_rel_err"] for s in samples)),
    }
    protocol = {
        "backend": "kernel",
        "kernel": kernel.name,
        "kernel_version": kernel.version(),
        "layer": layer,
        "token_counts": settings.token_counts,
        "warmup": settings.warmup,
        "repeats": settings.repeats,
        "timing": "cuda events, L2 flushed before each call",
        "activation_dtype": bench.activation_dtype,
        "baselines": bench.baselines,
        "speedup_over": bench.baselines[0],
        "skipped": skipped,
        "device": torch.cuda.get_device_name(device),
    }
    return QualityResult(metrics=metrics, samples=samples, protocol=protocol, produces={Artifact.PER_SAMPLE})


EVALUATOR = KernelEvaluator()
