from pydantic import Field

from effeval.core.catalog import BenchmarkDef, BenchmarkParams, KernelWeights, TaskDef, TaskMetric, TaskSize
from effeval.core.types import Artifact, Mode


class Params(BenchmarkParams):
    token_counts: list[int] = Field(
        default=[1, 16, 256, 4096], min_length=1, description="GEMM M dimension: tokens per call."
    )
    warmup: int = Field(default=25, ge=0, description="Untimed calls before timing.")
    repeats: int = Field(default=100, ge=1, description="Timed calls per shape; the median is reported.")


BENCHMARK = TaskDef(
    name="kernel/gemm_w4a16",
    evaluator="kernel",
    target="gemm_w4a16",
    version="1",
    category="kernel",
    description=(
        "W4A16 GEMM speed and error on the linear shapes of one decoder layer of the model (the middle "
        "one), for each token count. The kernel's output is compared with the dequantized weights in float32 "
        "(kernel error) and with the original weights (quantization error); latency is timed with CUDA "
        "events after warmup, L2 flushed per call, against float16 and bfloat16 torch matmul on the same GPU."
    ),
    modes=[Mode.BENCHMARK],
    default_mode=Mode.BENCHMARK,
    num_fewshot=0,
    produces={Mode.BENCHMARK: frozenset({Artifact.PER_SAMPLE})},
    metrics=[
        TaskMetric(name="speedup", higher_is_better=True),
        TaskMetric(name="kernel_rel_err", higher_is_better=False),
        TaskMetric(name="quant_rel_err", higher_is_better=False),
    ],
    sample_scores=[
        "latency_ms",
        "latency_p10_ms",
        "latency_p90_ms",
        "speedup",
        "kernel_rel_err",
        "quant_rel_err",
    ],
    reference=False,
    size=TaskSize(samples=28, prompt_tokens=0),  # 7 linear shapes x 4 token counts
    benchmark=BenchmarkDef(
        kernel="marlin",
        activation_dtype="float16",  # Marlin multiplies FP16 activations by INT4 weights
        baselines=["float16", "bfloat16"],
        weights=KernelWeights(type="int", bits=[4], symmetric=[True], group_sizes=[128, None]),
    ),
    params=Params,
)
