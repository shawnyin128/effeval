"""Kernel benchmark task: planning rules on CPU, and the benchmark loop with a stand-in kernel and timer."""

import pytest

from effeval.core.catalog import builtin_catalog
from effeval.core.spec import run_key
from effeval.plan import freeze

CATALOG = builtin_catalog()
W4 = {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_group", "group_size": 128}


def spec(quant=W4, gpu="H100-SXM"):
    return {
        "model": "qwen3-8b-base",
        "method": {
            "name": "rtn",
            "preset": None,
            "config": {"rules": [{"sites": ["linear.*.weight"], "quant": quant}]},
        },
        "field": "quantization",
        "tasks": ["kernel/gemm_w4a16"],
        "hardware": {"gpu": gpu},
    }


def test_benchmark_job_has_no_reference_and_its_run_key_follows_the_gpu():
    plan, issues = freeze(spec(), CATALOG, owner="a", server=None, lookup=None)
    assert plan is not None, issues
    assert [(j.role, j.spec.evaluator.name, j.compare) for j in plan.jobs] == [("main", "kernel", [])]
    other, _ = freeze(spec(gpu="H200-SXM"), CATALOG, owner="a", server=None, lookup=None)
    assert run_key(plan.jobs[0].spec) != run_key(other.jobs[0].spec)


@pytest.mark.parametrize(
    ("quant", "gpu", "code"),
    [
        ({**W4, "symmetric": False}, "H100-SXM", "kernel.unsupported_config"),
        ({**W4, "group_size": 64}, "H100-SXM", "kernel.unsupported_config"),
        ({**W4, "bits": 3}, "H100-SXM", "kernel.unsupported_config"),
        (W4, None, "kernel.hardware_required"),
    ],
)
def test_configs_the_kernel_cannot_run_are_rejected_at_plan_time(quant, gpu, code):
    plan, issues = freeze(spec(quant, gpu), CATALOG, owner="a", server=None, lookup=None)
    assert plan is None and code in {i.code for i in issues if i.severity == "error"}


def test_per_channel_symmetric_int4_is_accepted():
    per_channel = {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_channel"}
    plan, issues = freeze(spec(per_channel), CATALOG, owner="a", server=None, lookup=None)
    assert plan is not None, issues


torch = pytest.importorskip("torch")


def test_kernel_grid_is_the_rtn_grid():
    from effeval.core.fragments import IntFormat
    from effeval.evaluators.kernel import quantize_for_kernel
    from effeval.ops.quant.quant import fake_quantize

    weight = torch.randn(256, 384, generator=torch.Generator().manual_seed(0))
    fmt = IntFormat.model_validate(W4)
    dequantized, scales, group = quantize_for_kernel(weight, fmt)
    assert group == 128 and scales.shape == (256, 3)
    assert torch.equal(dequantized, fake_quantize(weight, fmt, "weight"))
    codes = dequantized.reshape(256, 3, 128) / scales.unsqueeze(-1)
    assert torch.allclose(codes, codes.round(), atol=1e-4)
    assert codes.round().min() >= -8 and codes.round().max() <= 7


def test_benchmark_loop_scores_every_shape_and_skips_misfits():
    from effeval.core.fragments import QuantRule
    from effeval.evaluators.kernel import Kernel, benchmark

    stand_in = Kernel(
        name="stand_in",
        version=lambda: "0",
        unsupported=lambda n, k: "K must be 256" if k != 256 else None,
        prepare=lambda dq, scales, group: lambda a: a.float() @ dq.float().t(),
    )
    calls = iter([2.0, 4.0, 1.0, 1.0] * 10)

    def timer(fn):
        fn()
        value = next(calls)
        return [value, value, value]

    bench = CATALOG.tasks["kernel/gemm_w4a16"].benchmark
    rule = QuantRule.model_validate({"sites": ["linear.*.weight"], "quant": W4})
    weights = {
        "linear.a.weight": torch.randn(512, 256, generator=torch.Generator().manual_seed(1)),
        "linear.b.weight": torch.randn(512, 384, generator=torch.Generator().manual_seed(2)),
    }
    samples, skipped = benchmark(weights, {s: rule for s in weights}, stand_in, bench, [1, 8], timer, seed=0)
    assert [s.id for s in samples] == ["linear.a.weight:1", "linear.a.weight:8"]
    assert skipped == ["linear.b.weight: K must be 256"]
    first = samples[0].scores
    assert first["latency_ms"] == 2.0 and first["float16_latency_ms"] == 4.0 and first["speedup"] == 2.0
    assert first["kernel_rel_err"] < 1e-2 < first["quant_rel_err"] < 0.5
    assert samples[0].target == {"m": 1, "n": 512, "k": 256}


@pytest.mark.compute
def test_middle_layer_weights_load_from_a_local_checkpoint(tmp_path):
    transformers = pytest.importorskip("transformers")
    from effeval.core.spec import ResolvedModel
    from effeval.evaluators.kernel import load_layer_weights

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=False,
    )
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.save_pretrained(tmp_path)
    resolved = ResolvedModel(
        name="t", checkpoint=str(tmp_path), revision=None, architecture="Qwen3ForCausalLM", dtype="float32"
    )
    layer, weights = load_layer_weights(resolved, CATALOG.architectures["Qwen3ForCausalLM"])
    assert layer == 2
    assert weights["linear.attn.k_proj.weight"].shape == (16, 32)
    assert torch.equal(weights["linear.mlp.down_proj.weight"], model.model.layers[2].mlp.down_proj.weight)
