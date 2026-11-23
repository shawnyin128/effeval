"""runtime.cuda_graphs and runtime.static_cache_tokens: schema, run_key identity and validation."""

import pytest
from pydantic import ValidationError

from effeval.core.spec import RuntimeSpec
from effeval.validate import validate
from test_spec_schema import (  # noqa: F401  (_speed_comparatives is an autouse fixture)
    RTN_RULES,
    _catalog,
    _speed_comparatives,
    check,
    codes,
    frozen,
    role,
    spec,
    speed_spec,
)

ATTN_K = {"type": "int", "bits": 4, "symmetric": False, "granularity": "per_token"}


def rtn(rules):
    return {"name": "rtn", "preset": None, "config": {"rules": rules}}


def attn_rule(**quant):
    return [{"sites": ["attn.k"], "quant": {**ATTN_K, **quant}}]


def main_key(raw):
    return role(frozen(raw)[0], "main").run_key


def test_runtime_defaults_and_capacity_floor():
    runtime = RuntimeSpec()
    assert runtime.cuda_graphs is True
    assert runtime.static_cache_tokens == 4096
    with pytest.raises(ValidationError):
        RuntimeSpec(static_cache_tokens=100)
    assert RuntimeSpec(static_cache_tokens=256).static_cache_tokens == 256


def test_speed_run_key_depends_on_cuda_graphs_and_capacity():
    base = main_key(speed_spec())
    assert main_key(speed_spec(runtime={"cuda_graphs": False})) != base
    assert main_key(speed_spec(runtime={"static_cache_tokens": 8192})) != base


def test_speculative_quality_run_key_depends_on_the_settings():
    base = main_key(spec())
    assert main_key(spec(runtime={"cuda_graphs": False})) != base


def test_quality_run_key_ignores_the_settings():
    plain = {"model": "qwen3-0.6b-base", "method": {"name": "gptq"}, "field": "quantization"}
    base = main_key(plain)
    assert main_key({**plain, "runtime": {"cuda_graphs": False}}) == base
    assert main_key({**plain, "runtime": {"static_cache_tokens": 8192}}) == base


def test_static_cache_unsupported_for_sequence_options_in_a_speculative_speed_job():
    for quant in ({"residual": 8}, {"granularity": "per_group", "group_size": 32, "group_axis": "token"}):
        assert "runtime.static_cache_unsupported" in check(speed_spec(method=rtn(attn_rule(**quant))))


def test_static_cache_unsupported_for_attn_scores_and_probs_sites():
    for site in ("attn.scores", "attn.probs", "attn.*"):
        rules = [{"sites": [site], "quant": ATTN_K}]
        assert "runtime.static_cache_unsupported" in check(speed_spec(method=rtn(rules)))


def test_static_cache_unsupported_checks_speculative_quality_jobs_too():
    assert "runtime.static_cache_unsupported" in check(spec(method=rtn(attn_rule(residual=8))))


def test_no_static_cache_error_with_cuda_graphs_off():
    raw = speed_spec(method=rtn(attn_rule(residual=8)), runtime={"cuda_graphs": False})
    assert "runtime.static_cache_unsupported" not in check(raw)


def test_no_static_cache_error_for_weight_only_and_supported_attention_rules():
    assert "runtime.static_cache_unsupported" not in check(speed_spec(method=rtn(RTN_RULES)))
    assert "runtime.static_cache_unsupported" not in check(speed_spec(method=rtn(attn_rule())))
    per_channel = attn_rule(granularity="per_channel")
    assert "runtime.static_cache_unsupported" not in check(speed_spec(method=rtn(per_channel)))


def test_no_static_cache_error_for_a_quality_job_with_sequence_options():
    raw = spec(method=rtn(attn_rule(residual=8)), decoding=None, field="quantization")
    assert "runtime.static_cache_unsupported" not in check(raw)


def test_cuda_graphs_on_an_engine_is_not_flagged_and_changes_identity():
    for runtime in ({"name": "vllm", "cuda_graphs": True}, {"name": "vllm", "cuda_graphs": False}):
        issues = validate(speed_spec(runtime=runtime), _catalog())
        assert "runtime.ineffective" not in codes(issues) + codes(issues, "info") + codes(issues, "warning")
    on = role(frozen(speed_spec(runtime={"name": "vllm", "cuda_graphs": True}))[0], "main")
    off = role(frozen(speed_spec(runtime={"name": "vllm", "cuda_graphs": False}))[0], "main")
    assert on.run_key != off.run_key


def test_engine_capacity_is_checked_with_cuda_graphs_off():
    def issue(budget):
        runtime = {"name": "vllm", "cuda_graphs": False}
        return capacity_issue(speed_spec(generation={"max_new_tokens": budget}, runtime=runtime))

    over = issue(4097)
    assert over is not None and "runtime.static_cache_tokens" in over.hint
    assert "cuda_graphs" not in over.hint
    assert issue(4096) is None


MX_E4M3 = {"type": "mx", "element": "e2m1", "block_size": 16, "scale": "e4m3"}
MX_E8M0 = {**MX_E4M3, "scale": "e8m0"}


def test_mx_e4m3_on_activation_sites_is_rejected_with_graphs():
    for site in ("attn.k", "linear.*.input", "linear.mlp.*.output"):
        raw = speed_spec(method=rtn([{"sites": [site], "quant": MX_E4M3}]))
        assert "runtime.static_cache_unsupported" in check(raw)
        off = speed_spec(method=rtn([{"sites": [site], "quant": MX_E4M3}]), runtime={"cuda_graphs": False})
        assert "runtime.static_cache_unsupported" not in check(off)


def test_mx_e4m3_weights_and_mx_e8m0_activations_are_accepted_with_graphs():
    for site, quant in (("linear.*.weight", MX_E4M3), ("linear.*.input", MX_E8M0), ("attn.k", MX_E8M0)):
        raw = speed_spec(method=rtn([{"sites": [site], "quant": quant}]))
        assert "runtime.static_cache_unsupported" not in check(raw)


def test_float_zero_point_on_key_axis_quantizers_is_rejected_with_graphs():
    for site in ("attn.k", "attn.v"):
        for granularity in ("per_channel", "per_tensor"):
            quant = {**ATTN_K, "granularity": granularity, "zero_point": "float"}
            raw = speed_spec(method=rtn([{"sites": [site], "quant": quant}]))
            assert "runtime.static_cache_unsupported" in check(raw)
            off = speed_spec(method=rtn([{"sites": [site], "quant": quant}]), runtime={"cuda_graphs": False})
            assert "runtime.static_cache_unsupported" not in check(off)


def test_float_zero_point_per_token_or_on_weights_is_accepted_with_graphs():
    per_token = {**ATTN_K, "zero_point": "float"}
    weight = {**ATTN_K, "granularity": "per_channel", "zero_point": "float"}
    for site, quant in (("attn.k", per_token), ("linear.*.weight", weight)):
        raw = speed_spec(method=rtn([{"sites": [site], "quant": quant}]))
        assert "runtime.static_cache_unsupported" not in check(raw)


def capacity_issue(raw):
    return next(
        (i for i in validate(raw, _catalog()) if i.code == "runtime.static_cache_capacity"),
        None,
    )


def test_a_generation_budget_beyond_the_cache_capacity_is_an_error_naming_the_fixes():
    raw = speed_spec(generation={"max_new_tokens": 4097})
    issue = capacity_issue(raw)
    assert issue is not None and issue.severity.value == "error"
    assert "runtime.static_cache_tokens" in issue.hint and "runtime.cuda_graphs: false" in issue.hint
    assert "runtime.static_cache_unsupported" not in check(raw)


def test_the_capacity_check_accepts_a_budget_equal_to_the_capacity_and_follows_the_setting():
    def issue(budget, **runtime):
        return capacity_issue(speed_spec(generation={"max_new_tokens": budget}, runtime=runtime))

    assert issue(4096) is None
    assert issue(4097, cuda_graphs=False) is None
    assert issue(4097, static_cache_tokens=4097) is None
    assert issue(300, static_cache_tokens=256) is not None


def test_the_task_budget_governs_over_the_generation_default():
    hardware = {"gpu": "H100-SXM"}
    over = spec(tasks=[{"name": "speed/spec_bench", "params": {"max_new_tokens": 5000}}], hardware=hardware)
    assert capacity_issue(over) is not None
    under = spec(
        tasks=[{"name": "speed/spec_bench", "params": {"max_new_tokens": 100}}],
        generation={"max_new_tokens": 5000},
        hardware=hardware,
    )
    assert capacity_issue(under) is None


def test_the_capacity_check_covers_speculative_quality_jobs_and_skips_other_jobs():
    assert capacity_issue(spec(generation={"max_new_tokens": 5000})) is not None
    plain = spec(decoding=None, field="quantization", generation={"max_new_tokens": 5000})
    assert capacity_issue(plain) is None
