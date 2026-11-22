import math

from catalogs import CATALOG
from effeval.core.spec import run_key
from effeval.core.types import Sample
from effeval.metrics.derived import compute
from effeval.plan import freeze


def test_accuracy_stderr_is_the_standard_error_of_correctness():
    samples = [Sample(id=str(i), scores={"correct": c}) for i, c in enumerate([1.0, 0.0, 1.0, 1.0])]
    assert math.isclose(float(compute(["accuracy_stderr"], samples)["accuracy_stderr"]), 0.25)
    assert compute(["accuracy_stderr"], samples[:1]) == {}


def spec(**extra):
    return {
        "model": "qwen3-0.6b-base",
        "method": {"name": "gptq"},
        "field": "quantization",
        "tasks": [{"name": "lm_eval/arc_challenge", "mode": "loglikelihood"}, "native/wikitext2_ppl"],
        **extra,
    }


def test_requested_metrics_ride_on_jobs_without_changing_the_run_key():
    plain, _ = freeze(spec(), CATALOG, owner="a", server=None, lookup=None)
    plan, issues = freeze(spec(metrics=["accuracy_stderr"]), CATALOG, owner="a", server=None, lookup=None)
    mains = {j.spec.task.name: j for j in plan.jobs if j.role == "main"}
    assert mains["lm_eval/arc_challenge"].spec.metrics == ["accuracy_stderr"]
    assert mains["native/wikitext2_ppl"].spec.metrics == []  # no per-sample correctness to compute it from
    assert "metric.not_applicable" in {i.code for i in issues}
    assert {run_key(j.spec) for j in plan.jobs} == {run_key(j.spec) for j in plain.jobs}


def test_unknown_requested_metric_is_an_error():
    plan, issues = freeze(spec(metrics=["bleu"]), CATALOG, owner="a", server=None, lookup=None)
    assert plan is None and "metric.unknown" in {i.code for i in issues}
