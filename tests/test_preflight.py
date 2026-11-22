import pytest
from catalogs import CATALOG
from effeval.core.catalog import TaskSize
from effeval.plan import freeze
from effeval.preflight import estimate

RAW = {
    "model": "qwen3-0.6b-base",
    "method": {"name": "gptq"},
    "field": "quantization",
    "hardware": {"gpu": "H100-SXM"},
}


def _estimate(task):
    plan, issues = freeze({**RAW, "tasks": [task]}, CATALOG, owner=None, server=None, lookup=None)
    assert plan is not None, issues
    return estimate(plan, CATALOG, None)


def test_a_task_without_a_size_prior_has_no_measure_estimate():
    result = _estimate("lm_eval/mmlu")
    assert all(job.measure_s is None for job in result.jobs)
    assert all("no size prior" in job.basis for job in result.jobs)
    assert result.total_s is None


def test_a_task_with_a_size_prior_still_gets_a_number():
    result = _estimate("native/wikitext2_ppl")
    assert all(job.measure_s is not None for job in result.jobs)
    assert result.total_s is not None

def test_a_fractional_limit_scales_the_measure_estimate_by_that_fraction():
    sized = CATALOG.tasks["lm_eval/arc_challenge"].model_copy(
        update={"size": TaskSize(samples=1000, prompt_tokens=100, choices=4)}
    )
    catalog = CATALOG.model_copy(update={"tasks": {**CATALOG.tasks, sized.name: sized}})

    def measure(limit):
        params = {} if limit is None else {"limit": limit}
        plan, issues = freeze(
            {**RAW, "tasks": [{"name": sized.name, "params": params}]}, CATALOG, owner=None, server=None, lookup=None
        )
        assert plan is not None, issues
        return estimate(plan, catalog, None).jobs[0].measure_s

    assert measure(0.5) == pytest.approx(measure(None) / 2, rel=0.01)
    assert measure(0.25) == pytest.approx(measure(250), rel=0.01)
