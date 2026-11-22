"""Time estimates for speed-mode jobs and for speculative jobs (target plus draft)."""

from effeval.core.catalog import TaskSize
from effeval.plan import freeze
from effeval.preflight import estimate
from test_spec_schema import SPEED_TASK, _catalog, _speed_comparatives, spec  # noqa: F401

GPU = "RTX-A6000"


def sized_catalog(size):
    task = SPEED_TASK.model_copy(update={"size": size})
    catalog = _catalog()
    return catalog.model_copy(update={"tasks": {**catalog.tasks, task.name: task}})


def jobs(catalog, **overrides):
    plan, issues = freeze(
        spec(tasks=[SPEED_TASK.name], hardware={"gpu": GPU}, **overrides),
        catalog,
        owner=None,
        server=None,
        lookup=None,
    )
    assert plan is not None, issues
    result = estimate(plan, catalog, None)
    by_id = {e.job_id: e for e in result.jobs}
    main = next(j for j in plan.jobs if j.role == "main")
    reference = next((j for j in plan.jobs if j.role == "reference"), None)
    return result, by_id[main.job_id], None if reference is None else by_id[reference.job_id]


SIZE = TaskSize(samples=80, subtasks=8, prompt_tokens=200, output_tokens=300)


def test_a_speed_job_gets_an_estimate_instead_of_an_error():
    _, main, reference = jobs(sized_catalog(SIZE))
    assert main.measure_s is not None and main.prepare_s is not None
    assert reference is not None and reference.measure_s is not None


def test_a_speculative_job_adds_the_draft_to_loading_and_never_estimates_below_the_baseline():
    _, main, reference = jobs(sized_catalog(SIZE))
    assert main.prepare_s > reference.prepare_s
    assert main.measure_s > reference.measure_s
    assert "one token per round" in main.basis
    assert "one token per round" not in reference.basis


def test_a_speed_job_without_a_size_prior_has_no_measure_estimate():
    result, main, reference = jobs(sized_catalog(None))
    assert main.measure_s is None and reference.measure_s is None
    assert result.total_s is None
    assert "no size prior" in main.basis


def test_a_longer_draft_run_costs_more():
    catalog = sized_catalog(SIZE)

    def measure(k):
        decoding = {
            "mode": "speculative",
            "draft": {"model": "qwen3-0.6b"},
            "num_speculative_tokens": k,
        }
        return jobs(catalog, decoding=decoding)[1].measure_s

    assert measure(8) > measure(2)
