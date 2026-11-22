"""Method config errors name the exact offending field, so the UI can outline that input."""

from effeval.core.catalog import builtin_catalog
from effeval.core.manifest import validate_method_config

WEIGHTS = {"sites": ["linear.*.weight"], "layers": "all"}


def rtn_issues(quant):
    manifest = builtin_catalog().manifests["rtn"]
    return validate_method_config(manifest, {"rules": [{**WEIGHTS, "quant": quant}]}, "method.config")


def test_a_bad_value_inside_a_format_union_points_at_that_field():
    (issue,) = rtn_issues({"type": "int", "bits": 9, "symmetric": False, "granularity": "per_channel"})
    assert issue.paths == ["method.config.rules.0.quant.bits"]
    assert "maximum" in issue.message and "any of the given schemas" not in issue.message


def test_an_unknown_setting_points_at_the_closest_format():
    (issue,) = rtn_issues({"type": "fp", "encoding": "e4m3", "granularity": "per_token", "bits": 4})
    assert issue.paths == ["method.config.rules.0.quant"]
    assert "bits" in issue.message


def test_a_valid_rule_has_no_issue():
    assert rtn_issues({"type": "int", "bits": 4, "symmetric": False, "granularity": "per_channel"}) == []
