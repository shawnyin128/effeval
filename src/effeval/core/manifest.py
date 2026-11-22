"""Method manifests: the only description of a method the server and UI ever see (architecture.md §5.5)."""

from __future__ import annotations

from typing import Any, Literal

import jsonschema
from pydantic import Field, ValidationError

from effeval.core.canonical import canonical_json, sha256_hex
from effeval.core.fragments import FRAGMENTS
from effeval.core.types import Capabilities, HandleKind, Issue, Severity, StrictModel


class MethodManifest(StrictModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str | None = Field(
        default=None, min_length=1, description="Display name with its standard capitalisation, e.g. 'GPTQ'."
    )
    version: str = Field(min_length=1)
    field: str
    runtime: Literal["hf"]
    entry: str | None = Field(
        description="'package.module:function' imported on the compute node; None for 'base' (no change)."
    )
    capabilities: Capabilities = Field(
        default_factory=lambda: Capabilities(handles=frozenset({HandleKind.IN_PROCESS}))
    )
    uses: list[str] = Field(default_factory=list, description="Shared fragments, see core/fragments.py.")
    architectures: list[str] | None = Field(
        default=None, description="HF architecture classes the method supports; None means any."
    )
    config_schema: dict[str, Any]
    display: list[str] = Field(default_factory=list, description="Config paths shown as fixed table columns.")
    default_preset: str | None = None
    custom_presets: dict[str, dict[str, Any]] = Field(default_factory=dict)
    schema_hash: str
    builtin: bool = False
    description: str = ""


def schema_hash(config_schema: dict[str, Any]) -> str:
    return "sha256:" + sha256_hex(canonical_json(config_schema))


def _closest_errors(error: jsonschema.ValidationError) -> list[jsonschema.ValidationError]:
    """For a failed oneOf/anyOf, the errors of the alternative that came closest (fewest errors), so the report
    names the offending field (e.g. `quant.bits`) instead of "not valid under any of the given schemas"."""
    if error.validator not in ("oneOf", "anyOf") or not error.context:
        return [error]
    branches: dict[Any, list[jsonschema.ValidationError]] = {}
    for sub in error.context:
        branches.setdefault(sub.schema_path[0], []).append(sub)
    closest = min(branches.values(), key=len)
    return [leaf for sub in closest for leaf in _closest_errors(sub)]


def validate_method_config(manifest: MethodManifest, config: dict[str, Any], path: str) -> list[Issue]:
    """Check a resolved method config against the manifest schema and the semantic rules of used fragments."""
    issues: list[Issue] = []
    validator = jsonschema.Draft202012Validator(manifest.config_schema)
    errors = [leaf for error in validator.iter_errors(config) for leaf in _closest_errors(error)]
    for error in sorted(errors, key=lambda e: [str(p) for p in e.absolute_path]):
        location = ".".join(str(p) for p in error.absolute_path)
        if error.validator == "required":
            missing = error.message.split("'")[1] if "'" in error.message else ""
            field_path = ".".join(filter(None, [path, location, missing]))
            code = "preset.required" if manifest.default_preset else "method.config_invalid"
            issues.append(
                Issue(
                    code=code,
                    severity=Severity.ERROR,
                    paths=[field_path],
                    message=f"{manifest.name}: required setting {field_path!r} is missing",
                    hint="Use the paper preset (method.preset: default) or set it explicitly."
                    if code == "preset.required"
                    else "",
                )
            )
        else:
            issues.append(
                Issue(
                    code="method.config_invalid",
                    severity=Severity.ERROR,
                    paths=[".".join(filter(None, [path, location]))],
                    message=f"{manifest.name}: {error.message}",
                )
            )
    if issues:
        return issues
    for fragment in manifest.uses:
        prop, adapter = FRAGMENTS[fragment]
        try:
            adapter.validate_python(config[prop])
        except ValidationError as exc:
            for err in exc.errors():
                location = ".".join(str(p) for p in err["loc"])
                issues.append(
                    Issue(
                        code="method.config_invalid",
                        severity=Severity.ERROR,
                        paths=[".".join(filter(None, [path, prop, location]))],
                        message=f"{manifest.name}: {err['msg']}",
                    )
                )
    return issues
