"""The method interface: a function decorated with `@method(...)` and a pydantic config class.

That is all a method provides. Its manifest (config form, validation, presets, shared settings) is derived
from the decorator and the config class. Packages announce their methods under the `effeval.methods` entry
point group, `effeval method sync` uploads what it finds, and compute nodes import `entry`
('pkg.module:function'). The server never imports anything here except manifest helpers.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from effeval.core.manifest import MethodManifest, schema_hash
from effeval.core.types import Capabilities, HandleKind


@dataclass(frozen=True)
class MethodImpl:
    name: str
    title: str | None
    version: str
    field: str
    runtime: str
    config_model: type[BaseModel] | None
    fn: Callable[..., Any]
    uses: tuple[str, ...]
    architectures: tuple[str, ...] | None
    display: tuple[str, ...]
    default_preset: str | None
    presets: dict[str, dict[str, Any]]
    description: str
    prepare: Callable[[Any, Any, Any], None] | None = None
    architecture_transform: Literal["qkv_shared"] | None = None

    @property
    def entry(self) -> str:
        return f"{self.fn.__module__}:{self.fn.__qualname__}"


_METHODS: dict[str, MethodImpl] = {}


def method(
    name: str,
    *,
    title: str | None = None,
    version: str,
    field: str,
    runtime: str = "hf",
    config: type[BaseModel] | None = None,
    uses: tuple[str, ...] = (),
    architectures: tuple[str, ...] | None = None,
    display: tuple[str, ...] = (),
    default_preset: str | None = None,
    presets: dict[str, dict[str, Any]] | None = None,
    description: str = "",
    prepare: Callable[[Any, Any, Any], None] | None = None,
    architecture_transform: Literal["qkv_shared"] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register `fn(model, config, ctx)` as method `name`; `config` is its pydantic settings class.

    `title` is the display name (e.g. 'GPTQ'); `name` stays the lowercase identifier. `version` must change
    whenever the method's numerical behavior changes. `uses` names shared settings
    (core/fragments.py) whose editors and checks the UI reuses; `presets` maps a preset name to config values;
    `display` lists config fields shown as table columns. `prepare(model, tokenizer, config)` runs before the
    job's preprocessing; the method function after it. `architecture_transform` names the structural rewrite
    (`core.lowrank.ARCHITECTURE_TRANSFORMS`) that `prepare` applies to the model.
    """

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _METHODS and _METHODS[name].fn is not fn:
            raise ValueError(f"method {name!r} is already registered by {_METHODS[name].entry}")
        _METHODS[name] = MethodImpl(
            name=name,
            title=title,
            version=version,
            field=field,
            runtime=runtime,
            config_model=config,
            fn=fn,
            uses=uses,
            architectures=architectures,
            display=display,
            default_preset=default_preset,
            presets=dict(presets or {}),
            description=description,
            prepare=prepare,
            architecture_transform=architecture_transform,
        )
        return fn

    return register


ENTRY_POINT_GROUP = "effeval.methods"


def discover() -> list[MethodImpl]:
    """Import every `effeval.methods` entry point and return the methods they register, sorted by name.

    Raises ImportError/AttributeError for entry points that fail to load, and ValueError for entry points
    that do not point at an @method-decorated function.
    """
    found = {}
    for point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        impl = registered(point.load())
        if impl is None:
            raise ValueError(f"entry point {point.name} = {point.value} is not decorated with @method")
        found[impl.name] = impl
    return [found[name] for name in sorted(found)]


def resolve_entry(entry: str) -> Callable[..., Any]:
    """Import 'package.module:function'. Raises ImportError/AttributeError naming the entry."""
    module_name, sep, attr = entry.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"entry must look like 'package.module:function', got {entry!r}")
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def registered(fn: Callable[..., Any]) -> MethodImpl | None:
    """Return the registration of a decorated function, or None for plain functions."""
    for impl in _METHODS.values():
        if impl.fn is fn:
            return impl
    return None


def manifest_from_impl(impl: MethodImpl, *, builtin: bool) -> MethodManifest:
    if impl.config_model is None:
        config_schema: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
    else:
        config_schema = impl.config_model.model_json_schema(ref_template="#/$defs/{model}")
    return MethodManifest(
        name=impl.name,
        title=impl.title,
        version=impl.version,
        field=impl.field,
        runtime=impl.runtime,
        entry=impl.entry,
        capabilities=Capabilities(handles=frozenset({HandleKind.IN_PROCESS})),
        uses=list(impl.uses),
        architectures=None if impl.architectures is None else list(impl.architectures),
        config_schema=config_schema,
        display=list(impl.display),
        default_preset=impl.default_preset,
        custom_presets=impl.presets,
        schema_hash=schema_hash(config_schema),
        builtin=builtin,
        description=impl.description,
        architecture_transform=impl.architecture_transform,
    )
