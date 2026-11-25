"""Preprocessing engine: applies a job's rotation and scaling steps to a loaded model, before the quantizer.

A source implementation is registered per (kind, source type) with `@source`. `apply` runs the steps of a
`ResolvedPreprocess` in order; for each step it calls every distinct registered function once with all the
step's entries it serves (source types that share a function are handled together, so one function can order
its own edits), collects `StageResult`s, and registers all online ops through `ops.online` in one call, before
any quantizer hook exists, so a quantizer attached later sees the transformed tensor.
"""

from __future__ import annotations

import contextlib
import importlib
import pkgutil
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from effeval.core.catalog import ArchitectureDef
from effeval.core.fragments import Calibration, QuantRule, select_layers
from effeval.core.graph import GraphNode, LayerGraph, ModelShape, positions
from effeval.core.preprocess import SOURCE_VERSIONS, Entry, ResolvedPreprocess, Step
from effeval.ops import online, sites
from effeval.ops.artifacts import ArtifactCache, default_cache
from effeval.ops.calibration import calibration_windows
from effeval.ops.introspect import model_shape
from effeval.ops.online import OnlineOp


class EngineError(ValueError):
    """A preprocessing step cannot be applied to this model (unsupported position, size or option)."""


class SourceVersionMismatch(RuntimeError):
    """The source code installed on this node does not have the algorithm version the plan froze."""


@dataclass(frozen=True)
class ArtifactRef:
    """A derived tensor set held by the artifact cache: its cache key and a human-readable description."""

    key: str
    description: str


@dataclass
class StageResult:
    """What one source function did: edit log lines, online ops to register, artifacts it used or created."""

    edits: list[str] = field(default_factory=list)
    online: list[OnlineOp] = field(default_factory=list)
    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)


class Net:
    """Typed access to the modules of a model through its architecture layout and layer graph."""

    def __init__(self, model: nn.Module, layout: ArchitectureDef, graph: LayerGraph, shape: ModelShape):
        if graph.globals is None:
            raise EngineError(f"{layout.name} has no inspected architecture graph")
        self.model, self.layout, self.graph, self.shape = model, layout, graph, shape
        self.layers: nn.ModuleList = model.get_submodule(layout.layers_path)
        self.embedding: nn.Embedding = model.get_submodule(graph.globals.embedding)
        self.final_norm: nn.Module = model.get_submodule(graph.globals.final_norm)
        self.lm_head: nn.Linear = model.get_submodule(graph.globals.lm_head)

    def module(self, layer: int, node: GraphNode) -> nn.Module:
        """The module of a graph node (linear or norm) in decoder layer `layer`; not for routed experts."""
        assert node.module is not None, f"graph node {node.id} is not a module"
        if node.experts:
            raise EngineError(f"graph node {node.id} is one module per expert; use Net.modules")
        return self.layers[layer].get_submodule(node.module)

    def modules(self, layer: int, node: GraphNode) -> list[nn.Module]:
        """Every module of a graph node in decoder layer `layer`: all routed experts for an `experts` node,
        and none when the layer does not have the node (`node.layers`, or a router or expert node in a
        dense layer).
        """
        assert node.module is not None, f"graph node {node.id} is not a module"
        if node.layers is not None and layer not in self.layer_indices(node.layers):
            return []
        moe = self.layout.moe
        if moe is not None and (node.experts or node.module == moe.router) and layer not in moe.layers:
            return []
        return [m for _, m in sites.expand_path(self.layers[layer], node.module)]

    def linear_node(self, canon: str) -> GraphNode:
        """The graph node of the linear with canonical name `canon` (e.g. `mlp.down_proj`)."""
        return next(n for n in self.graph.of_kind("linear") if n.canon == canon)

    def layer_indices(self, layers: Any) -> list[int]:
        """Decoder layers a `LayerSelector` picks."""
        return select_layers(layers, len(self.layers))

    def stream_consumers(self) -> list[tuple[GraphNode, list[GraphNode]]]:
        """Hidden-axis norms reading the residual stream, each with the linears it feeds."""
        stream = {n.id for n in self.graph.nodes if n.kind in ("residual", "add")}
        out = []
        for norm in self.graph.of_kind("norm"):
            if norm.axis == "hidden" and any(p.id in stream for p, _ in self.graph.producers(norm.id)):
                out.append((norm, [c for c, _ in self.graph.consumers(norm.id) if c.kind == "linear"]))
        return out

    def stream_producers(self) -> list[GraphNode]:
        """Linears writing into the residual stream, directly or through a `route` node (routed experts).

        The router feeds a `route` node only its weights, so it is not a producer.
        """
        stream = {n.id for n in self.graph.nodes if n.kind in ("residual", "add")}
        router = None if self.layout.moe is None else self.layout.moe.router

        def reaches_stream(node: GraphNode, producer: GraphNode) -> bool:
            if node.id in stream:
                return True
            return (
                node.kind == "route"
                and producer.module != router
                and any(c.id in stream for c, _ in self.graph.consumers(node.id))
            )

        return [
            n
            for n in self.graph.of_kind("linear")
            if any(reaches_stream(c, n) for c, _ in self.graph.consumers(n.id))
        ]


@dataclass
class Context:
    """Everything a source function may use besides the model and the entries."""

    tokenizer: Any
    net: Net
    model_id: dict[str, Any]
    artifacts: ArtifactCache
    role: str = "target"
    downstream: list[QuantRule] | None = None
    step: Step | None = None
    upstream: list[dict[str, Any]] = field(default_factory=list)
    _windows: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    pending_online: list[OnlineOp] = field(default_factory=list)
    """Online ops collected so far (earlier steps and functions), not yet attached to the model."""

    @property
    def calibration(self) -> Calibration | None:
        return None if self.step is None else self.step.calibration

    def windows(self) -> list[torch.Tensor]:
        """Calibration token windows of the current step (memoised per calibration fragment)."""
        calibration = self.calibration
        if calibration is None:
            raise EngineError("this step has no calibration fragment")
        key = calibration.model_dump_json()
        if key not in self._windows:
            self._windows[key] = calibration_windows(calibration, self.tokenizer.name_or_path)
        return self._windows[key]


@dataclass
class Applied:
    """Result of `apply`: per-source results and the online-op attachment (undo with `.remove()`)."""

    results: list[StageResult] = field(default_factory=list)
    online_ops: sites.Attachment | None = None

    @property
    def edits(self) -> list[str]:
        return [line for r in self.results for line in r.edits]

    @property
    def artifacts(self) -> dict[str, ArtifactRef]:
        return {name: ref for r in self.results for name, ref in r.artifacts.items()}


SourceFn = Callable[[nn.Module, LayerGraph, list[Any], Context], StageResult]
_SOURCES: dict[tuple[str, str], SourceFn] = {}


@contextlib.contextmanager
def attached_online(model: nn.Module, ops: list[OnlineOp]) -> Iterator[None]:
    """Keep the online `ops` (e.g. `Context.pending_online`) attached to `model` for the duration of the block."""
    attachment = online.register(model, ops) if ops else None
    try:
        yield
    finally:
        if attachment is not None:
            attachment.remove()


def source(source_type: str, kind: str) -> Callable[[SourceFn], SourceFn]:
    """Register `fn(model, graph, entries, ctx) -> StageResult` for a source type declared in core.preprocess.

    Stack the decorator to serve several source types with one function. Raises ValueError for an undeclared
    or already registered (kind, source type).
    """

    def register(fn: SourceFn) -> SourceFn:
        if source_type not in SOURCE_VERSIONS.get(kind, {}):
            raise ValueError(f"{kind} source {source_type!r} is not declared in SOURCE_VERSIONS")
        if (kind, source_type) in _SOURCES:
            raise ValueError(f"{kind} source {source_type!r} is already registered")
        _SOURCES[(kind, source_type)] = fn
        return fn

    return register


def check_versions(preprocess: ResolvedPreprocess) -> None:
    """Raise SourceVersionMismatch when a frozen source version differs from the installed one."""
    for name, frozen in preprocess.versions.items():
        kind, _, source_type = name.partition(".")
        installed = SOURCE_VERSIONS.get(kind, {}).get(source_type)
        if installed != frozen:
            raise SourceVersionMismatch(
                f"preprocess source {name!r}: installed code is version {installed}, the plan froze {frozen}"
            )


def make_context(
    model: nn.Module,
    tokenizer: Any,
    model_id: dict[str, Any],
    *,
    role: str = "target",
    downstream: list[QuantRule] | None = None,
    artifacts: ArtifactCache | None = None,
) -> Context:
    """Context for `model`: architecture layout and graph from the catalog, shape from the model's config."""
    layout = sites.architecture(model)
    graph = layout.effective_graph()
    if graph is None:
        raise EngineError(f"{layout.name} has no inspected architecture graph; run `effeval arch extract`")
    net = Net(model, layout, graph, model_shape(model.config))
    return Context(
        tokenizer=tokenizer,
        net=net,
        model_id=model_id,
        artifacts=artifacts if artifacts is not None else default_cache(),
        role=role,
        downstream=downstream,
    )


def check_position(ctx: Context, kind: str, entry: Entry) -> None:
    """Raise EngineError unless the graph says `kind` at `entry.mode` is valid at `entry.position`."""
    table = positions(ctx.net.graph, ctx.net.shape)
    cell = table.get(entry.position)
    if cell is None:
        raise EngineError(f"unknown position {entry.position!r}; positions: {sorted(table)}")
    verdict = cell[kind][entry.mode]
    if verdict != "ok":
        raise EngineError(f"{kind} {entry.mode} at {entry.position} is not available: {verdict}")


def apply(model: nn.Module, preprocess: ResolvedPreprocess, ctx: Context) -> Applied:
    """Apply every step of `preprocess` to `model` in order, then register the collected online ops.

    Weight and norm edits happen in place. Raises SourceVersionMismatch, EngineError (unsupported position,
    size or option) or KeyError (no implementation registered for a source type).
    """
    _discover()
    check_versions(preprocess)
    applied = Applied()
    online_ops: list[OnlineOp] = []
    done: list[dict[str, Any]] = []
    ctx.pending_online = online_ops
    for step in preprocess.steps:
        ctx.step = step
        ctx.upstream = list(done)
        groups: dict[SourceFn, list[Entry]] = {}
        for entry in step.entries:
            fn = _SOURCES.get((step.kind, entry.source.type))
            if fn is None:
                raise KeyError(f"no implementation registered for {step.kind} source {entry.source.type!r}")
            groups.setdefault(fn, []).append(entry)
        for fn, entries in groups.items():
            result = fn(model, ctx.net.graph, entries, ctx)
            applied.results.append(result)
            online_ops += result.online
        done.append(step.model_dump(mode="json"))
    ctx.step = None
    if online_ops:
        applied.online_ops = online.register(model, online_ops)
    return applied


def apply_preprocess(
    model: nn.Module, tokenizer: Any, preprocess: ResolvedPreprocess, model_id: dict[str, Any]
) -> Applied:
    """`make_context` + `apply` for a job's frozen preprocess; the entry point of runtime.hf.prepare."""
    return apply(model, preprocess, make_context(model, tokenizer, model_id))


_discovered = False


def _discover() -> None:
    """Import every sibling module so its `@source` registrations run (new methods only add a file)."""
    global _discovered
    if _discovered:
        return
    package = importlib.import_module(__package__)
    for info in pkgutil.iter_modules(package.__path__):
        if info.name != "engine" and not info.name.startswith("_"):
            importlib.import_module(f"{__package__}.{info.name}")
    _discovered = True
