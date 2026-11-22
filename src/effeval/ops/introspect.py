"""Extract architecture layouts and model registry entries from Hugging Face metadata.

Everything comes from what transformers already exposes, on a meta-device model (no weights are loaded):
`base_model_prefix` + `config.base_model_pp_plan` locate the decoder layers, the base model's
`_can_record_outputs["attentions"]` names the attention class, `_supports_attention_backend` says whether the
attention function can be swapped, and the decoder layer's `nn.Linear` modules are the linear sites.

The layer graph comes from `torch.export` of one decoder layer of a tiny model of the same class built from
the real config: ATen nodes are collapsed into module nodes through `nn_module_stack`, and the data flow
between them gives the edges.
"""

from __future__ import annotations

import copy
import os

import torch
import transformers
from huggingface_hub import HfApi
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from effeval.core.catalog import ArchitectureDef, ModelEntry, model_code
from effeval.core.graph import GraphGlobals, GraphNode, LayerGraph, ModelShape

DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}


def meta_model(config: PretrainedConfig) -> PreTrainedModel:
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config)


def _layers_path(model: PreTrainedModel, config: PretrainedConfig) -> str:
    base = model.base_model_prefix
    plan = getattr(config, "base_model_pp_plan", None) or {}
    for key in plan:
        module = model.get_submodule(f"{base}.{key}")
        if isinstance(module, nn.ModuleList) and len(module) == config.num_hidden_layers:
            return f"{base}.{key}"
    raise ValueError(f"{type(model).__name__}: no decoder layer list in base_model_pp_plan {list(plan)}")


def _attention_path(model: PreTrainedModel, layer: nn.Module) -> str | None:
    recorded = getattr(model.base_model, "_can_record_outputs", None) or {}
    target = recorded.get("attentions")
    if target is None:
        return None
    target_class = getattr(target, "target_class", target)
    paths = [name for name, module in layer.named_modules() if isinstance(module, target_class)]
    if len(paths) != 1:
        raise ValueError(f"expected one {target_class.__name__} per decoder layer, found {paths}")
    return paths[0]


_TINY_DIMS = {
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 97,
    "num_hidden_layers": 1,
}
_ACTIVATIONS = {"silu", "gelu", "relu", "tanh", "sigmoid"}
_SDPA_SITES = ("attn.q", "attn.k", "attn.v")


def _tiny_config(config: PretrainedConfig) -> PretrainedConfig:
    """Copy of `config` with every dimension shrunk (flags kept) and one layer, using the sdpa backend."""
    tiny = copy.deepcopy(config)
    for key, value in _TINY_DIMS.items():
        if hasattr(tiny, key):
            setattr(tiny, key, value)
    layer_types = getattr(tiny, "layer_types", None)
    if layer_types:
        tiny.layer_types = layer_types[:1]
    tiny._attn_implementation = "sdpa"
    return tiny


def _module_name(root: nn.Module, target: nn.Module) -> str:
    return next(name for name, module in root.named_modules() if module is target)


def _graph_globals(tiny: PreTrainedModel, layers_path: str, norm_kind: str) -> GraphGlobals:
    layers = tiny.get_submodule(layers_path)
    final = [
        _module_name(tiny, m)
        for m in tiny.base_model.children()
        if m is not layers
        and "norm" in type(m).__name__.lower()
        and isinstance(getattr(m, "weight", None), nn.Parameter)
    ]
    if len(final) != 1:
        raise ValueError(f"{type(tiny).__name__}: expected one final norm, found {final}")
    return GraphGlobals(
        embedding=_module_name(tiny, tiny.get_input_embeddings()),
        final_norm=final[0],
        lm_head=_module_name(tiny, tiny.get_output_embeddings()),
        norm_kind=norm_kind,
    )


class _LayerCall(nn.Module):
    """Decoder layer with the positional signature `torch.export` needs: (hidden, cos, sin) -> hidden."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer = layer

    def forward(self, x, cos, sin):
        out = self.layer(x, attention_mask=None, position_embeddings=(cos, sin))
        return out[0] if isinstance(out, tuple) else out


def _unique_ids(names: dict[str, str]) -> dict[str, str]:
    """Preferred id per key; a key whose preferred id collides gets its dotted path with underscores."""
    counts: dict[str, int] = {}
    for preferred in names.values():
        counts[preferred] = counts.get(preferred, 0) + 1
    return {k: (v if counts[v] == 1 else k.replace(".", "_")) for k, v in names.items()}


def _layer_graph(
    config: PretrainedConfig, layers_path: str, attention: str | None, linears: dict[str, str]
) -> LayerGraph:
    """Graph of one decoder layer of the architecture `config` instantiates, from `torch.export`."""
    if attention is None:
        raise ValueError("no attention module found in the decoder layer")
    tiny = AutoModelForCausalLM.from_config(_tiny_config(config)).eval()
    layer = tiny.get_submodule(layers_path)[0]
    rotary = getattr(tiny.base_model, "rotary_emb", None)
    if rotary is None:
        raise ValueError(f"{type(tiny).__name__}: base model has no rotary_emb")
    x = torch.randn(1, 5, _TINY_DIMS["hidden_size"])
    cos, sin = rotary(x, torch.arange(5)[None])
    with torch.no_grad():
        exported = torch.export.export(_LayerCall(layer).eval(), (x, cos, sin), strict=False)
    placeholders = {n.name: n for n in exported.graph.nodes if n.op == "placeholder"}
    user_inputs = [placeholders[name] for name in exported.graph_signature.user_inputs]
    rope_inputs = set(user_inputs[1:])

    def path_of(node) -> str:
        stack = node.meta.get("nn_module_stack") or {}
        name = list(stack.values())[-1][0] if stack else "layer"
        return name[len("layer.") :] if name.startswith("layer.") else ""

    def op_of(node) -> str:
        return str(node.target)

    def aten_name(node) -> str:
        parts = op_of(node).split(".")
        return parts[1] if parts[0] == "aten" and len(parts) > 1 else ""

    calls = [n for n in exported.graph.nodes if n.op == "call_function" and "_assert" not in op_of(n)]
    norm_paths = {
        path_of(n) for n in calls if path_of(n) and ("rsqrt" in op_of(n) or "layer_norm" in op_of(n))
    }
    act_paths = {
        path_of(n)
        for n in calls
        if path_of(n) not in linears
        and path_of(n) not in norm_paths
        and path_of(n) != attention
        and aten_name(n) in _ACTIVATIONS
    }
    linear_ids = _unique_ids({p: c.split(".")[-1].removesuffix("_proj") for p, c in linears.items()})
    norm_ids = _unique_ids({p: p.split(".")[-1] for p in sorted(norm_paths)})

    # `producers[n]` = ids of the module nodes whose output `n` carries, looking through plain tensor ops.
    producers: dict = {user_inputs[0]: ("x_in",)}
    depends_on_rope_inputs = set(rope_inputs)
    after_attention: set = set()
    unit_of: dict = {}
    adds = 0
    for n in calls:
        path = path_of(n)
        inputs = list(n.all_input_nodes)
        if any(a in depends_on_rope_inputs for a in inputs):
            depends_on_rope_inputs.add(n)
        fed_by_modules = [a for a in inputs if producers.get(a)]
        if aten_name(n) in ("to", "_to_copy"):
            unit = None
        elif "scaled_dot_product_attention" in op_of(n):
            unit = "attn"
        elif path in linears:
            unit = linear_ids[path]
        elif path in norm_paths:
            unit = norm_ids[path]
        elif path in act_paths:
            unit = "act"
        elif path == attention and n in depends_on_rope_inputs and not after_attention.intersection(inputs):
            unit = "rope"
        elif path == "" and op_of(n).startswith("aten.add"):
            adds += 1
            unit = f"add{adds}"
        elif op_of(n).startswith("aten.mul") and path != attention and len(fed_by_modules) >= 2:
            unit = "mul"
        else:
            unit = None
        if unit == "attn" or after_attention.intersection(inputs):
            after_attention.add(n)
        if unit is None:
            merged: list[str] = []
            for a in inputs:
                merged += [u for u in producers.get(a, ()) if u not in merged]
            producers[n] = tuple(merged)
        else:
            unit_of[n] = (unit, path)
            producers[n] = (unit,)

    norm_kind = "layer" if any("layer_norm" in op_of(n) for n in calls) else "rms"
    hidden, head_dim = _TINY_DIMS["hidden_size"], _TINY_DIMS["head_dim"]
    nodes: dict[str, GraphNode] = {"x_in": GraphNode(id="x_in", kind="residual")}
    edges: list[tuple[str, str, str | None]] = []

    def new_node(unit: str, path: str, n) -> GraphNode:
        if unit in linear_ids.values():
            return GraphNode(id=unit, kind="linear", module=path, canon=linears[path])
        if unit in norm_ids.values():
            width = n.args[0].meta["val"].shape[-1]
            axis = {hidden: "hidden", head_dim: "head_dim"}.get(width)
            if axis is None:
                raise ValueError(f"norm {path}: cannot tell the normalised axis from width {width}")
            gain = isinstance(getattr(layer.get_submodule(path), "weight", None), nn.Parameter)
            return GraphNode(id=unit, kind="norm", module=path, gain=gain, axis=axis)
        if unit == "act":
            return GraphNode(id=unit, kind="act", module=path, fn=aten_name(n))
        return GraphNode(id=unit, kind="rope" if unit == "rope" else "mul" if unit == "mul" else "add")

    def add_edge(src: str, dst: str, site: str | None) -> None:
        if (src, dst, site) not in edges:
            edges.append((src, dst, site))

    for n in calls:
        if n not in unit_of:
            continue
        unit, path = unit_of[n]
        if unit == "attn":
            if "qk" not in nodes:
                nodes["qk"] = GraphNode(id="qk", kind="matmul", out_site="attn.scores")
                nodes["softmax"] = GraphNode(id="softmax", kind="softmax", out_site="attn.probs")
                nodes["pv"] = GraphNode(id="pv", kind="matmul", out_site="attn.out")
                add_edge("qk", "softmax", "attn.scores")
                add_edge("softmax", "pv", "attn.probs")
            for i, arg in enumerate(n.args[:3]):
                for src in producers[arg]:
                    add_edge(src, "qk" if i < 2 else "pv", _SDPA_SITES[i])
            continue
        if unit not in nodes:
            nodes[unit] = new_node(unit, path, n)
        site = f"linear.{nodes[unit].canon}.input" if nodes[unit].kind == "linear" else None
        for arg in n.all_input_nodes:
            for src in producers.get(arg, ()):
                if src != unit:
                    add_edge("pv" if src == "attn" else src, unit, site)

    _label_blocks(nodes, edges, attention)
    return LayerGraph(
        globals=_graph_globals(tiny, layers_path, norm_kind), nodes=list(nodes.values()), edges=edges
    )


def _label_blocks(
    nodes: dict[str, GraphNode], edges: list[tuple[str, str, str | None]], attention: str
) -> None:
    for node in nodes.values():
        if node.kind in ("rope", "matmul", "softmax"):
            node.block = "attn"
        elif node.kind in ("linear", "act", "mul"):
            in_attention = node.module is not None and node.module.startswith(attention + ".")
            node.block = "attn" if in_attention else "mlp"
    for node in nodes.values():
        if node.kind == "norm":
            consumer = next(nodes[d] for s, d, _ in edges if s == node.id and nodes[d].kind != "add")
            node.block = consumer.block


def _is_moe(layer: nn.Module) -> bool:
    return any(
        "moe" in type(m).__name__.lower() or name.rsplit(".", 1)[-1] == "experts"
        for name, m in layer.named_modules()
    )


def extract_architecture(config: PretrainedConfig, source: str) -> ArchitectureDef:
    """Layout of the architecture class `config` instantiates. Raises ValueError if undeterminable."""
    model = meta_model(config)
    layers_path = _layers_path(model, config)
    layer = model.get_submodule(layers_path)[0]
    attention = _attention_path(model, layer)
    linears: dict[str, str] = {}
    for path, module in layer.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if attention is not None and path.startswith(attention + "."):
            linears[path] = "attn." + path[len(attention) + 1 :]
        else:
            linears[path] = path
    if len(set(linears.values())) != len(linears):
        raise ValueError(f"canonical linear names collide: {linears}")
    covered = {f"{path}.weight" for path in linears}
    uncovered = [name for name, p in layer.named_parameters() if p.dim() >= 2 and name not in covered]
    graph = (
        LayerGraph(unsupported=["moe"])
        if _is_moe(layer)
        else _layer_graph(config, layers_path, attention, linears)
    )
    return ArchitectureDef(
        name=type(model).__name__,
        layers_path=layers_path,
        linears=linears,
        attention_path=attention,
        supports_attention_backend=bool(getattr(model, "_supports_attention_backend", False)),
        uncovered_weights=uncovered,
        graph=graph,
        extracted_with={"transformers": transformers.__version__, "source": source},
    )


def model_shape(config: PretrainedConfig) -> ModelShape:
    """Dimensions of the checkpoint from its config; raises ValueError if the config has no `head_dim`."""
    text = config.get_text_config()
    head_dim = getattr(text, "head_dim", None)
    if head_dim is None:
        raise ValueError(f"{type(config).__name__} has no head_dim; refusing to guess hidden // heads")
    return ModelShape(
        hidden=text.hidden_size,
        intermediate=text.intermediate_size,
        num_layers=text.num_hidden_layers,
        heads=text.num_attention_heads,
        kv_heads=getattr(text, "num_key_value_heads", None) or text.num_attention_heads,
        head_dim=head_dim,
        vocab=text.vocab_size,
        tie_word_embeddings=bool(config.tie_word_embeddings),
    )


def model_entry(
    checkpoint: str, revision: str | None, description: str = ""
) -> tuple[ModelEntry, ArchitectureDef]:
    """Registry entry for a checkpoint (HF id or local directory) plus its extracted architecture.

    Hub checkpoints are pinned to the commit sha that `revision` (default: main) resolves to.
    """
    pinned = revision
    if not os.path.isdir(checkpoint):
        pinned = HfApi().model_info(checkpoint, revision=revision).sha
    config = AutoConfig.from_pretrained(checkpoint, revision=pinned)
    architecture = extract_architecture(config, source=checkpoint)
    declared = (config.architectures or [architecture.name])[0]
    if declared != architecture.name:
        raise ValueError(
            f"{checkpoint}: config declares {declared} but transformers builds {architecture.name}"
        )
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype not in DTYPES:
        raise ValueError(f"{checkpoint}: unsupported native dtype {dtype!r}")
    params = sum(p.numel() for p in meta_model(config).parameters())
    entry = ModelEntry(
        code=model_code(checkpoint),
        checkpoint=checkpoint,
        revision=pinned,
        architecture=architecture.name,
        dtype=DTYPES[dtype],
        params=params,
        tie_word_embeddings=bool(config.tie_word_embeddings),
        shape=model_shape(config),
        description=description,
    )
    return entry, architecture
