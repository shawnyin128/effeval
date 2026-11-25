"""Online (runtime) transforms applied to activations before the quantizer sees them.

Order contract: `register` installs forward-pre-hooks and attention operand ops; a quantizer attached later
by `sites.attach` runs after them, so the quantizer always sees the transformed tensor. Call `register`
before `sites.attach`, and remove the returned attachments in reverse order.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from effeval.core.fragments import select_layers
from effeval.ops import rotation, sites


@dataclass(frozen=True)
class OnlineOp:
    """Transform `fn` applied to the activation at `site`, in the layers selected by `layers`.

    `site` is a canonical site name: `linear.<group>.<proj>.input` (applied to the linear's input) or
    `attn.q` / `attn.k` / `attn.v` (applied to the attention operand, after RoPE and the KV cache update).
    """

    site: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    layers: str | list[int] = "all"


class RightMultiply(nn.Module):
    """Online op x @ R over the last dimension, computed in `compute_dtype` and returned in x's dtype."""

    def __init__(self, matrix: torch.Tensor, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.register_buffer("matrix", matrix.detach().clone().to(compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.to(self.compute_dtype) @ self.matrix.to(x.device)).to(x.dtype)


class KronRightMultiply(nn.Module):
    """Online op x @ diag(s) @ (H_K^T (x) H_m) / sqrt(K * m) on x [..., K * m], without the dense matrix.

    `base` is the unnormalised [K, K] matrix H_K, H_m the Sylvester Hadamard matrix of size `m` (a power of
    two), and `signs` an optional [K * m] +-1 vector s. Computed in `compute_dtype` and returned in x's dtype.
    """

    def __init__(
        self,
        base: torch.Tensor,
        m: int,
        signs: torch.Tensor | None = None,
        compute_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.register_buffer("base", base.detach().clone().to(compute_dtype) / math.sqrt(base.shape[0]))
        self.register_buffer("sylvester", rotation.hadamard(m).to(compute_dtype))
        self.register_buffer("signs", None if signs is None else signs.detach().clone().to(compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = x.to(self.compute_dtype)
        if self.signs is not None:
            y = y * self.signs.to(x.device)
        y = y.reshape(*shape[:-1], self.base.shape[0], self.sylvester.shape[0])
        y = torch.einsum("...ac,ba,cd->...bd", y, self.base.to(x.device), self.sylvester.to(x.device))
        return y.reshape(shape).to(x.dtype)


class HeadMix(nn.Module):
    """Online op x @ (R (x) I_head_dim) on x [..., num_heads * head_dim]: mixes heads at each channel d."""

    def __init__(self, matrix: torch.Tensor, head_dim: int, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.head_dim = head_dim
        self.compute_dtype = compute_dtype
        self.register_buffer("matrix", matrix.detach().clone().to(compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        heads = x.to(self.compute_dtype).reshape(*shape[:-1], -1, self.head_dim)
        mixed = torch.einsum("...hd,hg->...gd", heads, self.matrix.to(x.device))
        return mixed.reshape(shape).to(x.dtype)


class HeadwiseRightMultiply(nn.Module):
    """Online op x_h @ R_h on x [B, Hx, T, D] with one [D, D] matrix per head, in `compute_dtype`.

    Raises ValueError when x does not have `matrices.shape[0]` heads.
    """

    def __init__(self, matrices: torch.Tensor, compute_dtype: torch.dtype = torch.float32):
        super().__init__()
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError(f"matrices must be [heads, D, D], got {tuple(matrices.shape)}")
        self.compute_dtype = compute_dtype
        self.register_buffer("matrices", matrices.detach().clone().to(compute_dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.matrices.shape[0] or x.shape[3] != self.matrices.shape[2]:
            raise ValueError(
                f"expected x [B, {self.matrices.shape[0]}, T, {self.matrices.shape[2]}], got {tuple(x.shape)}"
            )
        out = torch.einsum("bhtd,hde->bhte", x.to(self.compute_dtype), self.matrices.to(x.device))
        return out.to(x.dtype)


class Chain(nn.Module):
    """Applies `fns` to the input one after another, first to last."""

    def __init__(self, fns: list[Callable[[torch.Tensor], torch.Tensor]]):
        super().__init__()
        self.fns = tuple(fns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fn in self.fns:
            x = fn(x)
        return x


def _pre_hook(fn: Callable[[torch.Tensor], torch.Tensor]):
    def hook(_module: nn.Module, args: tuple) -> tuple:
        return (fn(args[0]), *args[1:])

    hook._effeval_graph_safe = True
    return hook


def share_pre_hooks(source: nn.Module, target: nn.Module) -> None:
    """Make `target` run the forward-pre-hooks of `source`, e.g. a stand-in for a linear carrying online ops.

    The hook registries are shared, not copied: handles returned by `register` keep working, and hooks added
    or removed on either module apply to both.
    """
    for name in ("_forward_pre_hooks", "_forward_pre_hooks_with_kwargs"):
        setattr(target, name, getattr(source, name))


def register(model: nn.Module, ops: list[OnlineOp]) -> sites.Attachment:
    """Install online ops on a model; returns an Attachment whose `remove()` undoes them.

    Linear-input ops become forward-pre-hooks; attention-operand ops are stored on the attention modules and
    applied by the effeval attention functions, so the attention backend is switched to `effeval_sdpa`.
    Several ops on the same attention operand are applied in the order given.
    Raises SiteError when an op matches no location.
    """
    candidates = sites._candidates(model)
    num_layers = len(sites.decoder_layers(model))
    attachment = sites.Attachment(model=model, hits=[], attention_attr=sites.ATTN_OPS_ATTR)
    attention_ops: dict[nn.Module, dict[str, Callable[[torch.Tensor], torch.Tensor]]] = {}
    try:
        for index, op in enumerate(ops):
            layer_ids = set(select_layers(op.layers, num_layers))
            matched = [
                loc
                for loc in candidates
                if loc.layer in layer_ids and fnmatch.fnmatchcase(loc.site, op.site)
            ]
            if not matched:
                raise sites.SiteError(
                    f"online op {index} (site {op.site!r}, layers {op.layers!r}) matches nothing"
                )
            for loc in matched:
                if loc.site.endswith(".input"):
                    attachment.handles.append(loc.module.register_forward_pre_hook(_pre_hook(op.fn)))
                elif loc.site in ("attn.q", "attn.k", "attn.v"):
                    module_ops = attention_ops.setdefault(loc.module, {})
                    previous = module_ops.get(loc.site)
                    module_ops[loc.site] = op.fn if previous is None else Chain([previous, op.fn])
                else:
                    raise sites.SiteError(f"online ops are not supported at site {loc.site!r}")
        if attention_ops:
            sites.register_attention_functions()
            attachment.previous_attn_implementation = model.config._attn_implementation
            model.set_attn_implementation(sites.SDPA_IMPL)
            for module, module_ops in attention_ops.items():
                setattr(module, sites.ATTN_OPS_ATTR, module_ops)
                attachment.attention_modules.append(module)
    except Exception:
        attachment.remove()
        raise
    return attachment
