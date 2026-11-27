# Vendored reference implementation for tests only.
# Source: https://github.com/SAI-Lab-NYU/WSVD (commit 9dce5ce82314cf6cc7ecc6b1ee6fbe19a28a4484), directory
# fake_quant/. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
# The vendored files were compared with `gh api repos/SAI-Lab-NYU/WSVD/contents/fake_quant/<file>?ref=<commit>`
# (byte-identical); every vendored slice below is a verbatim copy of those lines apart from the listed
# modifications and a shifted indentation.
# Vendored (line ranges refer to that commit; the module builds the slices in the order of the official call chain):
#   utils.py 71-78 (set_seed)
#   hadamard_utils.py 63-84 (matmul_hadU), 90-95 (random_hadamard_matrix), 263-264 (is_pow2); get_hadK keeps only the
#     power-of-two path (the else branch at 56-60)
#   quant_utils.py 8-64 (get_minq_maxq .. AsymSTEQuantize), 346-535 (WeightQuantizer: find_paramsb, find_params,
#     quantize, fake_quantize, enabled, ready)
#   optimizer_utils.py 1-178 (whole file: SGDG and its helpers)
#   rotation_utils.py 782-809 (random_orthogonal_matrix, get_orthogonal_matrix), 1104-1119 (apply_exact_had_to_UV)
#   act_aware_utils.py (calib_input_distributionlowresources, cholesky branch) hook 345-366, hook registration 499-507,
#     hook removal 513-521, Cholesky finalisation 524-557
#   grad_info_utils.py: sharing of the k_proj Cholesky factor to q/v 191-202 (from calib_grad_info), requires_grad setup
#     245-259, text batch 264-273, separate per-head S_grad accumulation 601-660, normalisation 941-952; per-head
#     separate SVD of prepare_fuse_svd 1149-1187; scores 1532-1537 and 1607-1625, `use_S_gradinfo_init` exit and
#     layer count 1703-1706, budget of the separate q/k/v path 1890-1927, result log 1929-1931 (from
#     svd_qkv_with_grad_info); get_top_k_scores 1935-2012; taylor_like_s_grad_squared 2143-2156,
#     get_layer_importance 2160-2166
#   local_ft_grad_utils.py: q/k/v SVD before rotation 120-188 (from calib_localft_grad_info), text batch 320-329,
#     per-projection weight-gradient accumulation (Fisher weights) 535-537 and 569-595, normalisation 667-686;
#     q projection training block 1000-1146 (local_ft_fused_svd), o_proj dense block 1449-1549,
#     set_grad_for_svd_modules and disable_grad_for_svd_modules 2122-2142, get_layer_importance 2404-2412,
#     create_and_fuse_scalingmatrix_to_W .. get_had_W_and_ABlinear 2446-2917
#   svd_utils.py 32-77 (SVDLinear.__init__, per-head local_ft branch), 959-1045 (SVDLinear.from_linear_localft)
# Modifications to the vendored code (everything else is verbatim):
#   (1) utils.get_dev() -> the module constant DEVICE = torch.device("cpu"); .cuda() -> .to(DEVICE);
#   (2) the CUDA Hadamard kernels are replaced by pure-torch stand-ins: get_hadK handles power-of-two sizes only
#       (the table-based sizes raise NotImplementedError); matmul_hadU_cuda(X, hadK, K) = (H(n) @ X.t()).t() with
#       H(n) = effeval.ops.rotation.hadamard(n) in float32, i.e. the dense matmul effeval uses instead of
#       fast_hadamard_transform; random_hadamard_matrix keeps the official matmul_hadU (float64 Sylvester, divided by
#       torch.tensor(n).sqrt()), so Q differs from effeval's random_hadamard in the last bits;
#   (3) the VLM loops are replaced by text drivers: calib_input_cholesky_text (the verbatim hook on every q/k/v Linear,
#       one model(window) per window instead of the layer-by-layer replay, then the verbatim finalisation including
#       the eigenvalue-shift fallback), calib_grad_info_text, svd_qkv_with_grad_info_text (no cache), prepare_fuse_svd
#       (separate per-head branch only), calib_localft_grad_info_text (per-head SVD before rotation and the Fisher
#       weights W_grad_info), local_ft_qkv_text, dense_prequant_text, wsvd_attention_text (the official order);
#       windows are LongTensors [1, T] (official: batch[0] of the dataloader); layer lists replace
#       model_utils.get_layers; there are no cache files, no output-mode fine-tuning, no smoothing options
#       (args.smooth_grad and args.smooth_final must be False: smooth_gradient_info is not vendored), no fused qkv/kv;
#   (4) the three q/k/v training blocks are one function train_projection_official(layer, idx, key, ...): the q block
#       with q generalised to key; num_heads is an argument (q: num_attention_heads, k and v: num_key_value_heads).
#       The official v block reads `num_key_value_heads if not args.is_q_headnum else num_attention_heads` like the q
#       block, which on a GQA model with --is_q_headnum views the v weight with the wrong head count (the swallowed
#       exception leaves v dense); the reference uses num_key_value_heads for v, as the k block does and as the SVD and
#       Fisher code of v do. The k and v blocks are otherwise identical to the q block (checked by a normalised diff);
#   (5) the try/except wrappers around each projection (which print the error and continue) and the bare excepts that
#       fall back to a dense weight (`U = weight.data; S = V = None`, `params_ = [Alinear.weight]`) are removed so
#       errors surface; the dense paths of o_proj, gate_proj, up_proj and down_proj are written as
#       dense_projection_official from the o_proj block (the four blocks are identical on the reachable path: no
#       svd_info_before_rot, Blinear is None, localft_iters 0 with qat_L_off), with the two fallbacks spelled out;
#       the non-fatal fallbacks inside numerical code (eigenvalue shift for a non-PD Gram, epsilon for a singular
#       Cholesky factor) are kept; the dense paths run after all q/k/v blocks instead of interleaved per layer
#       (the projections are independent and args.is_quant_aware_ft is True in both orders);
#   (6) prints, plots, loss dictionaries and torch.cuda.empty_cache in the vendored blocks are removed or turned into
#       logging.debug; train_projection_official returns (loss values, R) and dense_projection_official returns the
#       loss values; the get_layer_importance of grad_info_utils is named get_layer_importance_gradinfo (the
#       local_ft_grad_utils one keeps the name);
#   (7) imports of the official modules (utils, model_utils, data_utils, hadamard_utils, rotation_utils, quant_utils,
#       optimizer_utils, gptq_utils, llava, tqdm) are removed with their module prefixes; apply_exact_had_to_A and
#       matmul_had_U_128_cuda (unused CUDA helpers) and the VLM branches of the batch loops are not vendored;
#       SVDLinear.__init__ keeps only the per-head local_ft branch with rank_split (other branches raise
#       NotImplementedError) and has no forward;
#   (8) total_rank_sum and total_linear_count are module-level dicts (official: reset in svd_lm_setup); the
#       SVDLinear modules are returned by wsvd_attention_text instead of being installed into the layers;
#   (9) wsvd_args() collects the README W8A8 command as an args namespace (not official code).
# ruff: noqa
import logging
import math
import os
import random
import typing
from typing import List, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from effeval.ops.rotation import hadamard as _effeval_hadamard

DEVICE = torch.device("cpu")

# Counters of svd_utils.svd_lm_setup (official: reset at the start of svd_lm_setup).
total_rank_sum = {"q_proj": 0, "v_proj": 0, "k_proj": 0, "o_proj": 0, "up_proj": 0, "gate_proj": 0, "down_proj": 0}
total_linear_count = {"q_proj": 0, "v_proj": 0, "k_proj": 0, "o_proj": 0, "up_proj": 0, "gate_proj": 0, "down_proj": 0}


# ----------------------------------------------------------------------------- utils.py
def set_seed(seed):
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python's built-in random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU & CUDA
    torch.cuda.manual_seed(seed)  # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimizations for determinism

# ----------------------------------------------------------------------------- hadamard_utils.py (pure-torch stand-ins)
def is_pow2(n):
    return (n & (n - 1) == 0) and (n > 0)

def get_hadK(n, transpose=False):
    hadK, K = None, None
    if not is_pow2(n):
        raise NotImplementedError(f"online Hadamard of size {n} needs the official CUDA kernels")
    K = 1

    return hadK, K

def matmul_hadU(X, transpose=False):
    n = X.shape[-1]
    hadK, K = get_hadK(n, transpose)
    input = X.clone().view(-1, n, 1)
    output = input.clone()
    while input.shape[1] > K:
        input = input.view(input.shape[0], input.shape[1] // 2, 2, input.shape[2])
        output = output.view(input.shape)
        output[:, :, 0, :] = input[:, :, 0, :] + input[:, :, 1, :]
        output[:, :, 1, :] = input[:, :, 0, :] - input[:, :, 1, :]
        output = output.view(input.shape[0], input.shape[1], -1)
        (input, output) = (output, input)
    del output

    if K > 1:
        # Do not explicitly repeat - OOM
        # input = torch.bmm(
        #     hadK.repeat(len(input), 1, 1).to(input.device).to(input.dtype), input)
        # Use bcast instead
        input = hadK.view(1, K, K).to(input) @ input

    return input.view(X.shape) / torch.tensor(n).sqrt()
def random_hadamard_matrix(size, device):
    # See https://cornell-relaxml.github.io/quip-sharp/ , Section "Randomized Hadamard Transformation"
    Q = torch.randint(low=0, high=2, size=(size,)).to(torch.float64)
    Q = Q * 2 - 1
    Q = torch.diag(Q)
    return matmul_hadU(Q).to(device)

def matmul_hadU_cuda(X, hadK, K):
    if K != 1 or hadK is not None:
        raise NotImplementedError("online Hadamard with K != 1 needs the official CUDA kernels")
    n = X.shape[-1]
    H = _effeval_hadamard(n).to(torch.float32)
    return (H.to(X) @ X.t()).t()


# ----------------------------------------------------------------------------- quant_utils.py
def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        scale = scale.to(x.device)
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass the gradient through
        return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        scale = scale.to(x.device)
        zero = zero.to(x.device)
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_paramsb(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape  # [bsz, channel, in_channel]

        # Reshape for per-channel processing
        if self.perchannel:
            x = x.reshape(x.shape[0], x.shape[1], -1)  # Flatten last two dims
        else:
            x = x.flatten(start_dim=1).unsqueeze(1)  # Flatten everything except batch dim

        # Compute min/max along last dimension (in_channel)
        tmp = torch.zeros(x.shape[:-1], device=dev)  # Shape: [bsz, channel]
        xmin = torch.minimum(x.min(dim=-1)[0], tmp)  # Min along last dimension
        xmax = torch.maximum(x.max(dim=-1)[0], tmp)  # Max along last dimension

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full(xmin.shape, float('inf'), device=dev)  # Shape: [bsz, channel]
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(-1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(-1), zero1.unsqueeze(-1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, dim=-1)  # Sum over last dimension (in_channel)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[1]  # Channel dimension
            self.scale = self.scale.repeat(1, tmp)  # Repeat over batch
            self.zero = self.zero.repeat(1, tmp)

        shape = [shape[0], shape[1]] + [1] * (len(shape) - 2)  # [bsz, channel, 1]
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return


    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x
    # def quantize(self, x):
    #     x_dtype = x.dtype
    #     if self.ready() and self.bits < 16:
    #         if self.sym:
    #             return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
    #         return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
    #     return x

    # Return int value and scale in addtional to fake quantized weight
    def fake_quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            scale = self.scale.to(x.device)
            q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
            return (scale * q).to(x_dtype), q, scale
        else:
            return None, None, None

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

# ----------------------------------------------------------------------------- optimizer_utils.py
# coding=utf-8
import random
import torch
from torch.optim.optimizer import Optimizer

# --------- Helpers (kept as in your file, minor safety tweaks) ---------
def norm(v, dim: int = 1):
    assert v.dim() == 2
    return v.norm(p=2, dim=dim, keepdim=True)

def unit(v, dim: int = 1, eps: float = 1e-8):
    vnorm = norm(v, dim)
    return v / vnorm.add(eps), vnorm

def matrix_norm_one(W: torch.Tensor) -> torch.Tensor:
    # returns scalar tensor in same dtype/device
    return torch.abs(W).sum(dim=0).max()

def Cayley_loop(X, W, tan_vec, t):
    # X: [n, p], W: [k, k], tan_vec: [n, p] or [k, n]? (use your original shapes)
    n, p = X.size()
    Y = X + t * tan_vec
    for _ in range(5):
        Y = X + t * torch.matmul(W, 0.5 * (X + Y))
    return Y.t()

def qr_retraction(tan_vec):  # tan_vec: p-by-n, p <= n
    p, n = tan_vec.size()
    tan_vec_t = tan_vec.t().contiguous()
    dtype = tan_vec_t.dtype
    tan_vec_t = tan_vec_t.to(dtype=torch.float32)
    q, r = torch.linalg.qr(tan_vec_t)
    d = torch.diag(r, 0)
    ph = d.sign()
    q = q * ph.expand_as(q)
    return q.t().contiguous().to(dtype=dtype)

_EPS = 1e-8

# --------- Optimizer ---------
class SGDG(Optimizer):
    r"""
    SGD-G on Stiefel manifold (Cayley transform) or plain SGD.

    Args:
        params: iterable of parameters or param groups
        lr (float)
        momentum (float, default 0.0)
        dampening (float, default 0.0)
        weight_decay (float, default 0.0)
        nesterov (bool, default False)
        stiefel (bool, default False)
        omega (float, default 0.0)  # kept for compatibility
        grad_clip (float|None, default None)
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        stiefel: bool = False,
        omega: float = 0.0,
        grad_clip=None,
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires momentum > 0 and dampening == 0")
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            stiefel=stiefel,
            omega=omega,
            grad_clip=grad_clip,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum     = group["momentum"]
            dampening    = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov     = group["nesterov"]
            stiefel      = group["stiefel"]
            lr           = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Common dtype/device
                p_dtype = p.dtype
                p_dev   = p.device

                # Reshape param & grad
                P = p.view(p.size(0), -1)                      # [m, k]
                g = p.grad.view(p.size(0), -1).to(p_dtype)     # [m, k]

                # Optional grad clip
                if group["grad_clip"] is not None:
                    torch.nn.utils.clip_grad_norm_([p], group["grad_clip"])

                # Build "unity" (Q) on same dtype/device as p
                Q, _ = unit(P)                                  # [m, k]
                Q = Q.to(dtype=p_dtype, device=p_dev)

                # Occasional QR retraction for numerical hygiene
                if stiefel and Q.size(0) <= Q.size(1):
                    if random.randint(1, 101) == 1:
                        Q = qr_retraction(Q).to(dtype=p_dtype, device=p_dev)

                    # Momentum buffer V with shape matching g.t(): [k, m]
                    state = self.state[p]
                    V = state.get("momentum_buffer", None)
                    if V is None:
                        V = torch.zeros_like(g.t(), dtype=p_dtype, device=p_dev)  # [k, m]
                        state["momentum_buffer"] = V

                    # Update momentum on the manifold branch
                    V.mul_(momentum).add_(-g.t())  # V = m*V - g^T

                    # All matmuls must align in dtype/device
                    # MX: [k, m] @ [m, k] -> [k, k]
                    MX   = torch.mm(V, Q)
                    XMX  = torch.mm(Q, MX)           # [m, k]
                    XXMX = torch.mm(Q.t(), XMX)       # [k, k]
                    W_hat = MX - 0.5 * XXMX
                    W = W_hat - W_hat.t()

                    # Step size
                    # matrix_norm_one returns scalar tensor
                    denom = matrix_norm_one(W).clamp_min(_EPS)
                    t = 1.0 / denom.item()          # scalar float
                    alpha = min(1.0 * t, float(lr)) # your original had 0.5*2

                    # Cayley update (expects shapes consistent with your functions)
                    p_new = Cayley_loop(Q.t(), W, V, alpha)  # should return [k, m] -> reshape
                    V_new = torch.mm(W, Q.t())               # [k, m]

                    p.copy_(p_new.view_as(p))
                    V.copy_(V_new)

                else:
                    # Euclidean (plain SGD) branch
                    d_p = g.clone()

                    if weight_decay != 0.0:
                        d_p.add_(weight_decay, P)

                    state = self.state[p]
                    buf = state.get("momentum_buffer", None)
                    if momentum != 0.0:
                        if buf is None:
                            buf = d_p.clone()
                            state["momentum_buffer"] = buf
                        else:
                            buf.mul_(momentum).add_(1.0 - dampening, d_p)
                        d_p = d_p.add(momentum, buf) if nesterov else buf

                    p.add_(-lr, d_p.view_as(P))

        return loss


# ----------------------------------------------------------------------------- rotation_utils.py
def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

def get_orthogonal_matrix(size, mode, seed=42, device=DEVICE):
    set_seed(seed)

    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')
def apply_exact_had_to_UV(UV, had_dim=-1, output=False):
    # note: should always pass UV as a single matrix (out, in)
    device = UV.device
    dtype = UV.dtype
    UV = UV.float().to(DEVICE)
    if had_dim == -1:
        if output:
            out_features = UV.shape[-2]
            had_K, K = get_hadK(out_features)
            UV = matmul_hadU_cuda(UV.t(), had_K, K).t()
        else:
            in_features = UV.shape[-1]
            had_K, K = get_hadK(in_features)
            UV = matmul_hadU_cuda(UV, had_K, K)
    return UV.to(device=device, dtype=dtype)

# ----------------------------------------------------------------------------- local_ft_grad_utils.py (helpers)
def get_layer_importance(grad, args):
    if args.smooth_grad and not args.smooth_final:
        grad = smooth_gradient_info(grad, args)
    if args.taylor_order == 1:
        return -1.0 * grad
    elif args.taylor_order == 2:
        return grad.pow(2)
    elif args.taylor_order == 12:
        return -2.0 * grad + grad.pow(2)
def create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, linear, keys='q'):
    device = DEVICE
    if hasattr(linear, "scaling_diag_matrix"):
        scaling_diag_matrix *= linear.scaling_diag_matrix.to(device)**args.act_alpha
        scaling_diag_matrix += 1e-6  # avoid zero division
        scaling_matrix_inv = None
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            W = W * scaling_diag_matrix.view(1, 1, -1)
        else:
            W = W * scaling_diag_matrix.view(1, -1)
    elif hasattr(linear, "scaling_diag_matrixS"):
        scaling_diag_matrix = linear.scaling_diag_matrixS.to(device)
        try:
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
        except RuntimeError as e:
            logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
            eps = 1e-6
            scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
        W = W @ scaling_diag_matrix.float()
    else:
        scaling_matrix_inv = None
        raise ValueError("No scaling_diag_matrix found")
    return W, scaling_diag_matrix, scaling_matrix_inv

def create_svd_decomposition(W, args, layer_indices_dict, linear, idx, num_heads=None, keys='q_proj'):
    U, S, Vt = torch.linalg.svd(W.to(torch.float32), full_matrices=False)
    if args.is_per_head_svd and keys in ['q_proj', 'k_proj', 'v_proj']:
        V = Vt.transpose(1, 2) # n, C_in, c
        n_params = linear.weight.numel()
        compressed_params = int(n_params/num_heads * args.rank_ratio)
        LocalFT_rank = compressed_params // (linear.in_features + linear.out_features/num_heads) # update rank calculation
        LocalFT_rank = int(np.ceil(LocalFT_rank/1.0)*1.0)
        if layer_indices_dict is not None:
            U_ = []
            S_ = []
            V_ = []
            LocalFT_rank = layer_indices_dict[idx][keys]
            for i in range(num_heads):
                try:
                    rank_ = LocalFT_rank[i]
                except:
                    rank_ = [1]
                U_.append(U[i, :, rank_])
                S_.append(S[i, rank_])
                V_.append(V[i, :, rank_])
            U = U_
            S = S_
            V = V_
            del U_, S_, V_
        else:
            # U: n, c, c(r)
            # S: n, c(r)
            # V:  n, C_in, c(r); 
            # print(f"U shape: {U.shape}, S shape: {S.shape}, V shape: {V.shape}")
            U = U[:, :, :LocalFT_rank]
            S = S[:, :LocalFT_rank]
            V = V[:, :, :LocalFT_rank]

    else:
        V = Vt.T
        # truncate rank:
        n_params = linear.weight.numel()
        compressed_params = int(n_params * args.rank_ratio)
        LocalFT_rank = compressed_params // (linear.in_features + linear.out_features) # update rank calculation
        if layer_indices_dict is not None:
            LocalFT_rank = layer_indices_dict[idx][keys]

            U = U[:, LocalFT_rank]
            S = S[LocalFT_rank]
            V = V[:, LocalFT_rank]
        else:
            U = U[:, :LocalFT_rank]
            S = S[:LocalFT_rank]
            V = V[:, :LocalFT_rank]

    return U, S, V

def fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='q', layer_indices_dict=None, num_heads=None):
    if scaling_matrix_inv is not None:
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            if layer_indices_dict is not None:
                V = [scaling_matrix_inv.T @ vv for vv in V]
            else:
                V = scaling_matrix_inv.T.float().unsqueeze(0).expand(num_heads, -1, -1) @ V # V: n, C_in, rank, scaling_matrix_inv: C_in, C_in
        else:
            V = scaling_matrix_inv.T @ V
    else:
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            if layer_indices_dict is not None:
                V = [vv / scaling_diag_matrix.view(-1, 1) for vv in V]
            else:
                V = V/ scaling_diag_matrix.view(1, -1, 1)  # V: n, C_in, rank, scaling_diag_matrix: C_in, 1
        else:
            V = V / scaling_diag_matrix.view(-1, 1)
    return V

def create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys='q', num_heads=None):
    device = DEVICE
    if S is None:
        Alinear = nn.Linear(U.size(1), U.size(0), bias=False, device=device)
        Alinear.weight.requires_grad = True
        Blinear = None
        with torch.no_grad():
            Alinear.weight.data.copy_(U.to(device).to(torch.bfloat16))
        if args.is_quant_aware_ft:
            if keys not in quantizers[idx]:
                quantizers[idx][keys] = {}
            if 'Alinear' not in quantizers[idx][keys]:
                quantizers[idx][keys]['Alinear'] = WeightQuantizer()
                quantizers[idx][keys]['Alinear'].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                quantizers[idx][keys]['Alinear'].find_params(Alinear.weight.data)
        Alinear.requires_grad = True
        return Alinear, Blinear, None

    if args.is_per_head_svd and keys in ['q', 'k', 'v']:
        Alinear = nn.ParameterList()
        Blinear = nn.ParameterList()
        R = nn.ParameterList()
        if args.is_quant_aware_ft:
            if keys not in quantizers[idx]:
                quantizers[idx][keys] = {}
            if 'Alinear' not in quantizers[idx][keys]:
                quantizers[idx][keys]['Alinear'] = []
                quantizers[idx][keys]['Blinear'] = []

        for i in range(num_heads):
            if args.rotate:
                rank_ = S[i].shape[-1]
                had_K = get_orthogonal_matrix(rank_, 'random', args.seed) # do not think much now
                R.append(nn.Parameter(had_K.to(torch.bfloat16).to(device), requires_grad=True))
            Alinear.append(nn.Parameter(U[i].mul(S[i].sqrt()).contiguous().to(torch.bfloat16).to(device), requires_grad=True))
            Blinear.append(nn.Parameter(V[i].t().mul(S[i].sqrt().view(-1, 1)).contiguous().to(torch.bfloat16).to(device), requires_grad=True))

            if args.is_quant_aware_ft:
                if len(quantizers[idx][keys]['Alinear']) < num_heads:
                    quantizers[idx][keys]['Alinear'].append(WeightQuantizer())
                    quantizers[idx][keys]['Blinear'].append(WeightQuantizer())
                    quantizers[idx][keys]['Alinear'][-1].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                    quantizers[idx][keys]['Blinear'][-1].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                    # if idx == 0:
                    #     quantizers[idx][keys]['act_lat'].append(quant_utils.ActQuantizer())
                    #     quantizers[idx][keys]['act_lat'][-1].configure(args.a_bits, groupsize=-1, sym=not(args.a_asym), clip_ratio=args.a_clip_ratio)

    else:
        Alinear = nn.Linear(U.size(1), U.size(0), bias=False, device=device)
        Blinear = nn.Linear(V.size(0), V.size(1), bias=False, device=device) # V: C_in, Rank
        R = None
        # Set requires_grad before assigning data
        Alinear.weight.requires_grad = True
        Blinear.weight.requires_grad = True

        # Use proper assignment that preserves gradients
        with torch.no_grad():
            if args.rotate:
                had_K = get_orthogonal_matrix(S.shape[-1], 'random', args.seed)
                R = nn.Parameter(had_K.to(torch.bfloat16).to(device), requires_grad=True)
            Alinear.weight.data.copy_(U.mul(S.sqrt()).contiguous().to(device).to(torch.bfloat16))
            Blinear.weight.data.copy_(V.t().mul(S.sqrt().view(-1, 1)).contiguous().to(device).to(torch.bfloat16))
        if args.is_quant_aware_ft:
            if keys not in quantizers[idx]:
                quantizers[idx][keys] = {}
            if 'Alinear' not in quantizers[idx][keys]:
                quantizers[idx][keys]['Alinear'] = WeightQuantizer()
                quantizers[idx][keys]['Blinear'] = WeightQuantizer()
                quantizers[idx][keys]['Alinear'].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                quantizers[idx][keys]['Blinear'].configure(args.w_bits, perchannel=True, sym=not(args.w_asym), mse=False)
                # quantizers[idx][keys]['Alinear'].find_params(Alinear.weight.data)
                # quantizers[idx][keys]['Blinear'].find_params(Blinear.weight.data)
        Alinear.requires_grad = True
        Blinear.requires_grad = True

    return Alinear, Blinear, R


def update_qat_params(Alinear, Blinear, args, quantizers, idx, keys='q', num_heads=None):
    if Blinear is None:
        # qat linear is always fting
        pass
    if args.is_per_head_svd and keys in ['q', 'k', 'v']:
        for i in range(num_heads):
            quantizers[idx][keys]['Alinear'][i].find_params(Alinear[i].data)
            quantizers[idx][keys]['Blinear'][i].find_params(Blinear[i].data)
    else:
        quantizers[idx][keys]['Alinear'].find_params(Alinear.weight.data)
        quantizers[idx][keys]['Blinear'].find_params(Blinear.weight.data)

def create_optimizer(params_, params_R, args, optimizers, schedulers, key='q'):
    key_R = key + '_R'
    if args.qat_optim_R and params_R is not None:
        optimizers[key_R] = SGDG(params_R, lr=args.qat_lr_R, stiefel=True)

    if args.svd_ft_optim == 'adam':
        optimizers[key] = torch.optim.Adam(params_, lr=args.localft_lr)
    elif args.svd_ft_optim == 'sgd':
        optimizers[key] = torch.optim.SGD(params_, lr=args.localft_lr)
    elif args.svd_ft_optim == 'adamw':
        optimizers[key] = torch.optim.AdamW(params_, lr=args.localft_lr)
    elif args.svd_ft_optim == 'sgdm':
        optimizers[key] = torch.optim.SGD(params_, lr=args.localft_lr, momentum=0.9)
    else:
        raise ValueError(f"Invalid svd_ft_optim: {args.svd_ft_optim}")

    # Add learning rate scheduler for q projection
    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
        if args.scheduler_type == 'step':
            schedulers[key] = torch.optim.lr_scheduler.StepLR(
                optimizers[key], 
                step_size=getattr(args, 'scheduler_step_size', 30), 
                gamma=getattr(args, 'scheduler_gamma', 0.8)
            )
            if args.qat_optim_R and params_R is not None:
                schedulers[key_R] = torch.optim.lr_scheduler.StepLR(
                    optimizers[key_R], 
                    step_size=getattr(args, 'scheduler_step_size', 30), 
                    gamma=getattr(args, 'scheduler_gamma', 0.8)
                )
        elif args.scheduler_type == 'cosine':
            schedulers[key] = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[key], 
                T_max=args.localft_iters * (args.nsamples // args.bs),
                eta_min=getattr(args, 'scheduler_eta_min', 1e-6)
            )
            if args.qat_optim_R and params_R is not None:
                schedulers[key_R] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[key_R], 
                    T_max=args.localft_iters * (args.nsamples // args.bs),
                    eta_min=getattr(args, 'scheduler_eta_min', 1e-6)
                )
        else:
            raise ValueError(f"Invalid scheduler_type: {args.scheduler_type}. Supported types: 'step', 'cosine'")

    return optimizers, schedulers

def iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys='q', reg_UV=None, R=None, qat_param_update=False):
    norm_loss = True
    reg = None
    # here within enable grad can abstract a function class
    if Blinear is None:
        if args.svd_ft_mode == 'output':
            _in = __in[bsz:bsz+args.bs]
            if args.weighted_svd:
                _out = (_in @ W.T) * weight
                y = (_in @ Alinear.weight) * weight
                # y = Alinear(_in) * weight
            else:
                _out = _in @ W.T
                y = _in @ Alinear.weight
        elif args.svd_ft_mode == 'weight':
            def regularization_W(Alinear, reg_UV, weight=None):
                if weight is not None:
                    loss = (weight * (Alinear - reg_UV[-1])).pow(2).mean()
                else:
                    loss = (Alinear - reg_UV[-1]).pow(2).mean()
                return loss
            if args.weighted_svd:
                _out = weight * W.detach()
                y = weight * Alinear.weight
                y = weight * (quantizers[idx][keys]['Alinear'].quantize(Alinear.weight))
                if reg_UV is not None:
                    reg = regularization_W(Alinear, reg_UV, None)
            else:
                _out = W.detach()
                y = quantizers[idx][keys]['Alinear'].quantize(Alinear.weight)
                if reg_UV is not None:
                    reg = regularization_W(Alinear.weight, reg_UV)
        else:
            raise ValueError(f"Invalid svd_ft_mode: {args.svd_ft_mode}")
    else:
        if args.svd_ft_mode == 'output':
            _in = __in[bsz:bsz+args.bs]
            if args.weighted_svd:
                _out = (_in @ W.T) * weight
                y = (_in @ (Alinear.weight @ Blinear.weight)).T * weight
            else:
                _out = _in @ W.T
                y = _in @ (Alinear.weight @ Blinear.weight).T
        elif args.svd_ft_mode == 'weight':
            if args.is_per_head_svd and keys in ['q', 'k', 'v']:
                def regularization_UV(Alinear, Blinear, reg_UV, weight):
                    loss = None
                    for A_i, B_i, A0_i, B0_i in zip(Alinear, Blinear, reg_UV[0], reg_UV[1]):
                        if loss is None:
                            loss = (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
                        else:
                            loss += (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
                    # else:
                    #     for A_i, B_i, A0_i, B0_i, w_i in zip(Alinear, Blinear, reg_UV[0], reg_UV[1], weight):
                    #         loss = (w_i *(A_i - A0_i)).pow(2).sum() + (w_i *(B_i - B0_i)).pow(2).sum()
                    return loss
                def per_head_low_rank_forward_W(Alinear, Blinear, Qs=None, R=None):
                    y_ = []
                    if args.is_quant_aware_ft:
                        if R is not None:
                            for A_i, B_i, Q_a_i, Q_b_i, R_i in zip(Alinear, Blinear, Qs[keys]['Alinear'], Qs[keys]['Blinear'], R):
                                A_r = A_i @ R_i
                                B_r = R_i.T @ B_i
                                if qat_param_update:
                                    Q_a_i.find_params(A_r.detach())
                                    Q_b_i.find_params(B_r.detach())
                                y_.append((Q_a_i.quantize(A_r) @ Q_b_i.quantize(B_r)).unsqueeze(0))
                        else:
                            for A_i, B_i, Q_a_i, Q_b_i in zip(Alinear, Blinear, Qs[keys]['Alinear'], Qs[keys]['Blinear']):
                                if qat_param_update:
                                    Q_a_i.find_params(A_i.detach())
                                    Q_b_i.find_params(B_i.detach())
                                y_.append((Q_a_i.quantize(A_i) @ Q_b_i.quantize(B_i)).unsqueeze(0))
                            # print("qat quantize")
                    else:
                        for A_i, B_i in zip(Alinear, Blinear):
                            y_.append((A_i @ B_i).unsqueeze(0))
                            # print("no qat quantize")
                    # print(y_[-1].shape)
                    return torch.cat(y_, dim=0)
                if args.weighted_svd:
                    _out = weight * W
                    if args.is_quant_aware_ft:
                        y = weight * per_head_low_rank_forward_W(Alinear, Blinear, Qs=quantizers[idx], R=R)
                    else:
                        y = weight * per_head_low_rank_forward_W(Alinear, Blinear)
                        # y = per_head_low_rank_forward_W(Alinear, Blinear)
                        # print(y.shape)
                        # y = weight * y
                    if reg_UV is not None and args.is_quant_aware_ft:
                        reg = regularization_UV(Alinear, Blinear, reg_UV, None)
                else:
                    _out = W
                    y = per_head_low_rank_forward_W(Alinear, Blinear)
                    if reg_UV is not None and args.is_quant_aware_ft:
                        reg = regularization_UV(Alinear, Blinear, reg_UV, None)
            else:
                if args.weighted_svd:
                    _out = weight * W
                    if args.is_quant_aware_ft:
                        y = weight * (quantizers[idx][keys]['Alinear'].quantize(Alinear.weight) @ quantizers[idx][keys]['Blinear'].quantize(Blinear.weight))
                    else:
                        y = weight * (Alinear.weight @ Blinear.weight)
                else:
                    _out = W
                    y = Alinear.weight @ Blinear.weight 
        else:
            raise ValueError(f"Invalid svd_ft_mode: {args.svd_ft_mode}")
    if reg_UV is not None and args.is_quant_aware_ft:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean() 
            # print(loss.item()/reg.item())
            if args.qat_uv_reg_scale:
                scale = (loss.item()/(reg.item()+1e-5)) 
            # print(loss.item(), reg.item())
            # print(loss.item()/(reg.item()+1e-5))
            loss += args.qat_uv_reg_alpha * reg
        else:
            loss = (y - _out).pow(2).mean() + args.qat_uv_reg_alpha * reg
    else:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean()
        else:
            loss = (y - _out).pow(2).mean()
    loss.backward()
    return loss


def iter_local_qat_loss(Alinear, Blinear, W, Q, bsz, weight, __in, args, quantizers, idx,  keys='q', reg_UV=None, R=None, qat_param_update=False):
    norm_loss = True
    reg = None
    # here within enable grad can abstract a function class

    if args.svd_ft_mode == 'output':
        raise ValueError(f"Invalid QAT mode for now: {args.svd_ft_mode}")
        # _in = __in[bsz:bsz+args.bs]
        # if args.weighted_svd:
        #     _out = (_in @ W.T) * weight
        #     y = (_in @ (Alinear.weight @ Blinear.weight)).T * weight
        # else:
        #     _out = _in @ W.T
        #     y = _in @ (Alinear.weight @ Blinear.weight).T

    elif args.svd_ft_mode == 'weight':
        if args.is_per_head_svd and keys in ['q', 'k', 'v']:
            W_had, Alinear_had, Blinear_had = get_had_W_and_ABlinear(W, Alinear, Blinear, Q, key=keys) # online hadamard rotation

            _out = W_had
            y = per_head_low_rank_forward_W(Alinear_had, Blinear_had, keys=keys, Qs=quantizers[idx], R=R, qat_param_update=qat_param_update) # further rotate in R, and quantize

            if reg_UV is not None and args.is_quant_aware_ft:
                reg = regularization_UV(Alinear_had, Blinear_had, reg_UV, None)

            if args.weighted_svd:
                _out = weight * _out.detach()
                y = weight * y
            else:
                _out = _out.detach()
                y = y

            # _out = _out.detach()
            # y = y
    else:
        raise ValueError(f"Invalid QAT mode: {args.svd_ft_mode}")

    if reg_UV is not None and args.is_quant_aware_ft:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean() 
            if args.qat_uv_reg_scale:
                scale = (loss.item()/(reg.item()+1e-5)) 
            loss += args.qat_uv_reg_alpha * reg
        else:
            loss = (y - _out).pow(2).mean() + args.qat_uv_reg_alpha * reg
    else:
        if norm_loss:
            loss = (y - _out).pow(2).mean() / _out.pow(2).mean()
        else:
            loss = (y - _out).pow(2).mean()

    loss.backward()
    return loss

def regularization_UV(Alinear, Blinear, reg_UV, weight):
    loss = None
    for A_i, B_i, A0_i, B0_i in zip(Alinear, Blinear, reg_UV[0], reg_UV[1]):
        if loss is None:
            loss = (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
        else:
            loss += (A_i - A0_i).pow(2).sum() + (B_i - B0_i).pow(2).sum()
    return loss

def per_head_low_rank_forward_W(Alinear, Blinear, keys='q', Qs=None, R=None, qat_param_update=True):
    y_ = []
    for A_i, B_i, Q_a_i, Q_b_i, R_i in zip(Alinear, Blinear, Qs[keys]['Alinear'], Qs[keys]['Blinear'], R):
        A_r = A_i @ R_i
        B_r = R_i.T @ B_i
        if qat_param_update:
            Q_a_i.find_params(A_r.detach())
            Q_b_i.find_params(B_r.detach())
        y_.append((Q_a_i.quantize(A_r) @ Q_b_i.quantize(B_r)).unsqueeze(0))
    return torch.cat(y_, dim=0)

def fuse_per_head_svd_rotated_weights(Alinear, Blinear, quantizers, idx, keys='q', num_heads=None, R=None):
    # Only rotate in-place, no quantization
    if R is None or num_heads is None:
        return
    import torch
    with torch.no_grad():
        for i in range(num_heads):
            # Promote to float32 for matmul stability, then cast back
            dev_a, dt_a = Alinear[i].device, Alinear[i].dtype
            dev_b, dt_b = Blinear[i].device, Blinear[i].dtype
            A_new = (Alinear[i].detach().to(torch.float32) @ R[i].to(torch.float32)).to(device=dev_a, dtype=dt_a)
            B_new = (R[i].T.to(torch.float32) @ Blinear[i].detach().to(torch.float32)).to(device=dev_b, dtype=dt_b)
            Alinear[i].copy_(A_new)
            Blinear[i].copy_(B_new)

def get_had_W_and_ABlinear(W, Alinear, Blinear, Q, key=None):  
    if key == 'v': # Only V proj need to apply hadamard
        a_dtype = Alinear[0].dtype
        device = Alinear[0].device
        Alinear_had = [apply_exact_had_to_UV(aa.to(device=DEVICE, dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=a_dtype) for aa in Alinear]

        W_had = [apply_exact_had_to_UV(ww.to(device=DEVICE, dtype=torch.float64), had_dim=-1, output=True).to(device=device, dtype=W.dtype) for ww in W]
        W_had = torch.stack(W_had, dim=0)
    else:
        Alinear_had = Alinear
        W_had = W

    b_dtype = Blinear[0].dtype
    device = Blinear[0].device
    Blinear_had = [torch.matmul(bb.to(device=DEVICE, dtype=torch.float64), Q).to(device=device, dtype=b_dtype) for bb in Blinear]
    W_had = [torch.matmul(ww.to(device=DEVICE, dtype=torch.float64), Q).to(device=device, dtype=W.dtype) for ww in W_had]
    W_had = torch.stack(W_had, dim=0)

    return W_had, Alinear_had, Blinear_had
def set_grad_for_svd_modules(model, args):
    model.train()
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['all', 'qkv', 'attn']:
                param.requires_grad = True
            elif 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o']:
                param.requires_grad = True
            elif ('up_proj' in name or 'gate_proj' in name) and args.svd_modules in ['all', 'mlp', 'gaup']:
                param.requires_grad = True
            elif 'down_proj' in name and args.svd_modules in ['all', 'mlp', 'down']:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

def disable_grad_for_svd_modules(model, args):
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False

# ----------------------------------------------------------------------------- svd_utils.py
class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", had_K=None, K=-1, had_mode='hadamard', is_ma_hack=False, original_linear=None, topk=1, is_per_head_svd=False, rank_split=None) -> None:
        super().__init__()
        self.rank_split = rank_split
        if is_per_head_svd:
            if rank_split is not None:
                self.head_num = len(V)
                self.ALinear = nn.ParameterList()
                if sigma_fuse == "UV":
                    V_ = []
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].mul(S[i].sqrt()).contiguous()))
                        V_.append(V[i].t().mul(S[i].sqrt().view(-1, 1)).contiguous())
                    V_ = torch.cat(V_, dim=0) # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == "U":
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].mul(S[i]).contiguous()))
                    V_ = torch.cat(V, dim=-1).t() # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == "V":
                    V_ = []
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].contiguous()))
                        V_.append(V[i].t().mul(S[i].sqrt().view(-1, 1)).contiguous())
                    V_ = torch.cat(V_, dim=0) # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == 'local_ft':
                    V_ = []
                    for i in range(self.head_num):
                        self.ALinear.append(nn.Parameter(U[i].contiguous()))
                        V_.append(V[i])
                    V = torch.cat(V_, dim=0)
                    del V_
                    S = None
                    del S
                    self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
                    self.BLinear.weight.data = V
                else:
                    raise ValueError(f"Unsupported sigma_fuse in per head SVD: {sigma_fuse}")
            else:
                raise NotImplementedError("per-head SVDLinear without rank_split is not vendored")
        else:
            raise NotImplementedError("SVDLinear without per-head SVD is not vendored")

    @staticmethod
    def from_linear_localft(
        linear: nn.Linear,
        param_ratio: float,
        sigma_fuse="local_ft",
        had_rank=False,
        had_mode='random',
        seed=0,
        module_name='o_proj',
        is_per_head_svd=False,
        is_rank_allocation=False,
    ):
        device = DEVICE
        svd_info = linear.svd_info

        # Get SVD information
        U = svd_info['U'].to(device)
        S = svd_info['S']
        V = svd_info['V'].to(device)
        if is_per_head_svd:
            rank = U[0].shape[1]
            num_heads = len(U)
            if had_rank:
                logging.info("Warning: head wise svd from linear is not supporting had_rank yet")
                pass
        else:
            rank = U.shape[1]
            if had_rank:
                set_seed(seed)
                if had_mode == 'hadamard':
                    K = 1
                    had_K = None
                elif had_mode == 'rh':
                    had_K = get_orthogonal_matrix(rank, 'hadamard')
                    K = rank
                elif had_mode == 'random':
                    had_K = get_orthogonal_matrix(rank, 'random')
                    K = rank

        # Update total rank sum and linear layer count
        if not is_rank_allocation:
            total_rank_sum[module_name] += rank
            logging.info(f"Layer {module_name} uses rank: {rank}")

        total_linear_count[module_name] += 1
        if is_per_head_svd:
            Us = []
            if is_rank_allocation:
                rank_split = []
            else:
                rank_split = None
            for i in range(num_heads):
                rank_ = U[i].shape[-1]
                if is_rank_allocation:
                    rank_split.append(rank_)
                    total_rank_sum[module_name] += rank_
                    logging.info(f"Layer uses rank: {rank_} on head {i}")
                Us.append(U[i])
            Us = [Us]
            Vs = [V]
            if not is_rank_allocation:
                Us = [torch.stack(list(us), dim=0) for us in Us] # # [n * [c, rank_] -> [n, c, rank_]
                Vs = [torch.stack(list(vs), dim=0) for vs in Vs] # [[n * [C_in, rank_]] -> [n, rank_, C_in]]
                # do not support UV of RA version, which have different rank_i
        else:
            Us = [U]
            Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        if had_rank:
            if is_per_head_svd:
                logging.info("Warning: head wise svd from linear is not supporting had_rank yet")
                pass
            else:
                new_linear = SVDLinear(Us[0], None, Vs[0], bias, sigma_fuse, had_K, K, had_mode)
        else:
            if is_per_head_svd:
                new_linear = SVDLinear(Us[0], S, Vs[0], bias, sigma_fuse, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
            else:
                new_linear = SVDLinear(Us[0], S, Vs[0], bias, sigma_fuse)

        new_linear.to(linear.weight.dtype)
        return new_linear.cpu()


# ----------------------------------------------------------------------------- grad_info_utils.py (helpers)
def get_top_k_scores(grad_scores_dict, k_list, keys=['k_proj'], is_per_head_svd=False):
    """
    Get indices and scores of top k important singular values for each key across all layers.

    Args:
        grad_scores_dict: dict of {layer_idx: {key: tensor of scores}}
        k_list: list of integers, top-k values for each key
        keys: list of keys to consider (e.g., ['k_proj', 'o_proj'])

    Returns:
        top_indices: dict of {key: list of (layer_idx, sv_idx)}
        top_scores: dict of {key: list of scores}
        layer_indices_dict: dict of {layer_idx: {key: [sv_idx, ...]}}
    """
    top_indices = {}
    top_scores = {}
    layer_indices_dict = {}
    assert len(keys) == len(k_list), "keys and k_list must have the same length" # [FIXED: num * [v] -> [num * v]]
    for key, k in zip(keys, k_list):
        all_scores = []
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            logging.info("added per head score here")
            for layer_idx, scores in grad_scores_dict.items():
                if key not in scores:
                    print(f"Layer {layer_idx} has no {key} grad scores")
                    continue
                layer_num = int(layer_idx.split('_')[1])
                score = scores[key]
                # score_ = score # n, c
                # logging.info(f'score shape {score.shape}') # [384?]
                head_num = score.shape[0]
                for head_idx in range(score.shape[0]):
                    for i, score_ in enumerate(score[head_idx]):
                        all_scores.append((layer_num, i, head_idx, score_.item()))
        else:
            for layer_idx, scores in grad_scores_dict.items():
                if key not in scores:
                    print(f"Layer {layer_idx} has no {key} grad scores")
                    continue
                layer_num = int(layer_idx.split('_')[1])
                for i, score in enumerate(scores[key]):
                    all_scores.append((layer_num, i, score.item()))

        # Sort and get top-k
        all_scores.sort(key=lambda x: x[-1], reverse=True)
        # [TODO: only support heterogenous per head svd rank allocation for now]
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            logging.info(f'{key} original for per head svd is {k}')
            k = k * head_num # [FIXME: check head_num use kv_headnum to accomodate GQA]
            logging.info(f'scale {key} by head_num: {head_num} to k: {k}')
        top_k = all_scores[:k]

        # Store top indices and scores
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            top_indices[key] = [(layer, idx, head_idx) for layer, idx, head_idx, _ in top_k] # [TODO:]here need some quick fix to support perhead and other global svd rank allocation
            top_scores[key] = [score for _, _, _, score in top_k]
            # Fill per-layer per-head dictionary
            for layer_idx, sv_idx, head_idx in top_indices[key]:
                if layer_idx not in layer_indices_dict:
                    layer_indices_dict[layer_idx] = {}
                if key not in layer_indices_dict[layer_idx]:
                    layer_indices_dict[layer_idx][key] = {}
                if head_idx not in layer_indices_dict[layer_idx][key]:
                    layer_indices_dict[layer_idx][key][head_idx] = []
                layer_indices_dict[layer_idx][key][head_idx].append(sv_idx)
        else:
            top_indices[key] = [(layer, idx) for layer, idx, _ in top_k] 
            top_scores[key] = [score for _, _, score in top_k]

            # Fill per-layer dictionary
            for layer_idx, sv_idx in top_indices[key]:
                if layer_idx not in layer_indices_dict:
                    layer_indices_dict[layer_idx] = {}
                if key not in layer_indices_dict[layer_idx]:
                    layer_indices_dict[layer_idx][key] = []
                layer_indices_dict[layer_idx][key].append(sv_idx)

    return top_indices, top_scores, layer_indices_dict

def taylor_like_s_grad_squared(S, S_grad, n, add_taylor_first=False):
    """
    Compute generalized S_grad_squared up to order n using:
    sum_{k=0}^{n} [2 * (-1)^k / (k+2)!] * S^k * S_grad^{k+2}
    """
    n = n - 2
    result = torch.zeros_like(S)
    if add_taylor_first:
        result = - 2 * S_grad / S
    for k in range(n + 1):
        coeff = 2 * ((-1) ** k) / math.factorial(k + 2)
        term = coeff * S.pow(k) * S_grad.pow(k + 2)
        result = result + term
    return result

def get_layer_importance_gradinfo(grad, args):
    if args.taylor_order == 1:
        return -1.0 * grad
    elif args.taylor_order == 2:
        return grad.pow(2)
    elif args.taylor_order == 12:
        return -2.0 * grad + grad.pow(2)

def prepare_fuse_svd(layers, args):
    """Per-head SVD of the whitened q/k/v weights (separate Q, K, V branch of grad_info_utils.prepare_fuse_svd)."""
    device = DEVICE
    alpha = args.act_alpha
    for idx, layer in enumerate(layers):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            q_linear = layer.self_attn.q_proj
            k_linear = layer.self_attn.k_proj
            v_linear = layer.self_attn.v_proj
            if args.qkv_fuse or args.kv_fuse:
                raise NotImplementedError("fused q/k/v SVD is not vendored")
            for proj_name, proj_layer in [('q_proj', q_linear), ('k_proj', k_linear), ('v_proj', v_linear)]:
                if args.is_per_head_svd:
                    if proj_name == 'q_proj':
                        num_heads = layer.self_attn.config.num_attention_heads
                    else:
                        num_heads = layer.self_attn.config.num_key_value_heads
                    w = proj_layer.weight.data.view(num_heads, -1, proj_layer.in_features).float().to(DEVICE)
                else:
                    w = proj_layer.weight.data.float().to(DEVICE)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(proj_layer.in_features, device=device)
                    if hasattr(k_linear, "scaling_diag_matrix") and k_linear.scaling_diag_matrix is not None:
                        scaling_diag_matrix *= k_linear.scaling_diag_matrix.to(device)**alpha
                        scaling_diag_matrix += 1e-6
                        if args.is_per_head_svd:
                            w = w * scaling_diag_matrix.view(1, 1, -1)
                        else:
                            w = w * scaling_diag_matrix.view(1, -1)
                    elif hasattr(k_linear, "scaling_diag_matrixS") and k_linear.scaling_diag_matrixS is not None:
                        scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(device)
                        try:
                            scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                        except Exception as e:
                            logging.info("Warning: scaling_diag_matrix is not full rank!")
                            scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(DEVICE)
                            scaling_matrix_inv = None
                        w = w @ scaling_diag_matrix.float()

                U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)
                if args.is_per_head_svd:
                    V = Vt.transpose(1, 2)
                else:
                    V = Vt.T

                proj_layer.svd_info = {
                    'U': U.cpu(),
                    'S': S.cpu(),
                    'V': V.cpu()
                }
                logging.debug(f"Layer {idx} {proj_name} SVD completed, S shape: {S.shape}")


@torch.no_grad()
def calib_input_cholesky_text(model, layers, windows, args):
    """act_aware_utils.calib_input_distributionlowresources for text windows: hooks on every q/k/v Linear, one forward per window."""
    method = args.calib_method
    model.eval()
    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        elif "cholesky" in method:
            inp = input[0].detach().float()
            if inp.dim() == 2:   # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.raw_scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max
    for i in range(len(layers)):
        layer = layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'o']:
                    continue # only skip when svd_modules is not all
                module.raw_scaling_diag_matrix = 0
                module.register_forward_hook(hook)
    for window in windows:
        model(window.to(DEVICE))
    for i in range(len(layers)):
        layer = layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'o']:
                    continue # only skip when svd_modules is not all
                module._forward_hooks.clear()
                module.raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.cpu()
    logging.info("Start Cholesky Decomposition...")
    for i in range(len(layers)):
        # layer_profile = {}
        for name, module in layers[i].named_modules():
            if isinstance(module, nn.Linear):
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'o']:
                    continue # only skip when svd_modules is not all
                raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(DEVICE)
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                except Exception as e:
                    print("Warning: eigen scaling_diag_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(DEVICE)
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                    eigenvalues = None
                    del eigenvalues
                if 'k_proj' in name: # only valid for k_proj in original svd_modules ==qkv 
                    # layers[i].self_attn.q_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    # layers[i].self_attn.v_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'o_proj' in name:
                    layers[i].self_attn.o_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'up_proj' in name:
                    layers[i].mlp.up_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    # layers[i].mlp.gate_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'down_proj' in name:
                    layers[i].mlp.down_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
                del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
                torch.cuda.empty_cache()


def calib_grad_info_text(model, layers, windows, args):
    """grad_info_utils.calib_grad_info for text windows: S_grad_info of every q/k/v projection (separate per-head branch)."""
    prepare_fuse_svd(layers, args)

    if not args.qkv_fuse and not args.kv_fuse:
        for idx, layer in enumerate(layers):
            if hasattr(layer.self_attn.k_proj, "scaling_diag_matrix"):
                if not hasattr(layer.self_attn.q_proj, "scaling_diag_matrix"):
                    layer.self_attn.q_proj.scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix
                if not hasattr(layer.self_attn.v_proj, "scaling_diag_matrix"):
                    layer.self_attn.v_proj.scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix
            elif hasattr(layer.self_attn.k_proj, "scaling_diag_matrixS"):
                if not hasattr(layer.self_attn.q_proj, "scaling_diag_matrixS"):
                    layer.self_attn.q_proj.scaling_diag_matrixS = layer.self_attn.k_proj.scaling_diag_matrixS
                if not hasattr(layer.self_attn.v_proj, "scaling_diag_matrixS"):
                    layer.self_attn.v_proj.scaling_diag_matrixS = layer.self_attn.k_proj.scaling_diag_matrixS

    model.eval()

    device = DEVICE
    model = model.to(device)

    accumulation_steps = 1   # Number of accumulated batches
    batch_count = 0          # Accumulated batch counter

    # Set model to training mode and only allow gradient computation for svd_modules layers
    model.train()
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['all', 'qkv', 'attn']:
                param.requires_grad = True
            elif 'o_proj' in name and args.svd_modules in ['attn', 'all', 'o']:
                param.requires_grad = True
            elif ('up_proj' in name or 'gate_proj' in name) and args.svd_modules in ['all', 'mlp', 'gaup']:
                param.requires_grad = True
            elif 'down_proj' in name and args.svd_modules in ['all', 'mlp', 'down']:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    for window in windows:
        input_ids = window.to(device)
        labels = input_ids.clone()
        if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
            labels = labels[:, 1:]
            input_ids = input_ids[:, :-1]
        with torch.enable_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs[0]
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()
        batch_count += 1  # Count each batch

        # Perform backpropagation when accumulated the set number of batches
        if batch_count % accumulation_steps == 0:
            # Update gradient information for S in each layer
            for idx, layer in enumerate(layers):
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    if not args.qkv_fuse and not args.kv_fuse:
                        # Separate Q, K, V handling
                        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                            proj = getattr(layer.self_attn, proj_name)
                            if hasattr(proj, 'svd_info') and proj.weight.grad is not None:
                                svd_info = proj.svd_info
                                if args.is_per_head_svd:
                                    if proj_name == 'q_proj':
                                        num_heads = layer.self_attn.config.num_attention_heads
                                    else:
                                        num_heads = layer.self_attn.config.num_key_value_heads
                                    grad_cat = proj.weight.grad.detach().view(num_heads, -1, proj.in_features).to(torch.bfloat16).to(device)
                                else:
                                    grad_cat = proj.weight.grad.detach().to(torch.bfloat16).to(device)


                                if args.act_aware:
                                    if hasattr(proj, "scaling_diag_matrix"):
                                        try:
                                            scaling_diag_matrix = proj.scaling_diag_matrix.to(device)
                                        except:
                                            proj.scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix
                                            scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrix.to(device)
                                        scaling_diag_matrix += 1e-6
                                        scaling_diag_matrix = scaling_diag_matrix**(-args.act_alpha)
                                        if args.is_per_head_svd:
                                            grad_cat = grad_cat * scaling_diag_matrix.view(1, 1, -1).to(torch.bfloat16)
                                        else:
                                            grad_cat = grad_cat * scaling_diag_matrix.view(1, -1).to(torch.bfloat16)
                                    elif hasattr(proj, "scaling_diag_matrixS"):
                                        try:
                                            scaling_diag_matrix = proj.scaling_diag_matrixS.to(device)
                                        except:
                                            proj.scaling_diag_matrixS = layer.self_attn.k_proj.scaling_diag_matrixS
                                            scaling_diag_matrix = layer.self_attn.k_proj.scaling_diag_matrixS.to(device)
                                        if hasattr(layer.self_attn.k_proj, "scaling_matrix_inverse_transpose"):
                                            scaling_matrix_inverse_transpose = layer.self_attn.k_proj.scaling_matrix_inverse_transpose
                                        else:
                                            scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                                            layer.self_attn.k_proj.scaling_matrix_inverse_transpose = scaling_matrix_inverse_transpose
                                        grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.bfloat16)

                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                # S_grad = torch.diag(U.T @ grad_cat @ V)
                                if args.is_per_head_svd:
                                    S_grad = torch.sum(U * (grad_cat @ V), dim=1) # n, r
                                else:
                                    S_grad = torch.sum(U * (grad_cat @ V), dim=0)

                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first)
                                else:
                                    S_grad_squared = get_layer_importance_gradinfo(S_grad,  args)

                                if not hasattr(proj, 'S_grad_info'):
                                    proj.S_grad_info = S_grad_squared
                                else:
                                    proj.S_grad_info += S_grad_squared 
            model.zero_grad()  # Clear gradients

    # Normalize S gradient information
    if batch_count > 0:
        for layer in layers:
            if hasattr(layer.self_attn.k_proj, 'S_grad_info'):
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    layer.self_attn.k_proj.S_grad_info = layer.self_attn.k_proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                    if not args.qkv_fuse and not args.kv_fuse:
                        # Separate Q, K, V handling
                        for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                            proj = getattr(layer.self_attn, proj_name)
                            if hasattr(proj, 'S_grad_info'):
                                proj.S_grad_info = proj.S_grad_info.div(batch_count//accumulation_steps).abs().sqrt()


def svd_qkv_with_grad_info_text(layers, args):
    """grad_info_utils.svd_qkv_with_grad_info (separate per-head branch, no cache): importance scores, budget, top-k.

    Returns (top_indices, top_scores, layer_indices_dict) with layer_indices_dict {layer: {proj: {head: [indices]}}}.
    """
    grad_alpha = args.grad_alpha
    grad_scores_dict = {}
    device = DEVICE  # Get CUDA device
    for idx, layer in enumerate(layers):
        layer_key = f"layer_{idx}"
        grad_scores_dict[layer_key] = {}
        if args.svd_modules in ['qkv', 'attn', 'all']:
            # Separate Q, K, V case (NEW)
            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                proj = getattr(layer.self_attn, proj_name)
                if hasattr(proj, 'svd_info') and hasattr(proj, 'S_grad_info'):
                    svd_info = proj.svd_info
                    S = svd_info['S'].to(device).to(torch.bfloat16)
                    S_grad = proj.S_grad_info.to(device).to(torch.bfloat16)

                    # Calculate importance score
                    importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                    grad_scores_dict[layer_key][proj_name] = importance_score.cpu()
                    grad_scores_dict[layer_key][f'{proj_name}_S'] = S.cpu()
                    if args.use_S_gradinfo_init:
                        proj.S_grad_info = importance_score.cpu()
                    logging.debug(f"Layer {idx} {proj_name} importance score computed, shape: {importance_score.shape}")
                    if args.is_rank_allocate_ft:
                        del proj.svd_info
                else:
                    logging.debug(f"Warning: Layer {idx} {proj_name} lacks necessary SVD information or gradient information, cannot compute importance score")
    if args.use_S_gradinfo_init:
        return # only log the S_grad_info as importance score for pruning
    # Get indices and scores of top k important singular values
    num_layers = len(layers)
    if args.svd_modules == 'qkv' and not args.qkv_fuse and not args.kv_fuse:
        if args.use_true_param_ratio:
            # Calculate parameter ratios for Q, K, V separately
            if args.is_per_head_svd:
                num_q_heads = layers[0].self_attn.config.num_attention_heads
                num_heads = layers[0].self_attn.config.num_key_value_heads
                q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features/num_q_heads)/num_q_heads
                k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features/num_heads)/num_heads
                v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features/num_heads)/num_heads
            else:
                q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
                v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)
            hidden_size = [q_param_ratio, k_param_ratio, v_param_ratio]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj', 'v_proj'], is_per_head_svd=args.is_per_head_svd)
        elif args.use_all_true_param_ratio:
            # Calculate parameter ratios for Q, K, V separately
            if args.is_per_head_svd:
                num_q_heads = layers[0].self_attn.config.num_attention_heads
                num_heads = layers[0].self_attn.config.num_key_value_heads
                q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)/num_q_heads
                k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)/num_heads
                v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)/num_heads
            else:
                q_param_ratio = layers[0].self_attn.q_proj.weight.numel() / (layers[0].self_attn.q_proj.in_features + layers[0].self_attn.q_proj.out_features)
                k_param_ratio = layers[0].self_attn.k_proj.weight.numel() / (layers[0].self_attn.k_proj.in_features + layers[0].self_attn.k_proj.out_features)
                v_param_ratio = layers[0].self_attn.v_proj.weight.numel() / (layers[0].self_attn.v_proj.in_features + layers[0].self_attn.v_proj.out_features)
            hidden_size = [q_param_ratio, k_param_ratio, v_param_ratio]
            total_rank = [num_layers * hsz for hsz in hidden_size]
            k_value = [int(args.rank_ratio * t_rank) for t_rank in total_rank]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj', 'v_proj'], is_per_head_svd=args.is_per_head_svd)
        else:
            hidden_size = layers[0].self_attn.k_proj.in_features
            total_rank = num_layers * hidden_size
            qkv_k_value = int(args.rank_ratio/3 * total_rank)
            k_value = [qkv_k_value, qkv_k_value, qkv_k_value]
            top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k_list=k_value, keys=['q_proj', 'k_proj', 'v_proj'])
    else:
        raise NotImplementedError("only the separate per-head q/k/v budget is vendored")

    for key, indices in top_indices.items():
        logging.info(f"Selected top {len(indices)} important singular values for {key}")
    return top_indices, top_scores, layer_indices_dict


def calib_localft_grad_info_text(model, layers, windows, args, layer_indices_dict=None):
    """local_ft_grad_utils.calib_localft_grad_info for text windows: per-head SVD of q/k/v before rotation and W_grad_info."""
    device = DEVICE
    for idx, layer in enumerate(layers):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            q_linear = layer.self_attn.q_proj
            k_linear = layer.self_attn.k_proj
            v_linear = layer.self_attn.v_proj

            if args.is_per_head_svd:
                num_heads = layer.self_attn.config.num_key_value_heads if not args.is_q_headnum else layer.self_attn.config.num_attention_heads
                W = q_linear.weight.data.view(num_heads, -1, q_linear.in_features).float().to(device)
            else:
                W = q_linear.weight.data.float().to(device)
            if not hasattr(q_linear, "svd_info"): 
                # Apply activation-aware scaling (if enabled)
                if args.act_aware: # q has the same input as k, so we can use the same scaling matrix
                    scaling_diag_matrix = torch.ones(q_linear.in_features, device=DEVICE)  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, k_linear, keys='q')

                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, q_linear, idx, keys='q_proj', num_heads=num_heads if args.is_per_head_svd else None)
                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, keys='q', layer_indices_dict=layer_indices_dict, num_heads=num_heads if args.is_per_head_svd else None)
                q_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
            if args.is_per_head_svd:
                num_heads = layer.self_attn.config.num_key_value_heads
                W = k_linear.weight.data.float().view(num_heads, -1, k_linear.in_features).to(device)
            else:
                W = k_linear.weight.data.float().to(device)
            if not hasattr(k_linear, "svd_info"):
                # Apply activation-aware scaling (if enabled)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(k_linear.in_features, device=DEVICE)  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, k_linear, keys='k')

                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, k_linear, idx, keys='k_proj', num_heads=num_heads if args.is_per_head_svd else None)

                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, layer_indices_dict=layer_indices_dict, num_heads=num_heads if args.is_per_head_svd else None)
                k_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }
            if args.is_per_head_svd:
                num_heads = layer.self_attn.config.num_key_value_heads
                W = v_linear.weight.data.float().view(num_heads, -1, v_linear.in_features).to(device)
            else:
                W = v_linear.weight.data.float().to(device)
            if not hasattr(v_linear, "svd_info"):
                # Apply activation-aware scaling (if enabled)
                if args.act_aware:
                    scaling_diag_matrix = torch.ones(v_linear.in_features, device=DEVICE)  # avoid zero division
                    W, scaling_diag_matrix, scaling_matrix_inv = create_and_fuse_scalingmatrix_to_W(W, scaling_diag_matrix, args, k_linear, keys='v')

                # SVD decomposition
                U, S, V = create_svd_decomposition(W, args, layer_indices_dict, v_linear, idx, keys='v_proj', num_heads=num_heads if args.is_per_head_svd else None)

                if args.act_aware:
                    V = fuse_invariance_S_to_V(V, scaling_matrix_inv, scaling_diag_matrix, args, layer_indices_dict=layer_indices_dict, num_heads=num_heads if args.is_per_head_svd else None)
                v_linear.svd_info_before_rot = {
                    'U': U,
                    'S': S,
                    'V': V
                }

    accumulation_steps = 1   # Number of accumulated batches
    batch_count = 0          # Accumulated batch counter

    # Set model to training mode and only allow gradient computation for svd_modules layers
    set_grad_for_svd_modules(model, args)

    for window in windows:
        input_ids = window.to(device)
        labels = input_ids.clone()
        if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
            labels = labels[:, 1:]
            input_ids = input_ids[:, :-1]
        with torch.enable_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs[0]
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()
        batch_count += 1  # Count each batch

        # Perform backpropagation when accumulated the set number of batches
        if batch_count % accumulation_steps == 0:
            # Update gradient information for S in each layer
            for idx, layer in enumerate(layers):
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    q_linear = layer.self_attn.q_proj
                    k_linear = layer.self_attn.k_proj
                    v_linear = layer.self_attn.v_proj
                    if q_linear.weight.grad is not None:
                        grad_cat = q_linear.weight.grad.detach().to(torch.bfloat16)
                        if args.is_per_head_svd:
                            num_heads = layer.self_attn.config.num_attention_heads
                            grad_cat = grad_cat.view(num_heads, -1, q_linear.in_features)
                        if not hasattr(layer.self_attn.q_proj, 'W_grad_info'):
                            layer.self_attn.q_proj.W_grad_info = get_layer_importance(grad_cat, args)
                        else:
                            layer.self_attn.q_proj.W_grad_info += get_layer_importance(grad_cat, args)
                    if k_linear.weight.grad is not None:
                        grad_cat = k_linear.weight.grad.detach().to(torch.bfloat16)
                        if args.is_per_head_svd:
                            num_heads = layer.self_attn.config.num_key_value_heads
                            grad_cat = grad_cat.view(num_heads, -1, k_linear.in_features)
                        if not hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                            layer.self_attn.k_proj.W_grad_info = get_layer_importance(grad_cat, args)
                        else:
                            layer.self_attn.k_proj.W_grad_info += get_layer_importance(grad_cat, args)
                    if v_linear.weight.grad is not None:
                        grad_cat = v_linear.weight.grad.detach().to(torch.bfloat16)
                        if args.is_per_head_svd:
                            num_heads = layer.self_attn.config.num_key_value_heads
                            grad_cat = grad_cat.view(num_heads, -1, v_linear.in_features)
                        if not hasattr(layer.self_attn.v_proj, 'W_grad_info'):
                            layer.self_attn.v_proj.W_grad_info = get_layer_importance(grad_cat, args)
                        else:
                            layer.self_attn.v_proj.W_grad_info += get_layer_importance(grad_cat, args)
            model.zero_grad()  # Clear gradients

    # Normalize S gradient information
    if batch_count > 0:
        layer_idx = 0
        for layer in layers:
            if hasattr(layer.self_attn.k_proj, 'W_grad_info'):
                if args.svd_modules in ['qkv', 'attn', 'all']:
                    if args.qkv_fuse:
                        layer.self_attn.k_proj.W_grad_info = layer.self_attn.k_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        # Apply final smoothing if enabled
                        if hasattr(args, 'smooth_final') and args.smooth_final:
                            layer.self_attn.k_proj.W_grad_info = smooth_gradient_info(layer.self_attn.k_proj.W_grad_info, args, layer_idx)
                    else:
                        layer.self_attn.q_proj.W_grad_info = layer.self_attn.q_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        layer.self_attn.k_proj.W_grad_info = layer.self_attn.k_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        layer.self_attn.v_proj.W_grad_info = layer.self_attn.v_proj.W_grad_info.div(batch_count//accumulation_steps).abs().sqrt()
                        # Apply final smoothing if enabled
                        if hasattr(args, 'smooth_final') and args.smooth_final:
                            layer.self_attn.q_proj.W_grad_info = smooth_gradient_info(layer.self_attn.q_proj.W_grad_info, args, layer_idx)
                            layer.self_attn.k_proj.W_grad_info = smooth_gradient_info(layer.self_attn.k_proj.W_grad_info, args, layer_idx)
                            layer.self_attn.v_proj.W_grad_info = smooth_gradient_info(layer.self_attn.v_proj.W_grad_info, args, layer_idx)
            layer_idx += 1


def train_projection_official(layer, idx, key, num_heads, args, quantizers, optimizers, schedulers, Q):
    """Local FT and QAT of one q/k/v projection (the q block of local_ft_grad_utils.local_ft_fused_svd, key = q, k or v).

    Returns (loss values of every step, R). R is the learned rotation of the final iteration, already folded into
    layer.self_attn.{key}_proj.svd_info when QAT is on.
    """
    device = DEVICE
    losses = []
    linear = getattr(layer.self_attn, key + '_proj')
    if args.is_per_head_svd:
        W = linear.weight.data.view(num_heads, -1, linear.in_features).float().to(device)
    else:
        W = linear.weight.data.float().to(device)
    if not hasattr(linear, "svd_info"):
        U = linear.svd_info_before_rot['U']
        S = linear.svd_info_before_rot['S']
        V = linear.svd_info_before_rot['V']
        #### A Blinear initialization
        Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys=key, num_heads=num_heads if args.is_per_head_svd else None)
    W = W.to(torch.bfloat16).to(device)

    if args.svd_ft_mode == 'output':
        __in = activations.pop(f"block_{idx}.attn{key}_in").to(device).to(torch.bfloat16)
    else:
        __in = None

    if args.weighted_svd:
        if Blinear is None and not args.weighted_none_svd_qat:
            args.weighted_svd = False
            logging.debug("skip weighted svd")
        else:
            weight = linear.W_grad_info.to(device).to(torch.bfloat16) # nb, override the actquantwrapper thing
            logging.debug(f"Weighted weight shape: {weight.shape}")
    if args.qat_optim_R:
        if args.is_per_head_svd:
            params_R = list(R.parameters())
        else:
            params_R = [R]
    else:
        params_R = None
    if args.is_per_head_svd:
        params_ = list(Alinear.parameters()) + list(Blinear.parameters())
        if args.qat_uv_reg:
            reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]
    else:
        params_ = [Alinear.weight, Blinear.weight]

    optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key=key)

    if not args.qat_L_off or Blinear is not None:
        pass
    else:
        temp_localft_iters = args.localft_iters
        args.localft_iters = 0
        logging.info(f"Layer {idx} Qproj Local FT disabled due to qat_L_off or Blinear is not None")
    if args.is_quant_aware_ft:
        qat_start_iter = int(args.localft_iters * args.qat_start_iter)
    else:
        qat_start_iter = None
    for _ in range(args.localft_iters):
        if qat_start_iter is not None:
            # add extra criteria to control qat start iter
            if _ < qat_start_iter:
                args.is_quant_aware_ft = False
                if _ == 0:
                    logging.debug(f"Layer {idx} Qproj QAT Local QAT disabled before  qat start iter")
            elif _ == qat_start_iter:
                if args.qat_uv_reg:
                    logging.info(f"Layer {idx} Qproj QAT Local QAT, update reg_UV at iter {_}")
                    reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]

                args.is_quant_aware_ft = True

                for pg in optimizers[key].param_groups:
                    pg['lr'] = args.qat_lr_UV
                if key in schedulers and schedulers[key] is not None:
                    schedulers[key].base_lrs = [pg['lr'] for pg in optimizers[key].param_groups]

                logging.debug(f"Layer {idx} Qproj QAT Local QAT enabled after qat start iter")
            else: # _ > qat_start_iter
                args.is_quant_aware_ft = True

        if (_+1) % args.qat_param_update_freq == 0 or _ == qat_start_iter:
            qat_param_update = True
        else:
            qat_param_update = False

        with torch.enable_grad():
            for bsz in range(0, args.nsamples, args.bs):
                if args.is_quant_aware_ft: # Local QAT
                    optimizers[key].zero_grad()
                    optimizers[key + '_R'].zero_grad()
                    loss = iter_local_qat_loss(Alinear, Blinear, W, Q, bsz, weight, __in, args, quantizers, idx,  keys=key, reg_UV=reg_UV if args.qat_uv_reg else None, R=R, qat_param_update=qat_param_update)
                    optimizers[key].step()
                    optimizers[key + '_R'].step()
                    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                        schedulers[key].step()
                        schedulers[key + '_R'].step()
                    if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                        logging.debug(f"Layer {idx} Qproj - Iter {_}, Loss: {loss.item():.6f}, U/V LR: {optimizers[key].param_groups[0]['lr']:.6f}, R LR: {optimizers[key + '_R'].param_groups[0]['lr']:.6f}")
                else: # Local FT
                    optimizers[key].zero_grad()
                    loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys=key)
                    optimizers[key].step()
                    if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                        schedulers[key].step()
                    if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                        logging.debug(f"Layer {idx} Qproj - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers[key].param_groups[0]['lr']:.6f}")

                losses.append(loss.item())

        if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
            args.is_quant_aware_ft = True

    if Blinear is None:
        q_dtype = linear.weight.dtype
        q_device = linear.weight.device
        quantizers[idx][key]['Alinear'].find_params(Alinear.weight.data)
        linear.weight.data = quantizers[idx][key]['Alinear'].quantize(Alinear.weight.data).to(q_dtype).to(q_device)
        if not args.weighted_none_svd_qat:
            args.weighted_svd = True
        if args.qat_L_off:
            args.localft_iters = temp_localft_iters
        logging.debug(f"Layer {idx} Qproj Local Qat FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
    else:
        # store SVD results
        if args.is_per_head_svd:
            if args.is_quant_aware_ft:
                fuse_per_head_svd_rotated_weights(Alinear, Blinear, quantizers, idx, keys=key, num_heads=num_heads, R=R)
            linear.svd_info = {
                'U': Alinear.cpu(),
                'S': None,
                'V': Blinear.cpu()
            }
            logging.debug(f"Layer {idx} Qproj Local FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
        else:
            linear.svd_info = {
                'U': Alinear.weight.data.cpu(),
                'S': None,
                'V': Blinear.weight.data.cpu()
            }
            logging.debug(f"Layer {idx} Qproj Local FT completed, Sigma shape: {S.shape}")
    return losses, R


def dense_projection_official(layer, linear, idx, key, args, quantizers, optimizers, schedulers):
    """The Blinear-is-None path of local_ft_grad_utils.local_ft_fused_svd for o_proj, gate_proj, up_proj, down_proj.

    Quantises linear.weight in place to args.w_bits per channel (no clip) after args.localft_iters FT steps
    (0 with args.qat_L_off). Returns the loss values.
    """
    device = DEVICE
    losses = []
    W = linear.weight.data.float().to(device)
    if not hasattr(linear, "svd_info"):
        U = linear.weight.data
        S = V = None

        #### A Blinear initialization
        Alinear, Blinear, R = create_ABlinear_and_quantizer(U, S, V, args, quantizers, idx, keys=key)
        # del linear.svd_info_before_rot #[FIXME:]

    W = W.to(torch.bfloat16)

    if args.svd_ft_mode == 'output':
        __in = activations.pop(f"block_{idx}.attn{key}_in").to(device).to(torch.bfloat16)
    else:
        __in = None

    if args.weighted_svd:
        if Blinear is None and not args.weighted_none_svd_qat:
            args.weighted_svd = False
            weight = None
            logging.debug("skip weighted svd")
        else:
            weight = linear.W_grad_info.to(device).to(torch.bfloat16)

    params_R = None
    params_ = [Alinear.weight]
    if args.qat_uv_reg:
        reg_UV = [W]

    optimizers, schedulers = create_optimizer(params_, params_R, args, optimizers, schedulers, key=key)

    if not args.qat_L_off or Blinear is not None:
        pass
    else:
        temp_localft_iters = args.localft_iters
        args.localft_iters = 0
        logging.info(f"Layer {idx} Oproj Local FT disabled due to qat_L_off or Blinear is not None")
    ### [FIXME: add extra function to compute loss iter, this should be the same for all svdmodules]
    if args.is_quant_aware_ft:
        qat_start_iter = int(args.localft_iters * args.qat_start_iter)
    else:
        qat_start_iter = None
    for _ in range(args.localft_iters):
        if qat_start_iter is not None:
            # add extra criteria to control qat start iter
            if _ < qat_start_iter:
                args.is_quant_aware_ft = False
            elif _ == qat_start_iter:
                if args.qat_uv_reg:
                    reg_UV = [[alin.detach().clone() for alin in Alinear], [blin.detach().clone() for blin in Blinear]]

                args.is_quant_aware_ft = True
            else:
                args.is_quant_aware_ft = True
        for bsz in range(0, args.nsamples, args.bs):
            optimizers[key].zero_grad()
            with torch.enable_grad():
                loss = iter_local_ft_loss(Alinear, Blinear, W, bsz, weight, __in, args, quantizers, idx,  keys=key, R=R if args.qat_optim_R else None)
                optimizers[key].step()    
            if hasattr(args, 'scheduler_type') and args.scheduler_type is not None:
                schedulers[key].step()
                if _ % getattr(args, 'scheduler_step_size', 30) == 0:
                    logging.debug(f"Layer {idx} Oproj - Iter {_}, Loss: {loss.item():.6f}, LR: {optimizers[key].param_groups[0]['lr']:.6f}")
            losses.append(loss.item())
        if qat_start_iter is not None and qat_start_iter >= args.localft_iters and _ == args.localft_iters - 1:
            args.is_quant_aware_ft = True
    if Blinear is None:
        o_dtype = linear.weight.dtype
        o_device = linear.weight.device
        quantizers[idx][key]['Alinear'].find_params(Alinear.weight.data)
        linear.weight.data = quantizers[idx][key]['Alinear'].quantize(Alinear.weight.data).to(o_dtype).to(o_device)   
        logging.debug(f"Layer {idx} Oproj Local Qat FT completed, Sigma shape: {S[0].shape if S is not None else 'None'}")
        if not args.weighted_none_svd_qat:
            args.weighted_svd = True
        if args.qat_L_off:
            args.localft_iters = temp_localft_iters
    else:
        # store SVD results
        linear.svd_info = {
            'U': Alinear.weight.data.cpu(),
            'S': None,
            'V': Blinear.weight.data.cpu()
        }
        logging.debug(f"Layer {idx} Oproj Local FT completed, Sigma shape: {S.shape}")
    return losses


def local_ft_qkv_text(layers, args, layer_indices_dict, Q):
    """Per layer, q then k then v: factor init, local FT, QAT and rotation fold. Returns the quantizers dict."""
    quantizers = {}
    for idx, layer in enumerate(layers):
        optimizers = {}
        schedulers = {}
        if args.is_quant_aware_ft:
            if idx not in quantizers:
                quantizers[idx] = {}
        config = layer.self_attn.config
        for key in ['q', 'k', 'v']:
            num_heads = config.num_attention_heads if key == 'q' else config.num_key_value_heads
            train_projection_official(layer, idx, key, num_heads, args, quantizers, optimizers, schedulers, Q)
    return quantizers


def dense_prequant_text(layers, args, quantizers):
    """Dense paths of local_ft_fused_svd for o_proj, up_proj, gate_proj and down_proj (svd_modules = 'all' under QAT)."""
    for idx, layer in enumerate(layers):
        optimizers = {}
        schedulers = {}
        if idx not in quantizers:
            quantizers[idx] = {}
        for key, linear in [
            ('o', layer.self_attn.o_proj),
            ('up', layer.mlp.up_proj),
            ('gate', layer.mlp.gate_proj),
            ('down', layer.mlp.down_proj),
        ]:
            dense_projection_official(layer, linear, idx, key, args, quantizers, optimizers, schedulers)


def wsvd_args(**overrides):
    """The released W8A8 QAT command (README.md, script/local_qat/wsvd_qat_llava_next_7b.sh) as an args namespace."""
    base = dict(
        seed=0,
        rotate=True,
        rotate_mode='hadamard',
        svd_modules='qkv',
        is_per_head_svd=True,
        is_q_headnum=True,
        qkv_fuse=False,
        kv_fuse=False,
        had_rank=False,
        act_aware=True,
        act_alpha=0.5,
        calib_method='cholesky',
        taylor_order=2,
        is_taylor=False,
        add_taylor_first=False,
        grad_alpha=1.0,
        label_shift=False,
        is_rank_allocate_ft=True,
        use_S_gradinfo_init=False,
        rank_ratio=0.5,
        use_true_param_ratio=True,
        use_all_true_param_ratio=False,
        group_ratio=0.0,
        weighted_svd=True,
        weighted_none_svd_qat=False,
        svd_ft_mode='weight',
        svd_ft_optim='adam',
        localft_iters=150,
        localft_lr=1e-4,
        is_quant_aware_ft=True,
        is_stage=False,
        qat_L_off=True,
        qat_uv_reg=False,
        qat_uv_reg_scale=False,
        qat_uv_reg_alpha=0.1,
        qat_start_iter=0.66666667,
        qat_optim_R=True,
        qat_param_update_freq=1,
        qat_lr_UV=1e-5,
        qat_lr_R=1.0,
        scheduler_type=None,
        smooth_grad=False,
        smooth_final=False,
        w_bits=8,
        w_asym=False,
        nsamples=64,
        bs=64,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def wsvd_attention_text(model, layers, windows, args):
    """The attention part of svd_lm_setup for text windows, in the official order.

    Gram (Cholesky) -> S_grad_info and allocation -> per-head SVD and Fisher weights -> local FT + QAT of q/k/v ->
    dense prequantisation of o/gate/up/down -> SVDLinear modules (kept in the returned dict, not installed).
    Returns a dict with layer_indices_dict, quantizers and svdlinears {(layer, proj): SVDLinear}.
    """
    args.svd_modules = 'qkv'
    calib_input_cholesky_text(model, layers, windows, args)
    calib_grad_info_text(model, layers, windows, args)
    _, _, layer_indices_dict = svd_qkv_with_grad_info_text(layers, args)
    calib_localft_grad_info_text(model, layers, windows, args, layer_indices_dict)
    disable_grad_for_svd_modules(model, args)
    Q = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode, args.seed)
    quantizers = local_ft_qkv_text(layers, args, layer_indices_dict, Q)
    dense_prequant_text(layers, args, quantizers)
    svdlinears = {}
    for idx, layer in enumerate(layers):
        for name in ['q_proj', 'k_proj', 'v_proj']:
            svdlinears[(idx, name)] = SVDLinear.from_linear_localft(
                getattr(layer.self_attn, name),
                param_ratio=args.rank_ratio,
                sigma_fuse='local_ft',
                had_rank=args.had_rank,
                had_mode='random',
                seed=args.seed,
                module_name=name,
                is_per_head_svd=args.is_per_head_svd,
                is_rank_allocation=args.is_rank_allocate_ft,
            )
    return {"layer_indices_dict": layer_indices_dict, "quantizers": quantizers, "svdlinears": svdlinears}
