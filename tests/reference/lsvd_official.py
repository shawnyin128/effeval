# Vendored reference implementation for tests only.
# Source: https://github.com/SAI-Lab-NYU/LSVD (commit f5675bd02af1d0ed047eb733dc1e72256d4408a6), directory
# fake_quant/utils/. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
# Vendored (line ranges refer to that commit):
#   utils.py 89-96 (set_seed)
#   quant_utils.py 8-65 (get_minq_maxq .. AsymSTEQuantize), 122-525 (ActQuantizer, ActQuantWrapper, WeightQuantizer),
#     550-589 (add_actquant, find_qlayers)
#   gptq_utils.py 19-28, 30-52 (LM_PROJ_SPECS, _should_skip_lm_weight_quant and its helpers), 1154-1262 (rtn_fwrd)
#   act_aware_utils.py 43-93 (_build_act_calib_quantizer, _get_calib_stats_input); from
#     calib_input_distributionlowresources: hook 577-606, hook registration 721-738, post-replay clean-up 743-759,
#     Cholesky finalisation 789-850
#   loss_aware_utils.py 21-27 (_is_lm_layer_name); from calib_grad_distribution: setup 875-892, make_output_grad_hook
#     901-947, hook registration 984-990, text batch 1046-1055, hook removal 1256-1258, Cholesky finalisation 1269-1323
#   grad_info_utils.py 198-241 (loss-aware helpers), 3073-3096 (taylor_like_s_grad_squared, get_layer_importance),
#     2780-2944 (get_top_k_scores); from prepare_fuse_svd: 1480-1492 and the separate q/k/v branch 1687-1756; from
#     calib_grad_info: scaling copy 503-516, requires_grad 592-608, separate q/k/v accumulation 948-1018,
#     normalisation 1401-1412; from svd_qkv_with_grad_info: 2124-2126, 2254-2255, scores 2329-2347, 2473,
#     budget 2699-2721, 2774-2776
#   svd_utils.py 34-45 (_match_module_weight_dtype), 158-169 (_copy_cached_calibration_stats), 171-319
#     (SVDLinear.__init__), 1521-1695 (from_linear_with_grad), 1947-2043 (SVDLinear.forward), 2044-2069 (rsetattr,
#     rgetattr, _unwrap_actquant_linear, _find_svd_candidate_layers), 2081-2097 (counter dicts), 2858-2909 (separate
#     q/k/v replacement loop of svd_lm_setup)
#   hybrid_ffn_utils.py 65 (_SVD_AWARE_MAX_TOKENS), 72-74, 131-200, 378-645, 910-984 (HybridSVDQuantFFN without
#     extra_repr), 1013-1341 (build_hybrid_ffn), 1348-1527, 1530-1536 (_make_fp16_linear), 2231-2517
#     (select_svd_channels_svd_aware), 2807-2824 (load_channel_groups), 2855-2883 (hybrid_ffn_rtn_targets); from
#     _collect_layer_inputs: 1848-1854, hooks 1868-1883, concatenation 1966-1984; from analyze_ffn_channel_groups
#     (Phase 1, svd_aware only): 2562-2576, 2628, 2634, 2640-2664, whitening 2665-2711, svd_aware branch 2746-2765,
#     entry 2776-2788; from hybrid_ffn_setup: per-layer replacement 1766-1822
# Modifications to the vendored code (everything else is verbatim, except that indentation of slices is shifted):
#   (1) utils.get_dev() -> the module constant DEVICE = torch.device("cpu"); .cuda() -> .to(DEVICE);
#   (2) new text drivers replacing the VLM loops: calib_input_cholesky_text (the verbatim hook on every k_proj, one
#       model(w) per window instead of the layer-by-layer replay, then the verbatim finalisation including the sharing
#       of the k_proj Cholesky factor to q/k/v of the Qwen legacy path), calib_grad_distribution_text (verbatim
#       setup, hooks, text-batch branch, finalisation), calib_grad_info_text (prepare_fuse_svd, requires_grad setup,
#       text-batch loss, verbatim separate q/k/v accumulation and normalisation), lsvd_attention_decompose (the
#       three calibrations in the official order, svd_qkv_with_grad_info, the verbatim replacement loop);
#       windows are LongTensors [1, T] (official: dataloader batches whose batch[0] is the token tensor);
#   (3) prepare_fuse_svd, svd_qkv_with_grad_info and rtn_fwrd take the list of decoder layers (model_utils.get_layers
#       is not available); svd_qkv_with_grad_info has no cache reads or writes and reads S_grad_info and svd_info
#       from the projections as calib_grad_info_text left them; debug payloads, cache files and the try/except around
#       the batch loops (which swallowed errors) are removed; the empirical-Fisher and raw-trace branches are dead
#       because act_calib_debug is not set;
#   (4) phase1_layer_groups(Wg, Wu, Wd, calib, args, idx) = the per-layer svd_aware block of
#       analyze_ffn_channel_groups with use_gpu_analysis = True on DEVICE (official: GPU, so the first-8192-token
#       cap and the X[::2][:4096] whitening rows apply); Wg, Wu, Wd are float weights instead of module weights;
#   (5) hybrid_ffn_setup(model, args, layer_inps, groups) takes the collected inputs and the groups dict ({"layers":
#       {str(idx): entry}}, as phase1_layer_groups entries) instead of reading JSON files or calling
#       _collect_layer_inputs; _collect_layer_inputs(model, layers, windows) keeps the text-only branch;
#   (6) the CUDA online-Hadamard kernels (hadamard_utils.get_hadK, matmul_hadU_cuda, matmul_hadU_cuda_nopow2,
#       fast_hadamard_transform) are replaced by stand-ins: pure torch for power-of-two sizes (get_hadK -> (None, 1),
#       matmul_hadU_cuda = the normalised Sylvester Hadamard of hybrid_ffn_utils._hadamard_transform_power2_lastdim),
#       NotImplementedError for every other size and for ActQuantWrapper's full_hadnopow2 and partial_had branches;
#       the try/except around get_hadK in hybrid_ffn_setup (skipped layers) and build_hybrid_ffn (hybrid_ffn_had)
#       is kept, so a non-power-of-two size there only logs a warning and leaves the layer without online Hadamard;
#   (7) imports of hadamard_utils, rotation_utils, model_utils, data_utils, calib_cache, debug_utils,
#       reasoning_calib_utils, molmo_utils, qwen_vlm_utils, tqdm and the VLM helpers are removed, as are
#       utils.cleanup_memory calls, tqdm wrappers and module prefixes (quant_utils., svd_utils.); names that only
#       dead paths use are not defined (debug_utils, select_svd_channels, _prepare_gradient_loss_inputs,
#       Int8WeightLinear, _make_real_int8_linear, _copy_runtime_hadamard_config, rotation_utils for the had_rank
#       branch of from_linear_with_grad, get_topk_location for the is_ma_hack branch of SVDLinear.forward);
#   (8) total_rank_sum and total_linear_count are module-level dicts (official: reset in svd_lm_setup).
# ruff: noqa
import logging
import math
import os
import random
import typing
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

DEVICE = torch.device("cpu")


class hadamard_utils:
    """Stand-in for the official CUDA Hadamard kernels: pure torch for power-of-two sizes, else NotImplementedError."""

    @staticmethod
    def get_hadK(n, transpose=False):
        if n & (n - 1) != 0 or n <= 0:
            raise NotImplementedError(f"online Hadamard of size {n} needs the official CUDA kernels")
        return None, 1

    @staticmethod
    def matmul_hadU_cuda(X, hadK, K):
        if K != 1 or hadK is not None:
            raise NotImplementedError("online Hadamard with K != 1 needs the official CUDA kernels")
        return _hadamard_transform_power2_lastdim(X)

    @staticmethod
    def matmul_hadU_cuda_nopow2(X, hadK, K):
        raise NotImplementedError("online Hadamard needs the official CUDA kernels")


class fast_hadamard_transform:
    @staticmethod
    def hadamard_transform(*args, **kwargs):
        raise NotImplementedError("online Hadamard needs the official CUDA kernels")
def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size
def set_seed(seed):
    np.random.seed(seed)  # Set NumPy seed
    random.seed(seed)  # Set Python's built-in random seed
    torch.manual_seed(seed)  # Set PyTorch seed for CPU & CUDA
    torch.cuda.manual_seed(seed)  # Set seed for CUDA (if using GPU)
    torch.cuda.manual_seed_all(seed)  # If multi-GPU, set for all GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable auto-optimizations for determinism
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
class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None


    def forward(self, x):
        x_dtype = x.dtype
        if self.bits == 16:
            return x
        elif self.sym:
            return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
        return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)


    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits, groupsize=-1, sym=False, clip_ratio=1.0):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)
        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module:torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.in_features = module.in_features # for convinence of localft in/out channel 
        self.out_features = module.out_features 
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_full_hadnopow2 = False
        self.online_partial_had = False
        self.online_random_had = False
        self.had_dim = 0
        self.fp32_had = False
        self.fuse_bias = False
        self.fuse_linear = False
        self.bias_param = None

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype
        if self.fuse_linear:
            x = self.bias_param(x.float()).to(x_dtype)
        if self.fuse_bias:
            a, b = self.bias_param
            x = (x + a * x + b).to(x_dtype)
        if self.online_random_had:
            if x.shape[-1] != self.had_K.shape[0]:
                init_shape = x.shape
                x = x.reshape((init_shape[0], -1, init_shape[-1]//self.had_K.shape[0],self.had_K.shape[0]))
                if self.fp32_had:
                    x = x.double()@self.had_K.to(x_dtype)
                else:
                    x = x @ self.had_K.to(x_dtype)
                x = x.reshape(init_shape)
            else:
                if self.fp32_had:
                    x = x.double()@self.had_K.to(x_dtype)
                else:
                    x = x @ self.had_K.to(x_dtype)
        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        if self.online_full_hadnopow2:

            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda_nopow2(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda_nopow2(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1/math.sqrt(init_shape[-1]//self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1, init_shape[-1]//self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1]//self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16: #Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        module_input = x
        module_weight = getattr(self.module, "weight", None)
        if (
            module_weight is not None
            and x.is_floating_point()
            and module_weight.is_floating_point()
            and x.dtype != module_weight.dtype
        ):
            module_input = x.to(module_weight.dtype)
        x = self.module(module_input).to(x_dtype)

        if self.out_quantizer.bits < 16: #Quantize the output, if needed
            self.out_quantizer.find_params(x)
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


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

    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
            return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
                x_dtype
            )
        return x

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
def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)

def find_qlayers(module, layers=[torch.nn.Linear,
                                ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res
LM_PROJ_SPECS = [
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "up_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "down_proj"),
]
def _is_svd_linear_module(module) -> bool:
    return hasattr(module, "ALinear") and hasattr(module, "BLinear")


def _layer_svd_prefixes(layer) -> set:
    prefixes = set()
    for parent_name, child_name in LM_PROJ_SPECS:
        parent = getattr(layer, parent_name, None)
        mod = getattr(parent, child_name, None) if parent is not None else None
        if _is_svd_linear_module(mod):
            prefixes.add(f"{parent_name}.{child_name}")
    return prefixes


def _should_skip_lm_weight_quant(name: str, layer, args) -> bool:
    if not bool(getattr(args, "gpt_svd_post_quant", False)) or not bool(getattr(args, "svd_lm", False)):
        return False
    for prefix in _layer_svd_prefixes(layer):
        if name == f"{prefix}.module" or name.startswith(f"{prefix}.ALinear") or name.startswith(f"{prefix}.BLinear"):
            return True
    return False
@torch.no_grad()
def rtn_fwrd(layers, dev, args, start_id=0):
    '''
    From GPTQ repo 
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in range(start_id, len(layers)):
        layer = layers[i].to(dev)

        if getattr(args, 'use_hybrid_ffn', False) and hasattr(layer.mlp, 'gate_svd'):
            real_int8 = bool(getattr(args, 'hybrid_real_int8', False))
            fp16_qg = bool(getattr(args, 'hybrid_ffn_fp16_quant_group', False))
            target_names = hybrid_ffn_rtn_targets(real_int8=real_int8, fp16_quant_group=fp16_qg)

            for name in target_names:
                try:
                    module = rgetattr(layer, name)
                except AttributeError:
                    logging.info(
                        "[hybrid_ffn][rtn] skip missing target %s in layer %d",
                        name,
                        i,
                    )
                    continue
                layer_weight_bits = args.w_bits
                if args.int8_down_proj and 'down_proj' in name:
                    layer_weight_bits = 8

                quantizer = WeightQuantizer()
                quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
                )
                W = module.weight.data
                W_dtype = W.dtype
                quantizer.find_params(W)
                module.weight.data = quantizer.quantize(W).to(
                    device=W.device, dtype=W_dtype
                )
                quantizers[f'model.layers.{i}.{name}'] = quantizer.cpu()

            layers[i] = layer.cpu()
            torch.cuda.empty_cache()
            del layer
            continue

        # Skipped hybrid-FFN layers (or non-hybrid mode) fall through to standard RTN below.

        # In FP16 quant-group mode, skip layers also keep their FFN in FP16.
        _fp16_skip_ffn = (
            bool(getattr(args, 'hybrid_ffn_fp16_quant_group', False))
            and bool(getattr(args, 'use_hybrid_ffn', False))
            and i in set(getattr(args, 'hybrid_ffn_skip_layers', None) or [])
        )
        _FFN_NAMES = {'gate_proj', 'up_proj', 'down_proj'}

        subset = find_qlayers(layer, layers=[torch.nn.Linear, torch.nn.ParameterList]) # SVDLinear

        for name in subset:
            if _should_skip_lm_weight_quant(name, layer, args):
                continue
            if _fp16_skip_ffn and any(fn in name for fn in _FFN_NAMES):
                continue
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            module = subset[name]

            if isinstance(module, torch.nn.ParameterList):
                for j, uu in enumerate(module):
                    quantizer = WeightQuantizer()
                    quantizer.configure(
                        layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
                    )
                    with torch.no_grad():
                        # uu is a Parameter
                        W = uu.data
                        W_dtype = W.dtype
                        quantizer.find_params(W)
                        uu.data.copy_(quantizer.quantize(W).to(device=W.device, dtype=W_dtype))
                    quantizers['model.layers.%d.%s.%d' % (i, name, j)] = quantizer.cpu()
            else: # nn.Linear
                quantizer = WeightQuantizer()
                quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
                )
                W = module.weight.data
                W_dtype = W.dtype
                quantizer.find_params(W)
                module.weight.data = quantizer.quantize(W).to(
                    device=W.device, dtype=W_dtype)
                quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
    return quantizers
def _build_act_calib_quantizer(args, method):
    # Apply fake-quantized stats regardless of calibration method when enabled.
    method_l = str(method).lower()
    enable = (
        bool(getattr(args, "act_calib_fake_quant", False))
        or "fakequant" in method_l
        or "fq" in method_l
    )
    if not enable:
        return None

    a_bits = int(getattr(args, "a_bits_calib", 16))
    if a_bits >= 16:
        logging.info("Activation fake-quant stats enabled but a_bits=%d, skipping fake quant.", a_bits)
        return None

    quantizer = ActQuantizer()
    quantizer.configure(
        bits=a_bits,
        groupsize=int(getattr(args, "a_groupsize", -1)),
        sym=not bool(getattr(args, "a_asym", False)),
        clip_ratio=float(getattr(args, "a_clip_ratio", 1.0)),
    )
    logging.info(
        "Activation fake-quant stats enabled: bits=%d, groupsize=%d, sym=%s, clip_ratio=%.4f",
        a_bits,
        int(getattr(args, "a_groupsize", -1)),
        str(not bool(getattr(args, "a_asym", False))),
        float(getattr(args, "a_clip_ratio", 1.0)),
    )
    return quantizer


def _get_calib_stats_input(x, quantizer):
    # Keep model forward untouched; only fake-quantize a detached copy for statistics.
    if quantizer is None:
        return x.detach()
    x_stats = x.detach()
    try:
        quantizer.find_params(x_stats)
        x_stats = quantizer(x_stats)
        quantizer.free()
        return x_stats
    except Exception as e:
        logging.warning("Activation fake-quant stats failed once, fallback to fp activations: %s", e)
        try:
            quantizer.free()
        except Exception:
            pass
        return x.detach()
def calib_input_cholesky_text(model, layers, windows, args):
    method = args.calib_method
    if args.a_bits_calib is None:
        args.a_bits_calib = args.a_bits
    legacy_qwen_kproj_cholesky = True
    model.eval()
    debug_hook_stats = False
    hook_name_by_id = {}
    hook_calls = {}
    stats_quantizer = _build_act_calib_quantizer(args, method)

    def hook(module, input, output):
        module_id = id(module)
        if module_id in hook_name_by_id:
            hook_calls[module_id] = hook_calls.get(module_id, 0) + 1
        stats_inp = _get_calib_stats_input(input[0], stats_quantizer)
        lead_tokens = int(stats_inp.numel() // max(int(stats_inp.shape[-1]), 1))
        module._debug_sample_count = int(getattr(module, "_debug_sample_count", 0) + int(stats_inp.shape[0] if stats_inp.dim() > 1 else 1))
        module._debug_token_count = int(getattr(module, "_debug_token_count", 0) + lead_tokens)
        if "abs_mean" in method:
            abs_mean = stats_inp.abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = stats_inp.abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        elif "cholesky" in method:
            inp = stats_inp.float()
            if inp.dim() == 2:   # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            if isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                module.raw_scaling_diag_matrix += adds_sum
            else:
                module.raw_scaling_diag_matrix = adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()

    missed_hook_layers = []
    for i in range(len(layers)):
        layer = layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                if legacy_qwen_kproj_cholesky and 'k_proj' not in name:
                    continue
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module.raw_scaling_diag_matrix = 0
                module_id = id(module)
                full_name = f"model.layers.{i}.{name}"
                hook_name_by_id[module_id] = full_name
                hook_calls[module_id] = 0
                module.register_forward_hook(hook)
    with torch.no_grad():
        for w in windows:
            model(w.to(DEVICE))
    for i in range(len(layers)):
        layer = layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                if legacy_qwen_kproj_cholesky and 'k_proj' not in name:
                    continue
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                module._forward_hooks.clear()
                if isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                    module.raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.cpu()
                else:
                    missed_hook_layers.append(f"model.layers.{i}.{name}")
    act_calib_debug_payload_lr = {}
    for i in range(len(layers)):
        for name, module in layers[i].named_modules():
            if isinstance(module, nn.Linear):
                if legacy_qwen_kproj_cholesky and 'k_proj' not in name:
                    continue
                if 'gate_proj' in name:
                    continue # qkv only store in k_proj, SwiGLU only store in up_proj, constantly skip
                elif 'o_proj' in name and args.svd_modules in ['qkv','kv','gaup','mlp', 'down']:
                    continue # only skip when svd_modules is qkv
                elif 'mlp' in name and args.svd_modules in ['attn', 'qkv', 'kv', 'o']:
                    continue # only skip when svd_modules is not all
                elif 'q_proj' in name and args.svd_modules in ['kv']:
                    continue # only skip q_proj when svd_modules is kv
                if not isinstance(module.raw_scaling_diag_matrix, torch.Tensor):
                    continue
                raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(DEVICE)
                diag_shift = torch.zeros(
                    raw_scaling_diag_matrix.shape[:-2] if raw_scaling_diag_matrix.ndim > 2 else (),
                    device=raw_scaling_diag_matrix.device,
                    dtype=raw_scaling_diag_matrix.dtype,
                )
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).double()
                except Exception as e:
                    print("Warning: eigen scaling_diag_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    diag_shift = (- eigenvalues[..., 0] + 1e-6).clamp_min(0.0)
                    raw_scaling_diag_matrix += diag_shift[..., None, None] * torch.eye(raw_scaling_diag_matrix.shape[-1]).to(DEVICE)
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).double()
                    eigenvalues = None
                    del eigenvalues
                if getattr(args, "act_calib_debug", False) and "cholesky" in method:
                    full_key = f"model.layers.{i}.{name}"
                    act_calib_debug_payload_lr[full_key] = debug_utils.act_calib_debug_pack_with_summary(
                        raw_cpu_double=raw_scaling_diag_matrix.detach().cpu(),
                        chol_cpu=scaling_diag_matrix.detach().cpu(),
                        normalization_count=max(getattr(module, "_debug_token_count", 0), 1),
                        sample_count=getattr(module, "_debug_sample_count", 0),
                        token_count=getattr(module, "_debug_token_count", 0),
                        hook_count=hook_calls.get(id(module), 0),
                        diag_shift_cpu=diag_shift.detach().cpu(),
                    )
                if 'k_proj' in name: # only valid for k_proj in original svd_modules ==qkv 
                    if legacy_qwen_kproj_cholesky:
                        layers[i].self_attn.q_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                        layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                        layers[i].self_attn.v_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    else:
                        layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'o_proj' in name:
                    layers[i].self_attn.o_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'up_proj' in name:
                    layers[i].mlp.up_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                elif 'down_proj' in name:
                    layers[i].mlp.down_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
                del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
                if hasattr(module, "_debug_sample_count"):
                    del module._debug_sample_count
                if hasattr(module, "_debug_token_count"):
                    del module._debug_token_count
                torch.cuda.empty_cache()
def _is_lm_layer_name(name):
    return (
        "model.layers" in name
        or "language_model.layers" in name
        or "text_model.layers" in name
        or "transformer.blocks" in name
    )
def calib_grad_distribution_text(model, layers, windows, args):
    device = DEVICE
    model.eval()
    model = model.to(device)
    model.train()
    # 先关掉所有参数梯度
    for _, p in model.named_parameters():
        p.requires_grad = False

    # 按索引开启需要梯度的少量层（比如只开第 0 层）
    layers = layers
    num_layers = len(layers)
    attn = layers[0].self_attn
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        if hasattr(attn, proj_name):
            for p in getattr(attn, proj_name).parameters():
                p.requires_grad = True
    num_layers = len(layers)
    attn = layers[0].self_attn
    for proj_name in ['q_proj', 'k_proj', 'v_proj']:
        if hasattr(attn, proj_name):
            for p in getattr(attn, proj_name).parameters():
                p.requires_grad = True

    handles = {}
    module_name_lookup = {id(module): name for name, module in model.named_modules()}
    num_heads_q = layers[0].self_attn.config.num_attention_heads #model.config.num_attention_heads
    num_heads_kv = layers[0].self_attn.config.num_key_value_heads
    head_dim = layers[0].self_attn.config.hidden_size // layers[0].self_attn.config.num_attention_heads
    hook_trigger_count = {}
    grad_debug_payload = None

    def make_output_grad_hook(name):
        def hook(module, grad_in, grad_out):
            hook_trigger_count[name] = hook_trigger_count.get(name, 0) + 1
            base_grad = grad_out[0].detach()
            token_count = int(base_grad.numel() // max(int(base_grad.shape[-1]), 1))
            sample_count = int(base_grad.shape[0] if base_grad.dim() > 1 else 1)
            module._debug_hook_count = int(getattr(module, "_debug_hook_count", 0) + 1)
            module._debug_sample_count = int(getattr(module, "_debug_sample_count", 0) + sample_count)
            module._debug_token_count = int(getattr(module, "_debug_token_count", 0) + token_count)
            bsz = grad_out[0].shape[0]
            if args.is_per_head_svd and ('q_proj' in name or 'k_proj' in name or 'v_proj' in name):
                # [batch_size, num_heads, seq_len, head_dim]
                if 'q_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_q, head_dim).permute(0, 2, 1, 3).contiguous().float()
                elif 'k_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_kv, head_dim).permute(0, 2, 1, 3).contiguous().float()
                elif 'v_proj' in name:
                    grads = grad_out[0].detach().view(bsz, -1, num_heads_kv, head_dim).permute(0, 2, 1, 3).contiguous().float()
                else:
                    raise ValueError(f"Invalid projection name: {name}")
            else:
                # [batch_size, seq_len, hidden_size]
                grads = grad_out[0].detach().float()
            # calculate covariance of output gradients
            grads_cov = torch.matmul(grads.transpose(-1, -2), grads).sum(dim=0) # [hidden_size, hidden_size] or [num_heads, head_dim, head_dim]
            if not hasattr(module, "output_grad_distribution"):
                module.register_buffer("output_grad_distribution", torch.zeros_like(grads_cov, dtype=torch.float32))
            module.output_grad_distribution += grads_cov
            empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
            input_proj = getattr(module, "_empirical_fisher_input_proj", None)
            if empirical_cfg is not None and isinstance(input_proj, torch.Tensor) and input_proj.numel() > 0:
                for head_idx, head_state in empirical_cfg["heads"].items():
                    if grads.ndim < 4 or int(head_idx) >= int(grads.shape[1]):
                        continue
                    output_projection = head_state["output_projection"].to(device=grads.device, dtype=grads.dtype)
                    grad_head = grads[:, int(head_idx), :, :]
                    grad_proj = torch.einsum("bth,hk->btk", grad_head, output_projection)
                    sketch = torch.einsum("btk,btr->bkr", grad_proj, input_proj)
                    flat = sketch.reshape(sketch.shape[0], -1).detach().float().cpu()
                    head_state["sketch_outer"] += flat.transpose(0, 1) @ flat
                    head_state["sketch_diag"] += torch.sum(flat * flat, dim=0)
                    head_state["sketch_trace_sum"] += float(torch.sum(flat * flat).item())
                    head_state["sample_count"] += int(flat.shape[0])
                    empirical_cfg["sample_count"] += int(flat.shape[0])
                delattr(module, "_empirical_fisher_input_proj")
            del grads, grads_cov
        return hook

    for idx, layer in enumerate(layers):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            for proj_name in ['q_proj', 'k_proj', 'v_proj']:
                if hasattr(layer.self_attn, proj_name):
                    proj = getattr(layer.self_attn, proj_name)
                    key = f"layer_{idx}.{proj_name}"
                    handles[key] = proj.register_full_backward_hook(make_output_grad_hook(key))
                    proj.output_grad_distribution = 0
                    hook_trigger_count[key] = 0

    accumulation_steps = 1
    batch_count = 0
    for batch_idx, w in enumerate(windows):
        input_ids = w.to(device)
        labels = input_ids.clone()
        if args.label_shift: # enable label shift like palu, or spinquant handle this in Trainer?
            labels = labels[:, 1:]
            input_ids = input_ids[:, :-1]
        with torch.enable_grad():
            outputs = model(input_ids, labels=labels)
            loss = outputs[0]
            loss = loss / accumulation_steps  # Normalize loss
            loss.backward()
        batch_count += 1
        if batch_count % accumulation_steps == 0:
            model.zero_grad()

    # Remove registered hooks
    for h in list(handles.values()):
        h.remove()
    # Cholesky Decomposition (skip modules where hook never ran, e.g. output_grad_distribution still 0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_lm_layer_name(name):
            if hasattr(module, 'output_grad_distribution'):
                if isinstance(module.output_grad_distribution, torch.Tensor):
                    G = module.output_grad_distribution.double().to(DEVICE)
                    diag_shift = torch.zeros(
                        G.shape[:-2] if G.ndim > 2 else (),
                        device=G.device,
                        dtype=G.dtype,
                    )
                    try:
                        output_grad_scaling_matrix = torch.linalg.cholesky(G, upper=True).float()
                    except Exception as e:
                        print("Warning: eigen output_grad_distribution is not positive!")
                        G = (G + G.transpose(-1, -2)) / 2
                        eigenvalues = torch.linalg.eigvalsh(G) # [..., D]
                        min_eig = eigenvalues[..., 0] # 每个批里的最小特征值
                        eps = 1e-6
                        shift = (-min_eig + eps).clamp_min(0.0) # 只对非正定的批次加抖动
                        diag_shift = shift
                        I = torch.eye(G.size(-1), device=G.device, dtype=G.dtype)  # [D, D]，可广播到批维
                        G = G + shift[..., None, None] * I               # [..., D, D]
                        output_grad_scaling_matrix = torch.linalg.cholesky(G, upper=True).float()
                        del eigenvalues, min_eig, eps, shift, I
                    if grad_debug_payload is not None:
                        grad_debug_payload[name] = debug_utils.grad_calib_debug_pack(
                            raw_cpu_double=G.detach().cpu(),
                            chol_cpu=output_grad_scaling_matrix.detach().cpu(),
                            normalization_count=max(getattr(module, "_debug_token_count", 0), 1),
                            sample_count=getattr(module, "_debug_sample_count", 0),
                            token_count=getattr(module, "_debug_token_count", 0),
                            hook_count=getattr(module, "_debug_hook_count", hook_trigger_count.get(name, 0)),
                            diag_shift_cpu=diag_shift.detach().cpu(),
                        )
                        empirical_cfg = getattr(module, "_empirical_fisher_cfg", None)
                        if empirical_cfg is not None:
                            grad_debug_payload[name]["empirical_fisher_debug"] = debug_utils.empirical_fisher_debug_pack(empirical_cfg)
                    module.output_grad_scaling_matrix = output_grad_scaling_matrix.cpu()
                    module.output_grad_distribution = None
                    del G, output_grad_scaling_matrix
                    torch.cuda.empty_cache()
                else:
                    # Hook never accumulated (e.g. all batches failed); clear placeholder
                    module.output_grad_distribution = None
                if hasattr(module, "_debug_sample_count"):
                    del module._debug_sample_count
                if hasattr(module, "_debug_token_count"):
                    del module._debug_token_count
                if hasattr(module, "_debug_hook_count"):
                    del module._debug_hook_count
                if hasattr(module, "_empirical_fisher_input_proj"):
                    del module._empirical_fisher_input_proj
                if hasattr(module, "_empirical_fisher_cfg"):
                    del module._empirical_fisher_cfg
def _loss_aware_block_diag(matrices):
    if not matrices:
        raise ValueError("Expected at least one matrix for block diagonal construction")
    first = matrices[0]
    if first.ndim == 2:
        return torch.block_diag(*matrices)
    if first.ndim == 3:
        blocks = []
        for head_idx in range(first.shape[0]):
            blocks.append(torch.block_diag(*[matrix[head_idx] for matrix in matrices]))
        return torch.stack(blocks, dim=0)
    raise ValueError(f"Unsupported matrix ndim for block diagonal construction: {first.ndim}")


def _loss_aware_output_grad_matrix(module, device):
    if module is None or not hasattr(module, "output_grad_scaling_matrix"):
        raise ValueError("Missing output_grad_scaling_matrix for loss-aware whitening")
    return module.output_grad_scaling_matrix.to(device)


def _loss_aware_output_grad_inv(output_grad_scaling_matrix, cache_owner=None):
    if cache_owner is not None and hasattr(cache_owner, "output_grad_scaling_matrix_inv"):
        cached = cache_owner.output_grad_scaling_matrix_inv
        if isinstance(cached, torch.Tensor):
            return cached.to(output_grad_scaling_matrix.device)
    output_grad_scaling_matrix_inv = torch.linalg.inv(output_grad_scaling_matrix.float())
    if cache_owner is not None:
        cache_owner.output_grad_scaling_matrix_inv = output_grad_scaling_matrix_inv.detach().cpu()
    return output_grad_scaling_matrix_inv


def _loss_aware_inv_transpose(output_grad_scaling_matrix_inv):
    return output_grad_scaling_matrix_inv.transpose(-1, -2)


def _apply_loss_aware_weight_whitening(weight, output_grad_scaling_matrix):
    return output_grad_scaling_matrix.float() @ weight


def _apply_loss_aware_grad_whitening(grad, output_grad_scaling_matrix_inv):
    output_grad_scaling_matrix_inv_t = _loss_aware_inv_transpose(output_grad_scaling_matrix_inv).to(
        grad.device
    ).to(torch.bfloat16)
    return output_grad_scaling_matrix_inv_t @ grad
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


def get_layer_importance(grad, args):
    if args.taylor_order == 1:
        return -1.0 * grad
    elif args.taylor_order == 2:
        return grad.pow(2)
    elif args.taylor_order == 12:
        return -2.0 * grad + grad.pow(2)
def get_top_k_scores(grad_scores_dict, k_list, keys=['k_proj'], is_per_head_svd=False, mixed_qkv_budget=False):
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
    top_indices = {key: [] for key in keys}
    top_scores = {key: [] for key in keys}
    layer_indices_dict = {}
    assert len(keys) == len(k_list), "keys and k_list must have the same length" # [FIXED: num * [v] -> [num * v]]
    should_mix_qkv = mixed_qkv_budget and set(keys) == {'q_proj', 'k_proj', 'v_proj'}

    if should_mix_qkv:
        shared_scores = []
        shared_k = 0
        selected_counts_by_proj = {key: 0 for key in keys}
        for key, k in zip(keys, k_list):
            head_num = None
            if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
                logging.info("added per head score here")
                for layer_idx, scores in grad_scores_dict.items():
                    if key not in scores:
                        print(f"Layer {layer_idx} has no {key} grad scores")
                        continue
                    layer_num = int(layer_idx.split('_')[1])
                    score = scores[key]
                    if head_num is None:
                        head_num = score.shape[0]
                    for head_idx in range(score.shape[0]):
                        for i, score_ in enumerate(score[head_idx]):
                            shared_scores.append((key, layer_num, i, head_idx, score_.item()))
            else:
                for layer_idx, scores in grad_scores_dict.items():
                    if key not in scores:
                        print(f"Layer {layer_idx} has no {key} grad scores")
                        continue
                    layer_num = int(layer_idx.split('_')[1])
                    for i, score in enumerate(scores[key]):
                        shared_scores.append((key, layer_num, i, None, score.item()))

            if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
                logging.info(f'{key} original for per head svd is {k}')
                if head_num is None:
                    head_num = 1
                    logging.warning("head_num is undefined for key=%s, fallback to 1.", key)
                scaled_k = k * head_num
                logging.info(f'scale {key} by head_num: {head_num} to k: {scaled_k}')
                shared_k += scaled_k
            else:
                shared_k += k

        if len(shared_scores) == 0:
            logging.warning("No grad scores collected for mixed qkv selection, skip top-k selection.")
            return top_indices, top_scores, layer_indices_dict

        shared_scores.sort(key=lambda x: x[-1], reverse=True)
        shared_k = min(shared_k, len(shared_scores))
        logging.info("Using mixed qkv budget with shared_k=%s", shared_k)
        top_k = shared_scores[:shared_k]

        for key, layer_idx, sv_idx, head_idx, score in top_k:
            selected_counts_by_proj[key] += 1
            top_scores[key].append(score)
            if head_idx is not None:
                top_indices[key].append((layer_idx, sv_idx, head_idx))
                if layer_idx not in layer_indices_dict:
                    layer_indices_dict[layer_idx] = {}
                if key not in layer_indices_dict[layer_idx]:
                    layer_indices_dict[layer_idx][key] = {}
                if head_idx not in layer_indices_dict[layer_idx][key]:
                    layer_indices_dict[layer_idx][key][head_idx] = []
                layer_indices_dict[layer_idx][key][head_idx].append(sv_idx)
            else:
                top_indices[key].append((layer_idx, sv_idx))
                if layer_idx not in layer_indices_dict:
                    layer_indices_dict[layer_idx] = {}
                if key not in layer_indices_dict[layer_idx]:
                    layer_indices_dict[layer_idx][key] = []
                layer_indices_dict[layer_idx][key].append(sv_idx)

        logging.info(
            "Mixed qkv selection counts: q=%s, k=%s, v=%s",
            selected_counts_by_proj.get('q_proj', 0),
            selected_counts_by_proj.get('k_proj', 0),
            selected_counts_by_proj.get('v_proj', 0),
        )
        return top_indices, top_scores, layer_indices_dict

    for key, k in zip(keys, k_list):
        all_scores = []
        head_num = None
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            logging.info("added per head score here")
            for layer_idx, scores in grad_scores_dict.items():
                if key not in scores:
                    print(f"Layer {layer_idx} has no {key} grad scores")
                    continue
                layer_num = int(layer_idx.split('_')[1])
                score = scores[key]
                if head_num is None:
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

        if len(all_scores) == 0:
            logging.warning("No grad scores collected for key=%s, skip top-k selection.", key)
            continue

        # Sort and get top-k
        all_scores.sort(key=lambda x: x[-1], reverse=True)
        # per-head (heterogeneous) rank allocation only
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            logging.info(f'{key} original for per head svd is {k}')
            if head_num is None:
                head_num = 1
                logging.warning("head_num is undefined for key=%s, fallback to 1.", key)
            k = k * head_num
            logging.info(f'scale {key} by head_num: {head_num} to k: {k}')
        k = min(k, len(all_scores))
        top_k = all_scores[:k]

        # Store top indices and scores
        if is_per_head_svd and key in ['k_proj', 'q_proj', 'v_proj']:
            top_indices[key] = [(layer, idx, head_idx) for layer, idx, head_idx, _ in top_k]
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
def prepare_fuse_svd(layers, args):
    print("Preprocessing QKV layer SVD decomposition...")
    device = DEVICE
    alpha = args.act_alpha
    svd_debug_payload = None
    svd_debug_S = None

    def _record_svd_debug(*args, **kwargs):
        return

    for idx, layer in enumerate(layers):
        if args.svd_modules in ['qkv', 'attn', 'all']:
            q_linear = layer.self_attn.q_proj
            k_linear = layer.self_attn.k_proj
            v_linear = layer.self_attn.v_proj
            # Handle separate Q, K, V
            for proj_name, proj_layer in [('q_proj', q_linear), ('k_proj', k_linear), ('v_proj', v_linear)]:
                try:
                    if args.is_per_head_svd:
                        if proj_name == 'q_proj':
                            num_heads = layer.self_attn.config.num_attention_heads
                        else:
                            num_heads = layer.self_attn.config.num_key_value_heads
                        w = proj_layer.weight.data.view(num_heads, -1, proj_layer.in_features).float().to(DEVICE)
                    else:
                        w = proj_layer.weight.data.float().to(DEVICE)
                    w_before = w.detach().clone()
                    input_inverse = None
                    if args.act_aware:
                        scaling_diag_matrix = torch.ones(proj_layer.in_features, device=device)
                        if hasattr(k_linear, "scaling_diag_matrix") and k_linear.scaling_diag_matrix is not None:
                            scaling_diag_matrix *= k_linear.scaling_diag_matrix.to(device)**alpha
                            scaling_diag_matrix += 1e-6
                            input_inverse = 1.0 / scaling_diag_matrix
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
                            input_inverse = scaling_matrix_inv
                            w = w @ scaling_diag_matrix.float()

                    output_grad_scaling_matrix_inv = None
                    if args.loss_aware:
                        output_grad_scaling_matrix = _loss_aware_output_grad_matrix(proj_layer, device)
                        w = _apply_loss_aware_weight_whitening(w, output_grad_scaling_matrix)
                        output_grad_scaling_matrix_inv = _loss_aware_output_grad_inv(
                            output_grad_scaling_matrix,
                            cache_owner=proj_layer,
                        )

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
                    if args.loss_aware:
                        proj_layer.svd_info['output_grad_scaling_matrix_inv'] = output_grad_scaling_matrix_inv.cpu()
                    _record_svd_debug(
                        idx,
                        proj_name,
                        w_before,
                        w,
                        S,
                        output_grad_inv=output_grad_scaling_matrix_inv,
                        input_inverse=input_inverse,
                    )
                    print(f"Layer {idx} {proj_name} SVD completed, S shape: {S.shape}")
                except Exception as e:
                    print(f"Layer {idx} {proj_name} SVD failed: {e}")
                    import traceback
                    traceback.print_exc()
def calib_grad_info_text(model, layers, windows, args):
    raw_fisher_trace_debug = False
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
    raw_fisher_trace_cache = None

    model.train()
    for name, param in model.named_parameters():
        if _is_lm_layer_name(name):
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['all', 'qkv', 'attn']:
                param.requires_grad = True
            elif ('k_proj' in name or 'v_proj' in name) and args.svd_modules in ['kv']:
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
    for batch_idx, w in enumerate(windows):
        input_ids = w.to(device)
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
                                if raw_fisher_trace_debug:
                                    trace_heads = layer.self_attn.config.num_attention_heads if proj_name == 'q_proj' else layer.self_attn.config.num_key_value_heads
                                    _accumulate_raw_fisher_trace_by_head(
                                        raw_fisher_trace_cache,
                                        idx,
                                        proj_name,
                                        proj,
                                        trace_heads,
                                    )
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
                                if args.loss_aware:
                                    output_grad_scaling_matrix_inv = svd_info['output_grad_scaling_matrix_inv'].to(device)
                                    grad_cat = _apply_loss_aware_grad_whitening(grad_cat, output_grad_scaling_matrix_inv)

                                U = svd_info['U'].to(device).to(torch.bfloat16)
                                V = svd_info['V'].to(device).to(torch.bfloat16)
                                if args.is_per_head_svd:
                                    S_grad = torch.sum(U * (grad_cat @ V), dim=1) # n, r
                                else:
                                    S_grad = torch.sum(U * (grad_cat @ V), dim=0)

                                if args.is_taylor:
                                    S = svd_info['S'].to(device).to(torch.bfloat16)
                                    S_grad_squared = taylor_like_s_grad_squared(S, S_grad, args.taylor_order, args.add_taylor_first)
                                else:
                                    S_grad_squared = get_layer_importance(S_grad,  args)

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
def svd_qkv_with_grad_info(layers, args):
    grad_alpha = args.grad_alpha
    raw_fisher_trace_debug = False
    grad_debug_payload = None
    selected_k_map = {}

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
                    print(f"Layer {idx} {proj_name} importance score computed, shape: {importance_score.shape}")
                    if args.is_rank_allocate_ft:
                        del proj.svd_info
                else:
                    print(f"Warning: Layer {idx} {proj_name} lacks necessary SVD information or gradient information, cannot compute importance score")

    num_layers = len(layers)

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
        top_indices, top_scores, layer_indices_dict = get_top_k_scores(
            grad_scores_dict,
            k_list=k_value,
            keys=['q_proj', 'k_proj', 'v_proj'],
            is_per_head_svd=args.is_per_head_svd,
            mixed_qkv_budget=getattr(args, "truncate_mixed_qkv_budget", False),
        )
        selected_k_map = dict(zip(['q_proj', 'k_proj', 'v_proj'], k_value))

    for key, indices in top_indices.items():
        logging.info(f"Selected top {len(indices)} important singular values for {key}")
    return top_indices, top_scores, layer_indices_dict
total_rank_sum = {
    'q_proj': 0,
    'v_proj': 0,
    'k_proj': 0,
    'o_proj': 0,
    'up_proj': 0,
    'gate_proj': 0,
    'down_proj': 0,
}
total_linear_count = {
    'q_proj': 0,
    'v_proj': 0,
    'k_proj': 0,
    'o_proj': 0,
    'up_proj': 0,
    'gate_proj': 0,
    'down_proj': 0,
}
def _match_module_weight_dtype(module, x):
    if hasattr(module, "quantizer") and hasattr(module, "module"):
        return x
    weight = getattr(module, "weight", None)
    if (
        weight is not None
        and x.is_floating_point()
        and weight.is_floating_point()
        and x.dtype != weight.dtype
    ):
        return x.to(weight.dtype)
    return x
def _copy_cached_calibration_stats(source_module, target_module, module_name=None):
    if source_module is None or target_module is None:
        return target_module
    for attr in ["scaling_diag_matrixS", "output_grad_scaling_matrix"]:
        if hasattr(source_module, attr):
            value = getattr(source_module, attr)
            if value is not None:
                setattr(target_module, attr, value.detach().cpu() if isinstance(value, torch.Tensor) else value)
    if module_name is not None:
        setattr(target_module, "gptsvd_source_name", module_name)
    return target_module
class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", had_K=None, K=-1, had_mode='hadamard', is_ma_hack=False, original_linear=None, topk=1, is_per_head_svd=False, rank_split=None) -> None:
        super().__init__()
        self.rank_split = rank_split
        if sigma_fuse == 'local_ft' and S is not None:
            raise ValueError(
                "local_ft SVDLinear expects already fused local-ft/QAT A/B weights with S=None; "
                "got non-None S, which usually means raw SVD info was saved after local-ft/QAT failed."
            )
        if is_per_head_svd:
            if bias is not None:
                self.bias = nn.Parameter(bias.contiguous())
            else:
                self.register_parameter("bias", None)
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
                        V_.append(V[i].t().mul(S[i].view(-1, 1)).contiguous())
                    V_ = torch.cat(V_, dim=0) # [Rank_sum, C_in]
                    self.BLinear = nn.Linear(V_.size(0), V_.size(1), bias=False)
                    self.BLinear.weight.data = V_
                    del V_
                elif sigma_fuse == 'local_ft':
                    V_ = []
                    for i in range(self.head_num):
                        u_i = U[i].contiguous()
                        v_i = V[i].contiguous()
                        rank_i = u_i.shape[-1]
                        if v_i.shape[0] != rank_i:
                            raise ValueError(
                                f"local_ft per-head V/Blinear must have shape (rank, hidden); "
                                f"got {tuple(v_i.shape)} for rank {rank_i}. "
                                "This usually means raw SVD V was used after local-ft/QAT failed."
                            )
                        self.ALinear.append(nn.Parameter(u_i))
                        V_.append(v_i)
                    V = torch.cat(V_, dim=0)
                    del V_
                    S = None
                    del S
                    self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
                    self.BLinear.weight.data = V
                else:
                    raise ValueError(f"Unsupported sigma_fuse in per head SVD: {sigma_fuse}")
            else:
                if sigma_fuse == 'local_ft':
                    rank = U.size(-1)
                    if V.size(1) != rank:
                        raise ValueError(
                            f"local_ft per-head V/Blinear must have shape (heads, rank, hidden); "
                            f"got {tuple(V.shape)} for rank {rank}. "
                            "This usually means raw SVD V was used after local-ft/QAT failed."
                        )
                    V = V.contiguous()
                    self.BLinear = nn.Linear(V.size(2), V.size(0)* V.size(1), bias=False)
                    # U: n, c, c(r)
                    # V:  n, c(r),C_in
                else:
                    self.BLinear = nn.Linear(V.size(1), V.size(2) * V.size(0), bias=False) # c_in, rank * n
                    # U: n, c, rank (c*n = C_out)
                    # S: n, rank 
                    # V: n, C_in, rank
                self.head_num = V.size(0)
                self.head_rank  = V.size(2)
                if sigma_fuse == "UV":
                    self.ALinear = nn.Parameter(U.mul(S.sqrt().unsqueeze(1)).contiguous())
                    self.BLinear.weight.data = V.mul(S.sqrt().unsqueeze(1)).transpose(1, 2).reshape(-1, V.size(1)).contiguous()
                elif sigma_fuse == "U":
                    self.ALinear = nn.Parameter(U.mul(S.unsqueeze(1)).contiguous())
                    self.BLinear.weight.data = V.transpose(1, 2).reshape(-1, V.size(1)).contiguous()
                elif sigma_fuse == "V":
                    self.ALinear = nn.Parameter(U.contiguous())
                    self.BLinear.weight.data = V.mul(S.unsqueeze(1)).transpose(1, 2).reshape(-1, V.size(1)).contiguous()
                elif sigma_fuse == "local_ft":
                    self.head_rank = V.size(1)
                    self.ALinear = nn.Parameter(U.contiguous())
                    self.BLinear.weight.data = V.reshape(-1, V.size(2)).contiguous()
                else:
                    raise ValueError(f"Unsupported sigma_fuse in per head SVD: {sigma_fuse}")
        else:
            self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)
            if bias is not None:
                self.ALinear.bias.data = bias
            if sigma_fuse == 'local_ft':
                self.BLinear = nn.Linear(V.size(1), V.size(0), bias=False)
            else:
                self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
            if sigma_fuse == "UV":
                self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
                self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
            elif sigma_fuse == "U":
                self.ALinear.weight.data = U.mul(S).contiguous()
                self.BLinear.weight.data = V.t().contiguous()
            elif sigma_fuse == "V":
                self.ALinear.weight.data = U.contiguous()
                self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
            elif sigma_fuse == "adaptive":
                eps = 1e-6
                scale = U.abs().max(dim=0).values/ (V.abs().max(dim=0).values + eps) # [c, r] -> [r]
                self.ALinear.weight.data = U.mul(S**(1/(scale+1))).contiguous()
                self.BLinear.weight.data = V.t().mul(S**(scale/(scale+1)).view(-1, 1)).contiguous()
            elif sigma_fuse == 'profile':
                self.ALinear.weight.data = U.mul(S).contiguous()
                self.BLinear.weight.data = V.t().contiguous()
                self.S = S
                self.U = U
                self.V = V
            elif sigma_fuse == 'local_ft':
                self.ALinear.weight.data = U
                self.BLinear.weight.data = V
            elif float(sigma_fuse) <=1.0 and float(sigma_fuse)>=0:
                self.ALinear.weight.data = U.mul(S**(1-float(sigma_fuse))).contiguous()
                self.BLinear.weight.data = V.t().mul((S**float(sigma_fuse)).view(-1, 1)).contiguous()
            else:
                raise RuntimeError(f"Error: unsupported sigma mode {sigma_fuse}")
        self.had_K = had_K
        self.K = K
        self.had_mode = had_mode

        # Add for collecting latent distribution attributes
        self.collect_latent = False
        self.alinear_hook_handle = None
        self.latent_stats = None

        # Add MA hack attributes
        self.is_ma_hack = is_ma_hack
        self.original_linear = original_linear
        self.topk = topk
        self.is_per_head_svd = is_per_head_svd

    @staticmethod
    def from_linear_with_grad(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        loss_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        singular_indices=None,
        seed=0,
        module_name='o_proj',
        is_ma_hack=False,
        topk=1,
        is_per_head_svd=False,
        is_q_headnum=False,
    ):

        device = DEVICE
        svd_info = linear.svd_info

        # Get SVD information
        U = svd_info['U'].to(device)
        S = svd_info['S'].to(device)
        V = svd_info['V'].to(device)

        # Use pre-calculated importance sorting results
        if singular_indices is not None:
            if is_per_head_svd:
                # U: n, c, c
                # S: n, c
                # V:  n, C_in, c 
                U_selected = []
                S_selected = []
                V_selected = []
                rank_split = []
                N_heads = U.shape[0]
                for head_idx in range(N_heads):
                    head_indices = singular_indices.get(head_idx, []) if isinstance(singular_indices, dict) else singular_indices[head_idx]
                    if len(head_indices) == 0:
                        head_indices = [0]
                        logging.warning(f"No singular index selected for head {head_idx}, fallback to top-1.")
                    U_selected.append(U[head_idx, :, head_indices])
                    S_selected.append(S[head_idx, head_indices])
                    V_selected.append(V[head_idx, :, head_indices])
                    rank = len(head_indices)
                    logging.info(f"Layer uses rank: {rank} on head {head_idx}")
                    total_rank_sum[module_name] += rank
                    rank_split.append(rank)
                total_linear_count[module_name] += 1

                if had_rank:
                    logging.info("Warning: head wise svd is not supported for had_rank yet")
            else:
                logging.info(f"Using {len(singular_indices)} pre-selected singular values")

                # Select important singular values and corresponding vectors
                U_selected = U[:, singular_indices]
                S_selected = S[singular_indices]
                V_selected = V[:, singular_indices]

                # Update rank to the number of selected singular values
                rank = len(singular_indices)

                if had_rank:
                    set_seed(seed)
                    if had_mode == 'hadamard':
                        K = 1
                        had_K = None
                    elif had_mode == 'rh':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                        K = rank    
                    elif had_mode == 'random':
                        had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                        K = rank

                # Update total rank sum and linear layer count
                total_rank_sum[module_name] += rank
                total_linear_count[module_name] += 1

                logging.info(f"Layer uses rank: {rank}")

            # Process activation-aware scaling
            scaling_diag_matrix = None
            if act_aware:
                try:
                    scaling_diag_matrix = linear.scaling_diag_matrix
                except:
                    scaling_diag_matrix = linear.scaling_diag_matrixS
            if act_aware and scaling_diag_matrix is not None:
                scaling_diag_matrix = scaling_diag_matrix.to(device)
                if scaling_diag_matrix.ndim == 1:
                    # ASVD
                    # One-dimensional vector, representing diagonal matrix diagonal elements
                    scaling_diag_matrix = scaling_diag_matrix**alpha
                    scaling_diag_matrix += 1e-6  # avoid zero division
                    if is_per_head_svd:
                        V_selected = [v_select / scaling_diag_matrix.view(-1, 1) for v_select in V_selected]
                    else:
                        V_selected = V_selected / scaling_diag_matrix.view(-1, 1)
                elif scaling_diag_matrix.ndim == 2:
                    # SVD-LLM
                    # Two-dimensional matrix, complete scaling matrix, need to right multiply inverse matrix
                    try:
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                    except RuntimeError as e:
                        logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
                        eps = 1e-6
                        scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
                        scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)

                    if is_per_head_svd:
                        V_selected = [scaling_diag_matrix_inv.T @ v_select for v_select in V_selected]
                    else:
                        V_selected = scaling_diag_matrix_inv.T @ V_selected

            if loss_aware:
                output_grad_scaling_matrix_inv = svd_info['output_grad_scaling_matrix_inv'].to(device)
                if is_per_head_svd:
                    U_selected = [output_grad_scaling_matrix_inv[head_idx].float() @ u_select for head_idx, u_select in enumerate(U_selected)]
                else:
                    U_selected = output_grad_scaling_matrix_inv.float() @ U_selected

            Us = [U_selected]
            Ss = [S_selected]
            Vs = [V_selected]

            # Process bias
            if linear.bias is not None:
                bias = linear.bias.data
            else:
                bias = None

            # Check NaN or Inf
            if not is_per_head_svd:
                for S in Ss:
                    if (S != S).any():
                        logging.info("nan in S")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                for U in Us:
                    if (U != U).any():
                        logging.info("nan in U")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                for V in Vs:
                    if (V != V).any():
                        logging.info("nan in V")
                        return (
                            nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                        )
                assert len(Us) == len(Ss) == len(Vs) == 1

            # Create new linear layer
            if had_rank:
                if is_per_head_svd:
                    logging.info("Warning: head wise svd is not supported for had_rank yet. Fallback to no had_rank.")
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                else:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, had_K, K, had_mode, is_ma_hack, linear if is_ma_hack else None, topk)
            else:
                if is_per_head_svd:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk, is_per_head_svd=is_per_head_svd, rank_split=rank_split)
                else:
                    new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, is_ma_hack=is_ma_hack, original_linear=linear if is_ma_hack else None, topk=topk)
            _copy_cached_calibration_stats(linear, new_linear, module_name=module_name)
            new_linear.to(linear.weight.dtype)
            return new_linear.cpu()

    def forward(self, inp):
        if self.is_ma_hack and self.original_linear is not None:
            # Compute both original and SVD outputs
            original_output = self.original_linear(inp)
            svd_output = self.BLinear(inp)
            svd_output = self.ALinear(svd_output)

            # Get top-k locations and values from original output
            original_abs = original_output.abs()
            topk_locs, topk_vals = self.get_topk_location(original_abs, self.topk)

            # Create a copy of SVD output to modify
            result = svd_output.clone()

            # Replace SVD output with top-k values from original output at corresponding locations
            for rank in range(abs(self.topk)):
                rank_str = str(rank)
                if rank_str in topk_locs and rank_str in topk_vals:
                    locations = topk_locs[rank_str]  # [bsz, 2] with (n_idx, c_idx)
                    values = topk_vals[rank_str]     # [bsz]

                    # Apply the replacement for each batch item
                    for bsz_idx in range(locations.shape[0]):
                        n_idx = locations[bsz_idx, 0].item()
                        c_idx = locations[bsz_idx, 1].item()
                        orig_val = original_output[bsz_idx, n_idx, c_idx]
                        print(f"replace result: {result[bsz_idx, n_idx, c_idx]} with {orig_val}")
                        # Replace the SVD output at the same location with original value
                        result[bsz_idx, n_idx, c_idx] = orig_val

            return result
        else:
            if self.is_per_head_svd:
                # per head SVD forward pass
                y = self.BLinear(inp)
                if self.rank_split is not None:


                    # y: (B, T, Rsum)
                    B, T, Rsum = y.shape
                    H = self.head_num
                    ranks = list(self.rank_split)
                    assert len(ranks) == H, f"len(rank_split)={len(ranks)} != head_num={H}"

                    # Sanity: weight ranks match
                    ranks_w = [w.shape[1] for w in self.ALinear]  # each w: (cout, r_h)
                    if ranks_w != ranks:
                        raise ValueError(f"Per-head rank mismatch: weight {ranks_w} vs split {ranks}")

                    Rmax = max(ranks)
                    cout = self.ALinear[0].shape[0]

                    # Split y into heads and pad to (B, T, H, Rmax)
                    y_split = torch.split(y, ranks, dim=-1)             # list of (B, T, r_h)
                    y_pad = y.new_zeros(B, T, H, Rmax)                  # (B,T,H,Rmax)
                    for h, (yh, r) in enumerate(zip(y_split, ranks)):
                        y_pad[:, :, h, :r] = yh

                    # Pad weights to (H, cout, Rmax)
                    w_pad = y.new_zeros(H, cout, Rmax)                  # same device/dtype as y
                    for h, (wh, r) in enumerate(zip(self.ALinear, ranks)):
                        # wh expected shape: (cout, r)
                        w_pad[h, :, :r] = wh

                    # Batched per-head matmul:
                    # (B,T,H,Rmax) x (H,cout,Rmax) -> (B,T,H,cout)
                    y = torch.einsum('bthr,hcr->bthc', y_pad, w_pad)

                    # Flatten heads: (B, T, H*cout)
                    y = y.reshape(B, T, H * cout)

                else:
                    if inp.dim() >= 1 and inp.shape[0] != 1 and not getattr(self, "_warned_per_head_batch", False):
                        logging.warning(
                            "Per-head SVD forward without rank_split currently assumes batch size 1; "
                            "observed input shape %s. Results for batch>1 may be incorrect.",
                            tuple(inp.shape),
                        )
                        self._warned_per_head_batch = True
                    y = y.view(-1, self.head_num, self.head_rank).transpose(0, 1) # (head_num, token, head_rank)
                    y = torch.matmul(y, self.ALinear.transpose(1, 2)) # (head_num, token, head_rank) @ (head_num, c, head_rank).T = (head_num, token, c) -> 
                    y = y.transpose(0, 1).flatten(1).unsqueeze(0)
                if self.bias is not None:
                    bias_view = (1,) * (y.dim() - 1) + (-1,)
                    y = y + self.bias.to(dtype=y.dtype).view(*bias_view)
                return y
            else:
                # Standard SVD forward pass
                out_dtype = inp.dtype if inp.is_floating_point() else None
                y = self.BLinear(_match_module_weight_dtype(self.BLinear, inp))
                if out_dtype is not None and y.dtype != out_dtype:
                    y = y.to(out_dtype)
                y = self.ALinear(_match_module_weight_dtype(self.ALinear, y))
                if out_dtype is not None and y.dtype != out_dtype:
                    y = y.to(out_dtype)
                return y
def rsetattr(obj, attr, value):
    """ Recursively set an attribute given a dotted path """
    pre, _, post = attr.rpartition('.')
    if pre:
        obj = rgetattr(obj, pre)  # Get the nested object first
    setattr(obj, post, value)

def rgetattr(obj, attr):
    """ Recursively get an attribute given a dotted path """
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj


def _unwrap_actquant_linear(module):
    if isinstance(module, ActQuantWrapper):
        return module.module
    return module


def _find_svd_candidate_layers(module):
    return find_qlayers(
        module, layers=[torch.nn.Linear, ActQuantWrapper]
    )
def lsvd_attention_decompose(model, layers, windows, args):
    calib_input_cholesky_text(model, layers, windows, args)
    calib_grad_distribution_text(model, layers, windows, args)
    calib_grad_info_text(model, layers, windows, args)
    top_indices, top_scores, layer_indices_dict = svd_qkv_with_grad_info(layers, args)
    rank, world_size = get_rank_and_world_size()
    for idx, layer in enumerate(layers):
        full = _find_svd_candidate_layers(layers[idx])
        for name, module in full.items():
            if ('q_proj' in name or 'k_proj' in name or 'v_proj' in name) and args.svd_modules in ['qkv', 'attn', 'all']:
                if 'q_proj' in name:
                    proj_name = 'q_proj'
                    if hasattr(layer.self_attn.k_proj, 'scaling_diag_matrixS'):
                        layer.self_attn.q_proj.scaling_diag_matrixS =  layer.self_attn.k_proj.scaling_diag_matrixS
                        layer.self_attn.v_proj.scaling_diag_matrixS =  layer.self_attn.k_proj.scaling_diag_matrixS
                    elif hasattr(layer.self_attn.k_proj, 'scaling_diag_matrix'):
                        layer.self_attn.q_proj.scaling_diag_matrix =  layer.self_attn.k_proj.scaling_diag_matrix
                        layer.self_attn.v_proj.scaling_diag_matrix =  layer.self_attn.k_proj.scaling_diag_matrix
                    proj = layer.self_attn.q_proj
                elif 'k_proj' in name:
                    proj_name = 'k_proj'
                    proj = layer.self_attn.k_proj
                elif 'v_proj' in name:
                    proj_name = 'v_proj'
                    proj = layer.self_attn.v_proj

                if not hasattr(proj, 'svd_info'):
                    logging.info(f"Layer {idx} {proj_name} has no pre-calculated SVD information, skipping")
                    continue

                if idx not in layer_indices_dict or proj_name not in layer_indices_dict[idx]:
                    logging.info(f"Layer {idx} {proj_name} is not in layer_indices_dict, skipping")
                    continue

                if rank == 0:
                    logging.info(f"Layer {idx} starting {proj_name} SVD, using {len(layer_indices_dict[idx][proj_name])} singular values")

                try:
                    new_linear = SVDLinear.from_linear_with_grad(
                        proj,
                        param_ratio=args.rank_ratio,
                        alpha=args.act_alpha,
                        act_aware=args.act_aware,
                        loss_aware=args.loss_aware,
                        rank_align=1.,
                        sigma_fuse=args.svd_mode,
                        had_rank=args.had_rank,
                        had_mode='random',
                        singular_indices=layer_indices_dict[idx][proj_name],
                        seed=args.seed,
                        module_name=proj_name,
                        is_per_head_svd=args.is_per_head_svd,
                        is_q_headnum=args.is_q_headnum,
                    )
                    rsetattr(layers[idx], f'self_attn.{proj_name}', new_linear)
                    if rank == 0:
                        logging.info(f"Layer {idx} {proj_name} SVD completed")
                except Exception as e:
                    logging.info(f"Layer {idx} {proj_name} SVD failed: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
_SVD_AWARE_MAX_TOKENS = 8192
def _get_hybrid_ffn_sigma_fuse(args) -> str:
    """Return the sigma split mode used by hybrid FFN SVD factors."""
    return getattr(args, "sigma_fuse", getattr(args, "svd_mode", "UV"))
def _compute_channel_proxy_errors(
    W: torch.Tensor,
    rank: int,
    w_bits: int,
    sym: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized proxy (svd_err, quant_err) for every output channel row in W.

    Returns two 1-D tensors of shape (n_rows,) on the same device as ``W``.
    """
    d_in = W.shape[1]
    norm_sq = W.square().sum(dim=1)

    frac_kept = min(1.0, rank / d_in)
    err_svd = norm_sq * (1.0 - frac_kept)

    if sym:
        n_levels = 2 ** (w_bits - 1) - 1
    else:
        n_levels = 2 ** w_bits - 1
    step = W.abs().amax(dim=1).clamp(min=1e-9) / n_levels
    err_quant = step.square() * (d_in / 12.0)
    return err_svd, err_quant


def _compute_channel_margins(
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    act_scale: torch.Tensor,
    svd_rank: int,
    w_bits: int,
    w_sym: bool,
) -> torch.Tensor:
    """Vectorized margin = weighted(quant_err - svd_err) for each FFN channel."""
    down_col_norm = W_down.norm(dim=0)
    down_col_norm = down_col_norm / (down_col_norm.mean() + 1e-6)

    act_scale = act_scale.to(W_gate.device, dtype=W_gate.dtype)
    Wg_scaled = W_gate * act_scale.unsqueeze(0)
    Wu_scaled = W_up * act_scale.unsqueeze(0)

    err_g_svd, err_g_q = _compute_channel_proxy_errors(Wg_scaled, svd_rank, w_bits, w_sym)
    err_u_svd, err_u_q = _compute_channel_proxy_errors(Wu_scaled, svd_rank, w_bits, w_sym)

    return down_col_norm * ((err_g_q + err_u_q) - (err_g_svd + err_u_svd))


def _prepare_svd_aware_calib_tokens(
    calib_inps: torch.Tensor,
    d_in: int,
    max_tokens: int = _SVD_AWARE_MAX_TOKENS,
) -> torch.Tensor:
    """Flatten and cap calibration tokens for svd_aware before device transfer."""
    X = calib_inps.float().reshape(-1, d_in)
    if X.shape[0] > max_tokens:
        X = X[:max_tokens]
    return X.contiguous()


def _flatten_calib_tokens(calib_inps: torch.Tensor, d_in: int) -> torch.Tensor:
    """Flatten calibration hidden states to 2-D tokens."""
    return calib_inps.float().reshape(-1, d_in).contiguous()


def _compute_act_scale_from_calib(calib_inps: torch.Tensor, d_in: int) -> torch.Tensor:
    """Compute per-input-dimension activation scale from calibration tensors."""
    X = _flatten_calib_tokens(calib_inps, d_in)
    return X.abs().mean(dim=0).contiguous() + 1e-6
def _quantize_rows(W: torch.Tensor, bits: int, sym: bool) -> torch.Tensor:
    """Per-row symmetric / asymmetric round-to-nearest quantization.

    Returns the de-quantized (float) result with the same shape as W.
    """
    if sym:
        maxq = 2 ** (bits - 1) - 1
        scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-5) / maxq
        q = torch.clamp(torch.round(W / scale), -(maxq + 1), maxq)
        return q * scale

    maxq = 2 ** bits - 1
    wmin = W.amin(dim=1, keepdim=True)
    wmax = W.amax(dim=1, keepdim=True)
    zeros = torch.zeros_like(wmin)
    wmin = torch.minimum(wmin, zeros)
    wmax = torch.maximum(wmax, zeros)
    both_zero = (wmin == 0) & (wmax == 0)
    wmin = torch.where(both_zero, torch.full_like(wmin, -1), wmin)
    wmax = torch.where(both_zero, torch.ones_like(wmax), wmax)
    scale = (wmax - wmin).clamp(min=1e-5) / maxq
    zero = (-wmin / scale).round()
    q = torch.clamp(torch.round(W / scale) + zero, 0, maxq)
    return scale * (q - zero)


def _fake_quantize_activation(
    x: torch.Tensor,
    bits: int,
    groupsize: int = -1,
    sym: bool = True,
    clip_ratio: float = 1.0,
) -> torch.Tensor:
    """Mirror ActQuantizer's per-token fake quantization for analysis.

    This intentionally returns de-quantized floating-point values, matching the
    runtime ActQuantWrapper path used during fake W/A quantization.
    """
    if bits is None or bits >= 16:
        return x
    if x.numel() == 0:
        return x

    x_dtype = x.dtype
    x_f = x.float()
    if sym:
        maxq = float(2 ** (bits - 1) - 1)
    else:
        maxq = float(2 ** bits - 1)

    if groupsize is not None and groupsize > 0:
        if x_f.shape[-1] % groupsize != 0:
            raise ValueError(
                f"activation groupsize {groupsize} does not divide last dim {x_f.shape[-1]}"
            )
        init_shape = x_f.shape
        grouped = x_f.reshape(-1, init_shape[-1] // groupsize, groupsize)
        xmax = grouped.amax(dim=-1, keepdim=True) * clip_ratio
        xmin = grouped.amin(dim=-1, keepdim=True) * clip_ratio
        if sym:
            xmax = torch.maximum(xmin.abs(), xmax)
            scale = xmax / maxq
            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
            q = torch.clamp(torch.round(grouped / scale), -(maxq + 1), maxq)
            return (q * scale).reshape(init_shape).to(x_dtype)

        both_zero = (xmin == 0) & (xmax == 0)
        xmin = torch.where(both_zero, torch.full_like(xmin, -1), xmin)
        xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)
        q = torch.clamp(torch.round(grouped / scale) + zero, 0, maxq)
        return (scale * (q - zero)).reshape(init_shape).to(x_dtype)

    flat = x_f.reshape(-1, x_f.shape[-1])
    zeros = torch.zeros(flat.shape[0], device=x_f.device, dtype=x_f.dtype)
    xmin = torch.minimum(flat.min(dim=1).values, zeros) * clip_ratio
    xmax = torch.maximum(flat.max(dim=1).values, zeros) * clip_ratio
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        scale = (xmax / maxq).unsqueeze(1)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(flat / scale), -(maxq + 1), maxq)
        return (q * scale).reshape_as(x_f).to(x_dtype)

    both_zero = (xmin == 0) & (xmax == 0)
    xmin = torch.where(both_zero, torch.full_like(xmin, -1), xmin)
    xmax = torch.where(both_zero, torch.ones_like(xmax), xmax)
    scale = ((xmax - xmin) / maxq).unsqueeze(1)
    zero = torch.round((-xmin).unsqueeze(1) / scale)
    q = torch.clamp(torch.round(flat / scale) + zero, 0, maxq)
    return (scale * (q - zero)).reshape_as(x_f).to(x_dtype)


def _power2_blocks(n: int) -> List[int]:
    """Greedy power-of-two decomposition for block-Hadamard rotations."""
    blocks: List[int] = []
    remaining = int(n)
    while remaining > 0:
        block = 1 << (remaining.bit_length() - 1)
        blocks.append(block)
        remaining -= block
    return blocks


def _hadamard_transform_power2_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Normalized Sylvester Hadamard transform on the last dim; dim must be pow2."""
    n = x.shape[-1]
    if n <= 1:
        return x
    y = x.reshape(-1, n).float()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        y0 = y[:, :, 0, :].clone()
        y1 = y[:, :, 1, :].clone()
        y[:, :, 0, :] = y0 + y1
        y[:, :, 1, :] = y0 - y1
        y = y.reshape(-1, n)
        h *= 2
    return (y.reshape(x.shape) / math.sqrt(n)).to(x.dtype)


def _block_hadamard_lastdim(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthogonal block-Hadamard rotation on the last dimension."""
    n = x.shape[-1]
    if n <= 1:
        return x
    out = x.clone()
    start = 0
    for block in _power2_blocks(n):
        end = start + block
        out[..., start:end] = _hadamard_transform_power2_lastdim(out[..., start:end])
        start = end
    return out


def _random_orthogonal_matrix(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Deterministic random orthogonal matrix for latent SVD reparameterization."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    q, r = torch.linalg.qr(torch.randn(size, size, generator=gen, dtype=torch.float32))
    # Fix QR sign ambiguity so the same seed is stable across LAPACK choices.
    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def _hybrid_ffn_latent_rotation_seed(args, layer_idx: int) -> int:
    base_seed = int(getattr(args, "seed", 0))
    layer_part = 0 if layer_idx is None or layer_idx < 0 else int(layer_idx)
    return base_seed + 1009 * layer_part + 7919


def _apply_svd_latent_rotation_to_factors(
    A: torch.Tensor,      # (d_out, rank)
    B: torch.Tensor,      # (rank, d_in)
    args,
    layer_idx: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply A <- A R^T and B <- R B without changing A @ B in FP."""
    if not bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False)):
        return A, B

    rank = A.shape[1]
    if rank <= 1:
        return A, B

    mode = getattr(args, "hybrid_ffn_svd_latent_rotate_mode", "hadamard")
    if mode == "hadamard":
        # Block-Hadamard is symmetric and orthogonal, supports arbitrary ranks,
        # and avoids materializing a dense rank x rank matrix.
        A_rot = _block_hadamard_lastdim(A)
        B_rot = _block_hadamard_lastdim(B.T).T
        return A_rot, B_rot

    if mode == "random":
        seed = _hybrid_ffn_latent_rotation_seed(args, layer_idx)
        R = _random_orthogonal_matrix(rank, A.device, A.dtype, seed)
        A_rot = A @ R.T
        B_rot = R @ B
        return A_rot, B_rot

    raise ValueError(f"Unsupported hybrid FFN SVD latent rotation mode: {mode}")


def _svd_factor_matrices_from_decomp(
    U: torch.Tensor,
    S: torch.Tensor,
    Vt: torch.Tensor,
    rank: int,
    sigma_fuse: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build concrete A/B matrices exactly matching SVDLinear's non-per-head modes."""
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    if sigma_fuse == "UV":
        return (
            U_r.mul(S_r.sqrt()).contiguous(),
            Vt_r.mul(S_r.sqrt().view(-1, 1)).contiguous(),
        )
    if sigma_fuse == "U" or sigma_fuse == "profile":
        return U_r.mul(S_r).contiguous(), Vt_r.contiguous()
    if sigma_fuse == "V":
        return U_r.contiguous(), Vt_r.mul(S_r.view(-1, 1)).contiguous()
    if sigma_fuse == "adaptive":
        eps = 1e-6
        V_r = Vt_r.T
        scale = U_r.abs().max(dim=0).values / (V_r.abs().max(dim=0).values + eps)
        return (
            U_r.mul(S_r ** (1 / (scale + 1))).contiguous(),
            Vt_r.mul((S_r ** (scale / (scale + 1))).view(-1, 1)).contiguous(),
        )

    try:
        sigma = float(sigma_fuse)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported sigma_fuse for hybrid FFN SVD analysis: {sigma_fuse}"
        ) from exc
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(
            f"Unsupported sigma_fuse for hybrid FFN SVD analysis: {sigma_fuse}"
        )
    return (
        U_r.mul(S_r ** (1 - sigma)).contiguous(),
        Vt_r.mul((S_r ** sigma).view(-1, 1)).contiguous(),
    )


def _apply_svd_latent_rotation_to_module(svd_mod: nn.Module, args, layer_idx: int) -> None:
    """Rotate the concrete SVDLinear A/B weights in-place, independent of sigma_fuse."""
    if not bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False)):
        return
    if not isinstance(getattr(svd_mod, "ALinear", None), nn.Linear):
        logging.warning(
            "[hybrid_ffn] layer %s: cannot apply latent rotation; ALinear is %s",
            layer_idx,
            type(getattr(svd_mod, "ALinear", None)).__name__,
        )
        return
    if not isinstance(getattr(svd_mod, "BLinear", None), nn.Linear):
        logging.warning(
            "[hybrid_ffn] layer %s: cannot apply latent rotation; BLinear is %s",
            layer_idx,
            type(getattr(svd_mod, "BLinear", None)).__name__,
        )
        return

    A = svd_mod.ALinear.weight.data.float()
    B = svd_mod.BLinear.weight.data.float()
    A_rot, B_rot = _apply_svd_latent_rotation_to_factors(A, B, args, layer_idx=layer_idx)
    svd_mod.ALinear.weight.data.copy_(A_rot.to(svd_mod.ALinear.weight.dtype))
    svd_mod.BLinear.weight.data.copy_(B_rot.to(svd_mod.BLinear.weight.dtype))
    logging.info(
        f"[hybrid_ffn] layer {layer_idx}: applied SVD latent "
        f"{getattr(args, 'hybrid_ffn_svd_latent_rotate_mode', 'hadamard')} "
        f"rotation (rank={A.shape[1]})"
    )
class HybridSVDQuantFFN(nn.Module):
    """Hybrid SVD-Quant gated FFN replacement for layer.mlp.

    After construction the module attributes are:
      gate_svd   – SVDLinear  (n_svd output channels of gate_proj)
      up_svd     – SVDLinear  (n_svd output channels of up_proj)
      gate_quant – ActQuantWrapper  (n_quant output channels of gate_proj)
      up_quant   – ActQuantWrapper  (n_quant output channels of up_proj)
      down_proj  – ActQuantWrapper  (full d_ffn input columns, permuted as perm)
      perm       – LongTensor [svd_idx | quant_idx]
      n_svd      – int
    """

    def __init__(
        self,
        gate_svd,
        up_svd,
        gate_quant,
        up_quant,
        down_proj,
        perm: torch.Tensor,
        n_svd: int,
        had_K=None,
        K: int = 1,
        fp32_had: bool = False,
    ):
        super().__init__()
        self.gate_svd   = gate_svd
        self.up_svd     = up_svd
        self.gate_quant = gate_quant
        self.up_quant   = up_quant
        self.down_proj  = down_proj
        self.register_buffer('perm', perm)
        self.n_svd = n_svd
        # Online Hadamard on FFN hidden dim (h = cat[h_svd, h_q]) before down_proj.
        # When enabled, W_down has H^{-1} pre-absorbed on its input side, so the
        # runtime Hadamard and the absorbed inverse cancel, restoring equivalence.
        self.online_had = had_K is not None
        self.K = K
        self.fp32_had = fp32_had
        if had_K is not None:
            self.register_buffer('had_K', had_K)
        else:
            self.register_buffer('had_K', None)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_in)
        Returns: (..., d_out)
        """
        # SVD group: low-rank projections
        g_svd = self.gate_svd(x)          # (..., n_svd)
        u_svd = self.up_svd(x)            # (..., n_svd)
        h_svd = F.silu(g_svd) * u_svd    # (..., n_svd)

        # Quant group: quantised weight projections
        g_q = self.gate_quant(x)          # (..., n_quant)
        u_q = self.up_quant(x)            # (..., n_quant)
        h_q = F.silu(g_q) * u_q          # (..., n_quant)

        # Concatenate in the same order as the permuted W_down columns
        h = torch.cat([h_svd, h_q], dim=-1)  # (..., d_ffn)

        # Optional online Hadamard on FFN hidden dim before down_proj.
        # Reduces outliers in down_proj input activations (mirrors standard QuaRot).
        if self.online_had:
            h_dtype = h.dtype
            if self.fp32_had:
                h = hadamard_utils.matmul_hadU_cuda(h.float(), self.had_K, self.K).to(h_dtype)
            else:
                h = hadamard_utils.matmul_hadU_cuda(h, self.had_K, self.K)

        # W_down already has columns ordered as [SVD | Quant]
        return self.down_proj(h)
def build_hybrid_ffn(
    layer,
    layer_idx: int,
    args,
    calib_inps: Optional[torch.Tensor] = None,
    layer_indices_dict: Optional[dict] = None,
    original_perm: Optional[torch.Tensor] = None,
    precomputed_group: Optional[dict] = None,
) -> HybridSVDQuantFFN:
    """Replace ``layer.mlp`` with a HybridSVDQuantFFN.

    Absorption of permutation matrices
    ------------------------------------
    The function applies two permutations and absorbs both into the weights
    so that no runtime reordering is needed:

      P   = original_perm  (optional; permutation already applied to the hidden
                            states by an upstream step, e.g. channel reordering
                            from QuaRot rotation – if None, the weights are
                            assumed to already be in the P-absorbed form)
      P_s = the grouping permutation we compute (svd_idx + quant_idx)

    Resulting absorbed weights (column-vector convention  y = W · x):

      Wa'   = P_s  @ Wa   @ P^T   (gate/up:  output rows grouped, input cols un-rotated)
      Wout' = P_s^T@ Wout               (down:     input cols un-grouped)
                                                (output side unchanged; next-layer P
                                                 absorbed by the downstream linear)

    When original_perm is None (default):
      P is the identity → Wa' = P_s @ Wa,  Wout' = Wout @ P_s^T
    which is exactly what the grouping already does, so no extra step is needed.

    Parameters
    ----------
    layer        : transformer decoder layer (has .mlp.{gate_proj,up_proj,down_proj})
    layer_idx    : index of this layer (for rank lookup / logging)
    args         : parsed argument namespace; relevant fields:
                     .hybrid_ffn_rank_ratio – optional rank ratio used only by hybrid_ffn
                     .rank_ratio    – fallback param ratio for SVD side when
                                      hybrid_ffn_rank_ratio is unset
                     .svd_ffn_ratio – fraction of d_ffn channels assigned to SVD group
                     .sigma_fuse    – how Σ is split between U and V  ('UV'/'U'/'V')
                     .act_aware     – bool; use activation scaling in SVD
                     .act_alpha     – float; scaling exponent
    calib_inps   : optional calibration activations fed to this layer (..., d_in)
    layer_indices_dict : optional {layer_idx: {'up_proj': [singular_idx, ...]}}
                         from loss-aware calibration (used as SVD rank selector)
    original_perm : optional 1-D LongTensor of length d_in.
                    If the upstream pipeline has permuted (but NOT yet absorbed into
                    weights) the model hidden states by index sequence `original_perm`,
                    pass it here so that the input-column side of gate/up is
                    un-permuted (absorb P) before the SVD/quant split.
                    In the standard QuaRot pipeline this is NOT needed because
                    rotation_utils already absorbs P into the weight matrices.
    """

    mlp        = layer.mlp
    gate_lin   = _unwrap(mlp.gate_proj)   # nn.Linear
    up_lin     = _unwrap(mlp.up_proj)
    down_aw    = mlp.down_proj             # ActQuantWrapper (already added)

    d_ffn = gate_lin.out_features
    d_in  = gate_lin.in_features
    d_out = _unwrap(down_aw).out_features

    # ── Determine n_svd ──────────────────────────────────────────────────
    svd_ffn_ratio = getattr(args, 'svd_ffn_ratio', 0.2)
    n_svd  = max(1, int(d_ffn * svd_ffn_ratio))
    n_svd  = min(n_svd, d_ffn - 1)          # at least one quant channel

    logging.info(
        f"[hybrid_ffn] layer {layer_idx}: d_ffn={d_ffn}, "
        f"n_svd={n_svd}, n_quant={d_ffn - n_svd}"
    )

    # ── Read weights (float for numerical ops) ────────────────────────────
    Wg = gate_lin.weight.data.float()   # (d_ffn, d_in)
    Wu = up_lin.weight.data.float()     # (d_ffn, d_in)
    Wd = _unwrap(down_aw).weight.data.float()  # (d_out, d_ffn)
    orig_dtype = gate_lin.weight.dtype

    # ── Absorb original_perm P (input-column side of gate / up) ──────────
    # If the hidden states were permuted as x' = x[P] but the weights have NOT
    # yet been updated to compensate, we must apply the inverse permutation to
    # the INPUT columns of gate/up so that Wg @ x' = Wg[:, P^{-1}] @ x.
    # For a permutation matrix P_inv = P^T: Wg_absorbed = Wg[:, inv_perm]
    # where inv_perm[P[i]] = i.
    if original_perm is not None:
        P_cpu = original_perm.cpu()
        assert P_cpu.shape[0] == d_in, (
            f"original_perm length {P_cpu.shape[0]} != d_in {d_in}"
        )
        # Build inverse permutation: inv[P[i]] = i
        inv_P = torch.empty_like(P_cpu)
        inv_P[P_cpu] = torch.arange(d_in)
        # Absorb: Wa' = Wa @ P^T  (i.e. reorder input columns by inv_P)
        Wg = Wg[:, inv_P]
        Wu = Wu[:, inv_P]
        logging.info(
            f"[hybrid_ffn] layer {layer_idx}: absorbed original_perm P "
            f"(input-column reorder, inv_perm[:5]={inv_P[:5].tolist()})"
        )

    # ── Channel selection (or load from precomputed groups) ──────────────
    if precomputed_group is not None:
        # Phase 2: use pre-computed channel assignments (no calibration needed).
        # If the requested svd_ffn_ratio differs from the stored ratio, dynamically
        # reslice using greedy_accept_order so different ratios can be tested from
        # a single Phase-1 JSON without re-running Phase 1.
        stored_n_svd = int(precomputed_group.get("n_svd", len(precomputed_group["svd_idx"])))
        if n_svd != stored_n_svd and "greedy_accept_order" in precomputed_group:
            gao      = precomputed_group["greedy_accept_order"]   # ordered list, len ≤ d_ffn
            n_quant  = d_ffn - n_svd
            all_ch   = list(range(d_ffn))
            if n_quant <= len(gao):
                quant_idx = gao[:n_quant]
            else:
                # More Quant channels requested than greedy explored; pad with
                # remaining channels (those never accepted → most SVD-friendly)
                accepted_set = set(gao)
                extra = [c for c in all_ch if c not in accepted_set]
                quant_idx = gao + extra[:n_quant - len(gao)]
            quant_set = set(quant_idx)
            svd_idx   = [c for c in all_ch if c not in quant_set]
            # Rebuild perm so SVD channels come first
            perm_list = svd_idx + quant_idx
            perm      = torch.tensor(perm_list, dtype=torch.long, device=Wg.device)
            rank      = None  # will be computed below from actual_n_svd
            actual_n_svd = len(svd_idx)
            logging.info(
                f"[hybrid_ffn] layer {layer_idx}: dynamic ratio reslice "
                f"(stored n_svd={stored_n_svd} → requested n_svd={n_svd}) – "
                f"SVD={actual_n_svd}, Quant={len(quant_idx)}"
            )
        else:
            svd_idx      = precomputed_group["svd_idx"]
            quant_idx    = precomputed_group["quant_idx"]
            perm         = torch.tensor(precomputed_group["perm"], dtype=torch.long, device=Wg.device)
            rank         = int(precomputed_group["svd_rank"])
            actual_n_svd = len(svd_idx)
            logging.info(
                f"[hybrid_ffn] layer {layer_idx}: loaded precomputed groups – "
                f"SVD={actual_n_svd}, Quant={len(quant_idx)}, rank={rank}"
            )
    else:
        # Single-pass or Phase 1+2 combined: compute SVD rank then run selector
        rank = _compute_svd_rank(n_svd, d_in, args, layer_idx,
                                  layer_indices_dict, key='up_proj')
        w_bits = int(getattr(args, 'w_bits', 4))
        w_sym  = not bool(getattr(args, 'w_asym', False))
        svd_idx, quant_idx, perm = select_svd_channels(
            Wg, Wu, Wd, n_svd,
            calib_inps = calib_inps,
            svd_rank   = rank,
            w_bits     = w_bits,
            w_sym      = w_sym,
        )
        actual_n_svd = len(svd_idx)
        logging.info(
            f"[hybrid_ffn] layer {layer_idx}: "
            f"SVD group={actual_n_svd} (cap={n_svd}), "
            f"Quant group={len(quant_idx)}, rank={rank}, "
            f"P_s perm[:5]={perm[:5].tolist()}"
        )

    # ── Apply grouping permutation P_s and slice ──────────────────────────
    # Absorbed formula (column-vector convention):
    #   Wa'   = P_s @ Wa          (output rows grouped by P_s)
    #   Wout' = Wout @ P_s^T      (input cols un-grouped so h=[SVD|Quant] feeds correctly)
    perm_cpu = perm.cpu()

    Wg_grp = Wg[perm_cpu, :]      # P_s @ Wg  →  (d_ffn, d_in)
    Wu_grp = Wu[perm_cpu, :]      # P_s @ Wu

    Wd_grp = Wd[:, perm_cpu]      # Wd @ P_s^T  →  (d_out, d_ffn)

    Wg_svd   = Wg_grp[:actual_n_svd, :]           # (actual_n_svd,  d_in)
    Wu_svd   = Wu_grp[:actual_n_svd, :]
    Wg_quant = Wg_grp[actual_n_svd:, :]           # (n_quant, d_in)
    Wu_quant = Wu_grp[actual_n_svd:, :]

    # Recompute rank for the actual (possibly smaller or dynamic-resliced) SVD group
    if actual_n_svd == 0:
        rank = 0
    elif rank is None or actual_n_svd != n_svd:
        rank = _compute_svd_rank(actual_n_svd, d_in, args, layer_idx,
                                  layer_indices_dict, key='up_proj')

    # ── Input whitening for SVD group (--hybrid_ffn_act_aware) ──────────────
    # Two modes (controlled by --hybrid_ffn_act_aware_mode):
    #   'chol': full Cholesky L of X^T X, sets scaling_diag_matrixS on gate/up
    #           _make_svd_linear applies W <- W @ L before SVD, absorbs L^{-1} into V
    #   'diag': per-feature RMS scale, sets scaling_diag_matrix on gate/up
    #           _make_svd_linear applies W <- W * scale before SVD, un-scales V (stable)
    _ffn_act_aware   = bool(getattr(args, 'hybrid_ffn_act_aware', False))
    _act_aware_mode  = getattr(args, 'hybrid_ffn_act_aware_mode', 'chol')
    _chol_S          = None
    _diag_scale_val  = None
    if _ffn_act_aware and calib_inps is not None:
        try:
            dev_c = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            _MAX_CALIB_TOK = 4096
            X_raw = calib_inps.reshape(-1, d_in)
            if X_raw.shape[0] > _MAX_CALIB_TOK:
                step = max(1, X_raw.shape[0] // _MAX_CALIB_TOK)
                X_raw = X_raw[::step][:_MAX_CALIB_TOK]
            X_d = X_raw.double().to(dev_c)
            if _act_aware_mode == 'diag':
                # Diagonal per-feature RMS scaling (no matrix inversion needed)
                var = (X_d ** 2).mean(dim=0)
                _diag_scale_val = var.clamp(min=1e-8).sqrt().cpu()
                gate_lin.scaling_diag_matrix = _diag_scale_val
                up_lin.scaling_diag_matrix   = _diag_scale_val
                logging.info(
                    f"[hybrid_ffn] layer {layer_idx}: computed diagonal whitening scale "
                    f"(X={X_d.shape}, mode=diag)"
                )
            else:
                # Full Cholesky (SVDLLM)
                H = (X_d.T @ X_d) / X_d.shape[0]
                damp = 0.01 * H.diagonal().mean().clamp(min=1e-8)
                H.diagonal().add_(damp)
                _chol_S = torch.linalg.cholesky(H).cpu()
                gate_lin.scaling_diag_matrixS = _chol_S
                up_lin.scaling_diag_matrixS   = _chol_S
                logging.info(
                    f"[hybrid_ffn] layer {layer_idx}: computed Cholesky whitening matrix "
                    f"(X={X_d.shape}, damp={damp:.2e})"
                )
                del H
            del X_d
            torch.cuda.empty_cache()
        except Exception as _ce:
            logging.warning(
                f"[hybrid_ffn] layer {layer_idx}: Whitening computation failed ({_ce}); "
                "falling back to raw SVD."
            )
            _ffn_act_aware = False

    _svd_fp16 = bool(getattr(args, 'hybrid_ffn_fp16_quant_group', False))
    gate_svd_mod = _make_svd_linear(Wg_svd, rank, args, gate_lin, layer_idx=layer_idx,
                                    act_aware_override=_ffn_act_aware, fp16=_svd_fp16)
    up_svd_mod   = _make_svd_linear(Wu_svd, rank, args, up_lin,   layer_idx=layer_idx,
                                    act_aware_override=_ffn_act_aware, fp16=_svd_fp16)

    # Clean up temporary whitening attributes to avoid confusion downstream
    if _chol_S is not None:
        for _lin in (gate_lin, up_lin):
            if hasattr(_lin, 'scaling_diag_matrixS'):
                del _lin.scaling_diag_matrixS
    if _diag_scale_val is not None:
        for _lin in (gate_lin, up_lin):
            if hasattr(_lin, 'scaling_diag_matrix'):
                del _lin.scaling_diag_matrix

    # ── Build quant sub-modules (gate_quant, up_quant) ───────────────────
    use_real_int8 = bool(getattr(args, 'hybrid_real_int8', False))
    use_fp16_quant_group = bool(getattr(args, 'hybrid_ffn_fp16_quant_group', False))
    if use_fp16_quant_group:
        gate_quant_mod = _make_fp16_linear(Wg_quant, d_in, gate_lin)
        up_quant_mod   = _make_fp16_linear(Wu_quant, d_in, up_lin)
        logging.info(f"[hybrid_ffn] layer {layer_idx}: quant group → FP16 (no quantization)")
    elif use_real_int8:
        gate_quant_mod = _make_real_int8_linear(Wg_quant, d_in, gate_lin)
        up_quant_mod   = _make_real_int8_linear(Wu_quant, d_in, up_lin)
        logging.info(f"[hybrid_ffn] layer {layer_idx}: quant group → real INT8")
    else:
        gate_quant_mod = _make_quant_linear(Wg_quant, d_in, gate_lin)
        up_quant_mod   = _make_quant_linear(Wu_quant, d_in, up_lin)

    # ── Write back permuted down_proj columns ────────────────────────────
    # Wout' = Wd @ P_s^T  (input columns now expect [SVD | Quant] ordered h)
    down_inner = _unwrap(down_aw)
    if use_real_int8:
        # Replace ActQuantWrapper down_proj with a real INT8 module
        down_int8 = Int8WeightLinear.from_weight(
            Wd_grp.to(orig_dtype),
            bias=down_inner.bias.data.clone() if down_inner.bias is not None else None,
        ).to(down_inner.weight.device if hasattr(down_inner, 'weight') else gate_lin.weight.device)
        _copy_runtime_hadamard_config(down_int8, down_aw)
        down_aw = down_int8
        logging.info(f"[hybrid_ffn] layer {layer_idx}: down_proj → real INT8")
    else:
        down_inner.weight.data = Wd_grp.to(orig_dtype)

    # ── Optional online Hadamard on FFN hidden dim ────────────────────────
    # When --hybrid_ffn_had is set, absorb H^{-1} into the input side of
    # down_proj and enable the corresponding runtime Hadamard in forward().
    # This mirrors the standard QuaRot online-Hadamard that was intentionally
    # skipped in rotate_mlp_output_hybrid_ffn (because the grouping permutation
    # and Hadamard don't commute when applied in the original order).  Here we
    # apply the Hadamard AFTER the concat step, so commutativity is not required.
    had_K_buf = None
    had_K_int  = 1
    fp32_had   = getattr(args, 'fp32_had', False)
    if getattr(args, 'hybrid_ffn_had', False):
        d_ffn_had = Wd_grp.shape[1]   # == n_svd + n_quant
        try:
            had_K_val, had_K_int = hadamard_utils.get_hadK(d_ffn_had)
            # Absorb H^{-1} (= H for Hadamard) into W_down's input dimension.
            W_ = down_inner.weight.data.float().to(DEVICE)
            W_ = hadamard_utils.matmul_hadU_cuda(W_, had_K_val, had_K_int)
            down_inner.weight.data = W_.to(orig_dtype).cpu()
            had_K_buf = had_K_val
            logging.info(
                f"[hybrid_ffn] layer {layer_idx}: absorbed H^{{-1}} into down_proj "
                f"(d_ffn={d_ffn_had}, K={had_K_int}); online Hadamard enabled"
            )
        except Exception as exc:
            logging.warning(
                f"[hybrid_ffn] layer {layer_idx}: --hybrid_ffn_had requested but "
                f"get_hadK({d_ffn_had}) failed ({exc}). Skipping Hadamard."
            )

    return HybridSVDQuantFFN(
        gate_svd   = gate_svd_mod,
        up_svd     = up_svd_mod,
        gate_quant = gate_quant_mod,
        up_quant   = up_quant_mod,
        down_proj  = down_aw,
        perm       = perm,
        n_svd      = actual_n_svd,
        had_K      = had_K_buf,
        K          = had_K_int,
        fp32_had   = fp32_had,
    )
def _unwrap(module):
    """Unwrap ActQuantWrapper → nn.Linear if needed."""
    if isinstance(module, ActQuantWrapper):
        return module.module
    return module


def _compute_svd_rank(
    n_svd: int,
    d_in: int,
    args,
    layer_idx: int,
    layer_indices_dict: Optional[dict],
    key: str = 'up_proj',
) -> int:
    """Compute the SVD rank for the SVD group.

    Priority:
    1. If loss-aware indices are available, use their length.
    2. If args.use_true_param_ratio, match param count of rank-r decomposition
       to param_ratio * (n_svd * d_in), where param_ratio prefers
       args.hybrid_ffn_rank_ratio over args.rank_ratio.
    3. Fall back to rank = int(param_ratio * min(n_svd, d_in)).
    """
    if (
        layer_indices_dict is not None
        and layer_idx in layer_indices_dict
        and key in layer_indices_dict[layer_idx]
        and layer_indices_dict[layer_idx][key]
    ):
        indices = layer_indices_dict[layer_idx][key]
        rank = len(indices)
        logging.info(f"[hybrid_ffn] layer {layer_idx}: using loss-aware rank={rank}")
        return max(1, rank)

    hybrid_param_ratio = getattr(args, 'hybrid_ffn_rank_ratio', None)
    if hybrid_param_ratio is None:
        param_ratio = float(getattr(args, 'rank_ratio', 0.5))
        ratio_src = 'rank_ratio'
    else:
        param_ratio = float(hybrid_param_ratio)
        ratio_src = 'hybrid_ffn_rank_ratio'

    if getattr(args, 'use_true_param_ratio', False):
        # r*(n_svd + d_in) = param_ratio * n_svd * d_in  →  solve for r
        r = int(param_ratio * n_svd * d_in / (n_svd + d_in))
    else:
        r = int(param_ratio * min(n_svd, d_in))

    r = max(1, min(r, min(n_svd, d_in)))
    logging.info(
        f"[hybrid_ffn] layer {layer_idx}: SVD rank={r} "
        f"(param_ratio={param_ratio}, source={ratio_src})"
    )
    return r


def _make_svd_linear(
    W: torch.Tensor,
    rank: int,
    args,
    ref_linear: nn.Linear,
    layer_idx: int = -1,
    act_aware_override: bool = None,
    fp16: bool = False,
):
    """Construct a SVDLinear from a weight slice W (n_out × d_in).

    act_aware_override
        When not None, overrides ``args.act_aware`` for this call only.
        Used by build_hybrid_ffn to enable SVDLLM-style Cholesky whitening
        (``--hybrid_ffn_act_aware``) independently of QKV ``--act_aware``.
    """

    # Determine effective act_aware for this call
    _act_aware = act_aware_override if act_aware_override is not None else bool(getattr(args, 'act_aware', False))

    device = W.device
    W_f = W.float()
    if W_f.shape[0] == 0 or rank <= 0:
        return _make_empty_linear(W.shape[1], W.shape[0], ref_linear)

    # Optional: activation-aware scaling of input features
    # Two modes:
    #   scaling_diag_matrix  – diagonal per-feature scale  (simple act_aware)
    #   scaling_diag_matrixS – full lower-triangular Cholesky L of X^T X (SVDLLM)
    act_scale = None
    use_cholesky = False
    scaling_diag = getattr(ref_linear, 'scaling_diag_matrix', None)
    scaling_diag_s = getattr(ref_linear, 'scaling_diag_matrixS', None)
    if _act_aware and scaling_diag is not None:
        act_scale = scaling_diag.float().to(device)
        W_f = W_f * act_scale.view(1, -1)
    elif _act_aware and scaling_diag_s is not None:
        S_mat = scaling_diag_s.float().to(device)
        W_f = (W_f.double() @ S_mat.double()).float()
        use_cholesky = True

    if torch.cuda.is_available():
        W_f = W_f.to(DEVICE)

    try:
        U, S, Vt = torch.linalg.svd(W_f, full_matrices=False)
    except Exception as e:
        logging.warning(f"[hybrid_ffn] SVD failed: {e}. Using identity SVDLinear.")
        fallback = nn.Linear(W.shape[1], W.shape[0], bias=False)
        fallback.weight.data = W.to(ref_linear.weight.dtype)
        return fallback

    U = U[:, :rank].to(device)       # (n_out, rank)
    S = S[:rank].to(device)          # (rank,)
    Vt = Vt[:rank, :].to(device)     # (rank, d_in)
    V = Vt.T.contiguous()            # (d_in, rank)
    W_f = W_f.to(device)

    # Undo activation scaling in V so that the SVDLinear represents the
    # original (un-whitened) weight factorisation.
    if act_scale is not None:
        # Simple diagonal case: W_whitened = W * scale → undo via V / scale
        V = V / act_scale.view(-1, 1)
    elif use_cholesky:
        # Cholesky case: W_whitened = W @ L → undo via V = L^{-T} @ V_whitened
        # (L is lower-triangular so L^{-T} = inv(L)^T)
        S_mat = scaling_diag_s.float().to(V.device)
        try:
            S_inv = torch.linalg.inv(S_mat)
            V = (S_inv.double().T @ V.double()).float()
        except Exception as e:
            logging.warning(
                f"[hybrid_ffn] layer {layer_idx}: Cholesky undo failed ({e}); "
                "V remains in whitened space — SVD reconstruction may be degraded."
            )

    sigma_fuse = _get_hybrid_ffn_sigma_fuse(args)
    if sigma_fuse == "local_ft":
        raise ValueError(
            "hybrid_ffn raw SVD path does not support sigma_fuse='local_ft'; "
            "use UV/U/V/profile/adaptive or a numeric sigma split."
        )
    svd_mod = SVDLinear(
        U=U, S=S, V=V,
        bias=None,
        sigma_fuse=sigma_fuse,
    )
    _apply_svd_latent_rotation_to_module(svd_mod, args, layer_idx=layer_idx)
    # Cast to original dtype
    dtype = ref_linear.weight.dtype
    svd_mod = svd_mod.to(device=device, dtype=dtype)
    if not fp16 and int(getattr(args, 'a_bits', 16)) < 16:
        # Hybrid FFN's SVD side is created after the original MLP wrappers are
        # discarded.  Match the existing per-head SVD convention: BLinear's
        # input quantizer handles x->A8, and BLinear's out_quantizer handles the
        # latent activation consumed by ALinear.
        if isinstance(svd_mod.BLinear, nn.Linear):
            svd_mod.BLinear = ActQuantWrapper(svd_mod.BLinear).to(device=device, dtype=dtype)
            svd_mod = svd_mod.to(device=device, dtype=dtype)
    return svd_mod


def _make_empty_linear(d_in: int, d_out: int, ref_linear: nn.Linear):
    """Return a zero-width projection used when a layer's SVD group is empty."""
    empty = nn.Linear(d_in, d_out, bias=False)
    empty = empty.to(ref_linear.weight.device, dtype=ref_linear.weight.dtype)
    return empty


def _make_quant_linear(W: torch.Tensor, d_in: int, ref_linear: nn.Linear):
    """Create an ActQuantWrapper around an nn.Linear with weight W."""

    n_out = W.shape[0]
    lin = nn.Linear(d_in, n_out, bias=(ref_linear.bias is not None))
    lin.weight.data = W.to(ref_linear.weight.dtype)
    if ref_linear.bias is not None:
        lin.bias = None   # bias handled at the down_proj level if present
    wrapper = ActQuantWrapper(lin)
    return wrapper.to(ref_linear.weight.device)
def _make_fp16_linear(W: torch.Tensor, d_in: int, ref_linear: nn.Linear):
    """Create a plain nn.Linear (FP16, no quantization) with weight W."""
    n_out = W.shape[0]
    lin = nn.Linear(d_in, n_out, bias=False)
    lin.weight.data = W.to(ref_linear.weight.dtype)
    return lin.to(ref_linear.weight.device)
def select_svd_channels_svd_aware(
    W_gate: torch.Tensor,          # (d_ffn, d_in)
    W_up:   torch.Tensor,          # (d_ffn, d_in)
    W_down: torch.Tensor,          # (d_out, d_ffn)
    n_svd:  int,
    calib_inps: torch.Tensor,      # (N, d_in) or (..., d_in)
    svd_rank: int = 0,
    w_bits: int = 8,
    w_sym: bool = True,
    a_bits: int = 16,
    a_groupsize: int = -1,
    a_sym: bool = True,
    a_clip_ratio: float = 1.0,
    latent_rotate_args = None,
    layer_idx: int = -1,
    chol_L: Optional[torch.Tensor] = None,     # (d_in, d_in) lower-triangular Cholesky of X^T X
    diag_scale: Optional[torch.Tensor] = None, # (d_in,) per-feature RMS scale (diagonal mode)
) -> Tuple[List[int], List[int], torch.Tensor, List[int]]:
    """**SVD-Aware** greedy channel selection.

    Core idea: start from a pure-SVD baseline and greedily move channels
    to the quantization group **only when** RTN quantization outperforms SVD
    on the FFN reconstruction MSE.  Channels where SVD is already good enough
    stay in the SVD group; only SVD-poor channels fall back to quantization.

    Uses RTN (round-to-nearest) weight quantization and optional ActQuantizer-
    style activation fake quantization, making Phase 1 consistent with a
    Phase 2 that also uses RTN and W8A8 runtime wrappers.

    Algorithm
    ---------
    1. Compute FP16 reference output ``y_ref``.
    2. Compute W-quantized SVD baseline ``y_svd``; set as starting point.
    3. Compute per-channel delta ``delta_h = h_rtn - h_svd`` (switching SVD→RTN).
    4. Order channels by SVD quality (worst SVD first) using closed-form error.
    5. Greedily accept channel i → Quant if it strictly reduces FFN MSE.
       Stop when ``n_quant = d_ffn - n_svd`` channels are assigned to Quant.

    Returns
    -------
    svd_idx, quant_idx, perm
    """
    d_ffn, d_in = W_gate.shape
    d_out = W_down.shape[0]
    assert 0 < n_svd < d_ffn
    assert calib_inps is not None, "svd_aware requires calib_inps"

    dev = W_gate.device
    Wg = W_gate.float().to(dev)
    Wu = W_up.float().to(dev)
    Wd = W_down.float().to(dev)

    X = _prepare_svd_aware_calib_tokens(calib_inps, d_in).to(dev)   # (N, d_in)
    N = X.shape[0]

    if svd_rank <= 0:
        svd_rank = max(1, int(0.5 * min(n_svd, d_in)))

    # ── 1. FP16 reference ────────────────────────────────────────────────
    with torch.no_grad():
        g_fp  = X @ Wg.T
        u_fp  = X @ Wu.T
        y_ref = (F.silu(g_fp) * u_fp) @ Wd.T          # (N, d_out)

    # Runtime ActQuantWrapper quantizes each linear's input independently.  The
    # gate/up quant branch and the SVD BLinear branch share the same MLP input.
    X_q = _fake_quantize_activation(
        X,
        bits=a_bits,
        groupsize=a_groupsize,
        sym=a_sym,
        clip_ratio=a_clip_ratio,
    )

    # ── 2. W8A8-SVD baseline (simulate Phase 2 quantized SVD factors) ───────
    # Phase 2 quantizes ALinear (U*S^0.5) and BLinear (V^T*S^0.5) to W8.
    # With hybrid SVD factor wrappers enabled, Phase 2 also quantizes the
    # BLinear input and the ALinear latent input to A8.
    #
    # When chol_L is provided (SVDLLM full-Cholesky, --hybrid_ffn_act_aware --hybrid_ffn_act_aware_mode chol):
    #   SVD is performed on the whitened weight W @ L, then Vt is un-whitened
    #   (Vt_orig = Vt_whitened @ L^{-1}) to get factor matrices in the original
    #   input space.  This matches Phase 2's _make_svd_linear behaviour.
    #
    # When diag_scale is provided (diagonal mode, --hybrid_ffn_act_aware --hybrid_ffn_act_aware_mode diag):
    #   SVD is performed on W * scale.view(1,-1), then Vt is un-whitened by
    #   simple element-wise division: Vt_orig = Vt_w / scale.view(1,-1).
    #   No matrix inversion needed — numerically stable.

    # Flags used both in Step 2 and Step 4 (margin ordering)
    _L_d: Optional[torch.Tensor] = None      # Cholesky L (double, on dev)
    _L_inv_d: Optional[torch.Tensor] = None  # L^{-1}    (double, on dev)
    _ds: Optional[torch.Tensor] = None       # diag scale (float, on dev)

    with torch.no_grad():
        if chol_L is not None:
            # Whiten weights: W_w = W @ L  (double for numerical stability)
            _L_d = chol_L.double().to(dev)
            try:
                _L_inv_d = torch.linalg.inv(_L_d)
            except Exception as _le:
                logging.warning(
                    f"[svd_aware] layer {layer_idx}: Cholesky L inversion failed ({_le}); "
                    "falling back to raw SVD for Phase-1 baseline."
                )
                _L_d, _L_inv_d = None, None

            if _L_d is not None:
                Wg_w = (Wg.double() @ _L_d).float()
                Wu_w = (Wu.double() @ _L_d).float()
                U_g, S_g, Vt_g_w = torch.linalg.svd(Wg_w, full_matrices=False)
                U_u, S_u, Vt_u_w = torch.linalg.svd(Wu_w, full_matrices=False)
                # Un-whiten: Vt_orig = Vt_whitened @ L^{-1}
                Vt_g = (Vt_g_w.double() @ _L_inv_d).float()
                Vt_u = (Vt_u_w.double() @ _L_inv_d).float()
                logging.debug(
                    f"[svd_aware] layer {layer_idx}: using Cholesky-whitened SVD baseline"
                )
            else:
                U_g, S_g, Vt_g = torch.linalg.svd(Wg, full_matrices=False)
                U_u, S_u, Vt_u = torch.linalg.svd(Wu, full_matrices=False)
        elif diag_scale is not None:
            # Diagonal whitening: W_w = W * scale.view(1,-1)
            _ds = diag_scale.float().to(dev)
            Wg_w = Wg * _ds.view(1, -1)
            Wu_w = Wu * _ds.view(1, -1)
            U_g, S_g, Vt_g_w = torch.linalg.svd(Wg_w, full_matrices=False)
            U_u, S_u, Vt_u_w = torch.linalg.svd(Wu_w, full_matrices=False)
            # Un-whiten: Vt_orig = Vt_w / scale  (element-wise, no matrix inversion)
            Vt_g = Vt_g_w / _ds.view(1, -1)
            Vt_u = Vt_u_w / _ds.view(1, -1)
            logging.debug(
                f"[svd_aware] layer {layer_idx}: using diagonal-scale-whitened SVD baseline"
            )
        else:
            U_g, S_g, Vt_g = torch.linalg.svd(Wg, full_matrices=False)
            U_u, S_u, Vt_u = torch.linalg.svd(Wu, full_matrices=False)

        sigma_fuse = (
            _get_hybrid_ffn_sigma_fuse(latent_rotate_args)
            if latent_rotate_args is not None else "UV"
        )
        # Construct the same concrete A/B factor matrices that Phase 2 SVDLinear
        # uses, then optionally rotate that latent basis before quantization.
        A_g, B_g = _svd_factor_matrices_from_decomp(U_g, S_g, Vt_g, svd_rank, sigma_fuse)
        A_u, B_u = _svd_factor_matrices_from_decomp(U_u, S_u, Vt_u, svd_rank, sigma_fuse)

        if latent_rotate_args is not None:
            A_g, B_g = _apply_svd_latent_rotation_to_factors(
                A_g, B_g, latent_rotate_args, layer_idx=layer_idx
            )
            A_u, B_u = _apply_svd_latent_rotation_to_factors(
                A_u, B_u, latent_rotate_args, layer_idx=layer_idx
            )

        # Apply W8 RTN quantization to factor matrices (matching Phase 2 rtn_fwrd)
        A_g_q = _quantize_rows(A_g, w_bits, w_sym)
        B_g_q = _quantize_rows(B_g, w_bits, w_sym)
        A_u_q = _quantize_rows(A_u, w_bits, w_sym)
        B_u_q = _quantize_rows(B_u, w_bits, w_sym)

        # Compute hidden activations via W8A8 factor chain:
        # x -> A8 -> BLinear(W8) -> latent -> A8 -> ALinear(W8)
        x_lat_g = X_q @ B_g_q.T                        # (N, rank)
        x_lat_g = _fake_quantize_activation(
            x_lat_g, bits=a_bits, groupsize=a_groupsize,
            sym=a_sym, clip_ratio=a_clip_ratio,
        )
        g_svd   = x_lat_g @ A_g_q.T                    # (N, d_ffn)
        x_lat_u = X_q @ B_u_q.T
        x_lat_u = _fake_quantize_activation(
            x_lat_u, bits=a_bits, groupsize=a_groupsize,
            sym=a_sym, clip_ratio=a_clip_ratio,
        )
        u_svd   = x_lat_u @ A_u_q.T                    # (N, d_ffn)
        h_svd = F.silu(g_svd) * u_svd                  # (N, d_ffn)
        h_svd_wd = h_svd
        y_svd = h_svd_wd @ Wd.T                        # (N, d_out)

    baseline_mse = float(((y_svd - y_ref) ** 2).mean().item())
    logging.info(
        f"[svd_aware] W{w_bits}A{a_bits}-SVD baseline MSE = {baseline_mse:.6f}"
    )

    # ── 3. RTN weight-quantized branch activations (per-channel delta) ───
    with torch.no_grad():
        Wg_q = _quantize_rows(Wg, w_bits, w_sym)
        Wu_q = _quantize_rows(Wu, w_bits, w_sym)

        g_q = X_q @ Wg_q.T
        u_q = X_q @ Wu_q.T
        h_q = F.silu(g_q) * u_q                        # (N, d_ffn)

        # delta_h[:, i] = h_rtn[:, i] - h_svd[:, i]  (switch SVD → RTN)
        delta_h = h_q - h_svd_wd

    # ── 4. Candidate ordering: SVD-worst first ────────────────────────────
    # Fix 1: When whitening is active, rank channels using whitened weights so
    # that the SVD quality ordering is consistent with the whitened SVD baseline
    # computed in Step 2.  Without this fix, raw-W margins would rank channels
    # differently from the whitened-SVD baseline, causing suboptimal selection.
    with torch.no_grad():
        if _L_d is not None:
            # Cholesky whitened ordering
            Wg_m = (Wg.double() @ _L_d).float()
            Wu_m = (Wu.double() @ _L_d).float()
            act_scale = (X.double() @ _L_d).float().abs().mean(dim=0) + 1e-6
        elif _ds is not None:
            # Diagonal whitened ordering
            Wg_m = Wg * _ds.view(1, -1)
            Wu_m = Wu * _ds.view(1, -1)
            act_scale = (X * _ds.view(1, -1)).abs().mean(dim=0) + 1e-6
        else:
            Wg_m = Wg
            Wu_m = Wu
            act_scale = X.abs().mean(dim=0) + 1e-6

    margins = _compute_channel_margins(
        Wg_m, Wu_m, Wd,
        act_scale=act_scale,
        svd_rank=svd_rank,
        w_bits=w_bits,
        w_sym=w_sym,
    )

    # Ascending: most negative margin = SVD error >> RTN error → try quant first
    candidate_order = torch.argsort(margins).tolist()

    # ── 5. Greedy swap: SVD → Quant (vectorised fast path) ───────────────
    # Instead of recomputing (N, d_out) trial outputs for every candidate,
    # we use the exact identity:
    #
    #   trial_mse = current_mse
    #             + 2/(N*d_out) * dot(delta_h[:,ch], score[:,ch])
    #             + mean(delta_h[:,ch]**2) * mean(Wd[:,ch]**2)
    #
    # where score = err_current @ Wd  (shape N × d_ffn).
    # When a channel is accepted we update score with a rank-1 correction:
    #   score += outer(delta_h[:,ch], Wd[:,ch] @ Wd)
    # This reduces per-candidate cost from O(N·d_out) to O(N).
    n_quant    = d_ffn - n_svd
    current_mse = baseline_mse
    quant_set: set = set()
    accepted = rejected = 0

    err_current = (y_svd - y_ref).contiguous()          # (N, d_out)
    score       = err_current @ Wd                      # (N, d_ffn)  – one matmul
    dh_sq       = (delta_h ** 2).mean(dim=0)            # (d_ffn,)
    wd_sq       = (Wd ** 2).mean(dim=0)                 # (d_ffn,)
    Nd_out      = float(N * d_out)

    # greedy_accept_order: channels in the order they were accepted into quant.
    # We run the full loop (no early stop) so the order can be reused to derive
    # any svd_ffn_ratio without re-running Phase 1.
    greedy_accept_order: List[int] = []

    for ch in candidate_order:
        dh_ch = delta_h[:, ch]                          # (N,)
        cross  = float((dh_ch @ score[:, ch]).item()) / Nd_out
        trial_mse = current_mse + 2.0 * cross + float(dh_sq[ch].item()) * float(wd_sq[ch].item())

        if trial_mse < current_mse:
            greedy_accept_order.append(ch)
            quant_set.add(ch)
            # rank-1 update to score: score += outer(dh_ch, Wd[:,ch] @ Wd)
            wd_cross = Wd[:, ch] @ Wd                   # (d_ffn,)
            score.add_(dh_ch.unsqueeze(1) * wd_cross.unsqueeze(0))
            current_mse = trial_mse
            accepted += 1
            logging.debug(f"[svd_aware] ch {ch:5d}: MSE → {trial_mse:.6f} ✓ → quant")
        else:
            rejected += 1

    del err_current, score, dh_sq, wd_sq

    improvement = baseline_mse - current_mse
    logging.info(
        f"[svd_aware] greedy done: {accepted} → quant / {rejected} kept SVD  "
        f"MSE {baseline_mse:.6f} → {current_mse:.6f}  "
        f"(Δ={improvement:+.6f}, rank={svd_rank}, W{w_bits}A{a_bits} RTN)"
    )

    # Use greedy_accept_order so any ratio can be derived from this single run.
    quant_idx = sorted(greedy_accept_order[:n_quant])
    svd_idx   = sorted(set(range(d_ffn)) - set(quant_idx))
    perm = torch.tensor(svd_idx + quant_idx, dtype=torch.long, device=dev)
    return svd_idx, quant_idx, perm, greedy_accept_order
def load_channel_groups(path: str) -> dict:
    """**Phase 2 helper**: Load pre-computed channel groups from a JSON file.

    Returns the groups dict with integer list fields restored.
    """
    import json
    with open(path, 'r') as f:
        groups = json.load(f)
    # Restore integer lists (JSON stores everything as native Python types)
    for layer_info in groups["layers"].values():
        layer_info["svd_idx"]   = [int(x) for x in layer_info["svd_idx"]]
        layer_info["quant_idx"] = [int(x) for x in layer_info["quant_idx"]]
        layer_info["perm"]      = [int(x) for x in layer_info["perm"]]
        layer_info["n_svd"]     = int(layer_info["n_svd"])
        layer_info["svd_rank"]  = int(layer_info["svd_rank"])
    logging.info(f"[hybrid_ffn] Loaded channel groups from {path} "
                 f"({len(groups['layers'])} layers)")
    return groups
def hybrid_ffn_rtn_targets(real_int8: bool = False, fp16_quant_group: bool = False) -> list:
    """Return the RTN fake-quant targets for a layer with HybridSVDQuantFFN.

    RTN fake-quant should quantize both parts of the hybrid FFN:
      - SVD side: the factorized A/B linear weights
      - Quant side: gate/up fake-quant linears and the full down_proj

    When ``real_int8=True``, quant-side modules have already been converted to
    ``Int8WeightLinear`` at setup time, so RTN should only touch the SVD side.

    When ``fp16_quant_group=True``, all FFN weights remain in FP16; return empty.
    """
    if fp16_quant_group:
        return []

    targets = [
        'mlp.gate_svd.ALinear',
        'mlp.gate_svd.BLinear',
        'mlp.up_svd.ALinear',
        'mlp.up_svd.BLinear',
    ]
    if real_int8:
        return targets

    return targets + [
        'mlp.gate_quant.module',
        'mlp.up_quant.module',
        'mlp.down_proj.module',
    ]
def _collect_layer_inputs(model, layers, windows):
    """Run the model up to each layer and collect hidden-state inputs.

    Returns a list of tensors, one per layer.  Each tensor has shape
    (N, T, d_in) where N = nsamples.
    """
    dev = DEVICE

    logging.info("[hybrid_ffn] Loading calibration data for layer-input collection …")

    # Hook-based capture: intercept the *input* to each MLP
    layer_inps = [[] for _ in layers]

    def make_hook(idx):
        def hook(module, inp, _out):
            # inp is a tuple; inp[0] is the hidden state with shape
            # (batch, seq, d_in). Sequence lengths vary across VLM samples, so
            # flatten tokens here and store a stable 2-D shape for concat.
            hidden = inp[0].detach().reshape(-1, inp[0].shape[-1]).cpu()
            layer_inps[idx].append(hidden)
        return hook

    handles = []
    for i, layer in enumerate(layers):
        mlp = layer.mlp
        handles.append(mlp.register_forward_hook(make_hook(i)))

    model.to(dev)
    model.eval()

    with torch.no_grad():
        for w in windows:
            model(w.to(dev), use_cache=False)

    for h in handles:
        h.remove()

    model.cpu()
    torch.cuda.empty_cache()

    # Concatenate samples per layer
    result = []
    missing_layers = []
    for i, inp_list in enumerate(layer_inps):
        if inp_list:
            result.append(torch.cat(inp_list, dim=0))  # (total_tokens, d_in)
        else:
            logging.warning(f"[hybrid_ffn] No calibration inputs captured for layer {i}")
            missing_layers.append(i)
            result.append(None)

    if missing_layers:
        raise RuntimeError(
            "[hybrid_ffn] Calibration failed: no layer inputs captured for "
            f"{len(missing_layers)}/{len(layers)} layers "
            f"(missing={missing_layers[:8]}{'...' if len(missing_layers) > 8 else ''})."
        )

    return result
def phase1_layer_groups(Wg, Wu, Wd, calib, args, idx):
    svd_ffn_ratio = getattr(args, 'svd_ffn_ratio', 0.2)
    sigma_fuse = _get_hybrid_ffn_sigma_fuse(args)
    w_bits = int(getattr(args, 'w_bits', 4))
    w_sym  = not bool(getattr(args, 'w_asym', False))
    a_bits = int(getattr(args, 'a_bits', 16))
    a_groupsize = int(getattr(args, 'a_groupsize', -1))
    a_sym = not bool(getattr(args, 'a_asym', False))
    a_clip_ratio = min(
        float(getattr(args, 'a_clip_ratio', 1.0)),
        float(getattr(args, 'lma_clip_ratio', 1.0)),
    )
    latent_rotate = bool(getattr(args, "hybrid_ffn_svd_latent_rotate", False))
    latent_rotate_mode = getattr(args, "hybrid_ffn_svd_latent_rotate_mode", "hadamard")

    calib_method = getattr(args, 'svd_ffn_calib_method', 'error_compare')
    analysis_dev = DEVICE
    d_ffn, d_in = Wg.shape
    grad = None
    act_scale = None
    n_svd = max(1, min(int(d_ffn * svd_ffn_ratio), d_ffn - 1))
    rank = _compute_svd_rank(n_svd, d_in, args, idx, None, key='up_proj')

    use_gpu_analysis = True
    if use_gpu_analysis:
        if calib_method == 'svd_aware' and calib is not None:
            calib = _prepare_svd_aware_calib_tokens(calib, d_in)
        elif calib_method == 'loss_calibration' and calib is not None:
            calib = _flatten_calib_tokens(calib, d_in)
        elif calib_method == 'gradient_loss':
            if calib is not None and grad is not None:
                calib, grad = _prepare_gradient_loss_inputs(calib, grad, d_in, Wd.shape[0])
            elif calib is not None:
                act_scale = _compute_act_scale_from_calib(calib, d_in)
        else:
            if calib is not None:
                act_scale = _compute_act_scale_from_calib(calib, d_in)

        Wg = Wg.to(analysis_dev, non_blocking=True)
        Wu = Wu.to(analysis_dev, non_blocking=True)
        Wd = Wd.to(analysis_dev, non_blocking=True)
        if calib is not None:
            calib = calib.to(analysis_dev, non_blocking=True)
        if grad is not None:
            grad = grad.to(analysis_dev, non_blocking=True)
        if act_scale is not None:
            act_scale = act_scale.to(analysis_dev, non_blocking=True)
    greedy_order = None   # populated only by svd_aware

    # ── Input whitening for Phase 1 (svd_aware + --hybrid_ffn_act_aware) ──
    # Compute the whitening statistics from calibration inputs so that
    # select_svd_channels_svd_aware uses the same whitened SVD quality metric
    # as Phase 2 (_make_svd_linear).  Two modes:
    #   'chol': full Cholesky L of X^T X  (SVDLLM, requires L^{-1})
    #   'diag': diagonal RMS scale         (stable, no matrix inversion)
    chol_L_layer = None
    diag_scale_layer = None
    _act_aware_mode = getattr(args, 'hybrid_ffn_act_aware_mode', 'chol')
    if (calib_method == 'svd_aware'
            and getattr(args, 'hybrid_ffn_act_aware', False)
            and calib is not None):
        try:
            _MAX_TOK_P1 = 4096
            X_raw = calib.reshape(-1, d_in)
            if X_raw.shape[0] > _MAX_TOK_P1:
                step = max(1, X_raw.shape[0] // _MAX_TOK_P1)
                X_raw = X_raw[::step][:_MAX_TOK_P1]
            X_d = X_raw.double().to(analysis_dev)
            if _act_aware_mode == 'diag':
                # Diagonal variance-based scaling
                var = (X_d ** 2).mean(dim=0)
                diag_scale_layer = var.clamp(min=1e-8).sqrt().cpu()
                logging.info(
                    f"[hybrid_ffn][analyze] layer {idx}: "
                    f"diagonal whitening scale computed (mode=diag)"
                )
            else:
                # Full Cholesky
                H = (X_d.T @ X_d) / X_d.shape[0]
                damp = 0.01 * H.diagonal().mean().clamp(min=1e-8)
                H.diagonal().add_(damp)
                chol_L_layer = torch.linalg.cholesky(H).cpu()
                logging.info(
                    f"[hybrid_ffn][analyze] layer {idx}: "
                    f"Cholesky whitening matrix computed for Phase-1 svd_aware"
                )
            del X_d
            torch.cuda.empty_cache()
        except Exception as _ce:
            logging.warning(
                f"[hybrid_ffn][analyze] layer {idx}: "
                f"Whitening computation failed ({_ce}); using raw SVD baseline."
            )
    assert calib_method == 'svd_aware'
    _whitening_label = (
        " + Cholesky whitening" if chol_L_layer is not None else
        " + diagonal whitening" if diag_scale_layer is not None else ""
    )
    logging.info(f"[hybrid_ffn][analyze] layer {idx}: using svd_aware (RTN){_whitening_label}")
    svd_idx, quant_idx, perm, greedy_order = select_svd_channels_svd_aware(
        Wg, Wu, Wd, n_svd,
        calib_inps  = calib,
        svd_rank    = rank,
        w_bits      = w_bits,
        w_sym       = w_sym,
        a_bits       = a_bits,
        a_groupsize  = a_groupsize,
        a_sym        = a_sym,
        a_clip_ratio = a_clip_ratio,
        latent_rotate_args = args,
        layer_idx  = idx,
        chol_L     = chol_L_layer.to(analysis_dev) if chol_L_layer is not None else None,
        diag_scale = diag_scale_layer.to(analysis_dev) if diag_scale_layer is not None else None,
    )

    layer_entry = {
        "svd_idx":   svd_idx,
        "quant_idx": quant_idx,
        "perm":      perm.tolist(),
        "n_svd":     len(svd_idx),
        "svd_rank":  rank,
        "d_ffn":     d_ffn,
        "d_in":      d_in,
    }
    if greedy_order is not None:
        # Full greedy acceptance order lets any svd_ffn_ratio be derived
        # offline via derive_ratio_groups.py without re-running Phase 1.
        layer_entry["greedy_accept_order"] = greedy_order
    return layer_entry
def hybrid_ffn_setup(model, args, layer_inps, groups, layer_indices_dict=None):
    layers = model.model.layers
    logging.info(f"[hybrid_ffn] Setting up {len(layers)} layers …")
    precomputed_groups = groups

    # ── Per-layer replacement ─────────────────────────────────────────────
    _skip_layers = set(getattr(args, 'hybrid_ffn_skip_layers', None) or [])
    if _skip_layers:
        logging.info(
            f"[hybrid_ffn] --hybrid_ffn_skip_layers: layers {sorted(_skip_layers)} "
            "will keep original (pure-quant) FFN — skipping HybridSVDQuantFFN."
        )

    for idx, layer in enumerate(layers):
        if idx in _skip_layers:
            # Restore the standard QuaRot online Hadamard on down_proj so the
            # skipped layer behaves identically to the pure-quant baseline.
            # The pre-loop code (main_llava_hybrid_ffn.py) disabled online_full_had
            # on ALL down_projs to let HybridSVDQuantFFN manage its own Hadamard.
            # For skipped layers we must re-enable it to match baseline behaviour.
            try:
                _down = layer.mlp.down_proj
                _gate = _unwrap(layer.mlp.gate_proj)
                d_ffn = _gate.weight.shape[0]   # intermediate_size
                _had_K, _had_int = hadamard_utils.get_hadK(d_ffn)
                _down.online_full_had = True
                _down.had_K  = _had_K
                _down.K      = _had_int
                _down.fp32_had = bool(getattr(args, 'fp32_had', False))
                logging.info(
                    f"[hybrid_ffn] layer {idx}: SKIPPED → keeping pure-quant FFN, "
                    f"restored online Hadamard on down_proj (d_ffn={d_ffn}, K={_had_int})"
                )
            except Exception as _he:
                logging.warning(
                    f"[hybrid_ffn] layer {idx}: SKIPPED → failed to restore Hadamard "
                    f"({_he}); down_proj may lack online Hadamard."
                )
            continue

        calib      = layer_inps[idx] if (layer_inps is not None) else None
        layer_info = (
            precomputed_groups["layers"].get(str(idx))
            if precomputed_groups is not None else None
        )

        try:
            hybrid = build_hybrid_ffn(
                layer              = layer,
                layer_idx          = idx,
                args               = args,
                calib_inps         = calib,
                layer_indices_dict = layer_indices_dict,
                precomputed_group  = layer_info,
            )
            layer.mlp = hybrid
            logging.info(f"[hybrid_ffn] layer {idx}: replaced mlp → HybridSVDQuantFFN")
        except Exception as e:
            logging.error(f"[hybrid_ffn] layer {idx}: FAILED ({e}), keeping original mlp")
            import traceback; traceback.print_exc()

    logging.info("[hybrid_ffn] Setup complete.")
