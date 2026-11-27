# Vendored reference implementation for tests only.
# Source: https://github.com/SAI-Lab-NYU/QSVD (commit 2a2d5ccd64ba775274ff78e8d325ebb52e77aff0), directory
# fake_quant/. Licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
# Vendored (line ranges refer to that commit):
#   utils.py 60-67 (set_seed)
#   quant_utils.py 8-66 (get_minq_maxq .. AsymSTEQuantize), 123-534 (ActQuantizer, ActQuantWrapper, WeightQuantizer),
#     559-603 (add_actquant, find_qlayers)
#   gptq_utils.py 646-686 (rtn_fwrd)
#   rotation_utils.py 17-33 (fuse_ln_linear), 1782-1848 (had_transform_rank, random_orthogonal_matrix,
#     get_orthogonal_matrix)
#   svd_utils.py 22-25 (get_rank_and_world_size), 27-94 (SVDLinear.__init__, apply_had_rank), 594-598
#     (SVDLinear.forward), 600-775 (from_linearqkv_with_grad, rsetattr, rgetattr), 948-1012 (qkv_fuse branch of
#     svd_lm_setup, inside qsvd_lm_decompose)
#   grad_info_utils.py 378-417 and 430-434 (gradient accumulation and normalisation, inside calib_grad_info_text),
#     448-650 (prepare_qkv_svd, svd_qkv_with_grad_info, get_top_k_scores)
#   act_aware_utils.py 52-54 (abs_mean hook, inside calib_input_distribution_text)
# Modifications to the vendored code (everything else is verbatim):
#   (1) utils.get_dev() and .cuda() are replaced by the module constant DEVICE = torch.device("cpu");
#   (2) get_orthogonal_matrix(size, mode, device=DEVICE): the QR runs on the CPU (official: on the GPU);
#   (3) prepare_qkv_svd, svd_qkv_with_grad_info and rtn_fwrd take the list of decoder layers instead of calling
#       model_utils.get_layers(model); cache-file reads and writes and torch.save are removed (the use_cache=False
#       path); the model_utils.get_layers calls inside the gradient accumulation block use the same layer list;
#   (4) new calib_input_distribution_text: the abs_mean hook of act_aware_utils.calib_input_distribution on every
#       nn.Linear whose name contains "model.layers", fed by model(w) per text window (official: model.generate on
#       VLM prompts with a Catcher that stops after the last MLP; same hook inputs);
#   (5) new calib_grad_info_text: the setup lines of calib_grad_info (prepare_qkv_svd, eval, train, requires_grad by
#       name), a text loss model(input_ids=w, labels=w)[0] per window (official: VLM batches), then the verbatim
#       accumulation block and normalisation;
#   (6) new qsvd_lm_decompose: calib_input_distribution_text, calib_grad_info_text, svd_qkv_with_grad_info, the
#       verbatim qkv_fuse branch of svd_lm_setup, then had_transform_rank (official order);
#   (7) ActQuantWrapper.forward: the CUDA online-Hadamard branches (full_had, full_hadnopow2, partial_had) raise
#       NotImplementedError; the pure-torch online_random_had branch is kept;
#   (8) imports of hadamard_utils, model_utils, data_utils, fisher_info_utils, tqdm, fast_hadamard_transform and the
#       VLM helpers are removed, as are the utils.cleanup_memory calls and the tqdm progress wrappers; the
#       Hadamard-based paths of get_orthogonal_matrix and SVDLinear.apply_had_rank (hadamard_utils) raise
#       NotImplementedError; references to rotation_utils., quant_utils. and svd_utils. module prefixes are dropped
#       because everything lives in this module.
# ruff: noqa
import logging
import math
import os
import random
import typing

import numpy as np
import torch
import torch.nn as nn
import transformers

DEVICE = torch.device("cpu")

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

    # def forward(self, x):
    #     x_dtype = x.dtype
    #     if self.bits == 16:
    #         return x
    #     elif self.sym:
    #         return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
    #     return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)

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
            # breakpoint()
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
        if self.online_full_had or self.online_full_hadnopow2 or self.online_partial_had:
            raise NotImplementedError('online Hadamard needs CUDA fast_hadamard_transform')

        if self.quantizer.bits < 16: #Quantize, if needed
            self.quantizer.find_params(x)
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

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








@torch.no_grad()
def rtn_fwrd(layers, dev, args, start_id=0):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    torch.cuda.empty_cache()

    quantizers = {}

    for i in range(start_id, len(layers)):
        layer = layers[i].to(dev)

        subset = find_qlayers(layer,
                                            layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.w_clip
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                next(iter(layer.parameters())).dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    return quantizers

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()

        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, 'bias'):
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def had_transform_rank(model):
    """
    Apply Hadamard transform to all suitable SVDLinear layers in the model
    
    Args:
        model: Model that needs Hadamard transform applied
    """
    
    print("Starting to apply Hadamard transform to the model...")
    
    # Find all SVDLinear layers in the model
    svd_modules = []
    for name, module in model.named_modules():
        if isinstance(module, SVDLinear):
            svd_modules.append((name, module))
    
    if not svd_modules:
        print("No SVDLinear layers found in the model, unable to apply Hadamard transform")
        return
    
    print(f"Found {len(svd_modules)} SVDLinear layers")
    
    # Apply Hadamard transform to each SVDLinear layer
    applied_count = 0
    for name, module in svd_modules:
        if hasattr(module, 'had_K') and module.had_K is not None:
            try:
                module.apply_had_rank()
                applied_count += 1
            except Exception as e:
                print(f"Error applying Hadamard transform to layer {name}: {e}")
    
    # Print statistics of Hadamard transform application
    print(f"Successfully applied Hadamard transform to {applied_count} SVDLinear layers")




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

def get_orthogonal_matrix(size, mode, device=DEVICE):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        raise NotImplementedError('random_hadamard_matrix needs hadamard_utils')
    else:
        raise ValueError(f'Unknown mode {mode}')

    

# Add global variables to track the sum of ranks
total_rank_sum = 0
total_linear_count = 0

class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", had_K=None, K=-1, had_mode='hadamard') -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
        self.truncation_rank = S.size(0)
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

    def apply_had_rank(self):
        had_K = self.had_K
        K = self.K
        had_mode = self.had_mode

        if K >0 and had_mode in ['rh', 'random']:
            W = self.ALinear # input
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=DEVICE, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, had_K).to(device="cpu", dtype=dtype)
            W = self.BLinear # output
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=DEVICE, dtype=torch.float64)
            W.weight.data = torch.matmul(had_K.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=DEVICE, dtype=torch.float64)
                W.bias.data = torch.matmul(had_K.T, b).to(device="cpu", dtype=dtype)
        elif K>0 and had_mode == 'hadamard':
            raise NotImplementedError('apply_exact_had_to_linear needs hadamard_utils')
        # del self.had_k

    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y

def from_linearqkv_with_grad(
    linear: nn.Linear,
    linear1: nn.Linear,
    linear2: nn.Linear,
    param_ratio: float,
    act_aware=False,
    ic_split=1,
    oc_split=1,
    alpha=1.,
    sigma_fuse="UV",
    rank_align=1.,
    had_rank=False,
    had_mode='hadamard',
    singular_indices=None,
    seed=0,
):
    global total_rank_sum, total_linear_count
    
    # Directly use pre-calculated SVD results and importance sorting results
    device = DEVICE
    
    # Check whether there are pre-calculated SVD results
    # Fix: Should not get qkv_svd_info from linear.self_attn, but from other places
    # We need to find the self_attn module that contains these linear layers
    
    # Assume these linear layers belong to the same self_attn module
    # Try to get SVD information from parent module
    parent_module = None
    for name, module in linear._modules.items():
        if hasattr(module, 'qkv_svd_info'):
            parent_module = module
            break
    
    # If linear cannot find, try to find in higher layers
    if parent_module is None:
        # Here we need a more general method to find the module that contains qkv_svd_info
        # Temporary solution: Assume these information are stored in the model at a known location
        # For example, can be obtained from the parameters passed in
        if hasattr(linear, 'parent_attn') and hasattr(linear.parent_attn, 'qkv_svd_info'):
            svd_info = linear.parent_attn.qkv_svd_info
        else:
            # If cannot find SVD information, print error and return original linear layer
            logging.info("Error: Cannot find QKV SVD information. Please ensure prepare_qkv_svd function has been called.")
            return linear, linear1, linear2
    else:
        svd_info = parent_module.qkv_svd_info
    
    # Get SVD information
    U = svd_info['U'].to(device)
    S = svd_info['S'].to(device)
    V = svd_info['V'].to(device)
    
    # Use pre-calculated importance sorting results
    if singular_indices is not None:
        logging.info(f"Using {len(singular_indices)} pre-selected singular values")
        
        # Select important singular values and corresponding vectors
        U_selected = U[:, singular_indices]
        S_selected = S[singular_indices]
        V_selected = V[:, singular_indices]
        
        # Update rank to the number of selected singular values
        rank = len(singular_indices)
        
        # logging.info("rank", rank)
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
        total_rank_sum += rank
        total_linear_count += 1
        
        logging.info(f"Layer uses rank: {rank}")
        
        # Process activation-aware scaling
        if act_aware and 'scaling_diag_matrix' in svd_info:
            scaling_diag_matrix = svd_info['scaling_diag_matrix'].to(device)
            if scaling_diag_matrix.ndim == 1:
                # ASVD
                # One-dimensional vector, representing diagonal matrix diagonal elements
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
                
                V_selected = scaling_diag_matrix_inv.T @ V_selected
            
        # Split U into three parts, corresponding to q, k, v
        U_selected = U_selected.view(3, -1, rank)
        Us = [U_selected[0], U_selected[1], U_selected[2]]
        Ss = [S_selected]
        Vs = [V_selected]
        
        # Process bias
        if linear.bias is not None:
            biasq = linear.bias.data
            biask = linear1.bias.data
            biasv = linear2.bias.data
        else:
            biasq = None
            biask = None
            biasv = None
        
        # Check NaN or Inf
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        
        assert len(Us)/3 == len(Ss) == len(Vs) == 1
        
        # Create new linear layer
        if had_rank:
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, had_K, K, had_mode)
            new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode)
            new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
        else:
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse)
            new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse)
            new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse)
        
        new_linearQ.to(linear.weight.dtype)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearQ.cpu(), new_linearK.cpu(), new_linearV.cpu()

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



def prepare_qkv_svd(layers, args):
    """
    Pre-process QKV layers with SVD decomposition and store results in attention modules
    
    Args:
        layers: List of decoder layers to be processed
        args: Parameter configuration
    """
    print("Preprocessing QKV layer SVD decomposition...")
    device = DEVICE
    alpha = args.act_alpha
    
    for idx, layer in enumerate(layers):
        q_linear = layer.self_attn.q_proj
        k_linear = layer.self_attn.k_proj
        v_linear = layer.self_attn.v_proj
        
        try:
            # Move weights to CUDA device
            w = torch.cat([
                q_linear.weight.data.float(), 
                k_linear.weight.data.float(), 
                v_linear.weight.data.float()
            ], dim=0).to(device)
            
            # Apply activation-aware scaling (if enabled)
            if args.act_aware:
                scaling_diag_matrix = torch.ones(k_linear.in_features, device=DEVICE)  # avoid zero division
                if hasattr(k_linear, "scaling_diag_matrix"):
                    # print("WARNING: scaling_diag_matrix is used")
                    scaling_diag_matrix *= (k_linear.scaling_diag_matrix.to(DEVICE)).to(torch.float32)**alpha
                    scaling_diag_matrix += 0  # avoid zero division
                    w = w * scaling_diag_matrix.view(1, -1)
                    scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1) # (k_linear.scaling_diag_matrix.to(DEVICE)).to(torch.float32)**(-alpha)
                elif hasattr(k_linear, "scaling_diag_matrixS"):
                    scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(DEVICE)
                    w = w @ scaling_diag_matrix.float()
                    scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
            V = Vt.T
            
            # Store SVD results
            layer.self_attn.qkv_svd_info = {
                'U': U.cpu(),
                'S': S.cpu(),
                'V': V.cpu()
            } 
            
            if hasattr(args, "act_aware") and args.act_aware:
                layer.self_attn.qkv_svd_info['scaling_diag_matrix'] = scaling_diag_matrix.cpu()
                layer.self_attn.qkv_svd_info['scaling_matrix_inverse_transpose'] = scaling_matrix_inverse_transpose.cpu()
            
            print(f"Layer {idx} QKV SVD completed, S shape: {S.shape}")
            
        except Exception as e:
            print(f"Layer {idx} QKV SVD failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("QKV layer SVD preprocessing completed")

def svd_qkv_with_grad_info(layers, args):
    """
    Perform SVD decomposition on QKV layer fusion and utilize gradient information to construct S importance scores
    
    Args:
        layers: List of model layers
        args: Parameter configuration
    
    Returns:
        grad_scores_dict: Dictionary containing gradient importance scores for S in each layer
    """
    grad_alpha = args.grad_alpha
    # Directly use pre-computed S gradient information
    grad_scores_dict = {}
    
    for idx, layer in enumerate(layers):
        if hasattr(layer.self_attn, 'qkv_svd_info') and hasattr(layer.self_attn, 'S_grad_info'):
            svd_info = layer.self_attn.qkv_svd_info
            S = svd_info['S']
            S_grad = layer.self_attn.S_grad_info
            
            # Ensure S and S_grad are on the same device (both moved to CUDA)
            device = DEVICE  # Get CUDA device
            S = svd_info['S'].to(device).to(torch.float16)
            S_grad = layer.self_attn.S_grad_info.to(device).to(torch.float16)
            
            # Calculate importance score: |S| * |S_grad|
            importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
            
            # Move result back to CPU for saving
            layer_key = f"layer_{idx}"
            grad_scores_dict[layer_key] = importance_score.cpu()
            
            print(f"Layer {idx} importance score computed, shape: {importance_score.shape}")
        else:
            print(f"Warning: Layer {idx} lacks necessary SVD information or gradient information, cannot compute importance score") 
    
    
    # Get indices and scores of top k important singular values
    num_layers = len(layers)
    hidden_size = layers[0].self_attn.q_proj.in_features # Assuming in_features is the full rank for MHA models
    total_rank = num_layers * hidden_size
    k_value = int(args.rank_ratio/2 * total_rank) # Factor of 2 is a legacy setting
        
    top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k=k_value)
    
    logging.info(f"Selected top {len(top_indices)} important singular values")
    return top_indices, top_scores, layer_indices_dict

# Add the following functions at the end of the file

def get_top_k_scores(grad_scores_dict, k):
    """
    Get indices and scores of top k important singular values across all layers
    
    Args:
        grad_scores_dict: Dictionary containing gradient importance scores for each layer
        k: Number of important singular values to select
    
    Returns:
        top_indices: List of (layer_index, singular_value_index) tuples for top k important singular values
        top_scores: List of corresponding importance scores
        layer_indices_dict: Dictionary of selected singular value indices for each layer
    """
    # Collect scores from all layers
    all_scores = []
    for layer_idx, scores in grad_scores_dict.items():
        layer_num = int(layer_idx.split('_')[1])
        for i, score in enumerate(scores):
            all_scores.append((layer_num, i, score.item()))
    
    # Sort by score in descending order
    all_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Select top k
    top_k = all_scores[:k]
    
    # Separate indices and scores
    top_indices = [(item[0], item[1]) for item in top_k]
    top_scores = [item[2] for item in top_k]
    
    # Create index dictionary for each layer
    layer_indices_dict = {}
    for layer_idx, singular_idx in top_indices:
        if layer_idx not in layer_indices_dict:
            layer_indices_dict[layer_idx] = []
        layer_indices_dict[layer_idx].append(singular_idx)
    
    return top_indices, top_scores, layer_indices_dict


def calib_input_distribution_text(model, windows):
    model.eval()

    def hook(module, input, output):
        abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
        module.scaling_diag_matrix += abs_mean.float()

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'model.layers' in name:
            module.scaling_diag_matrix = 0
            handles.append(module.register_forward_hook(hook))
    model = model.to(DEVICE)
    with torch.no_grad():
        for w in windows:
            model(w.to(DEVICE))
    for h in handles:
        h.remove()


def calib_grad_info_text(model, layers, windows, args):
    accumulation_steps = 1
    batch_count = 0
    prepare_qkv_svd(layers, args)
    model.eval()
    device = DEVICE
    model = model.to(device)
    model.train()
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

    for w in windows:
        w = w.to(device)
        with torch.enable_grad():
            loss = model(input_ids=w, labels=w)[0]
            loss = loss / accumulation_steps
            loss.backward()
        batch_count += 1
        if batch_count % accumulation_steps == 0:
            # Update gradient information for S in each layer
            for idx, layer in enumerate(layers):
                if hasattr(layer.self_attn, 'qkv_svd_info'):
                    svd_info = layer.self_attn.qkv_svd_info
                    q_linear = layer.self_attn.q_proj
                    k_linear = layer.self_attn.k_proj
                    v_linear = layer.self_attn.v_proj
                    
                    if (q_linear.weight.grad is not None and 
                        k_linear.weight.grad is not None and 
                        v_linear.weight.grad is not None):
                        grad_cat = torch.cat([
                            q_linear.weight.grad.detach().to(torch.float32),
                            k_linear.weight.grad.detach().to(torch.float32),
                            v_linear.weight.grad.detach().to(torch.float32)
                        ], dim=0).to(device)
                        
                        if args.act_aware:
                            scaling_matrix_inverse_transpose = svd_info['scaling_matrix_inverse_transpose'].to(device)
                            if scaling_matrix_inverse_transpose.ndim == 1:
                                # 1D vector representing diagonal matrix elements
                                grad_cat = grad_cat * scaling_matrix_inverse_transpose.view(1, -1).to(torch.float32)  # Scale each column
                            elif scaling_matrix_inverse_transpose.ndim == 2:
                                # 2D matrix representing full scaling matrix (possibly non-diagonal)
                                grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.float32)  # Right multiply matrix
            
                        U = svd_info['U'].to(device).to(torch.float32)
                        V = svd_info['V'].to(device).to(torch.float32)
                        # S_grad = torch.diag(U.T @ grad_cat @ V)
                        # more efficient - compute only diagonal elements
                        S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                        S_grad_squared = S_grad.pow(2)
                        
                        if not hasattr(layer.self_attn, 'S_grad_info'):
                            layer.self_attn.S_grad_info = S_grad_squared
                        else:
                            layer.self_attn.S_grad_info += S_grad_squared
            
            model.zero_grad()  # Clear gradients

    if batch_count > 0:
        for layer in layers:
            if hasattr(layer.self_attn, 'S_grad_info'):
                layer.self_attn.S_grad_info = layer.self_attn.S_grad_info.div(batch_count//accumulation_steps).sqrt()


def qsvd_lm_decompose(model, layers, windows, args):
    calib_input_distribution_text(model, windows)
    calib_grad_info_text(model, layers, windows, args)
    rank, world_size = get_rank_and_world_size()
    top_indices, top_scores, layer_indices_dict = svd_qkv_with_grad_info(layers, args) # top_indices: (layer_idx, singular_value_idx)
    for idx, layer in enumerate(layers):
        full = find_qlayers(layers[idx], layers=[torch.nn.Linear])
        for name, module in full.items():
            if args.qkv_fuse:
                if 'q_proj' in name: # fuse QKV
                    # Check whether there are pre-calculated SVD information
                    if not hasattr(layer.self_attn, 'qkv_svd_info'):
                        logging.info(f"Layer {idx} has no pre-calculated SVD information, skipping")
                        continue
                    
                    # Ensure idx is in layer_indices_dict
                    if idx not in layer_indices_dict:
                        logging.info(f"Layer {idx} is not in layer_indices_dict, skipping")
                        continue
                    
                    # Print more debugging information
                    if rank == 0:
                        logging.info(f"Layer {idx} starting QKV fusion SVD, using {len(layer_indices_dict[idx])} singular values")
                    
                    # Temporarily attach SVD information to linear layer, so from_linearqkv can access
                    layer.self_attn.q_proj.parent_attn = layer.self_attn
                    layer.self_attn.k_proj.parent_attn = layer.self_attn
                    layer.self_attn.v_proj.parent_attn = layer.self_attn
                    
                    try:
                        qlinear, klinear, vlinear = from_linearqkv_with_grad(
                            layer.self_attn.q_proj,
                            layer.self_attn.k_proj,
                            layer.self_attn.v_proj,
                            param_ratio=args.rank_ratio,
                            alpha=args.act_alpha,
                            act_aware=args.act_aware,
                            rank_align=1.,
                            sigma_fuse=args.svd_mode,
                            had_rank=args.had_rank,
                            had_mode='random', # ‘rh'
                            singular_indices=layer_indices_dict[idx],
                            seed=args.seed
                        )
                        
                        # Clean up temporary attributes
                        delattr(layer.self_attn.q_proj, 'parent_attn')
                        delattr(layer.self_attn.k_proj, 'parent_attn')
                        delattr(layer.self_attn.v_proj, 'parent_attn')
                        
                        rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                        rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                        rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                        if rank == 0:
                            logging.info(f"Layer {idx} QKV fusion SVD completed")
                    except Exception as e:
                        logging.info(f"Layer {idx} QKV fusion SVD failed: {e}")
                        import traceback
                        traceback.logging.info_exc()
                        
                        # Clean up temporary attributes
                        if hasattr(layer.self_attn.q_proj, 'parent_attn'):
                            delattr(layer.self_attn.q_proj, 'parent_attn')
                        if hasattr(layer.self_attn.k_proj, 'parent_attn'):
                            delattr(layer.self_attn.k_proj, 'parent_attn')
                        if hasattr(layer.self_attn.v_proj, 'parent_attn'):
                            delattr(layer.self_attn.v_proj, 'parent_attn')
                            
    had_transform_rank(model)
