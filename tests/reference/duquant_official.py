# Vendored reference implementation for tests only.
# Source: https://github.com/Hsu1023/DuQuant (commit d56cfc6fe97c34c0eb100fec82fe439865905679),
# files quantize/quantizer.py (class UniformAffineQuantizer, line ranges below), utils.py (exchange_row_col,
# get_rot, get_hadamard) and get_rot.py (base rotation matrix loop), quantize/const.py.
# Licensed under the MIT License, reproduced here:
#
# MIT License
#
# Copyright (c) 2024 Haobo Xu, Haokun Lin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adaptations (everything not listed is verbatim; each adapted line carries an "ADAPTED:" comment):
#  1. quantizer.py imports (`from utils import ...`, `quantize.const`) are replaced by the definitions below.
#  2. utils.get_rot loads `Rot.pkl` (written by get_rot.py on CUDA). Here `Rot[n]` is built on CPU by the loop of
#     get_rot.py (`build_rot_base`), with `device='cuda'` removed; `get_rot` keeps the QR of torch.randn on the
#     global CPU RNG and drops the try/except wrapper around it.
#  3. UniformAffineQuantizer.__init__ is reduced to the attributes the excerpted methods read
#     (quantizer.py:22-109 sets them from constructor arguments; same names, same defaults used by main.py).
#  4. forward() (quantizer.py:343-385) is not vendored; `quantize` below chains per_token_dynamic_calibration and
#     fake_quant exactly as forward() does after init_duquant (forward lines 380-384).
#  5. permutation_random, calculate_std, register_* and copy_duquant_params are not vendored (unused).
# ruff: noqa
import math
import pickle
import random

import numpy as np
import torch
import torch.nn as nn

# quantize/const.py:1-2
CLIPMIN = 1e-5
CLIPMAX = 1e4

# quantizer.py:13-18
def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x



# get_rot.py:8-37, loop for one n; ADAPTED: CPU instead of cuda, returns R instead of pickling it
def build_rot_base(n):
    v_i = 1 / np.sqrt(n)

    R = torch.eye(n)  # ADAPTED: .to(device) removed

    def exchange(i, j, tensor):
        indices_row = torch.arange(tensor.size(0))
        indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
        tensor = tensor[indices_row]

        indices_col = torch.arange(tensor.size(1))
        indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
        tensor = tensor[:, indices_col]
        return tensor

    for i in range(n - 1):  # ADAPTED: tqdm removed
        rot_i = torch.eye(n)  # ADAPTED: .to(device) removed
        cos = 1 / np.sqrt(i + 2)
        sin = np.sqrt(i + 1) / np.sqrt(i + 2)
        rot_i[i, i] = rot_i[i + 1, i + 1] = cos
        rot_i[i, i + 1] = -sin
        rot_i[i + 1, i] = sin
        R = torch.matmul(rot_i, R)

    R = exchange(0, n - 1, R)  # ADAPTED: .cpu() dropped (already CPU)
    return R


# utils.py:86-97 (verbatim)
def exchange_row_col(_tensor, i, j):
    tensor = _tensor.detach().clone()
    assert isinstance(tensor, torch.Tensor)
    indices_row = torch.arange(tensor.size(0))
    indices_row[i], indices_row[j] = indices_row[j].item(), indices_row[i].item()
    tensor = tensor[indices_row]

    indices_col = torch.arange(tensor.size(1))
    indices_col[i], indices_col[j] = indices_col[j].item(), indices_col[i].item()
    tensor = tensor[:, indices_col]
    return tensor


# utils.py:98-113; ADAPTED: Rot.pkl lookup replaced by build_rot_base, try/except removed
Rot = {}
def get_rot(n, device='cpu'):
    if Rot.get(n) is None:
        Rot[n] = build_rot_base(n)  # ADAPTED: was pickle.load(open("Rot.pkl", "rb"))[n]
    R = Rot[n].to(device)
    random_matrix = torch.randn(n-1, n-1).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q = torch.cat([torch.zeros(n-1, 1).to(device), q], dim=1)
    q = torch.cat([torch.zeros(1, n).to(device), q], dim=0)
    q[0, 0] = 1
    R = torch.matmul(R,q)
    return R


# utils.py:116-123 (verbatim)
def get_hadamard(n): 
    if n == 1:
        return torch.tensor([[1.]], dtype=torch.float32)
    else:
        assert n % 1 == 0, "The size should be divided by 2."
        H_n_minus_1 = get_hadamard(n//2)
        return torch.cat([torch.cat([H_n_minus_1, H_n_minus_1], dim=1),
                          torch.cat([H_n_minus_1, -H_n_minus_1], dim=1)], dim=0) / math.sqrt(2)


class UniformAffineQuantizer(nn.Module):
    # ADAPTED: reduced __init__ (see note 3); max_rotation_step default 1024 as in quantizer.py:39, main.py passes 256
    def __init__(self, n_bits=8, symmetric=False, group_size=None, swc=None, lac=None, block_size=128,
                 max_rotation_step=1024, permutation_times=1, rotate=True):
        super().__init__()
        self.symmetric = symmetric
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.deficiency = 0
        self.group_size = group_size
        self.swc = swc
        self.lac = lac
        self.lwc = False
        self.block_size = block_size
        self.rotate = rotate
        self.max_rotation_step = max_rotation_step
        self.permutation_times = permutation_times
        self.quant_method = 'duquant'
        self.init_duquant_params = torch.tensor(0)
        self.R, self.permutation_list = [], []

    def quantize(self, x):
        # ADAPTED: forward() lines 380-384 without the fp16/enable branches
        self.per_token_dynamic_calibration(x)
        return self.fake_quant(x, self.scale, self.round_zero_point)

    # quantizer.py:116-141 (verbatim)
    def fake_quant(self, x, scale, round_zero_point):
        if self.deficiency > 0:
            pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
            x = torch.cat((x,pad_zeros),dim=1)
        
        if self.group_size:
            assert len(x.shape)==2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
            
        x_int = round_ste(x.float() / scale).half()    # avoid overflow
        
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax) 

        x_dequant = x_int
        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:,:-self.deficiency]
        return x_dequant  

    # quantizer.py:173-342 (verbatim): permutation_zigzag, rotation, calculate_std, online_duquant_cali, init_duquant
    def permutation_zigzag(self, weight):
        weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        weight_max = weight.abs().max(dim=0).values
        weight_mean = weight_max.mean().item()
        pairs = [(i, weight_max[i].item()) for i in range(hidden_dim)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        def zigzag(numbers):
            cur = 0
            up = True
            l = [[] for i in range(hidden_dim // self.block_size)]
            for i in range(len(numbers)):
                l[cur].append(numbers[i])
                if up:
                    cur += 1
                    if cur == len(l):
                        cur -= 1
                        up = False
                else:
                    cur -= 1
                    if cur == -1:
                        cur += 1
                        up = True
            return l
        pairs = zigzag(pairs)

        for i in range(len(pairs)):
            pairs[i].sort(key=lambda x: x[1], reverse=True)

        perm = torch.zeros(hidden_dim, dtype=torch.long)
        for i in range(len(pairs)):
            perm[i * self.block_size:(i+1) * self.block_size] = torch.tensor([_[0] for _ in pairs[i]])
        weight = weight[:, perm]
        return weight, perm

    def rotation(self, weight, max_rotation_step=None, other=None, score_func=None):
        if max_rotation_step is None:
            max_rotation_step = self.max_rotation_step
        weight = weight.detach().clone()
        _weight = weight.detach().clone()
        hidden_dim = weight.shape[-1]
        exchange_ids = []
        peak_values = []
        weight = weight.reshape(-1, self.block_size)
        if other is not None:
            _other = other.detach().clone()
            other = other.reshape(-1, self.block_size)


        Rot = get_rot(self.block_size, weight.device)
        for j in range(max_rotation_step):
            if score_func is not None:
                weight_max = weight.abs().max(dim=0).values
                other_max = other.abs().max(dim=0).values
                r = score_func(weight_max, other_max).argmax().item()
                peak_values.append(score_func(weight_max[r], other_max[r]).item())
                
            else:
                r, c = divmod(weight.argmax().item(), weight.shape[-1])
                r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
                peak_values.append((weight[r, c] - weight[r2, c2]).item())
            exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
            exchange_ids.append(exchange_id)
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(weight)
            weight = torch.matmul(weight, R)
            if other is not None:
                other = torch.matmul(other, R)

        if score_func is not None:
            weight_max = weight.abs().max(dim=0).values
            other_max = other.abs().max(dim=0).values
            r = score_func(weight_max, other_max).argmax().item()
            peak_values.append(score_func(weight_max[r], other_max[r]).item())
        else:
            r, c = divmod(weight.argmax().item(), weight.shape[-1])
            r2, c2 = divmod(weight.argmin().item(), weight.shape[-1])
            peak_values.append((weight[r, c] - weight[r2, c2]).item())
        exchange_id = r if weight[r,c].abs() > weight[r2, c2].abs() else r2
        exchange_ids.append(exchange_id)

        weight = _weight.detach().clone()
        if other is not None:
            other = _other.detach().clone()
        select_length = torch.argmin(torch.tensor(peak_values)).item()
        exchange_ids = exchange_ids[:select_length]
        peak_values = peak_values[:select_length+1]

        R_ = torch.eye(self.block_size).to(weight)
        for exchange_id in exchange_ids:
            R = Rot.clone()
            R = exchange_row_col(R, 0, exchange_id % self.block_size).to(R_)
            R_ = torch.matmul(R_, R)
        weight = torch.matmul(weight.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        if other is not None:
            other = torch.matmul(other.reshape(-1, self.block_size), R_).reshape(-1,hidden_dim)
        return (weight, exchange_ids, R_) if other is None else (weight, other, exchange_ids, peak_values[select_length], R_)
    
    def calculate_std(self, weight):
        weight = weight.abs().max(dim=0).values
        groups = [weight[j * self.block_size: (j+1) * self.block_size] for j in range(weight.shape[0] // self.block_size)]
        group_means = [sum(group) / len(group) for group in groups]
        mean = sum(group_means) / len(group_means)
        variance = sum((x - mean) ** 2 for x in group_means) / len(group_means)
        return math.sqrt(variance)

    def online_duquant_cali(self, weight):
        weight = weight.detach().clone()
        T = {}
        
        self.permutation_list = None
        self.R = None
        for i in range(self.permutation_times):
            weight, _, R = self.rotation(weight)
            if self.R is None:
                self.R = R.unsqueeze(0)
            else:
                self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
            
            weight, perm = self.permutation_zigzag(weight)
            if self.permutation_list is None:
                self.permutation_list = perm.unsqueeze(0)
            else:
                self.permutation_list = torch.cat((self.permutation_list, perm.unsqueeze(0)), dim=0)

        weight, _, R = self.rotation(weight)
        if self.R is None:
            self.R = R.unsqueeze(0)
        else:
            self.R = torch.cat((self.R, R.unsqueeze(0)), dim=0)
        return weight

    def init_duquant(self, x: torch.Tensor):
        if self.quant_method is None:
            return x
        if self.rotate is None:
                x_shape = x.shape   # (n_tokens, hidden_dim) / (out_features, in_features)
                hadamard = self.H.to(x)
                x = x.reshape(-1, self.block_size)
                x = x.matmul(hadamard).view(x_shape)
        elif self.quant_method == 'duquant':
            if self.rotate:
                if not self.init_duquant_params:
                    x = self.online_duquant_cali(x)
                    self.init_duquant_params = torch.tensor(1)
                else:
                    x_size = x.shape
                    x_type = x.dtype
                    if self.permutation_list is not None:
                        for i in range(len(self.permutation_list)):
                            x = x.reshape(-1, self.block_size)
                            R = self.R[i].to(x)
                            x = x.matmul(R).reshape(x_size).squeeze(0)
                            # if False:
                            if True:
                                if len(self.permutation_list.shape) == 3:
                                    perm = (self.permutation_list[i, 0].to(x.device), self.permutation_list[i, 1].to(x.device))
                                    x[:, perm[0]], x[:, perm[1]] = x[:, perm[1]], x[:, perm[0]]
                                else:
                                    perm = self.permutation_list[i].to(x.device)
                                    x = x[:, perm]
                    if len(self.R) > 0:
                        x = x.reshape(-1, self.block_size)
                        R = self.R[-1].to(x)
                        x = x.matmul(R).reshape(x_size) 
        else:
            raise NotImplementedError
        return x
            


    # quantizer.py:387-419 (verbatim): per_token_dynamic_calibration
    def per_token_dynamic_calibration(self, x):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1,self.group_size)
            else:
                pad_zeros = torch.zeros((x.shape[0],self.deficiency),dtype=x.dtype,device=x.device)
                x = torch.cat((x,pad_zeros),dim=1)
                x = x.reshape(-1,self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True).to(x.device)
        xmax =  x.amax(reduce_shape, keepdim=True).to(x.device)
        if self.swc:
            xmax = self.swc*xmax
            xmin = self.swc*xmin
        elif self.lwc:
            xmax = self.sigmoid(self.upbound_factor.to(x.device))*xmax
            xmin = self.sigmoid(self.lowbound_factor.to(x.device))*xmin
        elif self.lac:
            xmax = self.lac*xmax
            xmin = self.lac*xmin

        if self.symmetric:
            abs_max = torch.max(xmax.abs(),xmin.abs())
            scale = abs_max / (2**(self.n_bits-1)-1)
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = (2**(self.n_bits-1)-1)*torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2**self.n_bits-1)
            self.scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
            zero_point = -(xmin) / (self.scale)
        self.round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round()
        
