# Vendored reference implementation for tests only.
# Source: https://github.com/z-lab/paroquant (branch legacy, commit 8e5ac77bff10e5a47fc5491eca0ab1c4370e3d85),
# files paroquant/optimize.py, paroquant/module.py, paroquant/quantizer.py, paroquant/util.py,
# paroquant/convert_utils.py, kernels/paroquant_kernels/interface.py and the layer loop of optimize.py.
# Upstream licence: MIT License, Copyright (c) 2025 Z Lab. Permission is hereby granted, free of charge, to
# any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions: The above copyright notice and this
# permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS
# PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# Modifications:
#   - `torch.ops.rotation.rotate` (CUDA kernel) is replaced by `rotate_op`, a naive pure-torch loop over
#     pairs that follows the kernel semantics read from `RotateTensorFunc.backward`; the autograd Function
#     itself (forward saved tensors, backward) is verbatim.
#   - `device="cuda"` defaults inside PseudoQuantizedLinear.__init__ are `device="cpu"`.
#   - `get_calib_dataset` takes `load_dataset` as an argument and calls it with the original arguments.
#   - The layer loop of optimize.py `main()` is `run_layerwise`: the device, model dtype, hyper-parameters and
#     in-memory samples are arguments, tqdm/logging/resume/saving are dropped and the pseudo-quantized layers
#     are kept in the model; the order of operations is unchanged.
# ruff: noqa
import math
import random
import warnings
from copy import deepcopy
from typing import Callable, Iterator, Literal, Optional, TypeVar, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


# ---- paroquant/util.py ----
def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def clamp_ste(
    x: torch.Tensor,
    min: Optional[torch.Tensor] = None,
    max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return (x.clamp(min, max) - x).detach() + x


class CosineAnnealingParam:
    def __init__(self, start_value: float, end_value: float, T_max: int):
        self.start_value = start_value
        self.end_value = end_value
        self.T_max = T_max
        self._step = -1

    def step(self) -> float:
        self._step += 1

        if self._step >= self.T_max:
            return self.end_value

        cos_val = math.cos(math.pi * self._step / self.T_max)
        return self.end_value + (self.start_value - self.end_value) * (1 + cos_val) / 2


_Linear_T = TypeVar("Linear", bound=nn.Module)


def get_named_linears(module: nn.Module, subclass: type[_Linear_T] = nn.Linear) -> dict[str, _Linear_T]:
    return {name: m for name, m in module.named_modules() if isinstance(m, subclass)}


def set_module_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def get_mixed_calib_dataset(
    datasets: list[str],
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
    load_dataset,
) -> list[torch.Tensor]:
    per_dataset_len = n_samples // len(datasets)
    results = []
    for i, dataset in enumerate(datasets):
        dataset_samples = per_dataset_len if i < len(datasets) - 1 else n_samples - len(results)
        results.extend(
            get_calib_dataset(
                data=dataset,
                tokenizer=tokenizer,
                n_samples=dataset_samples,
                block_size=block_size,
                seed=seed,
                split=split,
                load_dataset=load_dataset,
            )
        )
    assert len(results) == n_samples, f"Expected {n_samples} samples, got {len(results)}"

    rand = random.Random(seed)
    rand.shuffle(results)

    return results


def get_calib_dataset(
    data="pileval",
    *,
    tokenizer,
    n_samples: int,
    block_size: int,
    seed: int,
    split: str,
    load_dataset,
) -> list[torch.Tensor]:
    if data == "pileval":
        if split != "validation":
            warnings.warn("The split argument is ignored when data is 'pileval'.")
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        dataset = dataset.shuffle(seed=seed)
    elif data == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        dataset = dataset.shuffle(seed=seed)
    elif data == "c4":
        if split == "train":
            dataset = load_dataset(
                "allenai/c4",
                data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
                split=split,
            )
        elif split == "validation":
            dataset = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00001-of-00008.json.gz"},
                split=split,
            )
        dataset = dataset.shuffle(seed=seed)
    elif data == "redpajama":
        test_split, val_split = 0.2, 0.1
        dataset = load_dataset(
            "liang2kl/RedPajama-Data-1T-Sample-Backup",
            split="train",
            trust_remote_code=True,
        )
        dataset = dataset.shuffle(seed=seed)
        test_size = int(len(dataset) * test_split)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - test_size - val_size
        if split == "test":
            dataset = dataset.select(range(len(dataset) - test_size, len(dataset)))
        elif split == "validation":
            dataset = dataset.select(range(len(dataset) - test_size - val_size, len(dataset) - test_size))
        elif split == "train":
            dataset = dataset.select(range(0, train_size))
        else:
            raise ValueError(f"Invalid split: {split}")
    else:
        raise NotImplementedError

    samples = []
    total_len = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        total_len += len(line_encoded)
        if total_len >= n_samples * block_size:
            break
    samples = torch.cat(samples, dim=1).squeeze(0)
    n_split = min(samples.shape[0] // block_size, n_samples)

    return [samples[i * block_size : (i + 1) * block_size] for i in range(n_split)]


@torch.no_grad()
def catch_first_layer_input(
    model: nn.Module,
    layers: nn.ModuleList,
    samples: torch.Tensor,
    batch_size: Optional[int],
) -> tuple[torch.Tensor, dict]:
    layer_kwargs = {}
    batched = batch_size is not None
    inps: list[torch.Tensor] = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            # Bypass __setattr__ of nn.Module
            object.__setattr__(self, "module", module)

        def forward(self, inp, **kwargs):
            inps.append(inp)
            if len(layer_kwargs) == 0:
                layer_kwargs.update(kwargs)
            raise ValueError

        def __getattr__(self, name):
            return getattr(self.module, name)

    layers[0] = Catcher(layers[0])
    batch_size = samples.shape[0] if not batched or batch_size <= 0 else batch_size
    num_batches = samples.shape[0] // batch_size
    samples_batch = samples.chunk(num_batches)
    for samples in samples_batch:
        try:
            model(samples.to(next(model.parameters()).device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    if not batched:
        inps = inps[0]

    layer_kwargs["use_cache"] = False
    if "past_key_value" in layer_kwargs:
        del layer_kwargs["past_key_value"]
    if "past_key_values" in layer_kwargs:
        del layer_kwargs["past_key_values"]

    return inps, layer_kwargs


class CachedTensorShards:
    def __init__(
        self,
        batches: list[torch.Tensor],
        num_shards: int,
        *,
        target_device: torch.device,
        offload_device: torch.device = torch.device("cpu"),
    ):
        assert len(batches) % num_shards == 0
        if batches[0].device != offload_device:
            self.batches = [b.to(offload_device) for b in batches]
        else:
            self.batches = batches
        self.num_shards = num_shards
        self.current_shard: int = None
        self.cached_shard: list[torch.Tensor] = None
        self.target_device = target_device

    def _switch_shard(self, shard_index: int) -> None:
        if self.current_shard == shard_index:
            return
        self.current_shard = shard_index
        start, end = self._get_shard_range(shard_index)
        self.cached_shard = self.batches[start:end]
        self.cached_shard = [b.to(self.target_device) for b in self.cached_shard]

    def _get_shard_range(self, index: int) -> tuple[int, int]:
        if self.num_shards == 1:
            return 0, len(self.batches)
        shard_size = len(self.batches) // self.num_shards
        start = shard_size * index
        if index == self.num_shards - 1:
            end = len(self.batches)
        else:
            end = shard_size * (index + 1)
        return start, end

    def __getitem__(self, index: int) -> torch.Tensor:
        shard_len = len(self.batches) // self.num_shards
        shard_index = index // shard_len
        if self.current_shard != shard_index:
            self._switch_shard(shard_index)
        offset = index % shard_len
        return self.cached_shard[offset]

    def __iter__(self) -> "Iterator":
        return self.Iterator(self)

    def __len__(self) -> int:
        return len(self.batches)

    class Iterator:
        def __init__(self, batches: "CachedTensorShards"):
            self.batches = batches
            self.current_index = 0

        def __iter__(self):
            return self

        def __next__(self) -> torch.Tensor:
            if self.current_index >= len(self.batches):
                raise StopIteration
            result = self.batches[self.current_index]
            self.current_index += 1
            return result

        def __len__(self) -> int:
            return len(self.batches)


# ---- paroquant/convert_utils.py ----
def _align_shape(pair, angle, grp_size=128, inc_mask=False):
    assert pair.size(0) == angle.size(0)
    assert pair.size(1) == 2

    grp = 0
    pair_ptr = 0

    pair_groups = []
    angle_groups = []
    mask_groups = []
    while True:
        if pair_ptr >= pair.size(0):
            break
        s = torch.zeros((grp_size), dtype=torch.int32)
        count = 0
        temp_pairs = torch.zeros((grp_size // 2, 2), dtype=torch.int32, device=pair.device)
        temp_angle = torch.zeros((grp_size // 2), dtype=torch.float, device=angle.device)
        temp_mask = torch.zeros((grp_size // 2), dtype=torch.int32, device=angle.device)
        while count < grp_size // 2:
            if (
                pair_ptr < pair.size(0)
                and pair[pair_ptr, 0] - grp * grp_size < grp_size
                and pair[pair_ptr, 1] - grp * grp_size < grp_size
            ):
                temp_pairs[count, :] = pair[pair_ptr, :]
                temp_angle[count] = angle[pair_ptr]
                if s[pair[pair_ptr, 0] % grp_size] == 1 or s[pair[pair_ptr, 1] % grp_size] == 1:
                    raise ValueError("illigal pair")
                s[pair[pair_ptr, :] % grp_size] = 1
                pair_ptr += 1
            else:
                t_pair = torch.tensor([-1, -1])
                for i in range(grp_size):
                    if s[i] == 0:
                        t_pair[0] = i
                        s[i] = 1
                        break
                for i in range(grp_size):
                    if s[i] == 0:
                        t_pair[1] = i
                        s[i] = 1
                        break
                if t_pair[0] == -1 or t_pair[1] == -1:
                    raise ValueError("can't find a dummy pair")
                temp_pairs[count, :] = t_pair
                temp_angle[count] = float(0)
                temp_mask[count] = 1
            count += 1
        grp += 1
        pair_groups.append(temp_pairs)
        angle_groups.append(temp_angle)
        mask_groups.append(temp_mask)

    rotation_pairs = torch.cat(pair_groups, axis=0).view(-1).contiguous()
    rotation_pairs = rotation_pairs % grp_size
    angles = torch.cat(angle_groups, axis=0)
    if inc_mask:
        masks = torch.cat(mask_groups, axis=0)
        return rotation_pairs, angles, masks
    return rotation_pairs, angles


def transform_to_kernel_data(
    pairs_group: list[torch.Tensor],
    angles_group: list[torch.Tensor],
    group_size: int = 128,
):
    assert len(pairs_group) == len(angles_group)
    krot = len(pairs_group)
    pairs = []
    angles = []
    masks = []

    for i in range(krot):
        pair, angle, mask = _align_shape(pairs_group[i], angles_group[i], grp_size=group_size, inc_mask=True)
        pairs.append(pair)
        angles.append(angle)
        masks.append(mask)

    rotation_pairs = torch.stack(pairs, dim=0).to(dtype=torch.short)
    angles = torch.stack(angles, dim=0).to(dtype=torch.float)
    masks = torch.stack(masks, dim=0).to(dtype=torch.bool)
    return rotation_pairs, angles, masks


# ---- kernels/paroquant_kernels/interface.py (rotation op replaced by a naive loop) ----
def rotate_op(x, idx_ij, theta, scale, group_size):
    """Pure-torch stand-in of torch.ops.rotation.rotate: x (B, H), idx_ij (KROT, H), theta (KROT, H//2)."""
    assert scale is None
    y = x.clone()
    krot, hidden = idx_ij.shape
    for k in range(krot):
        cos_all = torch.cos(theta[k])
        sin_all = torch.sin(theta[k])
        for g in range(hidden // group_size):
            base = g * group_size
            for p in range(group_size // 2):
                i = base + int(idx_ij[k, base + 2 * p])
                j = base + int(idx_ij[k, base + 2 * p + 1])
                c = cos_all[g * (group_size // 2) + p]
                s = sin_all[g * (group_size // 2) + p]
                a = y[:, i].clone()
                b = y[:, j].clone()
                y[:, i] = c * a + s * b
                y[:, j] = c * b - s * a
    return y


class RotateTensorFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx_ij, theta, scale=None, group_size=128):
        ctx.orig_shape, ctx.orig_dtype = x.shape, x.dtype
        ctx.has_scale = scale is not None
        ctx.group_size = group_size

        y = rotate_op(x.reshape(-1, x.shape[-1]), idx_ij, theta, scale, group_size).view(x.shape)
        saved = (x, idx_ij, theta, y)
        if ctx.has_scale:
            saved += (scale,)
        ctx.save_for_backward(*saved)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        saved = list(ctx.saved_tensors)
        x, idx_ij, theta, y = saved[:4]
        scale = saved[4] if ctx.has_scale else None
        GROUP_SIZE = ctx.group_size
        g = grad_out.view(-1, x.size(-1))

        KROT, H = idx_ij.shape
        num_groups = H // GROUP_SIZE
        B = y.numel() // H
        t = y.view(B, H)
        grad_theta = torch.zeros_like(theta)

        for i in range(KROT - 1, -1, -1):
            theta_neg = -theta[[i]]
            idx_chunk = idx_ij[[i]]
            t_prev = rotate_op(t, idx_chunk, theta_neg, None, GROUP_SIZE)
            g_prev = rotate_op(g, idx_chunk, theta_neg, None, GROUP_SIZE)
            idx_chunk = idx_chunk.view(num_groups, GROUP_SIZE)
            group_offsets = (torch.arange(num_groups, device=idx_ij.device, dtype=torch.int32) * GROUP_SIZE).unsqueeze(1)
            di = idx_chunk[:, 0:GROUP_SIZE:2] + group_offsets
            dj = idx_chunk[:, 1:GROUP_SIZE:2] + group_offsets

            a = t_prev[:, di.reshape(-1)].view(t_prev.size(0), num_groups, GROUP_SIZE // 2)  # (B, G, GS/2)
            b = t_prev[:, dj.reshape(-1)].view(t_prev.size(0), num_groups, GROUP_SIZE // 2)
            ga = g[:, di.reshape(-1)].view(g.size(0), num_groups, GROUP_SIZE // 2)
            gb = g[:, dj.reshape(-1)].view(g.size(0), num_groups, GROUP_SIZE // 2)

            grad_s = (ga * b - gb * a).sum(dim=0)  # (G, GS/2)
            grad_c = (ga * a + gb * b).sum(dim=0)  # (G, GS/2)

            theta_chunk = theta[i]  # shape (H//2,)
            theta_groups = theta_chunk.view(num_groups, GROUP_SIZE // 2)  # (G, GS/2)
            s_vals = theta_groups.sin()
            c_vals = theta_groups.cos()

            grad_theta_chunk = grad_s * c_vals - grad_c * s_vals
            grad_theta[i] = grad_theta_chunk.reshape(-1).to(theta.dtype)
            t, g = t_prev, g_prev

        if ctx.has_scale:
            s = scale.unsqueeze(0)  # (1, H)
            grad_x_flat = g * s
            grad_scale = (x.view(B, H) * g).sum(dim=0).to(scale.dtype).to(scale.device)
        else:
            grad_x_flat, grad_scale = g, None

        grad_x = grad_x_flat.view(ctx.orig_shape).to(ctx.orig_dtype)

        return grad_x, None, grad_theta, grad_scale, None


def scaled_pairwise_rotation(x, idx_ij, theta, scales=None, group_size=128):
    return RotateTensorFunc.apply(x, idx_ij, theta, scales, group_size)


# ---- paroquant/quantizer.py ----
def _calc_scales_and_zero_points(weight: torch.Tensor, group_size: int, qmin: int, qmax: int) -> None:
    assert weight.dim() == 2, weight.shape
    if weight.dtype != torch.float32:
        weight = weight.float()
    x = weight.reshape(-1, group_size)
    min_val = x.amin(dim=1, keepdim=True)
    max_val = x.amax(dim=1, keepdim=True)
    scale = clamp_ste(max_val - min_val, min=1e-5) / qmax
    zero_point = min_val / scale

    assert torch.isnan(zero_point).sum() == 0, zero_point

    return scale, zero_point


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        n_bits: int,
        group_size: int,
    ):
        super().__init__()
        if not isinstance(n_bits, torch.Tensor):
            n_bits = torch.tensor(n_bits)
        if not isinstance(group_size, torch.Tensor):
            group_size = torch.tensor(group_size)
        self.register_buffer("n_bits", n_bits)
        self.register_buffer("group_size", group_size)
        assert weight.shape[-1] % group_size == 0

        scale, zero_point_float = _calc_scales_and_zero_points(weight, group_size, self.qmin, self.qmax)
        self.scale = nn.Parameter(scale)
        self.zero_point_float = nn.Parameter(zero_point_float)

        self.enable_checkpoint = False

    @property
    def qmin(self) -> int:
        return 0

    @property
    def qmax(self) -> torch.Tensor:
        return 2**self.n_bits - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and self.enable_checkpoint:
            return checkpoint(
                self.pseudo_quantize,
                x,
                self.n_bits,
                self.group_size,
                self.scale,
                self.zero_point_float,
                use_reentrant=False,
            )
        else:
            return self.pseudo_quantize(
                x,
                self.n_bits,
                self.group_size,
                self.scale,
                self.zero_point_float,
            )

    def optim_params(self) -> list[nn.Parameter]:
        return [self.scale, self.zero_point_float]

    def set_optim_enabled(self, enabled: bool):
        for param in self.optim_params():
            param.requires_grad = enabled

    @staticmethod
    def pseudo_quantize(
        x: torch.Tensor,
        n_bits: Union[int, torch.Tensor],
        group_size: Union[int, torch.Tensor],
        scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = x.dtype
        if x.dtype != torch.float32:
            x = x.float()
        assert torch.isnan(x).sum() == 0, x

        qmin, qmax = 0, 2**n_bits - 1
        if scale is None or zero_point is None:
            scale, zero_point = _calc_scales_and_zero_points(x, group_size, qmin, qmax)

        scale = clamp_ste(scale, min=1e-5, max=1e5)
        round_zero_point = clamp_ste(-round_ste(zero_point), qmin, qmax)
        dim1, dim2 = x.shape
        x = x.reshape(-1, group_size)
        x_int = round_ste(x / scale)
        assert torch.isnan(x_int).sum() == 0, (x_int, scale.min(), scale.max())
        x_int = x_int + round_zero_point
        x_int = clamp_ste(x_int, qmin, qmax)
        x_dequant = x_int
        x_dequant = x_dequant - round_zero_point
        x_dequant = x_dequant * scale
        x_dequant = x_dequant.reshape(dim1, dim2)
        assert torch.isnan(x_dequant).sum() == 0, x_dequant
        assert torch.isinf(x_dequant).sum() == 0, x_dequant
        return x_dequant.to(dtype)


# ---- paroquant/module.py ----
class PseudoQuantizedLinear(nn.Module):
    """Pseudo-quantized linear layer using custom kernels for fine-tuning."""

    def __init__(
        self,
        linear: nn.Linear,
        rotation_pairs: Optional[list[torch.Tensor]],
        channel_scales: Optional[torch.Tensor],
        *,
        group_size: int,
        n_bits: int,
        num_rotations: int,
    ) -> None:
        super().__init__()
        self.enable_checkpoint = False
        self.weight = nn.Parameter(linear.weight.clone())
        self.in_feat = self.weight.shape[1]
        self.out_feat = self.weight.shape[0]
        num_groups = self.in_feat // group_size
        assert self.in_feat % group_size == 0
        assert self.weight.dtype == torch.float16 or self.weight.dtype == torch.float32
        if rotation_pairs is not None:
            pairs_grouped, angles_grouped, mask = rotation_pairs
            assert pairs_grouped.size(0) == num_rotations
            assert angles_grouped.size(0) == num_rotations
            assert mask.shape == angles_grouped.shape

        num_rotations = (
            torch.tensor(num_rotations, device="cpu") if not isinstance(num_rotations, torch.Tensor) else num_rotations
        )
        self.register_buffer("num_rotations", num_rotations)

        if linear.bias is None:
            self.register_buffer("bias", None)
        else:
            self.bias = nn.Parameter(linear.bias.clone())
        if rotation_pairs is None:
            # generate dummy pairs & angles if rotation_pairs is None
            angles_grouped = torch.zeros(num_rotations, self.in_feat // 2, device="cpu", dtype=self.weight.dtype)
            single_group_idx_ij = torch.randperm(group_size, device="cpu", dtype=torch.short)
            one_krot_layer_idx_ij = single_group_idx_ij.repeat(num_groups)
            pairs_grouped = one_krot_layer_idx_ij.unsqueeze(0).expand(num_rotations, -1).contiguous()
            mask = torch.zeros_like(angles_grouped, device="cpu", dtype=torch.bool)
        else:
            pairs_grouped, angles_grouped, mask = rotation_pairs
        self.pairs_grouped = nn.Parameter(pairs_grouped.clone(), requires_grad=False)
        self.angles_grouped = nn.Parameter(angles_grouped.clone())

        self.register_buffer("mask", mask)
        if channel_scales is None:
            # generate dummy channel scales
            channel_scales = torch.ones(self.in_feat, device="cpu", dtype=self.weight.dtype)
        self.channel_scales = nn.Parameter(channel_scales)
        assert self.pairs_grouped.dtype == torch.short, self.pairs_grouped.dtype
        assert self.channel_scales.dtype == self.weight.dtype, (
            self.channel_scales.dtype,
            self.weight.dtype,
        )
        assert self.mask.dtype == torch.bool, self.mask.dtype

        self.quantizer: Optional[UniformAffineQuantizer] = None
        self.register_buffer("quantizer_optim_enabled", torch.tensor(False))
        n_bits = n_bits if isinstance(n_bits, torch.Tensor) else torch.tensor(n_bits)
        group_size = group_size if isinstance(group_size, torch.Tensor) else torch.tensor(group_size)
        self.register_buffer("n_bits", n_bits)
        self.register_buffer("group_size", group_size)

    def forward(self, x: torch.Tensor, weight_update: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight = self.weight
        if weight_update is not None:
            weight = weight + weight_update
        weight = self._pseudo_quantize(weight)
        x = self.checkpointed(torch.matmul, x, weight.T)
        if getattr(self, "bias", None) is not None:
            x = x + self.bias
        return x

    def _pseudo_quantize(self, weight: torch.Tensor) -> torch.Tensor:
        weight = weight * self.channel_scales
        weight = self.checkpointed(
            scaled_pairwise_rotation,
            weight,
            self.pairs_grouped,
            self.angles_grouped,
            None,
            self.group_size,
        )
        if self.quantizer_optim_enabled or self.quantizer is not None:
            assert self.quantizer is not None, "Quantizer should be initialized."
            self.quantizer.enable_checkpoint = self.enable_checkpoint
            weight = self.quantizer(weight)
        else:
            weight = self.checkpointed(
                UniformAffineQuantizer.pseudo_quantize,
                weight,
                self.n_bits,
                self.group_size,
            )
        if self.pairs_grouped is not None:
            flipped_pairs = torch.flip(self.pairs_grouped, dims=[0])
            flipped_angles = torch.flip(self.angles_grouped, dims=[0])
            weight = self.checkpointed(
                scaled_pairwise_rotation,
                weight,
                flipped_pairs,
                -flipped_angles,
                None,
                self.group_size,
            )
        if self.channel_scales is not None:
            weight = weight / self.channel_scales.view(1, -1)
        return weight

    def set_optim_enabled(
        self,
        *,
        weight: bool = False,
        bias: bool = False,
        angles: bool = False,
        channel_scales: bool = False,
        quantizer: bool = False,
    ) -> None:
        self.weight.requires_grad = weight
        if self.bias is not None:
            self.bias.requires_grad = bias
        if self.angles_grouped is not None:
            self.angles_grouped.requires_grad = angles
        if self.channel_scales is not None:
            self.channel_scales.requires_grad = channel_scales

        if quantizer and self.quantizer is None:
            # Initialize the quantizer with the current rotated weight
            weight = self.weight
            if self.channel_scales is not None:
                weight = weight * self.channel_scales.view(1, -1)
            if self.pairs_grouped is not None:
                weight = scaled_pairwise_rotation(
                    weight,
                    self.pairs_grouped,
                    self.angles_grouped,
                    None,
                    self.group_size,
                )

            self.quantizer = UniformAffineQuantizer(
                weight,
                n_bits=self.n_bits,
                group_size=self.group_size,
            )

        if self.quantizer:
            self.quantizer.set_optim_enabled(quantizer)
        self.quantizer_optim_enabled = torch.tensor(quantizer)

    def get_optim_params(self, name: str) -> list[nn.Parameter]:
        if name == "weight":
            return [self.weight]
        elif name == "bias":
            return [self.bias]
        elif name == "angles":
            if self.angles_grouped is None:
                return []
            return [self.angles_grouped]
        elif name == "channel_scales":
            if self.channel_scales is None:
                return []
            return [self.channel_scales]
        elif name == "quantizer":
            return self.quantizer.optim_params()
        else:
            raise ValueError(f"Unknown parameter group: {name}")

    @torch.no_grad()
    def reset_angles_by_mask(self):
        self.angles_grouped.data[self.mask] = 0

    def checkpointed(self, fn, *args) -> torch.Tensor:
        if torch.is_grad_enabled() and self.enable_checkpoint:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @torch.no_grad()
    def pseudo_weight(self) -> torch.Tensor:
        return self._pseudo_quantize(self.weight).detach()


@torch.no_grad()
def reset_angles_by_mask(module: nn.Module) -> None:
    linears = get_named_linears(module, PseudoQuantizedLinear)
    for linear in linears.values():
        linear.reset_angles_by_mask()


# ---- paroquant/optimize.py ----
def _get_independent_channel_pairs(
    pairs: torch.Tensor, dim: int, num_rotations: int, num_pairs_each: int
) -> list[list[tuple[int, int]]]:
    pairs = pairs.cpu().tolist()
    rotations_pairs = []
    # Record the available pairs in a matrix.
    available = torch.ones(dim, dim)
    available.fill_diagonal_(0)

    for _ in range(num_rotations):
        independent_pairs = []
        # We use a copy to track the available pairs in this rotation.
        # It's different from the original available matrix.
        available_in_rotation = available.clone()
        # Greedily select pairs that are independent
        for i, j in pairs:
            if len(independent_pairs) == num_pairs_each:
                break
            if available_in_rotation[i, j] == 0:
                continue
            # Simply select the first available pair.
            independent_pairs.append((i, j))
            # Selecting (i, j) in this rotation prohibits
            # selecting any other pairs that share i or j
            # in this rotation.
            available_in_rotation[i, :] = 0
            available_in_rotation[j, :] = 0
            available_in_rotation[:, i] = 0
            available_in_rotation[:, j] = 0
            # Mark the pair as unavailable for future selections.
            # i and j are still available for any other pairs
            # in next rotations.
            available[i, j] = 0
            available[j, i] = 0

        rotations_pairs.append(independent_pairs)

    return rotations_pairs


def get_random_rotation_pairs(
    sensitivity_input: torch.Tensor,
    group_size: int,
    num_rotations: int,
    num_pairs_factor: float,
    seed: int,
) -> list[tuple[int, int]]:
    sorted_pairs: list[list[tuple[int, int]]] = []
    rand = random.Random(seed)
    group_num = sensitivity_input.shape[0]

    for group_idx in range(group_num):
        sorted_pairs.append([])
        for i in range(group_size):
            for j in range(i + 1, group_size):
                sorted_pairs[group_idx].append((i, j))

        rand.shuffle(sorted_pairs[group_idx])

    sorted_pairs = torch.tensor(sorted_pairs, device=sensitivity_input.device)

    pairs_k_groups = [[] for _ in range(num_rotations)]
    num_pairs_per_group = int(group_size * num_pairs_factor)

    for i in range(group_num):
        offset = i * group_size
        pairs_g_k_groups = _get_independent_channel_pairs(sorted_pairs[i], group_size, num_rotations, num_pairs_per_group)
        for r_idx in range(num_rotations):
            pairs_g = pairs_g_k_groups[r_idx]
            for j, (col1, col2) in enumerate(pairs_g):
                pairs_g[j] = (col1 + offset, col2 + offset)
            pairs_k_groups[r_idx].extend(pairs_g)

    return pairs_k_groups


def optimize_module(
    module: nn.Module,
    train_set_batches: tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]],
    val_set_batches: tuple[Iterator[torch.Tensor], Iterator[torch.Tensor]],
    kwargs: dict,
    optim_params: list[dict],
    *,
    loss_fn: Literal["mse", "smooth_l1"],
    n_iter: int,
    early_stop: Optional[int],
    post_optim_callback: Optional[Callable[[nn.Module], None]] = None,
) -> None:
    train_input_batches, train_output_batches = train_set_batches
    val_input_batches, val_output_batches = val_set_batches

    total_steps = n_iter * len(train_input_batches)
    schedulers = [
        CosineAnnealingParam(
            start_value=param_group["lr"],
            end_value=param_group["lr"] / 20,
            T_max=total_steps,
        )
        for param_group in optim_params
    ]
    optimizer = torch.optim.AdamW(optim_params)
    scaler = torch.amp.GradScaler()
    if loss_fn == "mse":
        loss = nn.MSELoss()
    elif loss_fn == "smooth_l1":
        loss = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss function: {loss_fn}")

    def module_output(input: torch.Tensor) -> torch.Tensor:
        out = module(input, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    @torch.no_grad()
    def loss_batches(input_batches: Iterator[torch.Tensor], output_batches: Iterator[torch.Tensor]) -> torch.Tensor:
        total_loss = None
        for input_batch, output_batch in zip(input_batches, output_batches):
            output_q = module_output(input_batch)
            loss_value = loss(output_batch, output_q)
            if total_loss is None:
                total_loss = loss_value
            else:
                total_loss += loss_value
        return total_loss / len(input_batches)

    with torch.no_grad():
        original_val_loss = loss_batches(val_input_batches, val_output_batches)

    best_val_loss = original_val_loss
    best_sd = deepcopy(module.state_dict())

    early_stop_counter = 0

    for _ in range(n_iter):
        for input_batch, output_batch in zip(train_input_batches, train_output_batches):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                output_q = module_output(input_batch)
                loss_value = loss(output_batch, output_q)

            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

            for i, scheduler in enumerate(schedulers):
                optimizer.param_groups[i]["lr"] = scheduler.step()

            if post_optim_callback:
                post_optim_callback(module)

        with torch.no_grad():
            val_loss_value = loss_batches(val_input_batches, val_output_batches)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_sd = deepcopy(module.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop is not None and early_stop_counter >= early_stop:
                break

    module.load_state_dict(best_sd)


# ---- optimize.py main(): layer loop ----
def run_layerwise(
    model,
    blocks,
    samples,
    val_samples,
    *,
    params_to_optimize,
    epochs,
    group_size,
    n_bit,
    num_rotations,
    batch_size,
    cache_shards,
    seed,
    weight_decay=0.01,
    betas=(0.9, 0.95),
    eps=1e-10,
    loss="smooth_l1",
    dtype=torch.float16,
    device="cpu",
):
    """Official layer loop: returns {(layer_idx, linear_name): W_eff} (pseudo_weight of every linear).

    `params_to_optimize` is a list of {param group: lr} dicts, `epochs` one count per stage.
    """
    og_layer_input_batches, kwargs = catch_first_layer_input(model, blocks, samples, batch_size=batch_size)
    og_layer_val_input_batches, _ = catch_first_layer_input(model, blocks, val_samples, batch_size=batch_size)

    @torch.no_grad()
    def forward_layer_batch(layer, input_batched, kwargs, store_device, dtype=dtype, cast_to_dtype=dtype):
        output_batched = []
        layer.to(device)
        for input_batch in input_batched:
            output = layer(input_batch.to(dtype).to(device), **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            if output.dtype != cast_to_dtype:
                output = output.to(cast_to_dtype)
            if output.device != store_device:
                output = output.to(store_device)
            output_batched.append(output)
        layer.cpu()
        return output_batched

    results = {}
    for layer_idx, layer in enumerate(blocks):
        og_layer_output_batches = forward_layer_batch(layer, og_layer_input_batches, kwargs, store_device="cpu")
        og_layer_val_output_batches = forward_layer_batch(layer, og_layer_val_input_batches, kwargs, store_device="cpu")

        if layer_idx > 0:
            layer_input_batches = new_layer_output_batches
            layer_val_input_batches = new_layer_val_output_batches
        else:
            layer_input_batches = og_layer_input_batches
            layer_val_input_batches = og_layer_val_input_batches

        train_input_batches = layer_input_batches
        train_output_batches = og_layer_output_batches

        train_input_batches = CachedTensorShards(train_input_batches, cache_shards, target_device=device)
        train_output_batches = CachedTensorShards(train_output_batches, cache_shards, target_device=device)

        val_input_batches = [b.to(device) for b in layer_val_input_batches]
        val_output_batches = [b.to(device) for b in og_layer_val_output_batches]

        for param in layer.parameters():
            param.requires_grad = False

        linear_modules = get_named_linears(layer)

        for name, old_module in linear_modules.items():
            old_module.to(device)

            num_pairs_factor = 0.5
            weight = old_module.weight.float()
            weight_grouped = weight.view(weight.shape[0], -1, group_size).permute(1, 0, 2)

            all_pairs = get_random_rotation_pairs(
                weight_grouped,
                group_size=group_size,
                num_rotations=num_rotations,
                num_pairs_factor=num_pairs_factor,
                seed=seed + layer_idx,
            )

            all_pairs = [torch.tensor(pairs, device="cpu", dtype=torch.int32) for pairs in all_pairs]
            initial_angles = [torch.zeros(pairs.shape[0], device="cpu") for pairs in all_pairs]
            initial_scales = torch.ones(1, weight.shape[1], dtype=dtype, device=device)

            npairs, angles, mask = transform_to_kernel_data(
                all_pairs,
                initial_angles,
                group_size=group_size,
            )
            npairs = npairs.to(device)
            angles = angles.to(device)
            mask = mask.to(device)
            rotation_pairs = [npairs, angles, mask]
            channel_scales = initial_scales

            new_module = PseudoQuantizedLinear(
                old_module,
                rotation_pairs,
                channel_scales,
                group_size=group_size,
                n_bits=n_bit,
                num_rotations=num_rotations,
            )

            set_module_by_name(layer, name, new_module)
            old_module.cpu()

        layer.to(device).float()

        for step, step_params_dict in enumerate(params_to_optimize):
            optim_params = []
            new_modules = get_named_linears(layer, subclass=PseudoQuantizedLinear)
            for new_module in new_modules.values():
                new_module.set_optim_enabled(
                    **{param_name: True for param_name in step_params_dict.keys()},
                )
                for param_name, lr in step_params_dict.items():
                    optim_params.append(
                        dict(
                            params=new_module.get_optim_params(param_name),
                            lr=lr,
                            weight_decay=weight_decay,
                            betas=betas,
                            eps=eps,
                        )
                    )

            optimize_module(
                layer,
                (train_input_batches, train_output_batches),
                (val_input_batches, val_output_batches),
                kwargs,
                optim_params,
                loss_fn=loss,
                n_iter=epochs[step],
                early_stop=None,
                post_optim_callback=reset_angles_by_mask,
            )

        del (
            train_input_batches,
            train_output_batches,
            val_input_batches,
            val_output_batches,
        )

        layer.to(dtype).to(device)

        new_layer_output_batches = forward_layer_batch(layer, layer_input_batches, kwargs, store_device="cpu")
        new_layer_val_output_batches = forward_layer_batch(layer, layer_val_input_batches, kwargs, store_device="cpu")

        og_layer_input_batches = og_layer_output_batches
        og_layer_val_input_batches = og_layer_val_output_batches

        for name, module in get_named_linears(layer, subclass=PseudoQuantizedLinear).items():
            results[(layer_idx, name)] = module.pseudo_weight().detach().clone()

        layer.cpu()
    return results
