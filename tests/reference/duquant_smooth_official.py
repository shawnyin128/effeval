# Vendored reference implementation for tests only.
# Source: https://github.com/Hsu1023/DuQuant (commit d56cfc6fe97c34c0eb100fec82fe439865905679),
# files models/transformation.py (`smooth_ln_fcs_inplace`, `smooth_fc_fc_inplace`, `smooth_fc_inplace`) and the
# scale computation of quantize/duquant.py#L209-L213.
# Copyright (c) 2024 Haobo Xu, Haokun Lin. Licensed under the MIT License
# (https://github.com/Hsu1023/DuQuant/blob/main/LICENSE): permission is granted, free of charge, to use, copy,
# modify, merge, publish and distribute this software, provided this copyright and permission notice is included
# in all copies or substantial portions; the software is provided "as is", without warranty of any kind.
# Modifications, each marked "ADAPTED" at its line:
#  - the shift (`shifts`) terms and the bias bookkeeping are removed: for Llama-like models DuQuant uses
#    shift = 0 (duquant.py#L214-L217 `torch.zeros_like(scale)`) and the layers have no bias, so they only add
#    zero buffers; `use_temporary_parameter` flags are dropped (training-time switches of DuQuant's QuantLinear).
#  - `smooth_scale` wraps the scale lines of duquant.py#L209-L213 in a function (official: inline in a loop over
#    QuantLinear modules, `act` taken from act_scales, `weight` from the module).
#  - `smooth_fc_inplace` sets `fc.act_quantizer.smooth_scales` as official; the test gives `fc` a namespace there
#    (official: DuQuant's activation quantizer, which divides its input by it at runtime, quantizer.py#L344-L345).
#  - everything else is verbatim.
# ruff: noqa
import torch

CLIPMIN = 1e-5  # quantize/const.py


@torch.no_grad()
def smooth_scale(act_scale, module_weight, alpha):  # ADAPTED: function wrapper
    act = act_scale.to(device=module_weight.device, dtype=module_weight.dtype).clamp(min=CLIPMIN)
    weight = module_weight.abs().max(dim=0)[0].clamp(min=CLIPMIN)
    scale = (act.pow(alpha) / weight.to(act.device).pow(1 - alpha)).clamp(min=CLIPMIN)
    return scale


@torch.no_grad()
def smooth_ln_fcs_inplace(ln, fcs, scales):  # ADAPTED: no shifts, no bias
    if not isinstance(fcs, list):
        fcs = [fcs]
    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.to(fc.weight.device).view(1, -1))


@torch.no_grad()
def smooth_fc_fc_inplace(fc1, fc2, scales):  # ADAPTED: no shifts, no bias
    fc1.weight.div_(scales.to(fc1.weight.device).view(-1, 1))
    fc2.weight.mul_(scales.to(fc2.weight.device).view(1, -1))


@torch.no_grad()
def smooth_fc_inplace(fc, scales):
    fc.act_quantizer.smooth_scales = scales.view(1, -1)
    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
