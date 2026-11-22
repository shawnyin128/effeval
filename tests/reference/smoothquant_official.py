# Vendored reference implementation for tests only.
# Source: https://github.com/mit-han-lab/smoothquant (commit c61476d728e42ae0d8a35e7e78494edcac3237b5),
# files smoothquant/smooth.py (`smooth_ln_fcs_llama_like`) and smoothquant/calibration.py (`get_act_scales`).
# Copyright (c) 2022 MIT HAN Lab. Licensed under the MIT License
# (https://github.com/mit-han-lab/smoothquant/blob/main/LICENSE): permission is granted, free of charge, to use,
# copy, modify, merge, publish and distribute this software, provided this copyright and permission notice is
# included in all copies or substantial portions; the software is provided "as is", without warranty of any kind.
# Modifications, each marked "ADAPTED" at its line:
#  - smooth.py: only the imports and the isinstance tuple (Mistral / Mixtral norm classes dropped, the tiny
#    test model is Llama); the body of smooth_ln_fcs_llama_like is verbatim.
#  - calibration.py: `get_act_scales` takes a list of token-id tensors instead of a dataset path, a tokenizer and
#    the datasets / shuffle / tokenizer calls; the hook code is verbatim; the tqdm progress bar is dropped.
# ruff: noqa
import functools

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaRMSNorm  # ADAPTED: Mistral/Mixtral imports dropped


@torch.no_grad()
def smooth_ln_fcs_llama_like(ln, fcs, act_scales, alpha=0.5):
    if not isinstance(fcs, list):
        fcs = [fcs]
    assert isinstance(ln, (LlamaRMSNorm,))  # ADAPTED: Mistral/Mixtral norm classes dropped from the tuple
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()
    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    ln.weight.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales  # ADAPTED: the scales are returned so the test can compare them (official returns None)


def get_act_scales(model, token_windows):  # ADAPTED: signature (official: model, tokenizer, dataset_path, ...)
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    for input_ids in token_windows:  # ADAPTED: official loops over tokenized dataset documents
        model(input_ids.to(device))

    for h in hooks:
        h.remove()

    return act_scales
