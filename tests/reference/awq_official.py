# Vendored reference implementation for tests only.
# Source: https://github.com/mit-han-lab/llm-awq (commit d6e797a42b9ef7778de8ee2352116e0f48a78d61),
# files awq/quantize/auto_scale.py (get_act_scale, scale_ln_fcs, scale_fc_fc, `_search_module_scale` inside
# `auto_scale_block`), awq/quantize/auto_clip.py (`auto_clip_layer`, `apply_clip`) and awq/quantize/quantizer.py
# (`pseudo_quantize_tensor`).
# Copyright (c) 2023 MIT HAN Lab. Licensed under the MIT License
# (https://github.com/mit-han-lab/llm-awq/blob/main/LICENSE): permission is granted, free of charge, to use,
# copy, modify, merge, publish and distribute this software, provided this copyright and permission notice is
# included in all copies or substantial portions; the software is provided "as is", without warranty of any kind.
# Modifications, each marked "ADAPTED" at its line:
#  - `_search_module_scale` is lifted out of `auto_scale_block` into a module-level function; `w_quantize_func`
#    (a closure over `pseudo_quantize_tensor` there) becomes an argument so a test can inject effeval's quantizer;
#    it returns (scales, ratio) instead of scales (the ratio is compared); the debug `print(history)` is dropped.
#    The tuple-output handling (`isinstance(org_out, tuple)`) already in the official code covers transformers 5
#    attention modules returning (output, weights).
#  - `org_sd` in the search keeps `.cpu()` and adds `.clone()`: on a CPU tensor `.cpu()` returns the live parameter, so
#    the in-place `mul_` would corrupt the saved state; on the official CUDA run `.cpu()` already copies, so the clone
#    reproduces the official behaviour on CPU.
#  - `auto_clip_layer` takes `quantize_func` (default: the vendored `pseudo_quantize_tensor` with `q_config`);
#    the `oc_batch_size` rule `256 if divisible else 64` plus its assert is replaced by a fallback to the whole
#    output dimension, because tiny test layers have fewer than 64 output channels; batching over output channels
#    does not change any value. The `gc` / `torch.cuda.empty_cache()` housekeeping is dropped.
#  - `apply_clip` takes the layer and its max_val directly (official: a module + name list, `.cuda()` / `.cpu()`).
#  - `scale_ln_fcs` / `scale_fc_fc` drop the NaN assertions; `scale_fc_fc` is verbatim otherwise.
#  - everything else is verbatim.
# ruff: noqa
import torch
import torch.nn as nn


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device).to(ln.weight.dtype)

    ln.weight.div_(scales)
    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device).to(fc1.weight.dtype)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))


@torch.no_grad()
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


@torch.no_grad()
def search_module_scale(block, linears2scale: list, x, w_quantize_func, kwargs={}):  # ADAPTED: signature
    # w: co, ci
    # x: n, ci
    x = x.to(next(block.parameters()).device)
    with torch.no_grad():
        org_out = block(x, **kwargs)
        if isinstance(org_out, tuple):
            org_out = org_out[0]

    x_max = get_act_scale(x)

    best_error = float("inf")
    best_ratio = -1
    best_scales = None

    n_grid = 20
    history = []

    org_sd = {k: v.cpu().clone() for k, v in block.state_dict().items()}  # ADAPTED: .clone(), see header
    for ratio in range(n_grid):
        ratio = ratio * 1 / n_grid
        scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
        scales = scales / (scales.max() * scales.min()).sqrt()
        for fc in linears2scale:
            fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
            fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
        out = block(x, **kwargs)
        if isinstance(out, tuple):
            out = out[0]

        loss = (
            (org_out - out).float().pow(2).mean().item()
        )  # float prevents overflow
        history.append(loss)
        is_best = loss < best_error
        if is_best:
            best_error = loss
            best_ratio = ratio
            best_scales = scales
        block.load_state_dict(org_sd)
    if best_ratio == -1:
        raise Exception  # ADAPTED: debug print dropped
    # print(best_ratio)
    best_scales = best_scales.view(-1)

    assert torch.isnan(best_scales).sum() == 0, best_scales
    return best_scales.detach(), best_ratio  # ADAPTED: the ratio is returned as well


@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512, quantize_func=None
):  # ADAPTED: quantize_func argument
    assert w.dim() == 2
    org_w_shape = w.shape
    # w           [co, ci]      -> [co, 1, n_group, group size]
    # input_feat  [n_token, ci] -> [1, n_token, n_group, group size]
    group_size = (
        q_config["q_group_size"] if q_config["q_group_size"] > 0 else w.shape[1]
    )
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(1, input_feat.shape[0], -1, group_size)
    input_feat = input_feat[:, 0 :: input_feat.shape[1] // n_sample_token]
    w = w.reshape(w.shape[0], 1, -1, group_size)

    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    if w.shape[0] % oc_batch_size != 0:  # ADAPTED: official asserts here; tiny layers use one batch
        oc_batch_size = w.shape[0]
    w_all = w
    best_max_val_all = []

    for i_b in range(w.shape[0] // oc_batch_size):
        w = w_all[i_b * oc_batch_size : (i_b + 1) * oc_batch_size]

        org_max_val = w.abs().amax(dim=-1, keepdim=True)  # co, 1, n_group, 1

        best_max_val = org_max_val.clone()
        min_errs = torch.ones_like(org_max_val) * 1e9
        input_feat = input_feat.to(w.device)
        org_out = (input_feat * w).sum(dim=-1)  # co, n_token, n_group

        for i_s in range(int(max_shrink * n_grid)):
            max_val = org_max_val * (1 - i_s / n_grid)
            min_val = -max_val
            cur_w = torch.clamp(w, min_val, max_val)
            if quantize_func is None:  # ADAPTED: the injected quantizer replaces the call below when given
                q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)
            else:
                q_w = quantize_func(cur_w)
            cur_out = (input_feat * q_w).sum(dim=-1)

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
            del cur_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_max_val[cur_best_idx] = max_val[cur_best_idx]
        best_max_val_all.append(best_max_val)

    best_max_val = torch.cat(best_max_val_all, dim=0)

    del input_feat
    del org_out
    return best_max_val.squeeze(1)


@torch.no_grad()
def apply_clip(layer, max_val):  # ADAPTED: takes the layer directly (official: module + name, moves to cuda)
    max_val = max_val.to(layer.weight.device).to(layer.weight.dtype)
    org_shape = layer.weight.shape
    layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
    layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
    layer.weight.data = layer.weight.data.reshape(org_shape)
