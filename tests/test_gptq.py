import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import reference.gptq_official as gptq_official  # noqa: E402
from effeval.core.fragments import ClipSearch, IntFormat, QuantRule  # noqa: E402
from effeval.methods.quantization.gptq import (  # noqa: E402
    GPTQConfig,
    HessianAccumulator,
    quantize_model,
    quantize_weight,
)
from effeval.ops import sites  # noqa: E402
from reference.gptq_official import GPTQ, Quantizer  # noqa: E402
from reference.quarot_quant_official import WeightQuantizer, sym_quant_dequant  # noqa: E402
from test_sites import tiny_qwen3  # noqa: E402

pytestmark = pytest.mark.compute


def _inputs(seed: int, columns: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 40, columns, generator=gen) for _ in range(6)]


@pytest.mark.parametrize(
    ("groupsize", "act_order", "static_groups"),
    [
        (None, False, False),
        (None, True, False),
        (32, False, False),
        (16, False, False),  # groups start mid-block: exercises fitting on the outer W
        (16, True, False),
        (16, True, True),
    ],
)
def test_quantize_weight_bit_identical_to_official_fasterquant(groupsize, act_order, static_groups):
    torch.manual_seed(0)
    layer = torch.nn.Linear(96, 48, bias=False)
    layer.weight.data[:, 7] = 0.0
    inputs = _inputs(1, 96)
    for x in inputs:
        x[..., 5] = 0.0  # a dead input column exercises the official dead-column branch

    official_layer = torch.nn.Linear(96, 48, bias=False)
    official_layer.weight.data.copy_(layer.weight.data)
    official = GPTQ(official_layer)
    official.quantizer = Quantizer()
    official.quantizer.configure(4, perchannel=True, sym=False, mse=False)
    ours = HessianAccumulator(96, torch.device("cpu"))
    for x in inputs:
        official.add_batch(x, official_layer(x))
        ours.add_batch(x)
    assert torch.equal(ours.H, official.H)

    official.fasterquant(
        blocksize=32,
        percdamp=0.01,
        groupsize=groupsize or -1,
        actorder=act_order,
        static_groups=static_groups,
    )
    fmt = IntFormat(
        type="int",
        bits=4,
        symmetric=False,
        granularity="per_group" if groupsize else "per_channel",
        group_size=groupsize,
    )
    got = quantize_weight(
        layer.weight.data,
        ours.H,
        fmt,
        blocksize=32,
        percdamp=0.01,
        act_order=act_order,
        static_groups=static_groups,
    )
    assert torch.equal(got, official_layer.weight.data)


class _QuaRotSymmetricQuantizer(Quantizer):
    """The official GPTQ Quantizer interface with QuaRot's symmetric grid (codes -2^(b-1)..2^(b-1)-1).

    The official GPTQ symmetric grid is a different one (zero point 2^(b-1), step 2*absmax/(2^b-1)); effeval's
    symmetric grid is QuaRot's, so the GPTQ loop is compared with that grid plugged in.
    """

    def configure(self, bits, perchannel=False, sym=True, mse=False, **_):
        self.bits = bits
        self.mse = mse

    def find_params(self, x, weight=False):
        grid = WeightQuantizer()
        grid.configure(self.bits, perchannel=True, sym=True, mse=self.mse)
        grid.find_params(x)
        self.scale, self.zero, self.maxq = grid.scale, grid.zero, grid.maxq


@pytest.mark.parametrize("mse", [False, True], ids=["minmax", "clip_search"])
@pytest.mark.parametrize("symmetric", [False, True], ids=["asym", "sym"])
@pytest.mark.parametrize("bits", [3, 4, 8])
@pytest.mark.parametrize(
    ("groupsize", "act_order", "static_groups"),
    [(None, False, False), (None, True, False), (128, False, False), (128, True, False), (128, True, True)],
)
def test_quantize_weight_bit_identical_to_official_fasterquant_across_grids(
    groupsize, act_order, static_groups, bits, symmetric, mse, monkeypatch
):
    torch.manual_seed(bits)
    layer = torch.nn.Linear(256, 32, bias=False)
    layer.weight.data[:, 0] *= 10.0  # outlier column: the clip search has something to clip
    inputs = _inputs(2, 256)

    official_layer = torch.nn.Linear(256, 32, bias=False)
    official_layer.weight.data.copy_(layer.weight.data)
    official = GPTQ(official_layer)
    if symmetric:
        official.quantizer = _QuaRotSymmetricQuantizer()
        monkeypatch.setattr(
            gptq_official, "quantize", lambda x, scale, _zero, maxq: sym_quant_dequant(x, scale, maxq)
        )
    else:
        official.quantizer = Quantizer()
    official.quantizer.configure(bits, perchannel=True, sym=symmetric, mse=mse)
    ours = HessianAccumulator(256, torch.device("cpu"))
    for x in inputs:
        official.add_batch(x, official_layer(x))
        ours.add_batch(x)
    official.fasterquant(
        blocksize=128,
        percdamp=0.01,
        groupsize=groupsize or -1,
        actorder=act_order,
        static_groups=static_groups,
    )
    fmt = IntFormat(
        type="int",
        bits=bits,
        symmetric=symmetric,
        granularity="per_group" if groupsize else "per_channel",
        group_size=groupsize,
        clip_search=ClipSearch(type="mse") if mse else None,
    )
    got = quantize_weight(
        layer.weight.data,
        ours.H,
        fmt,
        blocksize=128,
        percdamp=0.01,
        act_order=act_order,
        static_groups=static_groups,
    )
    assert torch.equal(got, official_layer.weight.data)


def _config(**overrides):
    base = {
        "rules": [
            {
                "sites": ["linear.*.weight"],
                "layers": "all",
                "quant": {"type": "int", "bits": 3, "symmetric": False, "granularity": "per_channel"},
            }
        ],
        "act_order": False,
        "static_groups": False,
        "percdamp": 0.01,
        "blocksize": 128,
        "calibration": {"dataset": "c4", "n_samples": 4, "seqlen": 16, "seed": 0},
    }
    base.update(overrides)
    return GPTQConfig.model_validate(base)


def test_quantize_model_puts_every_targeted_weight_on_a_3bit_grid_and_leaves_others():
    model = tiny_qwen3()
    embed_before = model.model.embed_tokens.weight.clone()
    lm_head_before = model.lm_head.weight.clone()
    gen = torch.Generator().manual_seed(3)
    windows = [torch.randint(0, 97, (1, 16), generator=gen) for _ in range(4)]
    quantize_model(model, windows, _config())
    for layer in model.model.layers:
        for linear in (layer.self_attn.q_proj, layer.mlp.down_proj):
            levels = torch.tensor([row.unique().numel() for row in linear.weight.data])
            assert levels.max() <= 8
    assert torch.equal(model.model.embed_tokens.weight, embed_before)
    assert torch.equal(model.lm_head.weight, lm_head_before)


def test_config_rejects_activation_sites_and_float_formats():
    with pytest.raises(ValueError, match=r"at least one rule on '\.weight' sites"):
        _config(
            rules=[
                QuantRule(
                    sites=["linear.*.input"],
                    quant={"type": "int", "bits": 4, "symmetric": False, "granularity": "per_token"},
                )
            ]
        )
    with pytest.raises(ValueError, match="integer format"):
        _config(
            rules=[
                QuantRule(
                    sites=["linear.*.weight"],
                    quant={"type": "fp", "encoding": "e4m3", "granularity": "per_channel"},
                )
            ]
        )


INPUT_RULE = {
    "sites": ["linear.*.input"],
    "quant": {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token"},
}


def test_config_splits_weight_rules_from_rules_attached_afterwards():
    cfg = _config(rules=[*_config().rules, INPUT_RULE])
    assert [r.sites for r in cfg.weight_rules] == [["linear.*.weight"]]
    assert [r.sites for r in cfg.other_rules] == [["linear.*.input"]]
    with pytest.raises(ValueError, match="all weights or all non-weights"):
        _config(rules=[{**INPUT_RULE, "sites": ["linear.*.weight", "linear.*.input"]}])
    with pytest.raises(ValueError, match="use clip_search"):
        _config(
            rules=[
                {
                    "sites": ["linear.*.weight"],
                    "quant": {
                        "type": "int",
                        "bits": 4,
                        "symmetric": True,
                        "granularity": "per_channel",
                        "clip_ratio": 0.9,
                    },
                }
            ]
        )


def _probe_ids() -> torch.Tensor:
    return torch.randint(0, 97, (1, 16), generator=torch.Generator().manual_seed(9))


def test_gptq_with_input_rule_equals_gptq_without_it_plus_the_hook():
    gen = torch.Generator().manual_seed(3)
    windows = [torch.randint(0, 97, (1, 16), generator=gen) for _ in range(4)]
    with_rule = tiny_qwen3()
    quantize_model(with_rule, windows, _config(rules=[*_config().rules, INPUT_RULE]))
    separate = tiny_qwen3()
    quantize_model(separate, windows, _config())
    sites.attach(separate, [QuantRule.model_validate(INPUT_RULE)])
    for (name, a), (_, b) in zip(with_rule.named_parameters(), separate.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    ids = _probe_ids()
    with torch.no_grad():
        assert torch.equal(with_rule(ids).logits, separate(ids).logits)
        weights_only = tiny_qwen3()
        quantize_model(weights_only, windows, _config())
        assert not torch.equal(with_rule(ids).logits, weights_only(ids).logits)  # the input hook is active


def test_gptq_hessian_is_built_from_the_input_after_forward_pre_hooks():
    gen = torch.Generator().manual_seed(4)
    windows = [torch.randint(0, 97, (1, 16), generator=gen) for _ in range(4)]
    model = tiny_qwen3()
    down = model.model.layers[0].mlp.down_proj
    weight = down.weight.data.clone()
    channel_scale = torch.ones(96)
    channel_scale[:48] = 10.0
    raw: list[torch.Tensor] = []
    down.register_forward_pre_hook(lambda _m, args: raw.append(args[0].detach().clone()))
    down.register_forward_pre_hook(lambda _m, args: (args[0] * channel_scale,))
    cfg = _config()
    quantize_model(model, windows, cfg)

    fmt = cfg.weight_rules[0].quant
    assert len(raw) >= len(windows)  # the first len(windows) calls are the Hessian pass of layer 0

    def solve(scale: torch.Tensor) -> torch.Tensor:
        hessian = HessianAccumulator(96, torch.device("cpu"))
        for x in raw[: len(windows)]:
            hessian.add_batch(x * scale)
        return quantize_weight(
            weight,
            hessian.H,
            fmt,
            blocksize=cfg.blocksize,
            percdamp=cfg.percdamp,
            act_order=False,
            static_groups=False,
        )

    assert torch.equal(down.weight.data, solve(channel_scale))
    assert not torch.equal(down.weight.data, solve(torch.ones(96)))  # the transform changes the solution
