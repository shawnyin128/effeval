"""Token-axis groups, float zero points and the residual window on the KV cache sites.

The reference functions below restate KIVI's quantization (github.com/jy-yuan/KIVI@876b4d2, quant/new_pack.py
`quant_and_pack_kcache`, `quant_and_pack_vcache`, `triton_quantize_and_pack_along_last_dim`, dequantization
`data * scale + mn`) independently of effeval's quantizer.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pydantic import ValidationError  # noqa: E402

from catalogs import CATALOG  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.fragments import FpFormat, IntFormat, QuantRule  # noqa: E402
from effeval.methods.quantization.rtn import RTNConfig  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402
from effeval.validate import validate  # noqa: E402

pytestmark = pytest.mark.compute

GROUP = 32
RESIDUAL = 128


def int_format(bits: int, **overrides) -> IntFormat:
    options = {
        "type": "int",
        "bits": bits,
        "symmetric": False,
        "zero_point": "float",
        "granularity": "per_group",
        "group_size": GROUP,
        **overrides,
    }
    return IntFormat.model_validate(options)


def key_format(bits: int) -> IntFormat:
    return int_format(bits, group_axis="token", residual=RESIDUAL)


def value_format(bits: int) -> IntFormat:
    return int_format(bits, residual=RESIDUAL)


def minmax_quant(data: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """KIVI's grid over `dim` of `data`: (x - min) / scale clamped and rounded, then `* scale + min`."""
    max_int = 2**bits - 1
    mn = data.min(dim=dim, keepdim=True)[0]
    mx = data.max(dim=dim, keepdim=True)[0]
    scale = (mx - mn) / max_int
    codes = ((data - mn) / scale).clamp(0, max_int).round()
    return codes * scale + mn


def reference_keys(k: torch.Tensor, bits: int) -> torch.Tensor:
    """Keys [B, H, T, D]: whole groups of GROUP tokens per channel among all but the last RESIDUAL tokens."""
    batch, heads, tokens, dim = k.shape
    quantized = max(tokens - RESIDUAL, 0)
    quantized -= quantized % GROUP
    if quantized == 0:
        return k
    head = k[:, :, :quantized].reshape(batch, heads, quantized // GROUP, GROUP, dim)
    head = minmax_quant(head, bits, dim=-2).reshape(batch, heads, quantized, dim)
    return torch.cat([head, k[:, :, quantized:]], dim=2)


def reference_values(v: torch.Tensor, bits: int) -> torch.Tensor:
    """Values [B, H, T, D]: groups of GROUP features per token among all but the last RESIDUAL tokens."""
    batch, heads, tokens, dim = v.shape
    quantized = max(tokens - RESIDUAL, 0)
    if quantized == 0:
        return v
    head = v[:, :, :quantized].reshape(batch, heads, quantized, dim // GROUP, GROUP)
    head = minmax_quant(head, bits, dim=-1).reshape(batch, heads, quantized, dim)
    return torch.cat([head, v[:, :, quantized:]], dim=2)


def cache(tokens: int, dim: int = 64) -> torch.Tensor:
    torch.manual_seed(tokens)
    return torch.randn(2, 3, tokens, dim) + 3.0  # all-positive groups: the grid must not be pinned to zero


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("tokens", [100, 159, 160, 200, 300])
def test_keys_match_kivi_per_channel_groups_along_tokens(bits, tokens):
    k = cache(tokens)
    out = fake_quantize(k, key_format(bits), kind="activation")
    assert torch.allclose(out, reference_keys(k, bits), atol=1e-6, rtol=0)


@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize("tokens", [100, 128, 129, 200, 300])
def test_values_match_kivi_per_token_groups_along_features(bits, tokens):
    v = cache(tokens)
    out = fake_quantize(v, value_format(bits), kind="activation")
    assert torch.allclose(out, reference_values(v, bits), atol=1e-6, rtol=0)


def test_residual_window_and_partial_groups_stay_untouched():
    k = cache(170)  # 42 tokens older than the window: one whole group of 32, 10 tokens left over
    out = fake_quantize(k, key_format(2), kind="activation")
    assert not torch.equal(out[:, :, :GROUP], k[:, :, :GROUP])
    assert torch.equal(out[:, :, GROUP:], k[:, :, GROUP:])
    v = cache(170)
    out = fake_quantize(v, value_format(2), kind="activation")
    assert not torch.equal(out[:, :, :42], v[:, :, :42])
    assert torch.equal(out[:, :, 42:], v[:, :, 42:])


def test_a_cache_shorter_than_the_window_is_returned_unchanged():
    k = cache(RESIDUAL)
    assert torch.equal(fake_quantize(k, key_format(2), kind="activation"), k)
    assert torch.equal(fake_quantize(k, value_format(2), kind="activation"), k)


def test_float_zero_point_is_exact_on_the_group_extremes_and_not_pinned_to_zero():
    x = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    fmt = int_format(2, group_size=4, residual=None)
    assert torch.allclose(fake_quantize(x, fmt, kind="activation"), x, atol=1e-6)
    integer = fmt.model_copy(update={"zero_point": None})
    assert not torch.allclose(fake_quantize(x, integer, kind="activation"), x, atol=1e-3)


def test_a_constant_group_is_reproduced_exactly():
    x = torch.full((1, 2, 160, 64), 1.5)
    assert torch.equal(fake_quantize(x, key_format(2), kind="activation"), x)


def test_dtype_and_shape_are_preserved():
    k = cache(200).to(torch.bfloat16)
    out = fake_quantize(k, key_format(4), kind="activation")
    assert out.dtype == torch.bfloat16 and out.shape == k.shape


def test_the_same_options_work_on_fp_formats():
    k = cache(200)
    fmt = FpFormat(
        type="fp",
        encoding="e2m1",
        granularity="per_group",
        group_size=GROUP,
        group_axis="token",
        residual=RESIDUAL,
    )
    out = fake_quantize(k, fmt, kind="activation")
    assert torch.equal(out[:, :, 64:], k[:, :, 64:])
    assert not torch.equal(out[:, :, :64], k[:, :, :64])


def test_sequence_options_are_rejected_on_weights_when_quantizing():
    with pytest.raises(ValueError, match="activations"):
        fake_quantize(torch.randn(64, 64), key_format(2), kind="weight")


@pytest.mark.parametrize(
    "site", ["linear.mlp.down_proj.weight", "linear.attn.q_proj.input", "attn.q", "attn.*"]
)
@pytest.mark.parametrize("option", [{"group_axis": "token"}, {"residual": 128}])
def test_rules_reject_sequence_options_outside_the_kv_sites(site, option):
    quant = {
        "type": "int",
        "bits": 4,
        "symmetric": False,
        "granularity": "per_group",
        "group_size": 32,
        **option,
    }
    with pytest.raises(ValidationError, match=r"attn\.k.*attn\.v"):
        QuantRule(sites=[site], layers="all", quant=quant)
    with pytest.raises(ValidationError, match=r"attn\.k.*attn\.v"):
        RTNConfig(rules=[{"sites": [site], "layers": "all", "quant": quant}])


def test_rules_accept_sequence_options_on_the_kv_sites():
    for site in (["attn.k"], ["attn.v"], ["attn.k", "attn.v"]):
        QuantRule(sites=site, layers="all", quant=key_format(2))


def test_format_options_are_validated():
    with pytest.raises(ValidationError, match="group_axis"):
        IntFormat(type="int", bits=4, symmetric=False, granularity="per_token", group_axis="token")
    with pytest.raises(ValidationError, match="asymmetric"):
        IntFormat(type="int", bits=4, symmetric=True, granularity="per_token", zero_point="float")
    with pytest.raises(ValidationError, match="clip_ratio"):
        IntFormat(
            type="int", bits=4, symmetric=False, granularity="per_token", zero_point="float", clip_ratio=0.9
        )


def test_existing_formats_are_unchanged_by_the_new_options_being_absent():
    fmt = IntFormat(type="int", bits=4, symmetric=False, granularity="per_group", group_size=32)
    assert (fmt.zero_point, fmt.group_axis, fmt.residual) == (None, None, None)
    assert fmt.model_dump(exclude_none=True).keys() == {
        "type",
        "bits",
        "symmetric",
        "granularity",
        "group_size",
        "clip_ratio",
    }


KIVI_PRESETS = ["kivi_k2v2", "kivi_k4v4"]


@pytest.mark.parametrize("name", KIVI_PRESETS)
def test_kivi_presets_resolve_and_validate(name):
    preset = builtin_catalog().presets["rtn"][name]
    rules = preset.values["method.config.rules"].value
    assert [rule["sites"] for rule in rules] == [["attn.k"], ["attn.v"]]
    raw = {
        "model": "qwen3-0.6b-base",
        "method": {"name": "rtn", "preset": name},
        "field": "quantization",
        "tasks": [{"name": "native/wikitext2_ppl"}],
    }
    assert [i.code for i in validate(raw, CATALOG) if i.severity.value == "error"] == []


def tiny_qwen3():
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=512,
    )
    return transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()


@torch.no_grad()
@pytest.mark.parametrize("name", KIVI_PRESETS)
def test_preset_changes_the_attention_operands_only_outside_the_residual_window(name, monkeypatch):
    bits = 2 if name == "kivi_k2v2" else 4
    rules = [
        QuantRule.model_validate(r)
        for r in builtin_catalog().presets["rtn"][name].values["method.config.rules"].value
    ]
    baseline_rule = QuantRule(
        sites=["attn.q"], quant={"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token"}
    )
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    real = sites.sdpa_attention_forward

    def record(module, query, key, value, *args, **kwargs):
        seen.append((key.clone(), value.clone()))
        return real(module, query, key, value, *args, **kwargs)

    monkeypatch.setattr(sites, "sdpa_attention_forward", record)
    input_ids = torch.randint(0, 97, (1, 200))
    model = tiny_qwen3()

    attachment = sites.attach(model, [baseline_rule])  # layer 0 K/V are untouched by a query-only rule
    model(input_ids=input_ids)
    attachment.remove()
    attachment = sites.attach(model, rules)
    model(input_ids=input_ids)
    attachment.remove()
    (base_k, base_v), (quant_k, quant_v) = seen[0], seen[2]

    assert base_k.shape == (1, 2, 200, 64)
    assert torch.allclose(quant_k, reference_keys(base_k, bits), atol=1e-5, rtol=0)
    assert torch.allclose(quant_v, reference_values(base_v, bits), atol=1e-5, rtol=0)
    assert torch.equal(quant_k[:, :, 64:], base_k[:, :, 64:]) and not torch.equal(
        quant_k[:, :, :64], base_k[:, :, :64]
    )
    assert torch.equal(quant_v[:, :, 72:], base_v[:, :, 72:]) and not torch.equal(
        quant_v[:, :, :72], base_v[:, :, :72]
    )
