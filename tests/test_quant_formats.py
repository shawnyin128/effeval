import pytest

torch = pytest.importorskip("torch")

from effeval.core.fragments import ClipSearch, FpFormat, IntFormat, MxFormat  # noqa: E402
from effeval.ops.quant.formats import fp_max, int_grid, round_to_fp  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402
from reference.gptq_official import Quantizer, quantize  # noqa: E402
from reference.quarot_quant_official import (  # noqa: E402
    WeightQuantizer,
    asym_quant_dequant,
    sym_quant_dequant,
)

pytestmark = pytest.mark.compute


def _gptq_reference(weight: torch.Tensor, bits: int, groupsize: int | None) -> torch.Tensor:
    """Quantize like the official GPTQ/RTN code path (perchannel, asymmetric)."""
    w = weight.float()
    if groupsize is None:
        q = Quantizer()
        q.configure(bits, perchannel=True, sym=False, mse=False)
        q.find_params(w, weight=True)
        return quantize(w, q.scale, q.zero, q.maxq)
    out = torch.empty_like(w)
    for start in range(0, w.shape[1], groupsize):
        q = Quantizer()
        q.configure(bits, perchannel=True, sym=False, mse=False)
        q.find_params(w[:, start : start + groupsize], weight=True)
        out[:, start : start + groupsize] = quantize(w[:, start : start + groupsize], q.scale, q.zero, q.maxq)
    return out


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_int_asymmetric_per_channel_bit_identical_to_official_gptq(bits, dtype):
    gen = torch.Generator().manual_seed(bits)
    weight = (torch.randn(64, 96, generator=gen) * 0.05).to(dtype)
    weight[3] = 0.0  # all-zero row exercises the xmin=-1/xmax=+1 fallback
    weight[5] = weight[5].abs()  # strictly non-negative row: zero must still be on the grid
    fmt = IntFormat(type="int", bits=bits, symmetric=False, granularity="per_channel")
    ours = fake_quantize(weight, fmt, kind="weight")
    reference = _gptq_reference(weight, bits, None).to(dtype)
    assert ours.dtype == dtype
    assert torch.equal(ours, reference)


def test_int_asymmetric_per_group_bit_identical_to_official_gptq():
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(32, 256, generator=gen)
    fmt = IntFormat(type="int", bits=4, symmetric=False, granularity="per_group", group_size=128)
    assert torch.equal(fake_quantize(weight, fmt, kind="weight"), _gptq_reference(weight, 4, 128))


def test_int_symmetric_levels_and_range():
    x = torch.tensor([[-1.0, -0.5, 0.0, 0.26, 1.0]])
    fmt = IntFormat(type="int", bits=3, symmetric=True, granularity="per_token")
    out = fake_quantize(x, fmt, kind="activation")
    scale = 1.0 / 3  # absmax / (2^(3-1) - 1)
    codes = out / scale
    assert torch.allclose(codes, torch.round(codes))
    assert codes.min() >= -4 and codes.max() <= 3
    assert torch.allclose(out[0, -1], torch.tensor(1.0))


@pytest.mark.parametrize(
    ("encoding", "torch_dtype"), [("e4m3", torch.float8_e4m3fn), ("e5m2", torch.float8_e5m2)]
)
def test_fp8_rounding_matches_torch_cast(encoding, torch_dtype):
    e, m = int(encoding[1]), int(encoding[3])
    special = "fn" if encoding == "e4m3" else "ieee"
    max_value = fp_max(e, m, special)
    gen = torch.Generator().manual_seed(1)
    # normals and subnormals inside the finite range, where torch casts with round-to-nearest-even
    x = torch.cat(
        [
            torch.randn(4096, generator=gen) * max_value / 8,
            torch.randn(4096, generator=gen) * 2.0 ** (2 - 2 ** (e - 1)),
            torch.tensor([0.0, -0.0, max_value, -max_value]),
        ]
    ).clamp(-max_value, max_value)
    ours = round_to_fp(x, e, m, max_value)
    reference = x.to(torch_dtype).float()
    assert torch.equal(ours, reference)


def test_fp_max_known_formats():
    assert fp_max(4, 3, "fn") == 448.0
    assert fp_max(5, 2, "ieee") == 57344.0
    assert fp_max(2, 1, "none") == 6.0
    assert fp_max(3, 2, "none") == 28.0
    assert fp_max(2, 3, "none") == 7.5


def test_e2m1_rounds_to_grid_with_ties_to_even_and_saturates():
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 100.0, -2.6])
    out = round_to_fp(x, 2, 1, 6.0)
    # ties go to the even mantissa: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4
    assert out.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0, -3.0]
    assert all(abs(v) in grid.tolist() for v in out.tolist())


def test_fp_scaled_maps_absmax_to_format_max():
    x = torch.tensor([[0.1, -3.0, 1.2], [0.0, 0.0, 0.0]])
    fmt = FpFormat(type="fp", encoding="e2m1", granularity="per_token")
    out = fake_quantize(x, fmt, kind="activation")
    assert out[0, 1] == -3.0  # absmax is exactly representable after scaling
    assert torch.equal(out[1], torch.zeros(3))


def test_mx_e8m0_power_of_two_scale():
    block = torch.tensor(
        [[6.0, 3.0, 1.0, 0.1]]
    )  # absmax 6 -> floor(log2 6) - emax(e2m1)=2 - 2 = 0 -> scale 1
    fmt = MxFormat(type="mx", element="e2m1", block_size=4, scale="e8m0")
    assert fake_quantize(block, fmt, kind="activation").tolist() == [[6.0, 3.0, 1.0, 0.0]]
    scaled = block * 8  # scale becomes 2^3, elements identical
    assert fake_quantize(scaled, fmt, kind="activation").tolist() == [[48.0, 24.0, 8.0, 0.0]]


def test_mx_e4m3_two_level_scale_keeps_block_absmax_close():
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(8, 64, generator=gen)
    fmt = MxFormat(type="mx", element="e2m1", block_size=16, scale="e4m3")
    out = fake_quantize(x, fmt, kind="activation")
    assert out.shape == x.shape
    rel = (out - x).abs().reshape(-1, 16).amax(1) / x.abs().reshape(-1, 16).amax(1)
    assert rel.max() < 0.35  # coarse sanity bound for 4-bit elements


def test_granularity_errors_are_explicit():
    w = torch.randn(4, 8)
    with pytest.raises(ValueError, match="per_token"):
        fake_quantize(
            w, IntFormat(type="int", bits=4, symmetric=True, granularity="per_token"), kind="weight"
        )
    with pytest.raises(ValueError, match="divisible"):
        fake_quantize(
            w,
            IntFormat(type="int", bits=4, symmetric=True, granularity="per_group", group_size=3),
            kind="weight",
        )


def test_activation_per_channel_reduces_across_positions():
    x = torch.tensor([[1.0, 100.0], [0.5, 50.0], [0.25, 25.0]])
    fmt = IntFormat(type="int", bits=8, symmetric=True, granularity="per_channel")
    out = fake_quantize(x, fmt, kind="activation")
    # each column has its own scale, so the small column keeps its resolution
    assert torch.allclose(out, x, rtol=1e-2)


def _heavy_tailed(rows: int, columns: int, seed: int) -> torch.Tensor:
    """Gaussian rows with a few outliers, so that clipping below the min-max range lowers the error."""
    gen = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=gen) * 0.05
    weight[:, 0] *= 12.0
    weight[::3, 1] *= 8.0
    return weight


def _quarot_grid(units: torch.Tensor, bits: int, symmetric: bool, mse: bool):
    quantizer = WeightQuantizer()
    quantizer.configure(bits, perchannel=True, sym=symmetric, mse=mse)
    quantizer.find_params(units.clone())
    return quantizer


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("bits", [3, 4, 8])
def test_clip_search_scale_and_zero_equal_quarot_weight_quantizer_mse(bits, symmetric):
    units = _heavy_tailed(48, 256, seed=bits)
    official = _quarot_grid(units, bits, symmetric, mse=True)
    ours = int_grid(units, bits, symmetric, 1.0, ClipSearch(type="mse"))
    assert torch.equal(ours.scale, official.scale)
    assert torch.equal(ours.zero, official.zero)
    if bits <= 4:  # at 8 bits the full range is already the best candidate for every row
        no_search = _quarot_grid(units, bits, symmetric, mse=False)
        assert not torch.equal(official.scale, no_search.scale)  # the search actually moves some rows


@pytest.mark.parametrize("symmetric", [False, True])
def test_clip_search_options_follow_quarot_grid_max_shrink_norm(symmetric):
    units = _heavy_tailed(32, 128, seed=7)
    quantizer = WeightQuantizer()
    quantizer.configure(4, perchannel=True, sym=symmetric, mse=True, norm=2.0, grid=40, maxshrink=0.5)
    quantizer.find_params(units.clone())
    ours = int_grid(units, 4, symmetric, 1.0, ClipSearch(type="mse", grid=40, max_shrink=0.5, norm=2.0))
    assert torch.equal(ours.scale, quantizer.scale)
    assert torch.equal(ours.zero, quantizer.zero)


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("group_size", [None, 128])
def test_fake_quantize_with_clip_search_equals_quarot_quantize_dequantize(symmetric, group_size):
    weight = _heavy_tailed(24, 256, seed=3)
    fmt = IntFormat(
        type="int",
        bits=4,
        symmetric=symmetric,
        granularity="per_channel" if group_size is None else "per_group",
        group_size=group_size,
        clip_search=ClipSearch(type="mse"),
    )
    units = weight if group_size is None else weight.reshape(-1, group_size)
    official = _quarot_grid(units, 4, symmetric, mse=True)
    expected = (
        sym_quant_dequant(units, official.scale, official.maxq)
        if symmetric
        else asym_quant_dequant(units, official.scale, official.zero, official.maxq)
    ).reshape(weight.shape)
    assert torch.equal(fake_quantize(weight, fmt, kind="weight"), expected)


def test_clip_search_beats_min_max_and_leaves_dead_rows_at_zero():
    weight = _heavy_tailed(16, 256, seed=5)
    weight[2] = 0.0
    plain = IntFormat(type="int", bits=4, symmetric=False, granularity="per_channel")
    searched = plain.model_copy(update={"clip_search": ClipSearch(type="mse")})
    err_plain = (fake_quantize(weight, plain, kind="weight") - weight).abs().pow(2.4).sum(1)
    err_searched = (fake_quantize(weight, searched, kind="weight") - weight).abs().pow(2.4).sum(1)
    assert (err_searched <= err_plain).all()
    assert (err_searched < err_plain).any()
    assert torch.equal(fake_quantize(weight, searched, kind="weight")[2], torch.zeros(256))


def test_clip_search_validation():
    with pytest.raises(ValueError, match="replaces clip_ratio"):
        IntFormat(
            type="int",
            bits=4,
            symmetric=True,
            granularity="per_channel",
            clip_ratio=0.9,
            clip_search=ClipSearch(type="mse"),
        )
    fmt = IntFormat(
        type="int", bits=4, symmetric=True, granularity="per_token", clip_search=ClipSearch(type="mse")
    )
    with pytest.raises(ValueError, match="weights only"):
        fake_quantize(torch.randn(2, 8), fmt, kind="activation")
    with pytest.raises(ValueError, match="at least 1"):
        int_grid(torch.randn(2, 8), 4, True, 1.0, ClipSearch(type="mse", grid=1, max_shrink=0.5))


@pytest.mark.parametrize("group_size", [None, 128])
@pytest.mark.parametrize("bits", [3, 4, 8])
def test_symmetric_rtn_equals_quarot_weight_quantizer(bits, group_size):
    weight = _heavy_tailed(24, 256, seed=bits + 10)
    fmt = IntFormat(
        type="int",
        bits=bits,
        symmetric=True,
        granularity="per_channel" if group_size is None else "per_group",
        group_size=group_size,
    )
    units = weight if group_size is None else weight.reshape(-1, group_size)
    official = _quarot_grid(units, bits, True, mse=False)
    expected = sym_quant_dequant(units, official.scale, official.maxq).reshape(weight.shape)
    assert torch.equal(fake_quantize(weight, fmt, kind="weight"), expected)


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_asymmetric_per_group_128_bit_identical_to_official_gptq_for_each_bit_width(bits):
    gen = torch.Generator().manual_seed(bits)
    weight = torch.randn(32, 256, generator=gen) * 0.05
    weight[:, 128:] = weight[:, 128:].abs()  # second group is non-negative: zero stays inside the range
    fmt = IntFormat(type="int", bits=bits, symmetric=False, granularity="per_group", group_size=128)
    assert torch.equal(fake_quantize(weight, fmt, kind="weight"), _gptq_reference(weight, bits, 128))
