import importlib.util
import os
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from effeval.core.fragments import CodebookFormat, IntFormat, QuantRule

torch = pytest.importorskip("torch")

from effeval.ops import codebook  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402


def _fmt(**overrides):
    values = dict(type="codebook", centroids=4, granularity="per_channel")
    return CodebookFormat(**{**values, **overrides})


def _official(project: str, relative: str):
    """Import a module of the official CodeQuant sources found under $EFFEVAL_CODEQUANT_REF/<project>."""
    root = os.environ.get("EFFEVAL_CODEQUANT_REF")
    if not root or not (Path(root) / project / relative).is_file():
        pytest.skip("EFFEVAL_CODEQUANT_REF does not point at the official CodeQuant sources")
    project_root = Path(root) / project
    before = set(sys.modules)
    sys.path.insert(0, str(project_root))
    try:
        path = project_root / relative
        spec = importlib.util.spec_from_file_location(f"official_{project}_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(project_root))
        for name in set(sys.modules) - before:
            if name == "utils" or name.startswith("utils."):
                del sys.modules[name]
    return module


# format and rule validation


def test_codebook_rule_is_accepted_on_weight_sites_only():
    rule = QuantRule(sites=["linear.attn.q_proj.weight", "moe.router.weight"], quant=_fmt())
    assert isinstance(rule.quant, CodebookFormat)
    for site in ("linear.attn.q_proj.input", "linear.*", "attn.k"):
        with pytest.raises(ValidationError, match=r"codebook applies to .weight sites only"):
            QuantRule(sites=["linear.mlp.up_proj.weight", site], quant=_fmt())
    QuantRule(
        sites=["linear.*.weight"],
        quant=IntFormat(type="int", bits=4, symmetric=True, granularity="per_channel"),
    )
    QuantRule(
        sites=["linear.*.input"], quant=IntFormat(type="int", bits=4, symmetric=True, granularity="per_token")
    )


def test_codebook_format_defaults_and_group_size_rules():
    fmt = (
        TypeAdapter(QuantRule)
        .validate_python(
            {
                "sites": ["linear.*.weight"],
                "quant": {"type": "codebook", "centroids": 16, "granularity": "per_channel"},
            }
        )
        .quant
    )
    assert (fmt.kmeans.iters, fmt.kmeans.tol, fmt.kmeans.seed, fmt.kmeans.compute_dtype) == (
        300,
        1e-6,
        42,
        "weight",
    )
    with pytest.raises(ValidationError, match="requires group_size"):
        _fmt(granularity="per_group")
    with pytest.raises(ValidationError, match="only valid with granularity"):
        _fmt(group_size=8)
    with pytest.raises(ValidationError):
        _fmt(centroids=257)
    with pytest.raises(ValidationError):
        _fmt(kmeans={"iters": 3, "unknown": 1})


# invariants


def _units(rows=5, columns=60, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, columns, generator=generator, dtype=dtype)


def test_assignments_are_the_nearest_centroid():
    weight = _units()
    units, mask = codebook.to_units(weight, None)
    centroids, assignments = codebook.kmeans(units, mask, 6)
    nearest = (weight.unsqueeze(2) - centroids.unsqueeze(1)).abs().argmin(dim=2)
    assert assignments.dtype == torch.uint8
    assert torch.equal(assignments.long(), nearest)


def test_centroids_are_member_means_at_convergence():
    units, mask = codebook.to_units(_units(), None)
    centroids, assignments = codebook.kmeans(units, mask, 4, iters=1000, tol=1e-12)
    checked = 0
    for row in range(units.shape[0]):
        for cluster in range(4):
            members = units[row][assignments[row] == cluster]
            if len(members):
                torch.testing.assert_close(centroids[row, cluster], members.mean(), atol=1e-9, rtol=0)
                checked += 1
    assert checked >= 15


def test_seed_decides_the_k_means_plus_plus_start():
    units, mask = codebook.to_units(_units(), None)
    first, _ = codebook.kmeans(units, mask, 5, iters=0, seed=3)
    again, _ = codebook.kmeans(units, mask, 5, iters=0, seed=3)
    other, _ = codebook.kmeans(units, mask, 5, iters=0, seed=4)
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    for row in range(units.shape[0]):
        assert set(first[row].tolist()) <= set(units[row].tolist())


def test_one_centroid_is_the_row_mean():
    units, mask = codebook.to_units(_units(), None)
    centroids, assignments = codebook.kmeans(units, mask, 1)
    torch.testing.assert_close(centroids[:, 0], units.mean(dim=1))
    assert not assignments.any()


def test_units_round_trip_and_padding_never_influences_the_fit():
    weight = _units(rows=3, columns=10)
    units, mask = codebook.to_units(weight, 4)
    assert units.shape == (9, 4) and mask.sum() == 30
    assert torch.equal(codebook.from_units(units, 4, tuple(weight.shape)), weight)
    garbage = units.clone()
    garbage[~mask] = 1e6
    clean_fit = codebook.kmeans(units, mask, 2)
    garbage_fit = codebook.kmeans(garbage, mask, 2)
    assert torch.equal(clean_fit[0], garbage_fit[0])
    assert torch.equal(clean_fit[1][mask], garbage_fit[1][mask])


def test_units_of_a_group_larger_than_the_row_are_whole_rows():
    weight = _units(rows=3, columns=10)
    for group_size in (None, 10, 64):
        units, mask = codebook.to_units(weight, group_size)
        assert units.shape == (3, 10) and mask.all()


def test_gather_scatter_mean_and_empty_clusters():
    units, mask = codebook.to_units(_units(rows=3, columns=12), 6)
    centroids = torch.arange(6.0, dtype=torch.float64).reshape(1, 6).expand(units.shape[0], -1).contiguous()
    assignments = (torch.arange(6).reshape(1, 6) % 3).expand(units.shape[0], -1).to(torch.uint8)
    values = codebook.gather(centroids, assignments)
    for g in range(units.shape[0]):
        assert values[g].tolist() == [float(i % 3) for i in range(6)]
    gradient = _units(rows=6, columns=6, seed=5)
    mask = torch.ones_like(gradient, dtype=torch.bool)
    mask[0, 5] = False
    mean = codebook.scatter_mean_grad(gradient, mask, assignments[:1].expand(6, -1), 4)
    for g in range(6):
        for k in range(4):
            members = [gradient[g, s] for s in range(6) if s % 3 == k and mask[g, s]]
            expected = sum(members) / len(members) if members else 0.0
            assert mean[g, k].item() == pytest.approx(float(expected), abs=1e-12)
    assert (mean[:, 3] == 0).all()


def test_input_diag_and_recluster_follow_the_weighted_distance():
    rows, columns = 4, 12
    weight = _units(rows, columns, dtype=torch.float32)
    x = _units(30, columns, dtype=torch.float32, seed=1)
    target = x + 0.3 * _units(30, columns, dtype=torch.float32, seed=2)
    a, b = codebook.input_diag(x, target)
    torch.testing.assert_close(a, (x * x).mean(0))
    torch.testing.assert_close(b, (x * target).mean(0))
    assert (codebook.input_diag(torch.zeros(3, columns), torch.zeros(3, columns))[0] == 1e-6).all()

    units, mask = codebook.to_units(weight, 5)
    centroids, _ = codebook.kmeans(units, mask, 2)
    got = codebook.recluster_diag(units, mask, centroids, a, b, 5, (rows, columns))
    for g in range(units.shape[0]):
        for s in range(5):
            column = (g % 3) * 5 + s
            if column >= columns:
                continue
            scores = [(b[column] * units[g, s] - a[column] * c) ** 2 for c in centroids[g]]
            assert got[g, s].item() == int(torch.stack(scores).argmin())
    nearest = codebook.recluster_diag(units, mask, centroids, a, a, 5, (rows, columns))
    assert torch.equal(
        nearest[mask], (units.unsqueeze(2) - centroids.unsqueeze(1)).abs().argmin(2)[mask].to(torch.uint8)
    )
    assert not torch.equal(got[mask], nearest[mask])


def test_fake_quantize_keeps_k_values_per_unit_and_the_dtype():
    weight = _units(rows=6, columns=48, dtype=torch.float32)
    out = fake_quantize(weight, _fmt(centroids=4), "weight")
    assert out.dtype == weight.dtype and out.shape == weight.shape
    assert all(len(set(row.tolist())) <= 4 for row in out)
    grouped = fake_quantize(weight, _fmt(centroids=8, granularity="per_group", group_size=16), "weight")
    assert all(len(set(unit.tolist())) <= 8 for unit in grouped.reshape(-1, 16))
    assert (weight - grouped).pow(2).mean() < (weight - out).pow(2).mean()
    more = fake_quantize(weight, _fmt(centroids=16), "weight")
    assert (weight - more).pow(2).mean() < (weight - out).pow(2).mean()
    bf16 = fake_quantize(weight.bfloat16(), _fmt(centroids=4), "weight")
    assert bf16.dtype == torch.bfloat16
    f32 = fake_quantize(weight.bfloat16(), _fmt(centroids=4, kmeans={"compute_dtype": "float32"}), "weight")
    assert f32.dtype == torch.bfloat16


def test_as_many_centroids_as_values_is_lossless_and_activations_are_rejected():
    weight = _units(rows=3, columns=8, dtype=torch.float32)
    assert torch.equal(fake_quantize(weight, _fmt(centroids=8), "weight"), weight)
    with pytest.raises(ValueError, match="weights"):
        fake_quantize(weight, _fmt(), "activation")
    assert torch.equal(fake_quantize(weight, _fmt(), "weight"), codebook.fake_quantize(weight, _fmt()))


# comparison with the official CodeQuant code ($EFFEVAL_CODEQUANT_REF)


@pytest.fixture(scope="module")
def official_cluster():
    return _official("cq1", "utils/cluster_utils.py")


def _weight(rows, columns, dtype, device="cpu", seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(rows, columns, generator=generator) * 0.02).to(dtype=dtype, device=device)


@pytest.mark.parametrize(
    ("rows", "columns", "group_size"), [(16, 64, None), (8, 64, 24), (8, 64, 16), (4, 30, 64), (6, 256, 128)]
)
def test_grouping_equals_official(official_cluster, rows, columns, group_size):
    weight = _weight(rows, columns, torch.float32)
    units, mask = codebook.to_units(weight, group_size)
    expected, expected_mask = official_cluster.group(weight, -1 if group_size is None else group_size)
    assert torch.equal(units, expected) and torch.equal(mask, expected_mask)
    assert torch.equal(codebook.from_units(units, group_size, (rows, columns)), weight)


def _compare_kmeans(official_cluster, weight, group_size, k):
    units, mask = codebook.to_units(weight, group_size)
    expected_c, expected_a = official_cluster.kmeans(
        units, mask, cluster_num=k, kmeans_iteration=300, tol=1e-6
    )
    centroids, assignments = codebook.kmeans(units, mask, k, 300, 1e-6, 42)
    assert torch.equal(centroids, expected_c)
    assert torch.equal(assignments, expected_a)


@pytest.mark.parametrize(("group_size", "k"), [(None, 16), (None, 4), (24, 16), (None, 1)])
def test_kmeans_equals_official_on_cpu_float32(official_cluster, group_size, k):
    _compare_kmeans(official_cluster, _weight(24, 96, torch.float32), group_size, k)


@pytest.fixture
def deterministic_scatter():
    """CUDA scatter_add_ adds in atomic order: without this even the official code varies run to run."""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.gpu
def test_kmeans_equals_official_on_cuda_bfloat16(official_cluster, deterministic_scatter):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    _compare_kmeans(official_cluster, _weight(64, 256, torch.bfloat16, "cuda"), None, 16)
    _compare_kmeans(official_cluster, _weight(32, 200, torch.bfloat16, "cuda"), 64, 4)


@pytest.mark.gpu
def test_kmeans_equals_official_on_cuda_float32(official_cluster, deterministic_scatter):
    """The CodeQuant presets cluster in float32 on the model's device."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    _compare_kmeans(official_cluster, _weight(64, 256, torch.float32, "cuda"), None, 16)
    _compare_kmeans(official_cluster, _weight(32, 200, torch.float32, "cuda"), 64, 4)


@pytest.mark.parametrize("group_size", [None, 24])
def test_recluster_equals_official(official_cluster, group_size):
    rows, columns = 8, 96
    weight = _weight(rows, columns, torch.float32)
    units, mask = codebook.to_units(weight, group_size)
    centroids, _ = codebook.kmeans(units, mask, 16)
    x = _weight(40, columns, torch.float32, seed=1) * 20
    target = x + _weight(40, columns, torch.float32, seed=2) * 5
    quant, _, expected = official_cluster.kmeans_recluster(
        org_group_weight=units,
        centroids=centroids,
        weight_group_size=-1 if group_size is None else group_size,
        mask=mask,
        inputs=x,
        target_inputs=target,
        org_weight_shape=(rows, columns),
    )
    a, b = codebook.input_diag(x, target)
    assignments = codebook.recluster_diag(units, mask, centroids, a, b, group_size, (rows, columns))
    assert torch.equal(assignments.long()[mask], expected[mask])
    assert torch.equal(codebook.gather(centroids, assignments)[mask], quant[mask])
    plain = codebook.recluster_diag(units, mask, centroids, a, a, group_size, (rows, columns))
    assert not torch.equal(assignments[mask], plain[mask])
