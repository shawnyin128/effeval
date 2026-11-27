"""WSVD structure contract: the `qkv_per_head` transform and the allocation floor index."""

import pytest

torch = pytest.importorskip("torch")

from conftest import build_tiny_qwen3  # noqa: E402
from effeval.core.catalog import builtin_catalog  # noqa: E402
from effeval.core.lowrank import (  # noqa: E402
    ARCHITECTURE_ATTR,
    ARCHITECTURE_TRANSFORMS,
    QKV_PER_HEAD,
    factor_qkv_per_head_architecture,
    transform_architecture,
)
from effeval.ops import sites  # noqa: E402
from effeval.ops.lowrank import install_per_head_qkv, per_head_allocation  # noqa: E402

pytestmark = pytest.mark.compute

DENSE = builtin_catalog().architectures["Qwen3ForCausalLM"]


def test_qkv_per_head_transform_is_the_per_head_layout_without_ffn_branches():
    assert QKV_PER_HEAD == "qkv_per_head"
    assert QKV_PER_HEAD in ARCHITECTURE_TRANSFORMS
    layout = transform_architecture(DENSE, QKV_PER_HEAD, {})
    assert layout == factor_qkv_per_head_architecture(DENSE)
    assert transform_architecture(DENSE, QKV_PER_HEAD, {"ffn": {"enabled": True}}) == layout
    assert not any("svd" in node.id for node in layout.graph.nodes)


def test_qkv_per_head_transform_equals_the_layout_install_attaches():
    model = build_tiny_qwen3(layers=1)
    arch = sites.architecture(model)
    heads = {"q_proj": 4, "k_proj": 2, "v_proj": 2}
    layer = model.model.layers[0]
    factors = {}
    for name, h in heads.items():
        out = getattr(layer.self_attn, name).weight.shape[0]
        hd = out // h
        blocks = [torch.randn(hd, 3) for _ in range(h)]
        factors[name] = (torch.randn(3 * h, 64), blocks)
    install_per_head_qkv(model, [factors])
    assert getattr(model, ARCHITECTURE_ATTR) == transform_architecture(arch, QKV_PER_HEAD, {})


def _pools_with_an_empty_head():
    layer0 = torch.zeros(2, 4)
    layer0[0] = torch.tensor([9.0, 8.0, 0.0, 0.0])
    layer1 = torch.zeros(2, 4)
    layer1[1] = torch.tensor([7.0, 6.0, 0.0, 0.0])
    return [[layer0, layer1]]


def test_an_empty_head_gets_the_floor_index_and_the_default_is_zero():
    pools = _pools_with_an_empty_head()
    default, floored = per_head_allocation(pools, [2], joint=False)
    assert default[0][0][1] == [0] and default[0][1][0] == [0] and floored == 2
    indices, floored = per_head_allocation(pools, [2], joint=False, floor_index=1)
    assert indices[0][0][1] == [1] and indices[0][1][0] == [1] and floored == 2
    assert indices[0][0][0] == [0, 1]


def test_floor_index_not_below_the_head_dimension_is_rejected():
    with pytest.raises(ValueError, match="floor_index"):
        per_head_allocation(_pools_with_an_empty_head(), [2], joint=False, floor_index=4)
