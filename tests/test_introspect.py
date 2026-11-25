import types
from pathlib import Path

import pytest
import yaml

from effeval.core.catalog import ModelEntry, builtin_catalog, model_code
from effeval.core.graph import LayerGraph, ModelShape, positions

FIXTURE = Path(__file__).parent / "data" / "qwen3_layer_graph.yaml"


@pytest.mark.parametrize(
    ("checkpoint", "code"),
    [
        ("Qwen/Qwen3-0.6B-Base", "qwen3-0.6b-base"),
        ("Qwen/Qwen3-8B", "qwen3-8b"),
        ("/data/ckpt/My-Model-v2/", "my-model-v2"),
    ],
)
def test_model_code_is_the_lower_cased_repo_name(checkpoint, code):
    assert model_code(checkpoint) == code


def test_model_entry_rejects_a_hand_picked_code():
    with pytest.raises(ValueError, match="must be 'qwen3-8b'"):
        ModelEntry(
            code="q3-8b",
            checkpoint="Qwen/Qwen3-8B",
            revision=None,
            architecture="Qwen3ForCausalLM",
            dtype="bfloat16",
            params=1,
            tie_word_embeddings=False,
        )


def test_a_code_cannot_be_redefined():
    catalog = builtin_catalog()
    clash = catalog.models["qwen3-8b"].model_copy(update={"revision": "another"})
    with pytest.raises(ValueError, match="already means"):
        catalog.with_models([clash])


@pytest.mark.compute
def test_stored_qwen3_layout_matches_extraction_from_hf_metadata():
    transformers = pytest.importorskip("transformers")
    from effeval.ops.introspect import extract_architecture

    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    live = extract_architecture(config, source="test")
    stored = builtin_catalog().architectures["Qwen3ForCausalLM"]
    assert live.model_dump(exclude={"extracted_with"}) == stored.model_dump(exclude={"extracted_with"})
    assert "attn.q" in stored.site_names() and "linear.attn.q_proj.weight" in stored.site_names()


def _mixtral_config():
    transformers = pytest.importorskip("transformers")
    return transformers.MixtralConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=4,
        num_experts_per_tok=2,
    )


@pytest.mark.compute
def test_moe_experts_class_without_an_adapter_reports_its_fused_weights_as_uncovered():
    from effeval.ops.introspect import extract_architecture

    arch = extract_architecture(_mixtral_config(), source="test")
    assert arch.moe is None
    assert any(name.startswith("mlp.experts") for name in arch.uncovered_weights)
    assert not any(path.startswith("mlp.experts") for path in arch.linears)


def _fixture_graph() -> LayerGraph:
    return LayerGraph.model_validate(yaml.safe_load(FIXTURE.read_text(encoding="utf-8")))


@pytest.mark.compute
def test_extracted_qwen3_graph_equals_the_fixture_for_tiny_and_real_dimensions():
    transformers = pytest.importorskip("transformers")
    from effeval.ops.introspect import extract_architecture

    for hidden, inter, heads, kv, head_dim in ((64, 96, 4, 2, 16), (1024, 3072, 16, 8, 128)):
        config = transformers.Qwen3Config(
            vocab_size=97,
            hidden_size=hidden,
            intermediate_size=inter,
            num_hidden_layers=2,
            num_attention_heads=heads,
            num_key_value_heads=kv,
            head_dim=head_dim,
        )
        assert extract_architecture(config, source="test").graph == _fixture_graph()


@pytest.mark.compute
def test_llama_graph_has_no_head_norm_and_the_same_positions_as_qwen3():
    transformers = pytest.importorskip("transformers")
    from effeval.ops.introspect import extract_architecture

    config = transformers.LlamaConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    graph = extract_architecture(config, source="test").graph
    assert not any(n.axis == "head_dim" for n in graph.nodes)
    assert [e for e in graph.edges if e[1] == "rope"] == [("q", "rope", None), ("k", "rope", None)]
    shape = ModelShape(
        hidden=64, intermediate=96, num_layers=2, heads=4, kv_heads=2, head_dim=16, vocab=97, tie_word_embeddings=False
    )
    llama = positions(graph, shape)
    qwen3 = positions(_fixture_graph(), shape)
    assert list(llama) == list(qwen3)
    for position, kinds in llama.items():
        for kind, modes in kinds.items():
            for mode, cell in modes.items():
                assert (cell == "ok") == (qwen3[position][kind][mode] == "ok"), (position, kind, mode)


@pytest.mark.compute
def test_moe_experts_class_without_an_adapter_yields_an_unsupported_graph():
    from effeval.ops.introspect import extract_architecture

    graph = extract_architecture(_mixtral_config(), source="test").graph
    assert graph.unsupported == ["moe"] and graph.nodes == []


@pytest.mark.compute
def test_model_shape_takes_head_dim_from_the_config():
    transformers = pytest.importorskip("transformers")
    from effeval.ops.introspect import model_shape

    config = transformers.Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=2,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        tie_word_embeddings=True,
    )
    shape = model_shape(config)
    assert (shape.hidden, shape.heads, shape.head_dim, shape.kv_heads) == (1024, 16, 128, 8)
    assert shape.num_layers == 2

    class NoHeadDim:
        def get_text_config(self):
            return types.SimpleNamespace(hidden_size=64)

    with pytest.raises(ValueError, match="head_dim"):
        model_shape(NoHeadDim())
