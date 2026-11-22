import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from effeval.core.fragments import IntFormat, QuantRule  # noqa: E402
from effeval.ops import sites  # noqa: E402
from effeval.ops.quant.quant import fake_quantize  # noqa: E402

pytestmark = pytest.mark.compute

INT4 = {"type": "int", "bits": 4, "symmetric": True, "granularity": "per_token"}


def tiny_qwen3(attn: str = "sdpa", tie: bool = False):
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
    )
    return transformers.AutoModelForCausalLM.from_config(config, attn_implementation=attn).eval()


def padded_batch():
    input_ids = torch.tensor([[5, 6, 7, 8, 9, 10], [0, 0, 11, 12, 13, 14]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    return input_ids, attention_mask


@torch.no_grad()
@pytest.mark.parametrize(("ours", "native"), [(sites.SDPA_IMPL, "sdpa"), (sites.EAGER_IMPL, "eager")])
def test_registered_attention_is_bit_identical_to_native_on_padded_batch(ours, native):
    model = tiny_qwen3(native)
    input_ids, attention_mask = padded_batch()
    reference = model(input_ids=input_ids, attention_mask=attention_mask).logits
    sites.register_attention_functions()
    model.set_attn_implementation(ours)
    got = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert torch.equal(got, reference)


@torch.no_grad()
def test_padding_changes_logits_so_the_mask_check_is_meaningful():
    model = tiny_qwen3("eager")
    input_ids, attention_mask = padded_batch()
    masked = model(input_ids=input_ids, attention_mask=attention_mask).logits
    unmasked = model(input_ids=input_ids).logits
    assert not torch.equal(masked[1], unmasked[1])


@torch.no_grad()
def test_attention_operand_site_changes_output_and_remove_restores():
    model = tiny_qwen3("sdpa")
    input_ids, attention_mask = padded_batch()
    before = model(input_ids=input_ids, attention_mask=attention_mask).logits
    rule = QuantRule(sites=["attn.q", "attn.k"], layers="1", quant=INT4)
    attachment = sites.attach(model, [rule])
    assert model.config._attn_implementation == sites.SDPA_IMPL
    assert attachment.hits[0].layers == [1]
    during = model(input_ids=input_ids, attention_mask=attention_mask).logits
    assert not torch.equal(during, before)
    attachment.remove()
    assert model.config._attn_implementation == "sdpa"
    assert torch.equal(model(input_ids=input_ids, attention_mask=attention_mask).logits, before)


@torch.no_grad()
def test_scores_site_switches_to_eager_path():
    model = tiny_qwen3("sdpa")
    rule = QuantRule(sites=["attn.scores"], layers="all", quant=INT4)
    attachment = sites.attach(model, [rule])
    assert model.config._attn_implementation == sites.EAGER_IMPL
    attachment.remove()


@torch.no_grad()
def test_weight_site_quantizes_in_place_and_later_rules_override():
    model = tiny_qwen3()
    original = model.model.layers[0].mlp.down_proj.weight.clone()
    int4 = IntFormat(type="int", bits=4, symmetric=False, granularity="per_channel")
    rules = [
        QuantRule(sites=["linear.*.weight"], layers="all", quant={**INT4, "granularity": "per_channel"}),
        QuantRule(sites=["linear.mlp.down_proj.weight"], layers="0", quant=int4),
    ]
    attachment = sites.attach(model, rules)
    assert torch.equal(
        model.model.layers[0].mlp.down_proj.weight, fake_quantize(original, int4, kind="weight")
    )
    assert attachment.hits[0].locations == 3 * 7  # 3 layers x q/k/v/o/gate/up/down
    assert attachment.hits[1].locations == 1


def test_unmatched_rule_raises():
    model = tiny_qwen3()
    with pytest.raises(sites.SiteError, match="matches nothing"):
        sites.resolve(model, [QuantRule(sites=["lm_head.weight"], layers="0-1", quant=INT4)])
    with pytest.raises(ValueError, match="exceed model depth"):
        sites.resolve(model, [QuantRule(sites=["linear.*.weight"], layers="5", quant=INT4)])


def test_layout_comes_from_the_architecture_table_and_unknown_classes_are_rejected():
    model = tiny_qwen3()
    assert sites.architecture(model).name == "Qwen3ForCausalLM"
    _, hits = sites.resolve(model, [QuantRule(sites=["linear.mlp.*.weight"], quant=INT4)])
    assert hits[0].sites == [
        "linear.mlp.down_proj.weight",
        "linear.mlp.gate_proj.weight",
        "linear.mlp.up_proj.weight",
    ]
    torch.manual_seed(0)
    llama = transformers.AutoModelForCausalLM.from_config(
        transformers.LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )
    with pytest.raises(sites.SiteError, match="LlamaForCausalLM has no layout"):
        sites.resolve(llama, [QuantRule(sites=["linear.*.weight"], quant=INT4)])
