"""Run keys and model-group keys of specs that existed before model preprocessing, frozen byte for byte.

A change to a key makes every stored result of these specs unreachable for reuse and comparison.

The run_keys of the specs that evaluate native/wikitext2_ppl were re-frozen once, when that task went from
version "1" to "2" because its loss is now computed on float32 logits (a bfloat16 model's perplexity under
version "1" was rounded to bfloat16 steps). The task version is part of run_key, so exactly those keys
changed: with the version set back to "1" the previous values were reproduced, and the keys of the spec
without that task (rtn_w8a8_input_rules) and every model-group key are unchanged.
"""

import pytest

from catalogs import CATALOG
from effeval.core.canonical import sha256_hex
from effeval.plan import freeze, model_group_key

GPTQ_MODEL = "qwen3-0.6b-base"
W8A8_RULES = [
    {
        "sites": ["linear.*.weight"],
        "quant": {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_channel"},
    },
    {
        "sites": ["linear.*.input"],
        "quant": {"type": "int", "bits": 8, "symmetric": True, "granularity": "per_token"},
    },
]

SPECS = {
    "base": {
        "model": GPTQ_MODEL,
        "method": {"name": "base"},
        "field": "quantization",
        "tasks": ["native/wikitext2_ppl"],
    },
    "rtn_gptq_paper": {
        "model": GPTQ_MODEL,
        "method": {"name": "rtn", "preset": "gptq_paper"},
        "field": "quantization",
    },
    "rtn_w8a8_input_rules": {
        "model": GPTQ_MODEL,
        "method": {"name": "rtn", "preset": None, "config": {"rules": W8A8_RULES}},
        "field": "quantization",
        "tasks": ["lm_eval/mmlu"],
    },
    "gptq_paper": {
        "model": GPTQ_MODEL,
        "method": {"name": "gptq", "preset": "paper"},
        "field": "quantization",
    },
    "gptq_paper_w3": {
        "model": GPTQ_MODEL,
        "method": {"name": "gptq", "preset": "paper_w3"},
        "field": "quantization",
    },
    "gptq_repeat_seeds": {
        "model": GPTQ_MODEL,
        "method": {"name": "gptq"},
        "field": "quantization",
        "repeats": {"seeds": [3, 7]},
    },
}

# (role, run_key, sha256 of model_group_key) per job, in plan order.
EXPECTED: dict[str, list[tuple[str, str, str]]] = {
    "base": [
        (
            "main",
            "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
    ],
    "rtn_gptq_paper": [
        (
            "reference",
            "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
        (
            "main",
            "rk_146b407230ddad799e0eb0359ec7530dc559dc6f9840f96803414d8d48f0ded3",
            "e7b172218c411679bcb103bc0cfc61310de79b0c719af8a505ebc9afaaa78434",
        ),
    ],
    "rtn_w8a8_input_rules": [
        (
            "reference",
            "rk_265e8b3382996d957e105c0032d8e103faaa3df80e38f23569271a1b5821500f",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
        (
            "main",
            "rk_c39ed34fde46d9c3938a061eeead5d5a8638e9dc5aaeb7781fafbff61bbc3610",
            "ee7a5f1272e892730626c22bb891b6917a1cd2457c9377e425d0c8b3852ee426",
        ),
    ],
    "gptq_paper": [
        (
            "reference",
            "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
        (
            "main",
            "rk_dc033ef0a9468ea5942dbd611125887f4b6488d85567eff4a5625bf0140c1d3c",
            "bd5bfafbff7cab6d6dd83b8e8c9bc410467805394a744c631a75ce534afe676b",
        ),
    ],
    "gptq_paper_w3": [
        (
            "reference",
            "rk_08a6c056f685a2ac859a6ba71ab3b037aafed6649c20b292c23d48fdb1b8b17b",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
        (
            "main",
            "rk_ef57fb87538ddcc7f0f0666df036a136c436eb002873d9ed6be75511dde807e7",
            "adb1c11cb285dcee853aa6ad5b597ee374af001ba71ee290ffe860655a488464",
        ),
    ],
    "gptq_repeat_seeds": [
        (
            "reference",
            "rk_3dc8700b4ef567af821974a07735014cc82fa046b5038870fbbea18200de4bb2",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
        (
            "reference",
            "rk_3ce6e273debadfb9a4b2500775f048b6e64cb8b4b6545d9b464bc44a38082f97",
            "a32d556efcfe2f7720f2516cdd871ea293ddd40a4d0e4593aa566b145bab0601",
        ),
        (
            "main",
            "rk_9d5e596906f95dc6a52399160f3b414e70f45c68f3755ae35cabd33dc0e6141c",
            "bd5bfafbff7cab6d6dd83b8e8c9bc410467805394a744c631a75ce534afe676b",
        ),
        (
            "main",
            "rk_9ce87b97c8cc262d49cc56b3ad70a5864b7f9f89c3085e3e27e638797db3c72d",
            "bd5bfafbff7cab6d6dd83b8e8c9bc410467805394a744c631a75ce534afe676b",
        ),
    ],
}


def _frozen(name):
    plan, issues = freeze(SPECS[name], CATALOG, owner="bob", server=None, lookup=None)
    assert plan is not None, issues
    return [(j.role, j.run_key, sha256_hex(model_group_key(j))) for j in plan.jobs]


@pytest.mark.parametrize("name", list(SPECS))
def test_run_keys_and_group_keys_of_existing_specs_are_frozen(name):
    assert _frozen(name) == EXPECTED[name]
