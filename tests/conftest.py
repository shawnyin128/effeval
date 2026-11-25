import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _isolated_artifact_cache(tmp_path, monkeypatch):
    """Point the artifact cache at a per-test directory and reset the process-wide instance."""
    from effeval.ops import artifacts

    monkeypatch.setenv(artifacts.CACHE_ENV, str(tmp_path / "artifact-cache"))
    monkeypatch.setattr(artifacts, "_default", None)


def _randomize_norm_gains(model, seed: int) -> None:
    """Give every RMSNorm gain a distinct value so that gain folding is observable."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    for module in model.modules():
        if type(module).__name__.endswith("RMSNorm"):
            module.weight.data.copy_(0.5 + torch.rand(module.weight.shape, generator=generator))


def build_tiny_qwen3(
    head_dim: int = 32, tie: bool = True, layers: int = 2, seed: int = 0, intermediate: int = 96
):
    """Tiny Qwen3 (hidden 64, 4 heads, 2 kv heads, given intermediate size) with random norm gains.

    The default has heads * head_dim = 128 != hidden, like Qwen3-0.6B (1024 vs 16 * 128).
    """
    import torch
    import transformers

    torch.manual_seed(seed)
    config = transformers.Qwen3Config(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        max_position_embeddings=64,
        tie_word_embeddings=tie,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
    _randomize_norm_gains(model, seed + 1)
    return model


def build_tiny_llama(layers: int = 2, seed: int = 0):
    """Tiny Llama (hidden 64, intermediate 96, 4 heads x head_dim 16, 2 kv heads, untied), random gains."""
    import torch
    import transformers

    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa").eval()
    _randomize_norm_gains(model, seed + 1)
    return model


@pytest.fixture
def register_architecture():
    """Register a model's architecture layout and graph in the builtin catalog for one test.

    Call `register_architecture(model)`; classes the catalog already has are left alone. Entries added here
    are removed after the test.
    """
    from effeval.core.catalog import builtin_catalog
    from effeval.ops.introspect import extract_architecture

    added: list[str] = []

    def register(model) -> None:
        known = builtin_catalog().architectures
        name = type(model).__name__
        if name not in known:
            known[name] = extract_architecture(model.config, source="test fixture")
            added.append(name)

    yield register
    for name in added:
        builtin_catalog().architectures.pop(name)
