import torch

from effeval.ops import artifacts
from effeval.ops.artifacts import ArtifactCache


def test_tensors_sharing_storage_survive_a_disk_round_trip(tmp_path):
    base = torch.arange(12.0)
    computed = {"a::mask": base[:6], "b::mask": base[:6], "c::view": base[6:], "d": torch.ones(3)}
    ArtifactCache(tmp_path).get_or_compute("k", lambda: computed)
    loaded = ArtifactCache(tmp_path).get_or_compute("k", lambda: {})
    assert loaded.keys() == computed.keys()
    for name, tensor in computed.items():
        assert torch.equal(loaded[name], tensor), name


def test_default_cache_persists_to_the_directory_named_by_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(artifacts.CACHE_ENV, str(tmp_path))
    monkeypatch.setattr(artifacts, "_default", None)
    assert artifacts.default_cache().directory == tmp_path
