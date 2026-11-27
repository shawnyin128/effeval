"""Cache of tensors derived from a model and calibration data (rotation matrices, permutations, scales).

Entries live in the process memory and, when the `EFFEVAL_CACHE` environment variable names a directory, also
on disk as safetensors files named by their key.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from effeval.core.canonical import canonical_json, sha256_hex
from effeval.core.fragments import Calibration

CACHE_ENV = "EFFEVAL_CACHE"


def artifact_key(
    *,
    model: Any,
    calibration: Calibration | None,
    statistic: str,
    upstream: list[Any],
    downstream: Any = None,
) -> str:
    """sha256 of the canonical JSON of everything a derived tensor depends on.

    `model` identifies the checkpoint and the dtype and device the tensors are computed in, `statistic`
    the quantity computed, `upstream` the preprocessing steps applied before the one being computed, and
    `downstream` the quantizer rules the computation optimises for (None when it does not look at them).
    All values must be JSON-serializable.
    """
    parts = {
        "model": model,
        "calibration": None if calibration is None else calibration.model_dump(mode="json"),
        "statistic": statistic,
        "upstream": upstream,
        "downstream": downstream,
    }
    return sha256_hex(canonical_json(parts))


class ArtifactCache:
    """Memoizes `dict[str, Tensor]` results by key, in memory and optionally in `directory`."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory
        self._memory: dict[str, dict[str, torch.Tensor]] = {}
        self.computed = 0
        self.memory_hits = 0
        self.disk_hits = 0

    @classmethod
    def from_env(cls) -> ArtifactCache:
        directory = os.environ.get(CACHE_ENV)
        return cls(Path(directory) if directory else None)

    def get_or_compute(
        self, key: str, compute: Callable[[], dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """The tensors stored under `key`, running `compute` (and storing its result) on a miss."""
        if key in self._memory:
            self.memory_hits += 1
            return self._memory[key]
        path = None if self.directory is None else self.directory / f"{key}.safetensors"
        if path is not None and path.exists():
            self.disk_hits += 1
            self._memory[key] = load_file(path)
            return self._memory[key]
        tensors = compute()
        self.computed += 1
        self._memory[key] = tensors
        if path is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            save_file({name: t.detach().cpu().clone().contiguous() for name, t in tensors.items()}, partial)
            partial.replace(path)
        return tensors


def weights_fingerprint(model: nn.Module) -> str:
    """Digest of the per-tensor sums of absolute values: identifies the weights a cached artifact
    came from."""
    sums = [p.detach().abs().sum(dtype=torch.float64).cpu() for _name, p in sorted(model.named_parameters())]
    return hashlib.sha256(json.dumps(torch.stack(sums).tolist()).encode()).hexdigest()


_default: ArtifactCache | None = None


def default_cache() -> ArtifactCache:
    """The process-wide cache, configured from `EFFEVAL_CACHE` on first use."""
    global _default
    if _default is None:
        _default = ArtifactCache.from_env()
    return _default
