"""Determinism settings applied once per compute-node process, before any CUDA work."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from effeval.core.spec import DeterminismSpec


def apply(spec: DeterminismSpec) -> None:
    """Seed every RNG and configure deterministic kernels.

    `deterministic_algorithms` makes PyTorch raise on ops without a deterministic implementation instead of
    silently running a nondeterministic one. CUBLAS_WORKSPACE_CONFIG must be set before cuBLAS initializes.
    """
    if spec.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)
    torch.use_deterministic_algorithms(spec.deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = spec.tf32
    torch.backends.cudnn.allow_tf32 = spec.tf32
