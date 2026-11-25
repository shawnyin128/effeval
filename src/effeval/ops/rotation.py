"""Rotation matrices and weight-fold helpers (float64), torch only.

Convention: activations are row vectors and a rotation R replaces x by x @ R. A consumer weight W [out, in]
then becomes W @ R (`fold_input`); a producer weight W [out, in] whose output is rotated becomes R.T @ W
(`fold_output`). Hadamard matrices are H_K^T (x) H_(2^m) / sqrt(n) for n = K * 2^m, with the base Hadamard
matrix H_K from the table below, identical to QuaRot's `matmul_hadU(eye(n))`. Sizes QuaRot does not cover use
the Paley type I matrices of PALEY_BASES (K - 1 a prime, 3 mod 4), tried only after QuaRot's bases.

The base matrices (K in HADAMARD_BASES) are taken from QuaRot fake_quant/hadamard_utils.py
(https://github.com/spcl/QuaRot, commit 5008669, Apache License 2.0; matrices from
http://www.neilsloane.com/hadamard/index.html), stored as packed sign bits, row-major, +1 = bit 1.
"""

from __future__ import annotations

import base64
import math

import torch

# QuaRot's get_hadK tries these in this order (40 before 20) when n % K == 0 and n / K is a power of two.
HADAMARD_BASES = (172, 156, 140, 108, 60, 52, 36, 28, 40, 20, 12)

# Paley type I orders; tried after HADAMARD_BASES, so sizes QuaRot supports keep their matrices.
PALEY_BASES = (68, 152, 200)

_BASE_BITS = {
    12: (
        "gA0d6OtH2j7R9ou0najtx2o7"
    ),
    20: (
        "gAANhXnsK8thXpsK/NhX5sK/NhX5sKvNhd5sKvNh15sKvNiV5sivNoV5vCvN4V5rCvM="
    ),
    28: (
        "//3//sNrDfYbWGuw6sPdh1Ye7Dqwt2LVibsmrI3aNWhu4avDdw1eG7hqsN7DXYb2Gn/8AArDRPLWGCeasNE81YSJ7qwET7Vh"
        "InmrGRONWciYat5Ew1TyLhqHkbDVPI2GieQ="
    ),
    36: (
        "///f//9GLtGL+jF2jF/Ri7Rivoxtox30Y20Yr6MraMl9GltGi+ji2jxfRxbR4vo4tosX0sW0mL6mLajF9jFtxi+xi2oxfoxb"
        "0Yv0Yt6MX6MWf//AAAtGLC502jFBc67RiAudtoxQXO20YILnraMUFzltGaC5i2jdBcxbROgu4toHQXsW0ToLmLaZ0FjFtc6C"
        "xi2OdBoxbXOg0YtLnQ6MWFzo"
    ),
    40: (
        "gAAIAADYV52FeewrzsK8thXrYV6bCvmwr82FfNhX5sK+bCvzYV82Ffmwr5sKvNhbzYXebC3mwq82GvNh15sNebCrzYq82JXm"
        "yV5sivNorzaFebhXm8K83CvN4V5uFeawrzsK84AAB///2FeSeobsK8E9Q7YV5J6hmwr2T1DNhXMnqObCsZPU82FQyer5sKBk"
        "9bzYVDJ63mwiGT2vNhUMntebAoZPq82FQyeV5sahk4rzZ1DJhXm3qGTCvNPUMuFeYeoZsK809Qw="
    ),
    52: (
        "p5R+NMsJBTyj86ZISKnhH50yQkVPCP7pghIqfEfXTBCZU+I8unCEyp8R5dKELlT4jy6EIfKnxFl1IQeVfiDLqQg8q/EmXEhJ"
        "5R+JMuJCTyj8aZYSFwPTye36Zbganm9v0y3A1PN7bpnuAqe723THcBU/3sumO4Sp3vZdMdxlTvey6Q7nKne9l0B3eVG97LoD"
        "u8qt7mXYHZ5Xb3MuwOzyu3qZfgcnl9vUyyzUJCnk4HFmoSFPNwOLMQkqebgUWYhJU93AoshCSp7uDRZCElTncGiyELKmO4NF"
        "kIeVIdyaLIQ8qA7s0SQh5UB3ZokhDysDuzQJCnlYHVmgSFPLwOe3lmo/FPK9uLNR+Ked7cWYj9U872osxH6p73tRZiPlT3ve"
        "izEfKnvetFuI+VPe8aL8R8qW980X4j5Ut75ovxDyrb2zRfiHle3tmg/FPK9vLNB+KeU="
    ),
    60: (
        "gAAAAAAAAA0VsQwfPcld6K2IYPnuSutFbEMHz3JX2itiGD57kr7RWxDB89yV9orYhg+e5Ku0VsQwfPcl3aK2IYPnuSrtFbEM"
        "Hz3J12itiGD57kq7RWxDB89yldorYhg+e5yu0VsQwfPcpXaK2IYPnukrtFbEMHz3yV2itiGD575K7RWxDB898ldorYhg+euS"
        "u0VsQwfP3JXaK2IYPn7krtFbEMHz9yV2itiGD5+5K7RWxDB8vcldorYhg+nuSu0VsQwfz3JXaK2IYP57krtFbEMH89yV2iti"
        "GD+e5K7RWxDB/PcldorYhgvnuSu0VsQwnz3JXaK2IYj57krtFbEMh89yV2itiGg+e5K7RWxDwfPcldorYh4PnuSu0VsQsHz3"
        "JXaK2ImD57krtFbEjB89yV2itihg+e5K7RWxwwfPcldorYoYPnuSu0VskMHz3JXaK2iGD57krtFbxDB89yV2it4hg+e5K7RW"
        "sQwfPcldor2IYPnuSu0V7EMHz3JXaKtiGD57krtF2xDB89yV2irYhg+e5K7R1sQwfPcldoq2IYPnuSu0lbEMHz3JXaitiGD5"
        "7krtxWxDB89yV2orYhg+e5K7"
    ),
    108: (
        "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"
        "zVEvw2p4FuqYIWMzl75qiX4b"
    ),
    140: (
        "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"
        "KvM="
    ),
    156: (
        "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"
        "xI+pxl1va6Y4mFKGR+UEwyCn"
    ),
    172: (
        "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"
        "67N15icLZ/m0OFNQvQrKL8KbZQ/Yz12brzE="
    ),
}


def is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def hadamard_factor(n: int) -> tuple[int, int]:
    """Split n = K * 2^m with K = 1 or K in HADAMARD_BASES + PALEY_BASES, trying QuaRot's bases first.

    Raises ValueError when n has no such factorisation.
    """
    for k in HADAMARD_BASES + PALEY_BASES:
        if n % k == 0 and is_pow2(n // k):
            return k, n // k
    if is_pow2(n):
        return 1, n
    raise ValueError(
        f"no Hadamard matrix of size {n}: n must be 2^m or K * 2^m with K in {HADAMARD_BASES + PALEY_BASES}"
    )


def _paley_i(q: int) -> torch.Tensor:
    """Paley type I Hadamard matrix of order q + 1 (float64 +-1) for a prime q congruent to 3 mod 4."""
    residues = torch.zeros(q, dtype=torch.bool)
    residues[(torch.arange(1, q) ** 2) % q] = True
    chi = torch.where(residues, 1, -1)
    chi[0] = 0
    idx = torch.arange(q)
    jacobsthal = chi[(idx[None, :] - idx[:, None]) % q]
    skew = torch.zeros(q + 1, q + 1, dtype=torch.int64)
    skew[0, 1:] = 1
    skew[1:, 0] = -1
    skew[1:, 1:] = jacobsthal
    return (skew + torch.eye(q + 1, dtype=torch.int64)).to(torch.float64)


def _base_matrix(k: int) -> torch.Tensor:
    if k in PALEY_BASES:
        return _paley_i(k - 1)
    raw = torch.frombuffer(bytearray(base64.b64decode(_BASE_BITS[k])), dtype=torch.uint8)
    shifts = torch.arange(7, -1, -1, dtype=torch.uint8)
    bits = ((raw[:, None] >> shifts) & 1).flatten()[: k * k]
    return (bits.to(torch.float64) * 2 - 1).view(k, k)


def _sylvester(m: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < m:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def hadamard(n: int) -> torch.Tensor:
    """Orthonormal Hadamard matrix of size n = K * 2^m (float64), the matrix QuaRot applies as x @ H."""
    k, m = hadamard_factor(n)
    sylvester = _sylvester(m)
    full = sylvester if k == 1 else torch.kron(_base_matrix(k).T.contiguous(), sylvester)
    return full / math.sqrt(n)


def hadamard_parts(n: int) -> tuple[torch.Tensor, int]:
    """(H_K, m) with hadamard(n) = H_K^T (x) H_(2^m) / sqrt(n) and H_K the unnormalised float64 base matrix.

    Raises ValueError when n has no Hadamard factorisation.
    """
    k, m = hadamard_factor(n)
    return (torch.ones(1, 1, dtype=torch.float64) if k == 1 else _base_matrix(k)), m


def random_signs(n: int, seed: int) -> torch.Tensor:
    """The float64 +-1 vector [n] drawn by `random_hadamard` for `seed`."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(low=0, high=2, size=(n,), generator=generator).to(torch.float64) * 2 - 1


def random_hadamard(n: int, seed: int) -> torch.Tensor:
    """diag(s) @ hadamard(n) with random signs s, the draw QuaRot makes after `set_seed(seed)`.

    Equal to QuaRot's `random_hadamard_matrix` when its global RNG was seeded with `seed` just before.
    """
    return random_signs(n, seed)[:, None] * hadamard(n)


def random_orthogonal(n: int, seed: int) -> torch.Tensor:
    """QR of a Gaussian matrix with column signs fixed by diag(R), as QuaRot's `random_orthogonal_matrix`."""
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64, generator=generator))
    return (q * torch.sign(torch.diag(r)).unsqueeze(0)).contiguous()


def block_hadamard(n: int, min_block: int = 64) -> torch.Tensor:
    """I_(n/b) (x) H_b with b the largest power of two dividing n.

    Raises ValueError when b < min_block.
    """
    block = n & -n
    if block < min_block:
        raise ValueError(f"largest power of two dividing {n} is {block}, below the minimum block {min_block}")
    return torch.kron(torch.eye(n // block, dtype=torch.float64), hadamard(block))


def kron_hadamard(num_heads: int, head_dim: int) -> torch.Tensor:
    """H_num_heads (x) H_head_dim, equal to hadamard(num_heads * head_dim) when head_dim is a power of two."""
    return torch.kron(hadamard(num_heads), hadamard(head_dim))


def fold_input(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """W @ R for a consumer whose input x became x @ R; computed in float64, returned in weight's dtype."""
    return (weight.double() @ rotation.to(weight.device).double()).to(weight.dtype)


def fold_output(weight: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """R.T @ W for a producer whose output y became y @ R; computed in float64, returned in weight's dtype."""
    return (rotation.to(weight.device).double().T @ weight.double()).to(weight.dtype)


def fold_input_per_head(weight: torch.Tensor, rotation: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Apply `fold_input` with the same [head_dim, head_dim] rotation to every head_dim-wide column block."""
    out_features, in_features = weight.shape
    blocks = weight.double().reshape(out_features, in_features // head_dim, head_dim)
    folded = blocks @ rotation.to(weight.device).double()
    return folded.reshape(out_features, in_features).to(weight.dtype)


def fold_output_per_head(weight: torch.Tensor, rotation: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Apply `fold_output` with the same [head_dim, head_dim] rotation to every head_dim-high row block."""
    out_features, in_features = weight.shape
    blocks = weight.double().reshape(out_features // head_dim, head_dim, in_features)
    folded = rotation.to(weight.device).double().T @ blocks
    return folded.reshape(out_features, in_features).to(weight.dtype)


def fold_input_per_head_blocks(weight: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Apply `fold_input` with matrices[i] to the i-th head_dim-wide column block; matrices [n, D, D]."""
    count, head_dim, _ = matrices.shape
    out_features, in_features = weight.shape
    if in_features != count * head_dim:
        raise ValueError(f"weight has {in_features} input columns, matrices cover {count * head_dim}")
    weight64 = weight.double()
    matrices64 = matrices.to(weight.device).double()
    blocks = [weight64[:, i * head_dim : (i + 1) * head_dim] @ matrices64[i] for i in range(count)]
    return torch.cat(blocks, dim=1).to(weight.dtype)


def fold_output_per_head_blocks(weight: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    """Apply `fold_output` with matrices[i] to the i-th head_dim-high row block; matrices [n, D, D]."""
    count, head_dim, _ = matrices.shape
    out_features, in_features = weight.shape
    if out_features != count * head_dim:
        raise ValueError(f"weight has {out_features} output rows, matrices cover {count * head_dim}")
    blocks = weight.double().reshape(count, head_dim, in_features)
    folded = matrices.to(weight.device).double().transpose(1, 2) @ blocks
    return folded.reshape(out_features, in_features).to(weight.dtype)
