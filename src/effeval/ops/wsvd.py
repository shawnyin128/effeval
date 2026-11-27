"""WSVD training ops: whitening, factor initialisation, STE fake quant, Stiefel SGD and per-projection FT/QAT.

Numerics follow the released WSVD code (`fake_quant/local_ft_grad_utils.py`, `optimizer_utils.py`,
`quant_utils.py`, `grad_info_utils.py`), including its bf16 accumulations and float32/float64 casts.
"""

from __future__ import annotations

import contextlib
import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch

from effeval.ops import rotation as rot

logger = logging.getLogger(__name__)
TRAIN_DTYPE = torch.bfloat16
_EPS = 1e-8


def cholesky_lower_f32(
    gram: torch.Tensor, *, shift_fallback: bool = False, label: str = ""
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(L, inv(L))`, both float32, with `L = cholesky(gram.double()).float()` and `inv` taken in float32.

    With `shift_fallback`, a failed factorisation is retried as in the released code: the double Gram gets
    `(-lambda_min + 1e-6) * I` added, with `I` a default-dtype (float32) identity, and a warning naming
    `label` is logged. Raises ValueError if `gram` is not positive definite (after the shift, if enabled).
    """
    gram = gram.double()
    try:
        lower = torch.linalg.cholesky(gram).float()
    except torch.linalg.LinAlgError as error:
        if not shift_fallback:
            raise ValueError(f"Gram matrix is not positive definite: {error}") from error
        shift = -torch.linalg.eigvalsh(gram)[0] + 1e-6
        logger.warning("%s: input Gram is not positive definite; shifted the diagonal by %.3e", label, shift)
        gram += shift * torch.eye(gram.shape[0], device=gram.device)
        try:
            lower = torch.linalg.cholesky(gram).float()
        except torch.linalg.LinAlgError as retry_error:
            raise ValueError(
                f"Gram matrix is not positive definite after the shift: {retry_error}"
            ) from retry_error
    return lower, torch.linalg.inv(lower)


def released_head_budget(
    numel: int, d_in: int, d_out: int, heads: int, num_layers: int, rank_ratio: float
) -> int:
    """Singular values kept per head over all layers.

    `int(rank_ratio * (num_layers * (numel / (d_in + d_out / heads) / heads)))`.
    """
    return int(rank_ratio * (num_layers * (numel / (d_in + d_out / heads) / heads)))


def init_factors(
    U: torch.Tensor,
    S: torch.Tensor,
    V: torch.Tensor,
    head_indices: Sequence[Sequence[int]],
    l_inv: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-head factors `A_h = U_h sqrt(S_h)`, `B_h = sqrt(S_h) (inv(L)^T V_h)^T` of the selected components.

    `U [H, hd, n]`, `S [H, n]`, `V [H, d, n]` are the float32 SVD of the whitened heads, `head_indices[h]`
    the kept component indices of head `h` (in the given order), `l_inv` is `inv(L) [d, d]` float32.
    Returns lists of `A_h [hd, r_h]` and `B_h [r_h, d]` in `TRAIN_DTYPE`.
    """
    a_list, b_list = [], []
    for h, indices in enumerate(head_indices):
        idx = list(indices)
        u, s, v = U[h][:, idx], S[h][idx], V[h][:, idx]
        v = l_inv.T @ v
        a_list.append(u.mul(s.sqrt()).contiguous().to(TRAIN_DTYPE))
        b_list.append(v.t().mul(s.sqrt().view(-1, 1)).contiguous().to(TRAIN_DTYPE))
    return a_list, b_list


def fisher_weight(accumulated: torch.Tensor, count: int) -> torch.Tensor:
    """`accumulated.div(count).abs().sqrt()` in the dtype of `accumulated`."""
    return accumulated.div(count).abs().sqrt()


def _maxq(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def row_scale(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row scale `[rows, 1]` (float32 for bf16/float32 `x`).

    `max(|min(x, 0)|, max(x, 0), 1e-5) / maxq` per row.
    """
    flat = x.flatten(1)
    zeros = torch.zeros(flat.shape[0], device=flat.device)
    xmin = torch.minimum(flat.min(1)[0], zeros)
    xmax = torch.maximum(flat.max(1)[0], zeros)
    xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
    return (xmax / _maxq(bits)).reshape(-1, 1)


class _STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, maxq):
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def ste_quantize(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """`scale * clamp(round(x / scale), -(maxq + 1), maxq)` as `x.dtype`; the gradient is the identity."""
    return _STEQuantize.apply(x, scale, _maxq(bits)).to(x.dtype)


def prequantize_dense(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-row symmetric quantization of `weight` rounded to bf16 (in float32), returned as `weight.dtype`."""
    with torch.no_grad():
        values = weight.to(TRAIN_DTYPE).float()
        return ste_quantize(values, row_scale(values, bits), bits).to(weight.dtype)


def _qr_retraction(matrix: torch.Tensor) -> torch.Tensor:
    dtype = matrix.dtype
    q, r = torch.linalg.qr(matrix.t().contiguous().to(torch.float32))
    q = q * torch.diag(r, 0).sign().expand_as(q)
    return q.t().contiguous().to(dtype)


class StiefelSGD(torch.optim.Optimizer):
    """SGD on the Stiefel manifold with the Cayley transform (the released `SGDG` with `stiefel=True`).

    Parameters are 2-D with rows <= cols. A QR retraction of the row-normalised parameter replaces it with
    probability 1/101 per parameter and step, drawn from `rng.randint(1, 101)`. Raises ValueError for a
    parameter that is not 2-D or has more rows than columns.
    """

    def __init__(self, params, lr: float, rng: random.Random, momentum: float = 0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum})
        self.rng = rng
        for group in self.param_groups:
            for p in group["params"]:
                if p.dim() != 2 or p.size(0) > p.size(1):
                    raise ValueError(
                        f"StiefelSGD needs 2-D parameters with rows <= cols, got {tuple(p.shape)}"
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            momentum, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                dtype = p.dtype
                matrix = p.view(p.size(0), -1)
                g = p.grad.view(p.size(0), -1).to(dtype)
                q = matrix / matrix.norm(p=2, dim=1, keepdim=True).add(_EPS)
                if self.rng.randint(1, 101) == 1:
                    q = _qr_retraction(q)
                state = self.state[p]
                v = state.get("momentum_buffer")
                if v is None:
                    v = torch.zeros_like(g.t())
                    state["momentum_buffer"] = v
                v.mul_(momentum).add_(-g.t())
                mx = torch.mm(v, q)
                xmx = torch.mm(q, mx)
                xxmx = torch.mm(q.t(), xmx)
                w_hat = mx - 0.5 * xxmx
                w = w_hat - w_hat.t()
                denom = torch.abs(w).sum(dim=0).max().clamp_min(_EPS)
                alpha = min(1.0 * (1.0 / denom.item()), float(lr))
                x = q.t()
                y = x + alpha * v
                for _ in range(5):
                    y = x + alpha * torch.matmul(w, 0.5 * (x + y))
                p.copy_(y.t().view_as(p))
                v.copy_(torch.mm(w, q.t()))
        return loss


@contextlib.contextmanager
def no_tf32() -> Iterator[None]:
    """Disables TF32 for CUDA matmuls and cuDNN inside the block and restores the previous flags."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


@dataclass(frozen=True)
class QATSettings:
    """QAT phase: starts at iteration `int(iters * start_fraction)` with factor learning rate `lr_factors`.

    `lr_rotation` is the Stiefel learning rate of the per-head rotation (used when `learned_rotation`),
    `scale_update_every` the period of quantizer-scale refreshes in iterations, `bits` the weight bit width.
    """

    start_fraction: float
    lr_factors: float
    lr_rotation: float
    scale_update_every: int
    bits: int
    learned_rotation: bool


@dataclass(frozen=True)
class FTSettings:
    """Local fine-tuning: `iters` iterations of `steps_per_iteration` Adam steps at `lr`.

    `seed` seeds the rotations and the SGDG retraction draws.
    """

    iters: int
    lr: float
    steps_per_iteration: int
    seed: int
    qat: QATSettings | None


@dataclass
class TrainResult:
    """Trained per-head factors (`A_h [hd, r_h]`, `B_h [r_h, d]`, rotation folded) and per-step losses."""

    A: list[torch.Tensor]
    B: list[torch.Tensor]
    ft_losses: list[float]
    qat_losses: list[float]


def _normalised_loss(y: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    return (y - out).pow(2).mean() / out.pow(2).mean()


def _hadamard_rows(a: torch.Tensor, hadamard: torch.Tensor) -> torch.Tensor:
    return (hadamard.T @ a.float()).to(a.dtype)


def train_projection(
    A: Sequence[torch.Tensor],
    B: Sequence[torch.Tensor],
    W: torch.Tensor,
    weight: torch.Tensor | None,
    settings: FTSettings,
    residual: torch.Tensor | None,
    value_hadamard: torch.Tensor | None,
) -> TrainResult:
    """Trains the per-head factors of one projection to reproduce `W` (weight-space FT, then optional QAT).

    `A[h] [hd, r_h]`, `B[h] [r_h, d]` and `W [H, hd, d]` are in `TRAIN_DTYPE`; `weight [H, hd, d]` is the
    elementwise Fisher weight (or None). The FT loss is `mean((w*(A_h B_h) - w*W)^2) / mean((w*W)^2)` with
    `w = weight`. With `settings.qat`, iterations from `int(iters * start_fraction)` instead fake-quantize
    `A_h R_h` and `R_h^T B_h` with a per-head orthogonal `R_h` (learned by `StiefelSGD` when
    `learned_rotation`, else the identity) against
    the target `W` rotated by `value_hadamard [hd, hd]` (value projection only) on the output side and by
    `residual [d, d]` float64 on the input side (both None: no rotation). `R_h` is folded into the returned
    (`A_h R_h`, `R_h^T B_h`). Inputs are not modified.
    """
    heads = W.shape[0]
    if len(A) != heads or len(B) != heads:
        raise ValueError(f"expected {heads} head factors, got {len(A)} A and {len(B)} B")
    qat = settings.qat
    a_params = [a.detach().clone().requires_grad_(True) for a in A]
    b_params = [b.detach().clone().requires_grad_(True) for b in B]
    optimizer = torch.optim.Adam(a_params + b_params, lr=settings.lr)

    rotations: list[torch.Tensor] = []
    rotation_optimizer = None
    learned = qat is not None and qat.learned_rotation
    if learned:
        rotations = [
            rot.random_orthogonal(a.shape[1], settings.seed).to(TRAIN_DTYPE).to(a.device).requires_grad_(True)
            for a in a_params
        ]
        rotation_optimizer = StiefelSGD(rotations, lr=qat.lr_rotation, rng=random.Random(settings.seed))

    out_ft = W if weight is None else weight * W
    out_qat = None
    if qat is not None:
        w_had = W
        if value_hadamard is not None:
            w_had = torch.stack([_hadamard_rows(w, value_hadamard) for w in W], dim=0)
        if residual is not None:
            w_had = torch.stack(
                [torch.matmul(w.to(torch.float64), residual).to(W.dtype) for w in w_had], dim=0
            )
        out_qat = w_had if weight is None else weight * w_had

    qat_start = int(settings.iters * qat.start_fraction) if qat is not None else None
    scales_a: list[torch.Tensor | None] = [None] * heads
    scales_b: list[torch.Tensor | None] = [None] * heads
    ft_losses: list[torch.Tensor] = []
    qat_losses: list[torch.Tensor] = []

    with torch.enable_grad():
        for it in range(settings.iters):
            in_qat = qat is not None and it >= qat_start
            if qat is not None and it == qat_start:
                for group in optimizer.param_groups:
                    group["lr"] = qat.lr_factors
            refresh = in_qat and ((it + 1) % qat.scale_update_every == 0 or it == qat_start)
            for _ in range(settings.steps_per_iteration):
                optimizer.zero_grad()
                if not in_qat:
                    y = torch.cat(
                        [(a @ b).unsqueeze(0) for a, b in zip(a_params, b_params, strict=True)], dim=0
                    )
                    y = y if weight is None else weight * y
                    loss = _normalised_loss(y, out_ft)
                    loss.backward()
                    optimizer.step()
                    ft_losses.append(loss.detach())
                    continue
                if rotation_optimizer is not None:
                    rotation_optimizer.zero_grad()
                parts = []
                for h in range(heads):
                    a = a_params[h]
                    if value_hadamard is not None:
                        a = _hadamard_rows(a, value_hadamard)
                    b = b_params[h]
                    if residual is not None:
                        b = torch.matmul(b.to(torch.float64), residual).to(b.dtype)
                    if learned:
                        a, b = a @ rotations[h], rotations[h].T @ b
                    if refresh:
                        scales_a[h] = row_scale(a.detach(), qat.bits)
                        scales_b[h] = row_scale(b.detach(), qat.bits)
                    parts.append(
                        (
                            ste_quantize(a, scales_a[h], qat.bits) @ ste_quantize(b, scales_b[h], qat.bits)
                        ).unsqueeze(0)
                    )
                y = torch.cat(parts, dim=0)
                y = y if weight is None else weight * y
                loss = _normalised_loss(y, out_qat)
                loss.backward()
                optimizer.step()
                if rotation_optimizer is not None:
                    rotation_optimizer.step()
                qat_losses.append(loss.detach())

    with torch.no_grad():
        if learned:
            a_out = [
                (a.detach().float() @ r.float()).to(a.dtype) for a, r in zip(a_params, rotations, strict=True)
            ]
            b_out = [
                (r.T.float() @ b.detach().float()).to(b.dtype)
                for b, r in zip(b_params, rotations, strict=True)
            ]
        else:
            a_out = [a.detach().clone() for a in a_params]
            b_out = [b.detach().clone() for b in b_params]

    def to_floats(losses: list[torch.Tensor]) -> list[float]:
        return torch.stack(losses).float().tolist() if losses else []

    return TrainResult(A=a_out, B=b_out, ft_losses=to_floats(ft_losses), qat_losses=to_floats(qat_losses))
