"""Shared utilities for merging methods.

The PEFT side stores each LoRA layer as two matrices (A: r x in_dim, B: out_dim x r)
plus a scaling factor. The full task-vector delta for layer L is:
    Delta_t^L = scaling_t * B_t @ A_t   (shape: out_dim x in_dim)

KnOTS and TVQ operate on these full deltas, then need to convert the merged
delta back to (B_merged, A_merged) factors of the same rank r.
"""

from __future__ import annotations

from typing import Tuple

import torch


def materialize_delta(
    A: torch.Tensor,
    B: torch.Tensor,
    scaling: float = 1.0,
) -> torch.Tensor:
    """Return the full rank-r delta B @ A (with scaling).

    Shapes:
        A: (r, in_dim)
        B: (out_dim, r)
    Returns:
        (out_dim, in_dim) tensor
    """
    return scaling * (B @ A)


def svd_truncate_to_rank(
    delta: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose `delta` (out_dim x in_dim) into rank-`rank` LoRA factors.

    Returns:
        A: (rank, in_dim)
        B: (out_dim, rank)
    such that B @ A approximates `delta` in the rank-`rank` Frobenius-optimal sense.

    Stays on the same device as `delta`. Casts to float32 for SVD numerical
    stability (bf16 SVD is unreliable), then casts back to the original dtype.
    """
    out_dim, in_dim = delta.shape
    target_rank = min(rank, out_dim, in_dim)

    orig_dtype = delta.dtype
    orig_device = delta.device

    # For BIG matrices (4096-dim production), use randomized truncated SVD —
    # 10-50x faster than full SVD when target_rank << dim. For small matrices
    # (tests, ≤256-dim) the full SVD is both cheap AND deterministic.
    delta_f32 = delta.to(torch.float32)
    min_dim = min(out_dim, in_dim)
    use_lowrank = min_dim > 256 and target_rank < min_dim // 4
    if use_lowrank:
        # niter=4 gives near-optimal accuracy for small r; default is 2
        U, S, V = torch.svd_lowrank(delta_f32, q=target_rank, niter=4)
        # V here is (in_dim, r) — note svd_lowrank returns V, not Vh
        U_r, S_r, Vh_r = U, S, V.T
    else:
        U, S, Vh_full = torch.linalg.svd(delta_f32, full_matrices=False)
        U_r = U[:, :target_rank]
        S_r = S[:target_rank]
        Vh_r = Vh_full[:target_rank, :]

    A = Vh_r.to(dtype=orig_dtype, device=orig_device)
    B = (U_r * S_r).to(dtype=orig_dtype, device=orig_device)
    return A, B


def fold_scaling_into_B(B: torch.Tensor, scaling: float) -> torch.Tensor:
    """Multiply B by a scaling factor (used when going from PEFT factors to delta)."""
    return B * scaling
