"""KnOTS — Knots Ties SVD (Stoica et al. 2025, ICLR).

Algorithm (per arXiv:2410.19735, §3):
For each layer:
  1. Materialize per-task delta: Δ_t = B_t @ A_t (scaling included).
  2. Concatenate along the row dim: Δ_cat = [Δ_1; Δ_2; ...; Δ_T] (T·out × in).
  3. SVD: Δ_cat = U Σ V^T. The right basis V spans the joint column-space.
  4. Project each Δ_t to the aligned coefficient space: C_t = Δ_t @ V.
     (Each row of C_t is task-t's coefficients in the shared V basis.)
  5. Apply an inner merge (Task Arithmetic by default) on the C_t to get C_merged.
  6. Reconstruct the merged delta: Δ_merged = C_merged @ V^T.
  7. SVD-truncate Δ_merged to rank-r LoRA factors.

The "Knots-Ties" variant uses TIES as the inner merge instead of linear.

WARNING (2026-08-02): `inner_combination="linear"` is algebraically Task
Arithmetic and carries NO information about subspace alignment.
`torch.linalg.svd(..., full_matrices=False)` returns a V whose columns span at
least the row space of every Delta_t, so the projection round trip is the
identity there:

    Delta_merged = (sum_t w_t Delta_t) V V^T = sum_t w_t Delta_t

Verified to 3e-06 in merging/tests/test_knots.py. Every shipped KnOTS cell used
this default, which is why they matched TA to 3-4 decimals, and the paper read
that agreement as evidence that "subspace alignment has nothing to exploit at
zero overlap" in four places. It is not evidence of anything.

Use `inner_combination="ties"` for a method that actually differs: sign
election in the rotated basis is not sign election in the original basis, and
KnOTS-TIES is the variant the KnOTS paper headlines. The registry default is
left at "linear" so historical cells stay reproducible; pick "ties" explicitly.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .ties import _trim_topk, _elect_sign
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _inner_merge_linear(coeffs: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    """coeffs[t]: (out_dim, k). weights: (T,). Returns (out_dim, k)."""
    out = torch.zeros_like(coeffs[0])
    for c, w in zip(coeffs, weights):
        out = out + float(w) * c
    return out


def _inner_merge_ties(
    coeffs: list[torch.Tensor], weights: list[float], density: float = 0.2,
) -> torch.Tensor:
    stacked = torch.stack([_trim_topk(c, density) for c in coeffs], dim=0)
    elected = _elect_sign(stacked, "total")
    sign_t = torch.sign(stacked)
    match = ((sign_t == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)).to(stacked.dtype)
    w_t = torch.tensor(weights, dtype=stacked.dtype).view(-1, 1, 1)
    num = (stacked * match * w_t).sum(dim=0)
    den = (match * w_t).sum(dim=0).abs().clamp_min(1e-12)
    return torch.where(den > 1e-12, num / den, torch.zeros_like(num))


def merge_knots(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    inner_combination: str = "linear",
    density: float = 0.2,
    **_kwargs,
) -> None:
    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]

        # 1+2. Per-task delta, concatenated (stays on whatever device get_delta returns)
        deltas = [model.get_delta(ad, layer) for ad in adapter_names]
        # SVD needs float32 for stability; cast there only
        deltas_f32 = [d.to(torch.float32) for d in deltas]
        delta_cat = torch.cat(deltas_f32, dim=0)  # (T·out, in)

        # 3. SVD of concatenated stack. We need V (right singular vectors).
        _U, _S, Vh = torch.linalg.svd(delta_cat, full_matrices=False)
        V = Vh.T  # (in_dim, k)

        # 4. Project each task's delta into the shared V basis
        coeffs = [d @ V for d in deltas_f32]

        # 5. Inner merge
        if inner_combination == "linear":
            C_merged = _inner_merge_linear(coeffs, weights)
        elif inner_combination == "ties":
            C_merged = _inner_merge_ties(coeffs, weights, density)
        else:
            raise ValueError(f"unknown inner_combination: {inner_combination!r}")

        # 6. Reconstruct, cast back to original dtype before truncation
        delta_merged = (C_merged @ V.T).to(deltas[0].dtype)

        # 7. SVD-truncate to rank-r
        A_new, B_new = svd_truncate_to_rank(delta_merged, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
