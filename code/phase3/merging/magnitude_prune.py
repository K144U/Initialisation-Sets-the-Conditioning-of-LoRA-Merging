"""Magnitude pruning + plain task arithmetic — E7 Phase 2b probe.

Per-layer, per-task: keep top-`density` fraction of |delta| coefficients,
zero rest. Then plain sum (task arithmetic), SVD-truncate to rank-r.

This is the "implicit-TIES via sparsity" hypothesis test: if matching b=2
TVQ's effective sparsity (~50% in the zero bucket, ~87% with destroyed
signal) via explicit magnitude pruning reproduces the b=2 'less-is-more'
dip, the mechanism is sparsity rather than the structural choices
TIES makes (sign election + disjoint mean).

Compared to TIES at the same density: TIES adds (i) sign election and
(ii) disjoint-mean re-averaging on the surviving entries. magnitude_prune
strips both — it's the cleanest test of "is sparsity alone the
mechanism?"
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _trim_topk(delta: torch.Tensor, density: float) -> torch.Tensor:
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)
    flat = delta.abs().flatten()
    k = max(1, int(round(density * flat.numel())))
    threshold = torch.topk(flat, k, largest=True).values[-1]
    mask = delta.abs() >= threshold
    return delta * mask.to(delta.dtype)


def merge_magnitude_prune(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    density: float = 0.5,
    **_kwargs,
) -> None:
    """Trim each task delta to top-`density` by magnitude; weighted sum;
    SVD-truncate to rank-r."""
    assert len(adapter_names) == len(weights)
    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [_trim_topk(model.get_delta(ad, layer), density)
                  for ad in adapter_names]
        # Plain weighted sum (task arithmetic on the pruned deltas)
        merged = deltas[0] * weights[0]
        for w, d in zip(weights[1:], deltas[1:]):
            merged = merged + w * d
        A_new, B_new = svd_truncate_to_rank(merged, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
