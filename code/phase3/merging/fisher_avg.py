"""Fisher-magnitude merging (RegMean / Fisher-merging proxy).

Per-coordinate Fisher proxy F_t[i,j] = Delta_t[i,j]^2 (squared adapter
magnitude). The merged delta at each coordinate is the Fisher-precision-
weighted average:

    Delta_merged[i,j] = (sum_t F_t[i,j] * Delta_t[i,j]) / sum_t F_t[i,j]

This is the cheap variant of Fisher merging (Matena & Raffel 2022) that
avoids gradient passes by using squared magnitudes as the per-coordinate
Fisher proxy — standard practice when the trained adapters are at or
near a local minimum and the residual gradients are dominated by
magnitude.

Then SVD-truncate the merged delta back to rank r, same as TA.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def merge_fisher_avg(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    eps: float = 1e-12,
    **_kwargs,
) -> None:
    """Per-coordinate Fisher-magnitude weighted average of task deltas.

    The `weights` arg is the per-task scalar that the orchestrator
    passes in (typically 1/T uniformly). We multiply each task's Fisher
    proxy by its weight to keep the same calling convention.
    """
    assert len(adapter_names) == len(weights), "adapter/weight length mismatch"

    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [model.get_delta(ad, layer) for ad in adapter_names]
        fishers = [(float(w) * d.pow(2)) for w, d in zip(weights, deltas)]
        # Per-coordinate denominator: sum of Fisher proxies plus eps to
        # guard against all-zero coordinates.
        denom = fishers[0].clone()
        for f in fishers[1:]:
            denom = denom + f
        denom = denom + eps
        # Numerator: F_t * Delta_t summed.
        num = fishers[0] * deltas[0]
        for f, d in zip(fishers[1:], deltas[1:]):
            num = num + f * d
        delta = num / denom
        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
