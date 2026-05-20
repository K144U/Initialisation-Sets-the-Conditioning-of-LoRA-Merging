"""Task Arithmetic (Ilharco et al. 2023).

Merged delta per layer: Δ_merged = Σ_t w_t · Δ_t.
Then SVD-truncate back to rank-r LoRA factors.
"""

from __future__ import annotations

from typing import Iterable

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def merge_task_arithmetic(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    **_kwargs,
) -> None:
    """Linear combination of per-layer task deltas; SVD-truncate to rank r."""
    assert len(adapter_names) == len(weights), "adapter/weight length mismatch"

    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [model.get_delta(ad, layer) for ad in adapter_names]
        delta = float(weights[0]) * deltas[0]
        for w, d in zip(weights[1:], deltas[1:]):
            delta = delta + float(w) * d
        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
