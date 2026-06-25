"""AdaMerging (Yang et al. ICLR 2024) — task-vector merging with
learned per-task coefficients.

The original AdaMerging optimizes per-task scalars α_t (or per-layer
α_t,l) to minimize an *entropy* objective on unlabeled test inputs.
That requires forward passes, which puts it outside the data-free
merge category our paper centers on.

We implement the *data-free closed-form* variant: per-task coefficients
α_t are set proportional to the per-task Frobenius norm
$\|\Delta_t\|_F$ (the natural data-free signal for "this task asks
more of the merge"), normalized to sum to 1. Then a TA-style linear
combination, followed by SVD truncation to rank-r LoRA factors.

This is what most public re-implementations effectively ship when no
unlabeled test distribution is available, and it sits structurally
between TA (uniform weights) and Fisher-magnitude (per-coordinate
weighted average).
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def merge_adamerging(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    **_kwargs,
) -> None:
    """Frobenius-norm-weighted linear combination of per-layer Δ_t."""
    assert len(adapter_names) == len(weights), "adapter/weight length mismatch"

    # Compute per-task Frobenius norms across all layers (global alpha_t)
    task_total_fro_sq = [0.0 for _ in adapter_names]
    for layer in model.layer_names():
        for ti, name in enumerate(adapter_names):
            d = model.get_delta(name, layer)
            task_total_fro_sq[ti] += float(d.pow(2).sum().item())
    task_alphas = [
        (task_total_fro_sq[ti] ** 0.5) * float(weights[ti])
        for ti in range(len(adapter_names))
    ]
    total = sum(task_alphas)
    if total > 0:
        task_alphas = [a / total for a in task_alphas]
    else:
        task_alphas = [1.0 / len(adapter_names)] * len(adapter_names)

    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        delta = task_alphas[0] * model.get_delta(adapter_names[0], layer)
        for ti in range(1, len(adapter_names)):
            delta = delta + task_alphas[ti] * model.get_delta(
                adapter_names[ti], layer)
        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
