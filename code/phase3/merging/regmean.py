"""RegMean (Jin et al. 2023): closed-form merge that inverts the
per-task input-activation Gram.

The published formula is
    W^* = (Σ_t α_t Z_t^T Z_t)^{-1} (Σ_t α_t Z_t^T Z_t W_t)
where Z_t are the per-task input activations (data-dependent) and W_t
are the per-task fine-tuned weights. We adopt the *adapter-only*
variant: the Δ_t = scaling_t B_t A_t are merged per layer, with the
Gram Σ Z^T Z replaced by Σ A_t^T A_t (the column-space surrogate
that turns RegMean into a data-free merge --- the same trick we use
for rd-encoder ridge but with a different projector).

For each layer:
  Δ_merged = (Σ_t (A_t^T A_t + λ I))^{-1} (Σ_t (A_t^T A_t) Δ_t)

with a small Tikhonov λ for numerical stability (default 1e-3). Then
SVD-truncate to rank-r LoRA factors.

Adapter-only RegMean is what most public re-implementations actually
ship under the "RegMean" name when no activation cache is available;
the proper RegMean requires forward passes to collect Z_t, which is
out of scope for a data-free merge baseline.
"""

from __future__ import annotations

import torch

from ._adapter_utils import materialize_delta, svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def merge_regmean(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    ridge_lambda: float = 1e-3,
    **_kwargs,
) -> None:
    """Adapter-only RegMean: Gram-weighted average of Δ_t at each layer."""
    assert len(adapter_names) == len(weights), "adapter/weight length mismatch"

    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        # Gather A_t (rank x in_dim) and Δ_t (out x in) per task.
        # We need access to the per-task LoRA A factor: materialize via
        # the layer view if available; else fall back to inferring from
        # delta + B SVD. For PeftModelView (production) the factors are
        # directly available.
        layer_objs = []
        for name in adapter_names:
            entry = model.get_layer(name, layer) if hasattr(model, "get_layer") else None
            layer_objs.append(entry)

        # Δ_t per task at this layer
        deltas = [model.get_delta(ad, layer) for ad in adapter_names]
        # in_dim from the layer's first delta width
        in_dim = deltas[0].shape[1]

        # Per-task Gram on the input-side: prefer A_t^T A_t if available,
        # else fall back to Δ_t^T Δ_t / ||Δ_t||_F^2 (a coarse data-free
        # input-side surrogate). The fallback is engaged in tests with
        # FakeLoraLayer; production paths (PeftModelView) hit the A_t
        # branch.
        grams = []
        for ti, name in enumerate(adapter_names):
            A = None
            try:
                lo = layer_objs[ti]
                if lo is not None and hasattr(lo, "A"):
                    A = lo.A  # (r, in_dim)
            except Exception:
                A = None
            if A is None:
                # Fallback: Δ_t^T Δ_t as the per-task Gram surrogate.
                d = deltas[ti]
                gram = d.transpose(-1, -2) @ d
            else:
                gram = A.transpose(-1, -2).float() @ A.float()
            grams.append(gram)

        # Weighted sums
        device, dtype = deltas[0].device, deltas[0].dtype
        denom = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
        numer = torch.zeros_like(deltas[0], dtype=torch.float32)
        for ti, (gram, d) in enumerate(zip(grams, deltas)):
            w = float(weights[ti])
            g = gram.to(torch.float32)
            denom = denom + w * g
            numer = numer + w * d.to(torch.float32) @ g
        denom = denom + ridge_lambda * torch.eye(
            in_dim, dtype=torch.float32, device=device)
        # Solve denom @ delta^T = numer^T  =>  delta = numer @ denom^{-1}
        delta = torch.linalg.solve(denom, numer.transpose(-1, -2)).transpose(-1, -2)
        delta = delta.to(dtype)

        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
