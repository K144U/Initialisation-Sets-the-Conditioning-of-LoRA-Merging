"""TIES-Merging (Yadav et al. 2023, NeurIPS).

For each layer:
  1. Trim: keep top-`density` magnitude per task vector, zero rest.
  2. Elect sign: per-parameter, majority sign across tasks (weighted by magnitude
     when majority_sign_method='total', frequency-weighted when ='frequency').
  3. Disjoint merge: average only the tasks whose sign matches the elected sign.

Then SVD-truncate to rank-r LoRA factors.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _trim_topk(delta: torch.Tensor, density: float) -> torch.Tensor:
    """Keep top-`density` fraction by absolute value; zero rest."""
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)
    flat = delta.abs().flatten()
    k = max(1, int(round(density * flat.numel())))
    # threshold = k-th largest abs value
    threshold = torch.topk(flat, k, largest=True).values[-1]
    mask = delta.abs() >= threshold
    return delta * mask.to(delta.dtype)


def _elect_sign(stacked: torch.Tensor, method: str = "total") -> torch.Tensor:
    """stacked: (T, out_dim, in_dim). Returns elected sign: (out_dim, in_dim), values in {-1, 0, +1}."""
    if method == "total":
        # Sign of the magnitude-weighted sum (== sign of sum since each entry is already signed)
        signed_sum = stacked.sum(dim=0)
        return torch.sign(signed_sum)
    if method == "frequency":
        # +1 if more entries are positive than negative; -1 otherwise.
        pos = (stacked > 0).sum(dim=0).to(torch.float32)
        neg = (stacked < 0).sum(dim=0).to(torch.float32)
        return torch.sign(pos - neg)
    raise ValueError(f"unknown majority_sign_method: {method!r}")


def merge_ties(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    density: float = 0.2,
    majority_sign_method: str = "total",
    **_kwargs,
) -> None:
    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        # 1. Trim per task (stay on whatever device get_delta returned)
        trimmed = [_trim_topk(model.get_delta(ad, layer), density) for ad in adapter_names]
        stacked = torch.stack(trimmed, dim=0)  # (T, out, in)
        device, dtype = stacked.device, stacked.dtype

        # 2. Elect sign
        elected = _elect_sign(stacked, majority_sign_method)

        # 3. Disjoint merge: per-param, take weighted mean only of tasks whose sign matches the elected
        sign_t = torch.sign(stacked)
        match = ((sign_t == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)).to(dtype)
        w_t = torch.tensor(weights, dtype=dtype, device=device).view(-1, 1, 1)
        weighted = stacked * match * w_t
        denom_sum = (match * w_t).sum(dim=0).abs()
        delta = torch.where(
            denom_sum > 1e-12,
            weighted.sum(dim=0) / denom_sum.clamp_min(1e-12),
            torch.zeros_like(denom_sum),
        )

        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
