"""DELLA: Drop Edits in LoRA Adapters (Deep et al. 2024).

DELLA = DARE-style random drop + TIES-style density trim + sign election.

For each layer:
  1. Random drop: with probability p, zero each coordinate; rescale
     survivors by 1/(1-p) to preserve expectation (DARE step).
  2. Trim: keep top-`density` magnitude per task vector, zero rest
     (TIES step).
  3. Elect sign: per-parameter magnitude-weighted majority sign.
  4. Disjoint merge: weighted mean of only the tasks whose sign matches
     the elected sign.

Then SVD-truncate to rank-r LoRA factors. Equivalent to running DARE
into TIES; the original paper claims the combination is strictly better
than either component on the merge benchmarks they test.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel
from .ties import _elect_sign, _trim_topk


def _dare_drop(delta: torch.Tensor, drop_p: float, gen: torch.Generator) -> torch.Tensor:
    """Random drop with probability drop_p; rescale survivors by 1/(1-drop_p)."""
    if drop_p <= 0.0:
        return delta
    if drop_p >= 1.0:
        return torch.zeros_like(delta)
    keep_p = 1.0 - drop_p
    mask = (torch.rand(delta.shape, generator=gen, device="cpu") < keep_p)
    mask = mask.to(delta.device, dtype=delta.dtype)
    return (delta * mask) / keep_p


def merge_della(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    drop_p: float = 0.2,
    density: float = 0.2,
    majority_sign_method: str = "total",
    seed: int = 20260624,
    **_kwargs,
) -> None:
    """DELLA = random drop -> density trim -> sign election -> disjoint merge."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        # 1. DARE random drop + rescale per task
        dropped = [_dare_drop(model.get_delta(ad, layer), drop_p, gen)
                   for ad in adapter_names]
        # 2. TIES density trim
        trimmed = [_trim_topk(d, density) for d in dropped]
        stacked = torch.stack(trimmed, dim=0)
        device, dtype = stacked.device, stacked.dtype

        # 3. Elect sign
        elected = _elect_sign(stacked, majority_sign_method)

        # 4. Disjoint merge
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
