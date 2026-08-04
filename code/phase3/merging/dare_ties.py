"""DARE-TIES — drop-and-rescale, then TIES (Yu et al. 2024, composed).

This is the official `mask_merging` wrapper with `mask_apply_method="ties_merging"`
(yule-BUAA/MergeLM, model_merging_methods/merging_methods.py): each task's delta
is masked with DARE's drop-and-rescale, and the masked deltas are then merged by
TIES unchanged.

Two independent knobs, deliberately kept separate:

  dare_density   fraction of delta parameters DARE KEEPS (1 - mask_rate)
  ties_density   fraction TIES keeps in its magnitude trim

Their interaction is structural and worth knowing when reading results, so it is
recorded in notes/prereg_dare_ties_2026-08-04.md rather than discovered later:

  dare_density > ties_density   both bind; TIES trims what DARE left
  dare_density = ties_density   they roughly coincide
  dare_density < ties_density   the TIES trim goes INERT, because fewer entries
                                are nonzero than TIES wants to keep

DARE's rescale is a uniform scalar, so it cannot change which entries the trim
keeps (top-k by magnitude is scale-invariant) nor which signs are elected. It
changes the merged magnitude only, via TIES's weighted-mean denominator.

Why this method exists: DARE + task arithmetic is an unbiased estimator of task
arithmetic and therefore cannot beat it in expectation (notes/audit_dare_
2026-08-04.md). That argument does NOT apply here, because trimming and sign
election are biased, so the mask changes which parameters survive. This is the
composition where DARE could genuinely help.

The mask draw reuses dare.py's operator and the same single advancing generator,
so masks are independent across tasks and layers, as DARE requires.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .ties import ties_merge_deltas
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _dare_mask(delta: torch.Tensor, density: float, g: torch.Generator) -> torch.Tensor:
    """Identical to the operator in dare.py."""
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)
    mask_cpu = (torch.rand(delta.shape, generator=g, dtype=torch.float32) < density)
    mask = mask_cpu.to(device=delta.device, dtype=delta.dtype)
    return (delta * mask) / density


def merge_dare_ties(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    dare_density: float = 0.2,
    ties_density: float = 0.2,
    majority_sign_method: str = "total",
    seed: int = 20260518,
    **_kwargs,
) -> None:
    new_factors: dict[str, FakeLoraLayer] = {}
    g = torch.Generator().manual_seed(int(seed))   # CPU generator, masks moved per-delta
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        masked = [_dare_mask(model.get_delta(ad, layer), dare_density, g)
                  for ad in adapter_names]
        delta = ties_merge_deltas(masked, weights, ties_density, majority_sign_method)
        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
