"""DARE — Drop And REscale (Yu et al. 2024).

For each layer and each task:
  1. Randomly drop a (1 - density) fraction of parameters.
  2. Rescale survivors by 1/density to keep the expected sum invariant.

Then apply Task Arithmetic on the rescaled deltas; SVD-truncate to rank-r.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _dare_drop_rescale(delta: torch.Tensor, density: float, generator: torch.Generator) -> torch.Tensor:
    if density >= 1.0:
        return delta
    if density <= 0.0:
        return torch.zeros_like(delta)
    mask = (torch.rand(delta.shape, generator=generator, dtype=torch.float32) < density).to(delta.dtype)
    return (delta * mask) / density


def merge_dare(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    density: float = 0.2,
    seed: int = 20260518,
    **_kwargs,
) -> None:
    new_factors: dict[str, FakeLoraLayer] = {}
    g = torch.Generator().manual_seed(int(seed))   # CPU generator; we'll move masks to delta's device
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = []
        for ad in adapter_names:
            d = model.get_delta(ad, layer)
            if density < 1.0 and density > 0.0:
                mask_cpu = (torch.rand(d.shape, generator=g, dtype=torch.float32) < density)
                mask = mask_cpu.to(device=d.device, dtype=d.dtype)
                d = (d * mask) / density
            elif density <= 0.0:
                d = torch.zeros_like(d)
            deltas.append(d)
        delta = float(weights[0]) * deltas[0]
        for w, d in zip(weights[1:], deltas[1:]):
            delta = delta + float(w) * d
        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
