"""TVQ — Task Vector Quantization (Kim et al. 2025, ICCV).

Uniform scalar quantization per per-layer task delta at `rate_bits` bits.
At b=32 the quantization is a no-op (control). Sum the quantized deltas
(Task Arithmetic) then SVD-truncate to rank-r LoRA factors.

The paper also describes a "Residual Task Vector Quantization" (decompose
into base + offset, mixed bit budget). We implement the simpler uniform
scalar variant for Phase 3; residual is a v2 refinement.
"""

from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _uniform_scalar_quantize(delta: torch.Tensor, bits: float) -> torch.Tensor:
    """Per-tensor min-max uniform scalar quantization. Returns dequantized tensor."""
    if bits >= 32:
        return delta  # no-op control
    if bits < 1:
        # 0-bit collapse: everything to zero (degenerate; included for robustness)
        return torch.zeros_like(delta)
    levels = (1 << int(round(bits))) - 1   # 2^b - 1
    min_val = delta.min()
    max_val = delta.max()
    if (max_val - min_val).abs() < 1e-12:
        return delta  # constant tensor — no quantization needed
    step = (max_val - min_val) / levels
    q = torch.round((delta - min_val) / step)
    return q * step + min_val


def merge_tvq(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    rate_bits: float = 32,
    **_kwargs,
) -> None:
    new_factors: dict[str, FakeLoraLayer] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [_uniform_scalar_quantize(model.get_delta(ad, layer), rate_bits) for ad in adapter_names]
        delta = float(weights[0]) * deltas[0]
        for w, d in zip(weights[1:], deltas[1:]):
            delta = delta + float(w) * d
        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
