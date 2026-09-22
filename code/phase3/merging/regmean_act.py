"""RegMean with real input-activation Grams (Jin et al., 2023).

Per LoRA-wrapped layer, with Delta_t = scaling_t B_t A_t (out x in):

    G~_t     = alpha G_t + (1 - alpha) diag(G_t)       RegMean's own shrinkage
    Delta*   = (sum_t w_t Delta_t G~_t) (sum_t w_t G~_t)^{-1}

This is RegMean's objective, min_W sum_t ||X_t W^T - X_t W_t^T||^2, restricted
to the update: the base weight W_0 satisfies it identically, so merging the
deltas and merging the full weights give the same answer. alpha = 0.9 is the
published default and no extra ridge is added unless ridge_lambda is set.

The Grams come from eval/activation_grams.py and are passed in as kwargs.
The result is SVD-truncated to rank r for parity with every other method in
this pipeline; the pre-truncation norm and the system's condition number are
recorded per layer so the solve can be inspected without re-running it.
"""
from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer
from eval.activation_grams import gram_key

DIAGNOSTICS: dict = {}


def merge_regmean_act(model, adapter_names, weights, merged_adapter_name,
                      grams: dict, alpha: float = 0.9, ridge_lambda: float = 0.0,
                      _gram_tokens=None, **_kwargs) -> None:
    DIAGNOSTICS.clear()
    new_factors = {}
    for layer in model.layer_names():
        _, in_dim, r = model.layer_topology[layer]
        key = gram_key(layer)
        deltas = [model.get_delta(a, layer).float() for a in adapter_names]
        device = deltas[0].device
        sumG = torch.zeros(in_dim, in_dim, dtype=torch.float32, device=device)
        sumGD = torch.zeros_like(deltas[0])
        for a, w, D in zip(adapter_names, weights, deltas):
            G = grams[a][key].to(device=device, dtype=torch.float32)
            G = alpha * G + (1.0 - alpha) * torch.diag(torch.diagonal(G))
            sumG += w * G
            sumGD += w * (D @ G)
            del G
        if ridge_lambda:
            sumG += ridge_lambda * torch.eye(in_dim, device=device)
        merged = torch.linalg.solve(sumG, sumGD.T).T
        mean_delta = sum(w * D for w, D in zip(weights, deltas)) / sum(weights)
        ev = torch.linalg.eigvalsh(sumG)
        DIAGNOSTICS[layer] = {
            "cond": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
            "amp_pre_trunc": float(merged.norm() / mean_delta.norm()),
        }
        A_new, B_new = svd_truncate_to_rank(merged.to(deltas[0].dtype), r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
        del sumG, sumGD, merged, deltas
    model.add_adapter(merged_adapter_name, new_factors)
