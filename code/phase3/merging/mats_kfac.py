"""MaTS (Tam et al., 2023) with the blockwise (KFAC) Fisher, solved by CG.

Following the official code (github.com/r-three/mats,
src/merging/conjugateGradient_fisherMerging.py): for each linear layer the
merged weight solves

    sum_t  G_in,t (W - W_t) G_out,t = 0

by conjugate gradient, with the matrix-vector product
Input_Gram @ W @ Output_Gram, `num_iterations` = 100 and initialisation from
the average, as in the repository's example commands. model_lambda = 1.0
(their default) leaves the input Gram's off-diagonal untouched.

Written here on the deltas and in (out x in) layout, the product is
X -> sum_t G_out,t X G_in,t and the right-hand side is sum_t G_out,t D_t G_in,t;
the base weight satisfies the equation identically, so merging deltas and
merging weights agree. The result is SVD-truncated to rank r for parity.
"""
from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer
from eval.activation_grams import gram_key

DIAGNOSTICS: dict = {}


def _cg(apply, b, x0, iters: int):
    x = x0.clone()
    r = b - apply(x)
    p = r.clone()
    rs = (r * r).sum()
    b_norm = b.norm()
    for _ in range(iters):
        Ap = apply(p)
        denom = (p * Ap).sum()
        if denom <= 0:
            break
        a = rs / denom
        x = x + a * p
        r = r - a * Ap
        rs_new = (r * r).sum()
        if rs_new.sqrt() <= 1e-10 * b_norm:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, float(r.norm() / b_norm)


def merge_mats(model, adapter_names, weights, merged_adapter_name, grams: dict,
               grams_out: dict, num_iterations: int = 100, model_lambda: float = 1.0,
               kfac: bool = True, _gram_tokens=None, **_kwargs) -> None:
    DIAGNOSTICS.clear()
    new_factors = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [model.get_delta(a, layer).float() for a in adapter_names]
        device = deltas[0].device
        gi, go = [], []
        for a in adapter_names:
            G = grams[a][gram_key(layer)].to(device)
            if model_lambda != 1.0:
                G = model_lambda * G + (1 - model_lambda) * torch.diag(torch.diagonal(G))
            gi.append(G)
            go.append(grams_out[a][layer].to(device))

        def apply(X):
            return sum(o @ X @ i for o, i in zip(go, gi))

        b = sum(o @ D @ i for o, D, i in zip(go, deltas, gi))
        x0 = sum(w * D for w, D in zip(weights, deltas)) / sum(weights)
        merged, rel_res = _cg(apply, b, x0, num_iterations)
        DIAGNOSTICS[layer] = {"cg_rel_residual": rel_res,
                              "amp_pre_trunc": float(merged.norm() / x0.norm())}
        A, B = svd_truncate_to_rank(merged.to(deltas[0].dtype), r)
        new_factors[layer] = FakeLoraLayer(A=A, B=B, scaling=1.0)
        del gi, go, deltas, b, merged
    model.add_adapter(merged_adapter_name, new_factors)
