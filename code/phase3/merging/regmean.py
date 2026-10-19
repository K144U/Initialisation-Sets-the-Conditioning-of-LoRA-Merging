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


RULE_LAMBDAS: dict = {}
DIAGNOSTICS: dict = {}

# notes/prereg_lambda_selection_2026-10-19.md: grid and estimator fixed there.
SELECT_GRID = torch.logspace(-8, 2, 200, dtype=torch.float64)


def select_ridge(A_list, deltas, weights, mode):
    """Data-free ridge for adapter-only RegMean, per layer.

    Ridge regression with samples = the T r columns of X = [sqrt(w_t) A_t^T]
    and targets Y = [sqrt(w_t) D_t A_t^T]; with X = U S V^T and y_i = Y v_i the
    filter factors f_i = s_i^2 / (s_i^2 + lambda) give
      ||R||^2 = sum (1 - f_i)^2 ||y_i||^2,   tr H = sum f_i,
      ||W||^2 = sum ||y_i||^2 s_i^2 / (s_i^2 + lambda)^2.
    gcv:    argmin n ||R||^2 / (n - tr H)^2
    lcurve: max curvature of (log ||R||, log ||W||) in log lambda (Hansen),
            second-order finite differences, two grid points trimmed per end.
    """
    X = torch.cat([float(w) ** 0.5 * A.double().T for A, w in zip(A_list, weights)], dim=1)
    Y = torch.cat([float(w) ** 0.5 * (D.double() @ A.double().T)
                   for A, D, w in zip(A_list, deltas, weights)], dim=1)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    yn2 = torch.sum((Y @ Vh.T) ** 2, dim=0)                    # ||y_i||^2
    lam = SELECT_GRID.to(X.device)
    s2 = (S ** 2).unsqueeze(0)
    f = s2 / (s2 + lam.unsqueeze(1))                           # grid x n
    r2 = torch.sum((1 - f) ** 2 * yn2, dim=1)
    w2 = torch.sum(yn2 * s2 / (s2 + lam.unsqueeze(1)) ** 2, dim=1)
    n = X.shape[1]
    if mode == "gcv":
        crit = n * r2 / (n - f.sum(dim=1)) ** 2
        i = int(torch.argmin(crit))
    elif mode == "lcurve":
        t = torch.log(lam)
        rho, eta = 0.5 * torch.log(r2.clamp_min(1e-300)), 0.5 * torch.log(w2.clamp_min(1e-300))
        d1r, d1e = torch.gradient(rho, spacing=(t,))[0], torch.gradient(eta, spacing=(t,))[0]
        d2r, d2e = torch.gradient(d1r, spacing=(t,))[0], torch.gradient(d1e, spacing=(t,))[0]
        curv = 2 * (d1r * d2e - d2r * d1e) / (d1r ** 2 + d1e ** 2).clamp_min(1e-300) ** 1.5
        curv[:2] = -float("inf")
        curv[-2:] = -float("inf")
        i = int(torch.argmax(curv))
    else:
        raise ValueError(mode)
    return float(lam[i]), {"lambda": float(lam[i]), "grid_index": i,
                           "smin": float(S[-1]), "smax": float(S[0])}


def merge_regmean(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    ridge_lambda: float = 1e-3,
    alpha: float | None = None,
    ridge_mode: str = "fixed",
    kappa_target: float = 100.0,
    **_kwargs,
) -> None:
    """Adapter-only RegMean: Gram-weighted average of Δ_t at each layer.

    alpha, if given, applies RegMean's published regulariser to each per-task
    Gram, G -> alpha G + (1 - alpha) diag(G), before summing. None keeps the
    historical behaviour (no shrinkage), so committed configs reproduce.

    ridge_mode="kappa_target" replaces the fixed ridge by, per layer, the
    smallest isotropic ridge that caps the condition number of the summed
    Gram on its range at kappa_target (notes/prereg_lambda_rule_2026-09-23.md):
        lambda = max(0, (l_max - k * l_min_nonzero) / (k - 1)).
    The lambda chosen per layer is recorded in RULE_LAMBDAS.
    """
    RULE_LAMBDAS.clear()
    DIAGNOSTICS.clear()
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
            if alpha is not None:
                g = alpha * g + (1.0 - alpha) * torch.diag(torch.diagonal(g))
            denom = denom + w * g
            numer = numer + w * d.to(torch.float32) @ g
        if ridge_mode == "kappa_target":
            ev = torch.linalg.eigvalsh(denom)
            nz = ev[ev > 1e-10 * ev[-1]]
            k = float(kappa_target)
            lam = max(0.0, float((nz[-1] - k * nz[0]) / (k - 1)))
            RULE_LAMBDAS[layer] = lam
            # a floor keeps the full-dimensional solve nonsingular when the
            # rule returns 0, i.e. when the range is already well conditioned
            ridge = max(lam, 1e-6)
        elif ridge_mode in ("gcv", "lcurve"):
            A_list = [lo.A for lo in layer_objs]
            ridge, info = select_ridge(A_list, deltas, weights, ridge_mode)
            RULE_LAMBDAS[layer] = ridge
            DIAGNOSTICS[layer] = info
        elif ridge_mode == "fixed":
            ridge = ridge_lambda
        else:
            raise ValueError(f"unknown ridge_mode {ridge_mode!r}")
        denom = denom + ridge * torch.eye(
            in_dim, dtype=torch.float32, device=device)
        # Solve denom @ delta^T = numer^T  =>  delta = numer @ denom^{-1}
        delta = torch.linalg.solve(denom, numer.transpose(-1, -2)).transpose(-1, -2)
        delta = delta.to(dtype)

        A_new, B_new = svd_truncate_to_rank(delta, r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
