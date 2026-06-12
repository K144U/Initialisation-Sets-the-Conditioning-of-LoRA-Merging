"""RD encoder — the paper's near-optimal merging encoder, on real adapters (E1).

Implements the achievability construction of Theorem 4 / Section 5.1 in the
projector-surrogate geometry (H_t = P_{V_t}, V_t = top-r right-singular
subspace of the task delta — the same operationalization as deff_analysis.py):

  per layer:
    1. V_t  <- top-r right-singular basis of Delta_t          (in_dim x r)
    2. Hbar = sum_t w_t V_t V_t^T = M_w M_w^T,  M_w = [sqrt(w_t) V_t]_t
       eigendecomposed via the (Tr x Tr) Gram of M_w -> Q (in x d_eff),
       Lambda (d_eff)
    3. centroid  tau_H = (sum_t w_t Delta_t) Q Lambda^{-1} Q^T
       (Delta_t V_t V_t^T = Delta_t exactly, since V_t IS Delta_t's right
       space — so the weighted numerator is just the weighted delta sum)
    4. eta = tau_H Q Lambda^{1/2} = N Q Lambda^{-1/2}          (out x d_eff)
    5. bits < 32: randomized-Hadamard rotate (seeded Rademacher signs +
       fast Walsh-Hadamard, the same incoherence transform as the synthetic
       validation in code/synthetic/day12_achievability.py), uniform scalar
       quantization in [-c*sigma, c*sigma] with sigma = rms of the rotated
       vector; invert.  bits >= 32: no quantization (the b=infinity cell —
       w* = the H-weighted centroid exactly).
    6. W* = eta_q Lambda^{-1/2} Q^T ;  SVD-truncate to rank r for the PEFT
       adapter slot (the SAME deployment constraint every other method in
       the registry carries). The per-layer truncated-mass fraction
       ||W* - trunc_r(W*)||_F^2 / ||W*||_F^2 is recorded so the cost of the
       rank cap is measurable (written to logs/rd_encoder_diag/ when
       RDMERGE_DIAG_DIR is set, else printed in aggregate).

Note vs. the E1 spec: the spec names a Gaussian-QR rotation; at real layer
sizes (out*d_eff ~ 2^18) a dense QR is infeasible, so we use the randomized
Hadamard transform — the standard structured substitute with the same
incoherence guarantee, and the one our own synthetic achievability
validation used. Recorded in decisions.md.

Rate accounting: R = bits * n per layer-module, n = next_pow2(out*d_eff).
"""

from __future__ import annotations

import hashlib
import json
import math
import os

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer, FakePeftModel


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _fwht(x: torch.Tensor) -> torch.Tensor:
    """In-place-style fast Walsh-Hadamard transform over the last dim
    (length must be a power of 2). Unnormalized: applying twice gives n*x."""
    n = x.shape[-1]
    h = 1
    x = x.clone()
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        x = x.view(-1, n)
        h *= 2
    return x.view(-1)


def _layer_seed(seed: int, layer: str) -> int:
    digest = hashlib.sha256(f"{seed}:{layer}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _right_basis(delta: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r right-singular basis of delta (in_dim x r), fp32."""
    d = delta.to(torch.float32)
    if min(d.shape) > 256 and r < min(d.shape) // 4:
        _, _, v = torch.svd_lowrank(d, q=min(r + 8, min(d.shape)), niter=4)
        return v[:, :r]
    _, _, vh = torch.linalg.svd(d, full_matrices=False)
    return vh[:r, :].T


def _quantize_rotated(eta: torch.Tensor, bits: float, c: float,
                      seed: int) -> torch.Tensor:
    """Randomized-Hadamard rotate, uniform-scalar quantize, invert."""
    out_dim, k = eta.shape
    flat = eta.reshape(-1).to(torch.float32)
    n = _next_pow2(flat.numel())
    x = torch.zeros(n, dtype=torch.float32, device=flat.device)
    x[: flat.numel()] = flat
    g = torch.Generator(device="cpu").manual_seed(seed)
    signs = (torch.randint(0, 2, (n,), generator=g, dtype=torch.int8)
             .to(flat.device).to(torch.float32) * 2 - 1)
    xr = _fwht(signs * x) / math.sqrt(n)

    sigma = xr.norm() / math.sqrt(n)
    if sigma < 1e-20:
        return eta
    levels = 2 ** int(round(bits))
    hi = c * sigma
    step = (2 * hi) / levels
    idx = torch.clamp(torch.floor((xr + hi) / step), 0, levels - 1)
    xq = -hi + (idx + 0.5) * step

    back = signs * _fwht(xq) / math.sqrt(n)
    return back[: flat.numel()].reshape(out_dim, k).to(eta.dtype)


def merge_rd_encoder(
    model: FakePeftModel,
    adapter_names: list[str],
    weights: list[float],
    merged_adapter_name: str,
    bits: float = 32,
    c: float = 5.0,
    seed: int = 20260611,
    eig_rel_floor: float = 1e-6,
    full_rank_patch: bool = False,
    realize: str = "rank_r",
    ridge_lambda: float = 0.0,
    **_kwargs,
) -> None:
    """The paper's encoder as a registry merge method. bits >= 32 = no
    quantization (the b=infinity centroid cell).

    realize:
      "rank_r"    (v1) SVD-truncate W* to the adapter rank r. Same
                  deployment constraint as every baseline method; discards
                  ~30% of solution mass on real adapters (trunc_mass diag).
      "rank_deff" (v3) inject a rank-d_eff adapter carrying W* EXACTLY via
                  its natural factorization B = eta_q (out x d_eff),
                  A = Lambda^{-1/2} Q^T (d_eff x in) -- no SVD, no
                  truncation, standard adapter forward path.
    ridge_lambda: Tikhonov regularization of the centroid, replacing
    Lambda^{-1} with (Lambda + lambda)^{-1} throughout (encode/decode/
    factors). lambda=0 is the raw theory centroid, which on real adapters
    blows up 25x to 94x (degenerate Hbar spectrum, see decisions.md
    2026-06-12); moderate lambda interpolates toward task arithmetic.
    full_rank_patch (v2) is RETIRED: mutating base weights does not
    propagate coherently through unsloth-patched forward paths (measured
    +10 nats, 2026-06-12); kept only for the fake-model unit test."""
    assert len(adapter_names) == len(weights)
    wsum = float(sum(weights))
    w = [float(x) / wsum for x in weights]

    new_factors: dict[str, FakeLoraLayer] = {}
    deff_factors: dict[str, FakeLoraLayer] = {}
    max_deff = 0
    diag: dict[str, dict] = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [model.get_delta(ad, layer).to(torch.float32)
                  for ad in adapter_names]

        bases = [_right_basis(d, r) for d in deltas]
        M_w = torch.cat([math.sqrt(wt) * V for wt, V in zip(w, bases)], dim=1)
        G = M_w.T @ M_w                                      # (Tr x Tr)
        S, U = torch.linalg.eigh(G)
        keep = S > eig_rel_floor * S.max()
        S, U = S[keep], U[:, keep]
        Q = M_w @ U @ torch.diag(S.rsqrt())                  # (in x d_eff)
        S_eff = S + float(ridge_lambda)

        N = deltas[0] * w[0]
        for wt, d in zip(w[1:], deltas[1:]):
            N = N + wt * d
        eta = (N @ Q) @ torch.diag(S_eff.rsqrt())            # (out x d_eff)

        if bits < 32:
            eta = _quantize_rotated(eta, bits, c, _layer_seed(seed, layer))

        W_star = (eta @ torch.diag(S_eff.rsqrt())) @ Q.T     # (out x in)

        A_new, B_new = svd_truncate_to_rank(W_star, r)
        truncated = B_new.to(torch.float32) @ A_new.to(torch.float32)
        total = float((W_star ** 2).sum())
        kept = float((truncated ** 2).sum())
        diag[layer] = {
            "d_eff": int(S.numel()),
            "Tr": r * len(adapter_names),
            "trunc_mass_frac": max(0.0, 1.0 - kept / total) if total > 0 else 0.0,
            "rate_bits": (float(bits) * _next_pow2(eta.numel())
                          if bits < 32 else None),
        }
        if full_rank_patch:
            residual = (W_star - truncated)
            base_w = model.base_weight(layer)
            base_w.data.add_(residual.to(dtype=base_w.dtype,
                                         device=base_w.device))
        if realize == "rank_deff":
            # exact factorization of W*: B64 @ A64 == eta_q Lambda^{-1/2} Q^T
            A64 = torch.diag(S_eff.rsqrt()) @ Q.T      # (d_eff x in)
            B64 = eta                                   # (out x d_eff)
            deff_factors[layer] = FakeLoraLayer(A=A64, B=B64, scaling=1.0)
            max_deff = max(max_deff, int(S.numel()))
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)

    if realize == "rank_deff":
        model.add_adapter_rank(merged_adapter_name, deff_factors,
                               rank=max_deff)
    else:
        model.add_adapter(merged_adapter_name, new_factors)

    fracs = [v["trunc_mass_frac"] for v in diag.values()]
    deffs = [v["d_eff"] for v in diag.values()]
    print(f"[rd_encoder] bits={bits} layers={len(diag)} "
          f"d_eff min/max={min(deffs)}/{max(deffs)} "
          f"trunc_mass mean={sum(fracs)/len(fracs):.4f} "
          f"max={max(fracs):.4f}", flush=True)
    diag_dir = os.environ.get("RDMERGE_DIAG_DIR")
    if diag_dir:
        os.makedirs(diag_dir, exist_ok=True)
        tag = f"rd_b{bits}_{merged_adapter_name}".replace("/", "_")
        with open(os.path.join(diag_dir, f"{tag}.json"), "w") as f:
            json.dump(diag, f, indent=1)
