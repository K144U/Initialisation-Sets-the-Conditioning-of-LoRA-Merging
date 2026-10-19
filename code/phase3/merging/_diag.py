"""Mechanism diagnostics shared by the 2026-10-19 merges.

For a pre-truncation merged update tau (out x in) at one layer:
  amp       = ||tau||_F / ||mean_t D_t||_F
  tail_frac = share of ||tau||^2 in the bottom quartile of the eigen-directions
              of Hbar on its range, Hbar = mean_t P_t, P_t the projector onto
              the row space of A_t (the definition of Table "mechanism").
  off_range_frac = share of ||tau||^2 outside the span of the A_t rows.
Sampled layers: 8 q_proj layers evenly spaced in depth, the drift analysis's
sampling, so values are comparable across experiments.
"""
from __future__ import annotations

import re

import torch


def _A(model, adapter, layer):
    if hasattr(model, "get_layer"):
        return model.get_layer(adapter, layer).A
    return model.adapters[adapter][layer].A


def diag_layers(names, n=8, module="q_proj"):
    cand = [nm for nm in names if module in nm] or list(names)

    def depth(nm):
        m = re.search(r"layers\.(\d+)\.", nm)
        return int(m.group(1)) if m else 0
    cand = sorted(cand, key=depth)
    if len(cand) <= n:
        return cand
    step = (len(cand) - 1) / (n - 1)
    return [cand[round(i * step)] for i in range(n)]


def mechanism_diag(model, adapter_names, layer, tau, deltas):
    tau = tau.double()
    mean_d = sum(d.double() for d in deltas) / len(deltas)
    out = {"amp": float(tau.norm() / mean_d.norm())}
    qs = []
    for a in adapter_names:
        q, _ = torch.linalg.qr(_A(model, a, layer).double().T)   # in x r
        qs.append(q)
    basis, _ = torch.linalg.qr(torch.cat(qs, dim=1))             # in x Tr
    hbar = sum(basis.T @ q @ q.T @ basis for q in qs) / len(qs)  # Tr x Tr
    ev, evec = torch.linalg.eigh(hbar)
    k = max(1, ev.numel() // 4)
    bottom = basis @ evec[:, :k]                                  # in x k
    total = float(torch.sum(tau * tau))
    out["tail_frac"] = float(torch.sum((tau @ bottom) ** 2)) / total if total > 0 else 0.0
    # share of tau outside the span of the A_t rows: invisible to any loss
    # built from the adapters' row spaces (Adam's per-coordinate step puts
    # mass there; an exact solve does not)
    in_range = float(torch.sum((tau @ basis) ** 2))
    out["off_range_frac"] = max(0.0, 1.0 - in_range / total) if total > 0 else 0.0
    out["kappa_hbar"] = float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf")
    return out
