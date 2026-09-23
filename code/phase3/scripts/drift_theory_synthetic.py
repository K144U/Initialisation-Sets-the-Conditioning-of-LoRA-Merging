#!/usr/bin/env python3
"""Synthetic check of the first-order drift formulas (Appendix, drift theory).

Random A_0 (r x d), displacements E_t at relative size eps, isotropic or
low-rank anisotropic, random B_t; compares the first-order kappa and
amplification of drift_theory.py against the exact values.
"""
import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("dt", Path(__file__).with_name("drift_theory.py"))
dt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dt)

torch.manual_seed(0)
d, r, T, out = 512, 16, 4, 64
A0 = torch.randn(r, d) / d ** 0.5
for eps, aniso in [(0.13, False), (0.13, True), (0.05, True)]:
    Es = []
    for _ in range(T):
        L = (torch.randn(r, 4) @ torch.randn(4, d) / d ** 0.5 + 0.05 * torch.randn(r, d) / d ** 0.5
             if aniso else torch.randn(r, d) / d ** 0.5)
        Es.append(eps * L * A0.norm() / L.norm())
    As = [A0 + E for E in Es]
    Ds = [torch.randn(out, r) @ A for A in As]
    ex = dt.exact(As, Ds)
    pr = dt.predicted(As, Ds, dt.orth_rows(A0))
    pc = dt.predicted(As, Ds, dt.consensus_W0(As, r))
    print(f"eps={eps} aniso={aniso}: kappa {ex['kappa']:.3e} "
          f"ratio {pr['kappa_hat'] / ex['kappa']:.3f} (consensus {pc['kappa_hat'] / ex['kappa']:.3f}) | "
          f"amp {ex['amp']:.2f} ratio {pr['amp_hat'] / ex['amp']:.3f} "
          f"(consensus {pc['amp_hat'] / ex['amp']:.3f})")
