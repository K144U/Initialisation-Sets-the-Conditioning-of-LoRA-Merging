"""Smoke test: does the SOCP Chebyshev solver match day14b's closed-form
solver for T=2? If yes, the SOCP is validated and we can run T=3/T=4.
"""

import numpy as np
from day10_general_ht import compute_Hbar_tauH_general
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day14b_cheb_T2_closedform import cheb_center_T2_closed
from day15_cheb_general_T import cheb_center_general_T_socp

rng = np.random.default_rng(12345)

print("T=2 sanity: SOCP vs closed-form (lower is better):", flush=True)
for specs in [["iso", "iso"], ["iso", "geom"], ["geom", "twoscale"],
              ["lin", "twoscale"], ["geom", "iso"]]:
    D_list = build_D_list(specs, r=4)
    max_diff = 0.0
    max_dist = 0.0
    for _ in range(20):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            2, 128, 4, 1.0, D_list, "shared", rng)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        w_cf = cheb_center_T2_closed(taus, Hts, Q)
        w_socp = cheb_center_general_T_socp(taus, Hts, Q)
        diff = float(np.linalg.norm(w_cf - w_socp))
        norm_cf = float(np.linalg.norm(w_cf))
        max_diff = max(max_diff, diff)
        max_dist = max(max_dist, diff / (norm_cf + 1e-12))
    print(f"  {'+'.join(s[:4] for s in specs)}: "
          f"max ||dw||={max_diff:.2e}  max rel={max_dist:.2e}", flush=True)
print("PASS if max rel < 1e-4", flush=True)
