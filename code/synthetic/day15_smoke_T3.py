"""T=3 Chebyshev accuracy: does max_t D_t - min_t D_t = 0 on the
active set (i.e., is KKT satisfied)?"""

import numpy as np
from day10_general_ht import compute_Hbar_tauH_general
from day11_task_dep_Dt import sample_task_dep_ht_tuple, build_D_list
from day15_cheb_general_T import cheb_center_general_T_socp

rng = np.random.default_rng(31415)

for T, specs in [(3, ["iso", "iso", "iso"]),
                 (3, ["iso", "geom", "twoscale"]),
                 (3, ["geom", "twoscale", "lin"]),
                 (4, ["iso", "iso", "iso", "iso"]),
                 (4, ["iso", "geom", "lin", "twoscale"])]:
    D_list = build_D_list(specs, r=4)
    active_diffs = []
    full_diffs = []
    A_sizes = []
    for _ in range(30):
        taus, Us, Hts = sample_task_dep_ht_tuple(
            T, 128, 4, 1.0, D_list, "shared", rng)
        Hbar, tauH, Q, d_eff, w_pos = compute_Hbar_tauH_general(taus, Hts)
        w_cheb = cheb_center_general_T_socp(taus, Hts, Q)
        dists = np.array([float((w_cheb - taus[t]) @ (Hts[t] @ (w_cheb - taus[t])))
                          for t in range(T)])
        max_d = dists.max()
        active = [t for t in range(T) if dists[t] >= max_d - 1e-3 * max(max_d, 1.0)]
        A_sizes.append(len(active))
        if len(active) >= 2:
            active_diffs.append(float(dists[active].max() - dists[active].min()))
        full_diffs.append(float(dists.max() - dists.min()))
    active_diffs = np.array(active_diffs)
    print(f"  T={T} {'+'.join(s[:4] for s in specs)}: "
          f"|A| mean={np.mean(A_sizes):.2f}  "
          f"KKT |D_max - D_min on active|: "
          f"mean={active_diffs.mean():.2e} max={active_diffs.max():.2e}")
print("PASS if max < 1e-6")
