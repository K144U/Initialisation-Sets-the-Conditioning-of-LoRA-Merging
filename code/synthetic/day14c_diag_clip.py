"""Day 14c diagnostic — rerun with c=20 to check if slope gap in
non-iso cells is driven by clipping the perp coords at c*sigma_pc."""

import numpy as np
from day11_task_dep_Dt import build_D_list
from day14c_fractional_bits import run_cell, print_cell


def main():
    B = 1.0
    n_trials = 300
    R_list = [8, 12, 16, 20, 24, 32]

    print("=" * 100)
    print("Day 14c DIAGNOSTIC — c=20 (wider clip range)")
    print("=" * 100)

    cases = [
        (2, 128, 4, ["iso", "iso"]),
        (2, 128, 4, ["iso", "geom"]),
        (2, 128, 4, ["geom", "twoscale"]),
        (2, 128, 4, ["lin", "twoscale"]),
        (2, 128, 4, ["geom", "iso"]),
    ]
    for T, d, r, specs in cases:
        D_list = build_D_list(specs, r)
        res = run_cell(T, d, r, D_list, "shared", R_list, n_trials, B=B, c=20.0)
        print_cell(f"T={T} r={r} {'+'.join(s[:4] for s in specs)} shared c=20",
                   res, R_list, r)


if __name__ == "__main__":
    main()
