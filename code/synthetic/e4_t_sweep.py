"""E4 — synthetic T sweep for the achievability constant (master plan Part II).

Question (paper Remark 5): is the linear-T factor in the achievability
constant C = T c^2 / 3 real, or an analysis artifact? The paper says this
measurement "would decide" it.

Design: run the day12 Monte Carlo over T in {2, 3, 4, 8, 16} x r in {4, 8},
homogeneous iso spectra, Stiefel-random overlap (the generic regime,
d_eff = Tr, floor ~ 0), d = 256 fixed so Tr <= d in every cell,
n_trials = 1000. For each cell record the achievability ratio
excess_max / LB(c_TQ=1) per bit-width b.

Decision rule (pre-registered):
  ratio ~flat in T  -> linear-T factor is an analysis artifact -> triggers
                       theory workstream T1 (prove a T-free / log-T bound).
  ratio ~linear in T -> the gap is real -> soften Remark 5, state as a
                       sharp open problem.
Verdict heuristic printed at the end: fit ratio ~ a*T + b per (r, bits) and
compare against a log-T fit by R^2; also report ratio(T=16)/ratio(T=2).

Run:  /cm/local/apps/python311/bin/python3 e4_t_sweep.py [--trials 1000]
Outputs: results/e4_t_sweep/e4_results.json, e4_table.txt (+ plot if
matplotlib is available).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import day12_achievability as day12
from day11_task_dep_Dt import build_D_list

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "results", "e4_t_sweep")

# T=3 excluded: its d_eff (12/24) is not a power of two, so Hadamard
# padding inflates the charged rate (R = b*n > b*d_eff) while the LB decays
# at 2^(-2R/d_eff) -> ratio grows ~2^(2b(n/d_eff-1)), an accounting artifact
# that poisons the T-trend (verified in the 3-trial dry run: T=3 ratios grow
# 4x per 2 bits; all pow2 cells flat in b). Pow2-clean cells still span 8x.
T_LIST = [2, 4, 8, 16]
R_LIST = [4, 8]
D_AMBIENT = 256
BITS = [2, 4, 6, 8, 10]
BASE_SEED = 20260611


def run_cell(T, r, n_trials):
    # Deterministic per-cell RNG (day12 uses a module-global).
    day12.RNG = np.random.default_rng(BASE_SEED + 1000 * T + r)
    specs = ["iso"] * T
    D_list = build_D_list(specs, r)
    rows = day12.run_achievability_cell(
        T, D_AMBIENT, r, D_list, "random", BITS, n_trials, B=1.0)
    for row in rows:
        lb = row["compress_thy_cTQ1"]
        row["ratio"] = row["excess_max"] / lb if lb > 1e-15 else float("nan")
        row["T"], row["r"] = T, r
    return rows


def fit_r2(x, y, transform):
    x_t = transform(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x_t, y, 1)
    pred = np.polyval(coeffs, x_t)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")
    return [float(c) for c in coeffs], r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows = []
    t0 = time.time()
    for r in R_LIST:
        for T in T_LIST:
            tc = time.time()
            rows = run_cell(T, r, args.trials)
            all_rows.extend(rows)
            ratios = {row["b"]: round(row["ratio"], 3) for row in rows}
            print(f"[e4] T={T:>2} r={r} d_eff={rows[0]['d_eff']:.1f} "
                  f"ratios(b)={ratios} ({time.time()-tc:.0f}s)", flush=True)

    # ---- per (r, b) trend analysis across T ----
    analysis = {}
    lines = ["=" * 90,
             f"E4 T-sweep: achievability ratio excess_max/LB(cTQ=1), "
             f"n_trials={args.trials}, d={D_AMBIENT}, seed={BASE_SEED}",
             "=" * 90]
    for r in R_LIST:
        for b in BITS:
            pts = [(row["T"], row["ratio"]) for row in all_rows
                   if row["r"] == r and row["b"] == b
                   and not math.isnan(row["ratio"])]
            if len(pts) < 3:
                continue
            Ts = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            lin_c, lin_r2 = fit_r2(Ts, ys, lambda x: x)
            log_c, log_r2 = fit_r2(Ts, ys, np.log2)
            growth = ys[Ts.index(16)] / ys[Ts.index(2)] if 2 in Ts and 16 in Ts else float("nan")
            analysis[f"r{r}_b{b}"] = {
                "T": Ts, "ratio": ys,
                "linear_fit": {"coeffs": lin_c, "r2": lin_r2},
                "logT_fit": {"coeffs": log_c, "r2": log_r2},
                "ratio_T16_over_T2": growth,
            }
            lines.append(
                f"r={r} b={b:>2}: ratios={['%.3f' % y for y in ys]} "
                f"| T16/T2={growth:5.2f} | lin R2={lin_r2:.3f} "
                f"slope={lin_c[0]:+.3f} | logT R2={log_r2:.3f}")

    # ---- verdict heuristic ----
    growths = [a["ratio_T16_over_T2"] for a in analysis.values()
               if not math.isnan(a["ratio_T16_over_T2"])]
    med_growth = float(np.median(growths)) if growths else float("nan")
    # Linear-T predicts T16/T2 = 8; flat predicts ~1.
    if med_growth < 2.0:
        verdict = ("FLAT(ish): median T16/T2 ratio growth = "
                   f"{med_growth:.2f} (linear-T predicts 8x). The linear-T "
                   "factor looks like an analysis artifact -> trigger T1.")
    elif med_growth > 5.0:
        verdict = (f"LINEAR(ish): median T16/T2 growth = {med_growth:.2f} "
                   "(~8x predicted). The T-factor looks real -> soften "
                   "Remark 5, state sharp open problem.")
    else:
        verdict = (f"INTERMEDIATE: median T16/T2 growth = {med_growth:.2f} "
                   "— between flat (1x) and linear (8x); check the logT fits "
                   "(sqrt-log-T from a max-of-T concentration term is the "
                   "natural candidate) before scheduling T1.")
    lines += ["-" * 90, "VERDICT: " + verdict]

    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(OUT_DIR, "e4_table.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(OUT_DIR, "e4_results.json"), "w") as f:
        json.dump({"params": {"T_list": T_LIST, "r_list": R_LIST,
                              "d": D_AMBIENT, "bits": BITS,
                              "n_trials": args.trials, "seed": BASE_SEED},
                   "rows": all_rows, "analysis": analysis,
                   "median_T16_over_T2": med_growth,
                   "verdict": verdict}, f, indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(R_LIST), figsize=(11, 4), sharey=True)
        for ax, r in zip(np.atleast_1d(axes), R_LIST):
            for b in BITS:
                key = f"r{r}_b{b}"
                if key not in analysis:
                    continue
                a = analysis[key]
                ax.plot(a["T"], a["ratio"], "o-", label=f"b={b}")
            ax.set_xscale("log", base=2)
            ax.set_xlabel("T (tasks)")
            ax.set_title(f"r={r}")
            ax.grid(alpha=0.3)
        np.atleast_1d(axes)[0].set_ylabel("excess_max / LB(c_TQ=1)")
        np.atleast_1d(axes)[0].legend(fontsize=8)
        fig.suptitle("E4: achievability ratio vs T "
                     "(flat = linear-T factor is analysis artifact)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "e4_ratio_vs_T.pdf"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(OUT_DIR, "e4_ratio_vs_T.png"), dpi=150,
                    bbox_inches="tight")
        print(f"[e4] plot saved to {OUT_DIR}")
    except ImportError:
        print("[e4] matplotlib unavailable — skipped plot (data in JSON)")

    print(f"[e4] done in {(time.time()-t0)/60:.1f} min -> {OUT_DIR}")


if __name__ == "__main__":
    main()
