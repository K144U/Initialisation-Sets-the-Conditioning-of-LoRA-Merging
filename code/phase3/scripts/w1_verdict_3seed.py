#!/usr/bin/env python3
"""W1 final verdict on matched 3-seed means: tuned TA vs rd-encoder ridge.

Combines the seed1 alpha* cell from eval_w1_alpha/ with the seed2/seed3 cells
from eval_w1_alpha3s/, giving a 3-seed tuned-TA mean directly comparable to the
paper's 3-seed rd-ridge numbers.

Decision rule, fixed before the cells ran. d = TA3(alpha*) - rd3 per base:
  d < -0.005  tuned TA genuinely beats the method on that base
  |d| <= 0.005  statistical tie
  d > +0.005  rd-ridge wins

Usage:  python code/phase3/scripts/w1_verdict_3seed.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"
SEEDS = ["seed1", "seed2", "seed3"]
ALPHA_STAR = {"llama31_8b": (0.75, "0p75"), "mistral_7b": (0.50, "0p5"),
              "qwen25_7b": (0.50, "0p5"), "yi15_9b": (0.50, "0p5")}
LAM = {"llama31_8b": "0p05", "mistral_7b": "0p13",
       "qwen25_7b": "0p13", "yi15_9b": "0p13"}
TIE = 0.005


def w(rel: str) -> float | None:
    p = RES / rel
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["worst_task_excess"])
    except Exception:
        return None


def main() -> int:
    print("=" * 96)
    print("W1 FINAL: coefficient-tuned Task Arithmetic vs rd-encoder ridge, matched 3-seed means")
    print("=" * 96)
    print(f"{'base':<13}{'a*':>6}{'TA(a*) 3-seed':>15}{'n':>3}"
          f"{'rd-ridge 3-seed':>17}{'TIES 3-seed':>13}{'d':>9}  verdict")

    n_ta_wins = n_tie = n_rd = n_pending = 0
    rows = {}
    for base, (a, tag) in ALPHA_STAR.items():
        ta_vals = [w(f"eval_w1_alpha/{base}__ta_alpha{tag}__seed1.json")]
        ta_vals += [w(f"eval_w1_alpha3s/{base}__ta_alphastar__{s}.json")
                    for s in ("seed2", "seed3")]
        ta_vals = [v for v in ta_vals if v is not None]

        rd_vals = []
        for s in SEEDS:
            v = (w(f"eval_seed_rdridge_regmean/{base}__rd_ridge__{s}.json")
                 or w(f"eval_ridge_seed/{base}__ridge_l{LAM[base]}__{s}.json"))
            if v is not None:
                rd_vals.append(v)
        ties = [w(f"eval_matrix_seeds/{base}__ties__{s}.json") for s in SEEDS]
        ties = [v for v in ties if v is not None]

        if len(ta_vals) < 3 or not rd_vals:
            n_pending += 1
            got = f"{statistics.mean(ta_vals):.4f}" if ta_vals else "--"
            print(f"{base:<13}{a:>6.2f}{got:>15}{len(ta_vals):>3}"
                  f"{(statistics.mean(rd_vals) if rd_vals else float('nan')):>17.4f}"
                  f"{(statistics.mean(ties) if ties else float('nan')):>13.4f}"
                  f"{'--':>9}  pending ({len(ta_vals)}/3 TA seeds)")
            continue

        ta3, rd3 = statistics.mean(ta_vals), statistics.mean(rd_vals)
        d = ta3 - rd3
        if d < -TIE:
            verdict, n_ta_wins = "TUNED TA WINS", n_ta_wins + 1
        elif abs(d) <= TIE:
            verdict, n_tie = "tie", n_tie + 1
        else:
            verdict, n_rd = "rd-ridge wins", n_rd + 1
        rows[base] = (ta3, rd3, d)
        print(f"{base:<13}{a:>6.2f}{ta3:>15.4f}{len(ta_vals):>3}{rd3:>17.4f}"
              f"{statistics.mean(ties):>13.4f}{d:>+9.4f}  {verdict}")
        sd = statistics.stdev(ta_vals) if len(ta_vals) > 1 else 0.0
        print(f"{'':<13}TA per-seed: "
              + "  ".join(f"{v:.4f}" for v in ta_vals) + f"   (sd {sd:.4f})")

    if n_pending:
        print(f"\n{n_pending} base(s) pending; rerun when the cells land.")
        return 1

    print(f"\n  tuned TA wins {n_ta_wins}, ties {n_tie}, rd-ridge wins {n_rd} of 4.")
    print("\n  Honest margin of rd-ridge below a TUNED TA (the paper reports")
    print("  56-91%, measured against TA pinned at 1/T = 0.25):")
    for base, (ta3, rd3, d) in rows.items():
        print(f"    {base:<13}{100 * (1 - rd3 / ta3):>6.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
