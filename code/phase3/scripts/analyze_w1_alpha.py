#!/usr/bin/env python3
"""W1 verdict: does a tuned merge coefficient explain rd-encoder ridge?

Reads results/phase3/eval_w1_alpha/ plus the published baseline cells and
prints the three comparisons the review turns on. Decision rules are fixed
here, before the cells run, so the verdict is not chosen after seeing them.

  V1  best-alpha TA vs rd-ridge, per base.
      W1 UPHELD on a base if best-alpha TA reaches within 0.005 nats of
      rd-ridge's 3-seed mean there (0.005 ~ the largest multi-seed CI
      half-width in App. H). UPHELD on >=2 bases => the contribution needs
      restating.

  V2  norm-matched rd-ridge vs published rd-ridge.
      If renorm costs < 0.010 nats on >=3 bases, the win is about direction,
      not scale, and W1's mechanism reading is refuted empirically as well as
      algebraically.

  V3  rank-16 rd-ridge vs the rank-16 baselines.
      If rank-16 rd-ridge still has the lowest worst-task excess per base,
      the storage-parity objection (audit A3) is answered.

Usage:  python code/phase3/scripts/analyze_w1_alpha.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SEEDS = ["seed1", "seed2", "seed3"]
TIE = 0.005      # V1 threshold, nats
RENORM_TOL = 0.010   # V2 threshold, nats


def worst(p: Path) -> float | None:
    if not p.exists():
        return None
    try:
        return float(json.load(open(p))["worst_task_excess"])
    except Exception:
        return None


def mean_over_seeds(dirname: str, pattern: str, base: str) -> tuple[float | None, int]:
    vals = [worst(RES / dirname / pattern.format(base=base, seed=s)) for s in SEEDS]
    vals = [v for v in vals if v is not None]
    return (statistics.mean(vals) if vals else None), len(vals)


def main() -> int:
    # Published references.
    rd_pub, ties, ta_default = {}, {}, {}
    for b in BASES:
        rd_pub[b], _ = mean_over_seeds("eval_seed_rdridge_regmean",
                                       "{base}__rd_ridge__{seed}.json", b)
        if rd_pub[b] is None:   # Llama's live in eval_ridge_seed
            v = [worst(RES / "eval_ridge_seed" / f"{b}__ridge_l0p05__{s}.json")
                 for s in SEEDS]
            v = [x for x in v if x is not None]
            rd_pub[b] = statistics.mean(v) if v else None
        ties[b], _ = mean_over_seeds("eval_matrix_seeds", "{base}__ties__{seed}.json", b)
        ta_default[b], _ = mean_over_seeds("eval_matrix_seeds",
                                           "{base}__task_arithmetic__{seed}.json", b)

    # V1: TA alpha sweep (seed1).
    print("=" * 78)
    print("V1  TA merge-coefficient sweep (seed1), worst-task NLL excess")
    print("=" * 78)
    alphas = ["0p1", "0p15", "0p25", "0p35", "0p5", "0p75", "1"]
    labels = ["0.10", "0.15", "0.25*", "0.35", "0.50", "0.75", "1.00"]
    print(f"{'base':<13}" + "".join(f"{l:>9}" for l in labels)
          + f"{'bestTA':>9}{'rd-ridge':>10}{'verdict':>12}")
    upheld = 0
    for b in BASES:
        row, vals = f"{b:<13}", {}
        for a, lab in zip(alphas, labels):
            v = worst(RES / "eval_w1_alpha" / f"{b}__ta_alpha{a}__seed1.json")
            vals[lab] = v
            row += f"{v:>9.4f}" if v is not None else f"{'--':>9}"
        got = {k: v for k, v in vals.items() if v is not None}
        ref = f"{rd_pub[b]:>10.4f}" if rd_pub[b] is not None else f"{'--':>10}"
        if got and rd_pub[b] is not None:
            best = min(got.values())
            verdict = "UPHELD" if best <= rd_pub[b] + TIE else "answered"
            upheld += verdict == "UPHELD"
            row += f"{best:>9.4f}{ref}{verdict:>12}"
        else:
            row += f"{'--':>9}{ref}{'pending':>12}"
        print(row)
    print(f"\n  alpha=0.25 is the paper's default and must reproduce Table 1's TA row.")
    print(f"  V1: W1 upheld on {upheld}/4 bases "
          f"(>=2 => the contribution needs restating).")

    # V2: norm-matched.
    print()
    print("=" * 78)
    print("V2  norm-matched rd-ridge (renorm='ta') vs published rd-ridge, 3-seed")
    print("=" * 78)
    print(f"{'base':<13}{'published':>11}{'renormed':>11}{'delta':>9}{'n':>4}  verdict")
    ok = 0
    for b in BASES:
        rn, n = mean_over_seeds("eval_w1_alpha", "{base}__rd_renorm__{seed}.json", b)
        if rn is None or rd_pub[b] is None:
            pub = f"{rd_pub[b]:>11.4f}" if rd_pub[b] is not None else f"{'--':>11}"
            print(f"{b:<13}{pub}{'--':>11}{'--':>9}{n:>4}  (cells pending)")
            continue
        d = rn - rd_pub[b]
        v = "direction" if d < RENORM_TOL else "scale matters"
        ok += d < RENORM_TOL
        print(f"{b:<13}{rd_pub[b]:>11.4f}{rn:>11.4f}{d:>+9.4f}{n:>4}  {v}")
    print(f"\n  V2: win attributable to direction on {ok}/4 bases "
          f"(>=3 => W1's mechanism reading is refuted).")

    # V3: rank parity.
    print()
    print("=" * 78)
    print("V3  rank-16 rd-ridge vs rank-16 baselines, 3-seed")
    print("=" * 78)
    print(f"{'base':<13}{'rd rank16':>11}{'rd rank64':>11}{'TIES':>9}{'TA':>9}{'n':>4}  verdict")
    wins = 0
    for b in BASES:
        r16, n = mean_over_seeds("eval_w1_alpha", "{base}__rd_rank16__{seed}.json", b)
        if r16 is None:
            print(f"{b:<13}{'--':>11}")
            continue
        others = [x for x in (ties[b], ta_default[b]) if x is not None]
        v = "wins" if others and r16 < min(others) else "loses"
        wins += v == "wins"
        print(f"{b:<13}{r16:>11.4f}"
              f"{(rd_pub[b] if rd_pub[b] is not None else float('nan')):>11.4f}"
              f"{(ties[b] if ties[b] is not None else float('nan')):>9.4f}"
              f"{(ta_default[b] if ta_default[b] is not None else float('nan')):>9.4f}"
              f"{n:>4}  {v}")
    print(f"\n  V3: rank-16 rd-ridge best on {wins}/4 bases "
          f"(4/4 => storage-parity objection answered).")

    out = RES / "w1_alpha_summary.json"
    out.write_text(json.dumps({
        "rd_published_3seed": rd_pub, "ties_3seed": ties, "ta_default_3seed": ta_default,
        "thresholds": {"V1_tie_nats": TIE, "V2_renorm_tol_nats": RENORM_TOL},
    }, indent=2))
    print(f"\n[analyze] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
