#!/usr/bin/env python3
"""A1 merge matrix: does anything survive on an independently initialised cohort?

Implements the decision rules fixed in notes/prereg_a1_matrix_2026-08-03.md
BEFORE any cell in results/phase3/eval_a1_indep/ was read. Do not change a
threshold in this file. If a rule is badly specified, report the result under
the rule as written and record the problem as a limitation.

Source paths, resolved after the pre-registration was committed (8dad101) and
before any value was read. Path resolution is not a threshold choice:

  indep1 cohort   results/phase3/eval_a1_indep/{base}__{method}__indep1.json
  shared cohort   baselines  eval_matrix_seeds/{base}__{method}__seed1.json
                  rd_ridge   eval_ridge_seed/{base}__ridge_l{lam}__seed1.json
                  rd_rank16  eval_w1_alpha/{base}__rd_rank16__seed1.json

lambda* is 0.05 on Llama-3.1 and 0.13 on the other three (CLAUDE.md section 10).

Q1  does rd-ridge's advantage survive?     rd_ridge vs best of five baselines
Q2  do method rankings change by cohort?   top-1 and top-3 set, indep1 vs shared
Q3  is the salvage arc confounded by rank? rd_ridge (r=64) vs rd_rank16 (r=16)

One seed per indep1 cell. No seed statistics are possible. Llama's per-seed sd
on the shared cohort is 0.0064 (W1s), so any Llama gap below 0.013 (2 sd) is
flagged provisional regardless of what the rules return.

Usage:  python code/phase3/scripts/analyze_a1_matrix.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BASELINES = ["task_arithmetic", "ties", "dare", "knots", "tvq_b2"]
ALL7 = BASELINES + ["rd_ridge", "rd_rank16"]

TIE = 0.005          # pre-registered, same value as the W1s band and A2 K1/K2
LLAMA_PROVISIONAL = 0.013   # 2 x the measured per-seed sd on Llama
LAMBDA_TAG = {"llama31_8b": "0p05", "mistral_7b": "0p13",
              "qwen25_7b": "0p13", "yi15_9b": "0p13"}


def read_excess(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["worst_task_excess"])
    except Exception:
        return None


def indep(base: str, method: str) -> float | None:
    return read_excess(RES / "eval_a1_indep" / f"{base}__{method}__indep1.json")


def shared(base: str, method: str) -> float | None:
    if method == "rd_ridge":
        # Same fallback chain as w1_verdict_3seed.py lines 59-60. The first
        # attempt at this analyzer used eval_ridge_seed/ alone, which is
        # Llama-only, so rd_ridge was silently absent from Q2 on three bases.
        # Fixed 2026-08-03 AFTER the first run. This is a path correction that
        # ADDS an erroneously excluded method; no threshold was changed.
        tag = LAMBDA_TAG[base]
        return (read_excess(RES / "eval_seed_rdridge_regmean" / f"{base}__rd_ridge__seed1.json")
                or read_excess(RES / "eval_ridge_seed" / f"{base}__ridge_l{tag}__seed1.json"))
    if method == "rd_rank16":
        return read_excess(RES / "eval_w1_alpha" / f"{base}__rd_rank16__seed1.json")
    return read_excess(RES / "eval_matrix_seeds" / f"{base}__{method}__seed1.json")


def fmt(x: float | None) -> str:
    return "   n/a" if x is None else f"{x:6.4f}"


def main() -> int:
    I = {b: {m: indep(b, m) for m in ALL7} for b in BASES}
    S = {b: {m: shared(b, m) for m in ALL7} for b in BASES}

    missing = [f"{b}/{m}" for b in BASES for m in ALL7 if I[b][m] is None]
    if missing:
        print(f"[warn] missing indep1 cells: {', '.join(missing)}")

    print("=" * 96)
    print("A1 MERGE MATRIX, cohort indep1, worst-task NLL excess (1 seed per cell)")
    print("=" * 96)
    hdr = f"{'base':<13}" + "".join(f"{m[:11]:>12}" for m in ALL7)
    print(hdr)
    for b in BASES:
        print(f"{b:<13}" + "".join(f"{fmt(I[b][m]):>12}" for m in ALL7))

    # ---------------- Q1 ----------------
    print()
    print("=" * 96)
    print("Q1  does rd-ridge's advantage survive on a properly initialised cohort?")
    print("=" * 96)
    print(f"{'base':<13}{'rd_ridge':>10}{'best baseline':>16}{'name':>18}"
          f"{'gap':>10}  result")
    wins = ties = losses = 0
    provisional = []
    for b in BASES:
        rd = I[b]["rd_ridge"]
        cands = {m: I[b][m] for m in BASELINES if I[b][m] is not None}
        if rd is None or not cands:
            print(f"{b:<13}{'n/a':>10}")
            continue
        bm = min(cands, key=cands.get)
        bv = cands[bm]
        gap = bv - rd                       # positive means rd_ridge is better
        if gap > TIE:
            res, _ = "WINS", wins
            wins += 1
        elif gap < -TIE:
            res = "LOSES"
            losses += 1
        else:
            res = "TIES"
            ties += 1
        if b == "llama31_8b" and abs(gap) < LLAMA_PROVISIONAL:
            res += "  (provisional, |gap| < 2sd)"
            provisional.append(b)
        print(f"{b:<13}{fmt(rd):>10}{fmt(bv):>16}{bm:>18}{gap:+10.4f}  {res}")

    if losses >= 2:
        v1 = "DOES NOT SURVIVE"
    elif wins + ties >= 3 and wins >= 2:
        v1 = "SURVIVES"
    elif wins + ties >= 3:
        v1 = "WEAKENED"
    else:
        v1 = "DOES NOT SURVIVE"
    print(f"\n  wins {wins}, ties {ties}, losses {losses}  ->  Q1 VERDICT: {v1}")

    # ---------------- Q2 ----------------
    print()
    print("=" * 96)
    print("Q2  do method rankings change between the shared-init and indep1 cohorts?")
    print("=" * 96)
    top1_diff = top3_diff = 0
    for b in BASES:
        iv = {m: I[b][m] for m in ALL7 if I[b][m] is not None}
        sv = {m: S[b][m] for m in ALL7 if S[b][m] is not None}
        common = sorted(set(iv) & set(sv))
        if len(common) < 4:
            print(f"{b:<13}too few matched methods ({len(common)}), skipped")
            continue
        ri = sorted(common, key=lambda m: iv[m])
        rs = sorted(common, key=lambda m: sv[m])
        t1 = ri[0] != rs[0]
        t3 = set(ri[:3]) != set(rs[:3])
        top1_diff += int(t1)
        top3_diff += int(t3)
        print(f"{b:<13}n={len(common)}")
        print(f"{'':<13}  shared top3: {', '.join(rs[:3])}")
        print(f"{'':<13}  indep1 top3: {', '.join(ri[:3])}")
        print(f"{'':<13}  top-1 change: {'YES' if t1 else 'no':<4}"
              f"   top-3 set change: {'YES' if t3 else 'no'}")

    v2 = ("RANKINGS CHANGE MATERIALLY" if (top1_diff >= 2 or top3_diff >= 3)
          else "RANKINGS ARE STABLE")
    print(f"\n  top-1 changes {top1_diff}/4, top-3 set changes {top3_diff}/4"
          f"  ->  Q2 VERDICT: {v2}")
    print("  (rank correlation deliberately NOT computed: with 7 methods")
    print("   SD(rho) = 1/sqrt(6) = 0.41, the same n-too-small trap as W5)")

    # ---------------- Q3 ----------------
    print()
    print("=" * 96)
    print("Q3  is the salvage arc confounded by rank? rd_ridge r=64 vs rd_rank16 r=16")
    print("=" * 96)
    print(f"{'base':<13}{'rd_ridge':>10}{'rd_rank16':>12}{'diff':>10}  note")
    within = worse = 0
    for b in BASES:
        a, c = I[b]["rd_ridge"], I[b]["rd_rank16"]
        if a is None or c is None:
            print(f"{b:<13}{'n/a':>10}")
            continue
        d = c - a                    # positive means rank16 is worse
        if abs(d) <= TIE:
            note = "within threshold"
            within += 1
        elif d > TIE:
            note = "rank16 worse, rank is part of the effect"
            worse += 1
        else:
            note = "rank16 BETTER than rank64"
        print(f"{b:<13}{fmt(a):>10}{fmt(c):>12}{d:+10.4f}  {note}")

    if worse >= 2:
        v3 = "RANK IS PART OF THE EFFECT (audit A3 upheld)"
    elif within >= 3:
        v3 = "RANK IS IMMATERIAL"
    else:
        v3 = "INCONCLUSIVE under the pre-registered rule"
    print(f"\n  within {within}, rank16-worse {worse}  ->  Q3 VERDICT: {v3}")

    print()
    print("=" * 96)
    print(f"Q1 {v1}   |   Q2 {v2}   |   Q3 {v3}")
    print("One seed per indep1 cell. Every conclusion is provisional pending")
    print("replication. lambda* was tuned on the shared cohort, so rd_ridge is")
    print("arguably handicapped here; that argues for a follow-up sweep, not")
    print("for adjusting anything now.")
    print("=" * 96)

    out = RES / "a1_matrix_summary.json"
    out.write_text(json.dumps(
        {"indep1": I, "shared_seed1": S,
         "q1": {"wins": wins, "ties": ties, "losses": losses, "verdict": v1},
         "q2": {"top1_changes": top1_diff, "top3_changes": top3_diff,
                "verdict": v2},
         "q3": {"within": within, "rank16_worse": worse, "verdict": v3},
         "thresholds": {"tie": TIE, "llama_provisional": LLAMA_PROVISIONAL}},
        indent=2))
    print(f"[a1] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
