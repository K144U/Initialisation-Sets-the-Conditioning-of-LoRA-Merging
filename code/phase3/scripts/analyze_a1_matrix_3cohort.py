#!/usr/bin/env python3
"""A1 merge matrix across three independently initialised cohorts.

Implements notes/prereg_a1_matrix_amendment_2026-08-04.md, which amends
notes/prereg_a1_matrix_2026-08-03.md. Both were committed before any indep2 or
indep3 value was read (amendment at 42c7ff0). Do not change a threshold in this
file. If a rule is badly specified, report the result under the rule as written
and record the problem as a limitation.

The single-cohort analyzer analyze_a1_matrix.py is left untouched: it is the
record of what was run on indep1 alone, and overwriting it would erase that.

  cohorts       results/phase3/eval_a1_indep/{base}__{method}__indep{1,2,3}.json
  shared init   baselines  eval_matrix_seeds/{base}__{method}__seed1.json
                rd_ridge   eval_seed_rdridge_regmean/ then eval_ridge_seed/
                rd_rank16  eval_w1_alpha/{base}__rd_rank16__seed1.json

Q1' rd-ridge vs the champion baseline, on 3-cohort means, with a noise gate
Q2' shared-init vs independent regime, top-1 and top-3 set
Q3' rd_ridge (r=64) vs rd_rank16 (r=16), on 3-cohort means
Q4  NEW: is ranking instability specific to the regime, or just single-seed noise

The noise gate is one-directional by construction: it can only turn a WIN or a
LOSS into a TIE, never promote a TIE. With n = 3 the sd carries 2 degrees of
freedom, so it is a coarse screen and not an inference.

Usage:  python code/phase3/scripts/analyze_a1_matrix_3cohort.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BASELINES = ["task_arithmetic", "ties", "dare", "knots", "tvq_b2"]
ALL7 = BASELINES + ["rd_ridge", "rd_rank16"]
COHORTS = ["indep1", "indep2", "indep3"]

TIE = 0.005          # pre-registered in the parent, unchanged by the amendment
GATE_SD = 2.0        # a WIN/LOSS must also exceed 2 x SE, downgrade-only
LAMBDA_TAG = {"llama31_8b": "0p05", "mistral_7b": "0p13",
              "qwen25_7b": "0p13", "yi15_9b": "0p13"}


def read_excess(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text())["worst_task_excess"])
    except Exception:
        return None


def indep(base: str, method: str, cohort: str) -> float | None:
    return read_excess(RES / "eval_a1_indep" / f"{base}__{method}__{cohort}.json")


def shared(base: str, method: str) -> float | None:
    if method == "rd_ridge":
        tag = LAMBDA_TAG[base]
        return (read_excess(RES / "eval_seed_rdridge_regmean" / f"{base}__rd_ridge__seed1.json")
                or read_excess(RES / "eval_ridge_seed" / f"{base}__ridge_l{tag}__seed1.json"))
    if method == "rd_rank16":
        return read_excess(RES / "eval_w1_alpha" / f"{base}__rd_rank16__seed1.json")
    return read_excess(RES / "eval_matrix_seeds" / f"{base}__{method}__seed1.json")


def fmt(x: float | None, w: int = 6) -> str:
    return " " * (w - 3) + "n/a" if x is None else f"{x:{w}.4f}"


def mean_of(vals: list[float | None]) -> float | None:
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def gate(dvals: list[float]) -> tuple[float, float, float]:
    """Return (mean, sd, SE) of the per-cohort differences."""
    m = statistics.fmean(dvals)
    sd = statistics.stdev(dvals) if len(dvals) > 1 else 0.0
    return m, sd, sd / (len(dvals) ** 0.5)


def call(md: float, se: float) -> tuple[str, bool]:
    """Pre-registered call plus the one-directional noise gate."""
    if md > TIE:
        raw = "WINS"
    elif md < -TIE:
        raw = "LOSES"
    else:
        return "TIES", False
    if abs(md) > GATE_SD * se:
        return raw, False
    return "TIES", True          # downgraded, inside noise


def verdict_q1(wins: int, ties: int, losses: int) -> str:
    if losses >= 2:
        return "DOES NOT SURVIVE"
    if wins + ties >= 3 and wins >= 2:
        return "SURVIVES"
    if wins + ties >= 3:
        return "WEAKENED"
    return "DOES NOT SURVIVE"


def main() -> int:
    # V[base][method][cohort]
    V = {b: {m: {c: indep(b, m, c) for c in COHORTS} for m in ALL7} for b in BASES}
    S = {b: {m: shared(b, m) for m in ALL7} for b in BASES}
    M = {b: {m: mean_of([V[b][m][c] for c in COHORTS]) for m in ALL7} for b in BASES}

    missing = [f"{b}/{m}/{c}" for b in BASES for m in ALL7 for c in COHORTS
               if V[b][m][c] is None]
    if missing:
        print(f"[warn] missing cells ({len(missing)}): {', '.join(missing[:12])}")

    print("=" * 100)
    print("A1 MERGE MATRIX, three independently initialised cohorts")
    print("worst-task NLL excess (nats); cell = mean of indep1/2/3, sd in brackets")
    print("=" * 100)
    print(f"{'base':<13}" + "".join(f"{m[:14]:>17}" for m in ALL7))
    for b in BASES:
        row = f"{b:<13}"
        for m in ALL7:
            vals = [V[b][m][c] for c in COHORTS if V[b][m][c] is not None]
            if not vals:
                row += f"{'n/a':>17}"
            else:
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row += f"{statistics.fmean(vals):9.4f} [{sd:.4f}]"
        print(row)

    print()
    print("per-cohort detail")
    for b in BASES:
        print(f"  {b}")
        for c in COHORTS:
            print(f"    {c:<8}" + "".join(f"{fmt(V[b][m][c], 9):>10}" for m in ALL7))
        print(f"    {'methods:':<8}" + "".join(f"{m[:9]:>10}" for m in ALL7))

    # ---------------- Q1' ----------------
    print()
    print("=" * 100)
    print("Q1'  does rd-ridge's advantage survive across three cohorts?")
    print("     champion baseline selected on the 3-cohort mean (the less biased")
    print("     estimator, and the one more favourable to us: see robustness A)")
    print("=" * 100)
    print(f"{'base':<13}{'rd_ridge':>10}{'champion':>11}{'name':>17}"
          f"{'mean d':>10}{'sd(d)':>9}{'2xSE':>9}  result")
    wins = ties = losses = 0
    q1_rows = []
    for b in BASES:
        cands = {m: M[b][m] for m in BASELINES if M[b][m] is not None}
        if M[b]["rd_ridge"] is None or not cands:
            print(f"{b:<13}{'n/a':>10}")
            continue
        champ = min(cands, key=cands.get)
        dvals = [V[b][champ][c] - V[b]["rd_ridge"][c] for c in COHORTS
                 if V[b][champ][c] is not None and V[b]["rd_ridge"][c] is not None]
        md, sd, se = gate(dvals)
        res, downgraded = call(md, se)
        if res == "WINS":
            wins += 1
        elif res == "LOSES":
            losses += 1
        else:
            ties += 1
        tag = "  (downgraded, inside noise)" if downgraded else ""
        print(f"{b:<13}{fmt(M[b]['rd_ridge']):>10}{fmt(cands[champ]):>11}{champ:>17}"
              f"{md:+10.4f}{sd:9.4f}{GATE_SD * se:9.4f}  {res}{tag}")
        q1_rows.append({"base": b, "champion": champ, "mean_d": md, "sd": sd,
                        "two_se": GATE_SD * se, "per_cohort_d": dvals,
                        "result": res, "downgraded": downgraded})

    v1 = verdict_q1(wins, ties, losses)
    print(f"\n  wins {wins}, ties {ties}, losses {losses}  ->  Q1' PRIMARY VERDICT: {v1}")

    # robustness A: champion picked separately inside each cohort
    print()
    print("  robustness A: champion re-selected inside each cohort (takes a min over")
    print("  noise three times, so it is biased AGAINST rd_ridge; reported per the")
    print("  amendment because the primary rule is the one that favours us)")
    wa = ta = la = 0
    rowsA = []
    for b in BASES:
        dvals = []
        names = []
        for c in COHORTS:
            cands = {m: V[b][m][c] for m in BASELINES if V[b][m][c] is not None}
            rd = V[b]["rd_ridge"][c]
            if rd is None or not cands:
                continue
            bm = min(cands, key=cands.get)
            names.append(bm)
            dvals.append(cands[bm] - rd)
        if not dvals:
            continue
        md, sd, se = gate(dvals)
        res, dg = call(md, se)
        if res == "WINS":
            wa += 1
        elif res == "LOSES":
            la += 1
        else:
            ta += 1
        print(f"    {b:<13}mean d {md:+8.4f}  champions {'/'.join(names):<40} {res}"
              + ("  (downgraded)" if dg else ""))
        rowsA.append({"base": b, "mean_d": md, "champions": names, "result": res})
    v1a = verdict_q1(wa, ta, la)
    print(f"    -> {wa}W/{ta}T/{la}L  robustness A verdict: {v1a}"
          + ("   AGREES" if v1a == v1 else "   DISAGREES with primary"))

    # robustness B: parent single-cohort rule applied to each cohort separately
    print()
    print("  robustness B: the parent's original single-cohort rule, applied to each")
    print("  cohort independently (no gate, no averaging)")
    percohort = {}
    for c in COHORTS:
        w = t = l = 0
        detail = []
        for b in BASES:
            cands = {m: V[b][m][c] for m in BASELINES if V[b][m][c] is not None}
            rd = V[b]["rd_ridge"][c]
            if rd is None or not cands:
                continue
            bm = min(cands, key=cands.get)
            gp = cands[bm] - rd
            if gp > TIE:
                w += 1
                detail.append(f"{b}:W")
            elif gp < -TIE:
                l += 1
                detail.append(f"{b}:L")
            else:
                t += 1
                detail.append(f"{b}:T")
        vc = verdict_q1(w, t, l)
        percohort[c] = {"wins": w, "ties": t, "losses": l, "verdict": vc}
        print(f"    {c:<8}{w}W/{t}T/{l}L  {' '.join(detail):<40} {vc}")
    vs = [percohort[c]["verdict"] for c in COHORTS]
    maj = max(set(vs), key=vs.count)
    maj_n = vs.count(maj)
    unstable = (maj_n >= 2 and maj != v1)
    print(f"    -> majority of three: {maj} ({maj_n}/3)"
          + ("   DISAGREES with primary -> Q1' reported as UNSTABLE"
             if unstable else "   agrees with primary"))
    if unstable:
        v1 = f"UNSTABLE (primary {v1}, majority {maj})"

    # ---------------- Q2' ----------------
    print()
    print("=" * 100)
    print("Q2'  do rankings change between the shared-init and independent regimes?")
    print("=" * 100)
    top1_diff = top3_diff = 0
    q2_rows = []
    for b in BASES:
        iv = {m: M[b][m] for m in ALL7 if M[b][m] is not None}
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
        print(f"{'':<13}  shared seed1 top3: {', '.join(rs[:3])}")
        print(f"{'':<13}  indep mean   top3: {', '.join(ri[:3])}")
        print(f"{'':<13}  top-1 change: {'YES' if t1 else 'no':<4}"
              f"   top-3 set change: {'YES' if t3 else 'no'}")
        q2_rows.append({"base": b, "shared_top3": rs[:3], "indep_top3": ri[:3],
                        "top1_change": t1, "top3_change": t3})

    v2 = ("RANKINGS CHANGE MATERIALLY" if (top1_diff >= 2 or top3_diff >= 3)
          else "RANKINGS ARE STABLE")
    print(f"\n  top-1 changes {top1_diff}/4, top-3 set changes {top3_diff}/4"
          f"  ->  Q2' VERDICT: {v2}")
    print("  unbalanced design: 3 cohorts averaged on one side, 1 seed on the other.")
    print("  Q4 is what makes this interpretable. Rank correlation stays forbidden.")

    # ---------------- Q3' ----------------
    print()
    print("=" * 100)
    print("Q3'  is the salvage arc confounded by rank? rd_ridge r=64 vs rd_rank16 r=16")
    print("=" * 100)
    print(f"{'base':<13}{'rd_ridge':>10}{'rd_rank16':>12}{'mean d':>10}{'2xSE':>9}  note")
    within = worse = 0
    q3_rows = []
    for b in BASES:
        dvals = [V[b]["rd_rank16"][c] - V[b]["rd_ridge"][c] for c in COHORTS
                 if V[b]["rd_rank16"][c] is not None and V[b]["rd_ridge"][c] is not None]
        if not dvals:
            print(f"{b:<13}{'n/a':>10}")
            continue
        md, sd, se = gate(dvals)          # positive means rank16 is worse
        res, dg = call(md, se)
        if res == "WINS":                 # rank16 worse by more than threshold
            note = "rank16 worse, rank is part of the effect"
            worse += 1
        elif res == "LOSES":
            note = "rank16 BETTER than rank64"
        else:
            note = "within threshold" + (" (downgraded, inside noise)" if dg else "")
            within += 1
        print(f"{b:<13}{fmt(M[b]['rd_ridge']):>10}{fmt(M[b]['rd_rank16']):>12}"
              f"{md:+10.4f}{GATE_SD * se:9.4f}  {note}")
        q3_rows.append({"base": b, "mean_d": md, "two_se": GATE_SD * se, "note": note})

    if worse >= 2:
        v3 = "RANK IS PART OF THE EFFECT (audit A3 upheld)"
    elif within >= 3:
        v3 = "RANK IS IMMATERIAL"
    else:
        v3 = "INCONCLUSIVE under the pre-registered rule"
    print(f"\n  within {within}, rank16-worse {worse}  ->  Q3' VERDICT: {v3}")

    # ---------------- Q4 ----------------
    print()
    print("=" * 100)
    print("Q4   is ranking instability specific to the regime, or just single-seed noise?")
    print("     top-1 and top-3 across indep1/2/3, which differ ONLY in the init draw")
    print("=" * 100)
    k = 0
    q4_rows = []
    for b in BASES:
        tops = []
        top3s = []
        for c in COHORTS:
            vals = {m: V[b][m][c] for m in ALL7 if V[b][m][c] is not None}
            if len(vals) < 4:
                continue
            order = sorted(vals, key=lambda m: vals[m])
            tops.append(order[0])
            top3s.append(frozenset(order[:3]))
        if not tops:
            continue
        unanimous_t1 = len(set(tops)) == 1
        unanimous_t3 = len(set(top3s)) == 1
        if not unanimous_t1:
            k += 1
        print(f"{b:<13}top-1 by cohort: {', '.join(tops)}")
        print(f"{'':<13}  unanimous top-1: {'yes' if unanimous_t1 else 'NO':<4}"
              f"   unanimous top-3 set: {'yes' if unanimous_t3 else 'NO'}")
        q4_rows.append({"base": b, "top1_by_cohort": tops,
                        "unanimous_top1": unanimous_t1,
                        "unanimous_top3": unanimous_t3})

    if k >= 2:
        v4 = "UNSTABLE WITHIN THE REGIME"
        v4_consequence = (
            "Q2' is NOT attributable to initialisation. The claim that published\n"
            "  merging benchmarks may be confounded by shared-init geometry is\n"
            "  WITHDRAWN. What the campaign found is that single-seed method\n"
            "  rankings on this metric are noise: a real methodological point, but a\n"
            "  different and smaller claim, and it must be written as that one.")
    else:
        v4 = "STABLE WITHIN THE REGIME"
        v4_consequence = (
            "Q2' does bear on initialisation. The parent's consequence text applies,\n"
            "  INCLUDING its precondition: before any claim about the field, PEFT's\n"
            "  default init and at least two published merging benchmarks must be\n"
            "  checked for the shared-seed pattern. Our own repo showing the pattern\n"
            "  is not evidence about anyone else's.")
    print(f"\n  non-unanimous top-1 on k = {k}/4 bases  ->  Q4 VERDICT: {v4}")
    print(f"  {v4_consequence}")

    print()
    print("=" * 100)
    print(f"Q1' {v1}")
    print(f"Q2' {v2}")
    print(f"Q3' {v3}")
    print(f"Q4  {v4}")
    print("n = 3 cohorts, replication of the INITIALISATION DRAW ONLY. Not of the")
    print("eval shuffle (pinned 20260518), the bases, the tasks, or lambda*. The sd")
    print("behind every gate carries 2 degrees of freedom.")
    print("=" * 100)

    out = RES / "a1_matrix_3cohort_summary.json"
    out.write_text(json.dumps(
        {"per_cohort": V, "cohort_means": M, "shared_seed1": S,
         "q1": {"wins": wins, "ties": ties, "losses": losses, "verdict": v1,
                "rows": q1_rows,
                "robustness_a": {"verdict": v1a, "rows": rowsA},
                "robustness_b": percohort},
         "q2": {"top1_changes": top1_diff, "top3_changes": top3_diff,
                "verdict": v2, "rows": q2_rows},
         "q3": {"within": within, "rank16_worse": worse, "verdict": v3,
                "rows": q3_rows},
         "q4": {"k_nonunanimous_top1": k, "verdict": v4, "rows": q4_rows},
         "thresholds": {"tie": TIE, "gate_sd": GATE_SD},
         "prereg": ["notes/prereg_a1_matrix_2026-08-03.md",
                    "notes/prereg_a1_matrix_amendment_2026-08-04.md"]},
        indent=2))
    print(f"[a1] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
