"""Wave 2 analyzer. Committed before any cell of either arm has landed.

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), sections R7 and R8.

R7, untruncated. The link between the kappa we measure and the degradation we
observe SURVIVES if, without rank truncation, the lambda = 0 shared-versus-
independent ratio is at least 2 on at least 3 of 4 bases AND the ridge gain
remains larger on the shared arm on at least 3 of 4. It FAILS if either falls
below on at least 2 of 4. On failure, every sentence tying kappa to degradation
is downgraded to a conjecture, in the abstract, section 1, 6.4 and 7.2.

R8, the gate. With indep2 and indep3 the independent arm has n = 3, so the
2 x SE gate the pre-registration specified can finally be applied to the ridge
gain difference. The shared arm is still one cohort (seed1), which is stated
rather than papered over: the SE is the independent arm's alone, so the gate is
one-sided in provenance even though it is applied symmetrically.

Usage:  python code/phase3/scripts/analyze_wave2.py [r7|r8]
"""
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path.home() / "projects" / "rdmerge"))
RES = ROOT / "results" / "phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
LAMBDAS = [(0.0, "0"), (0.01, "0p01"), (0.03, "0p03"), (0.05, "0p05"),
           (0.13, "0p13"), (0.30, "0p30"), (1.00, "1p00")]
TIE = 0.005
MIN_RATIO = 2.0


def val(subdir, prefix, base, tag, cohort):
    p = RES / subdir / f"{base}__{prefix}_l{tag}__{cohort}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("worst_task_excess")


def curve(subdir, prefix, base, cohort):
    vs = [val(subdir, prefix, base, tag, cohort) for _, tag in LAMBDAS]
    if any(v is None for v in vs):
        return None
    i = min(range(len(vs)), key=lambda k: vs[k])
    return {"vals": vs, "lambda_star": LAMBDAS[i][0], "floor": vs[0],
            "gain": vs[0] - vs[i]}


def report_r7():
    sub, pre = "eval_ridge_untrunc", "rdu"
    rows = {}
    for base in BASES:
        for cohort in ("seed1", "indep1"):
            c = curve(sub, pre, base, cohort)
            if c is None:
                print(f"INCOMPLETE: {base} {cohort} missing cells, no verdict")
                return 2
            rows[(base, cohort)] = c

    print("=" * 100)
    print("R7: ridge sweep WITHOUT post-merge rank truncation (realize=rank_deff)")
    print("=" * 100)
    hdr = "".join(f"{lam:>9}" for lam, _ in LAMBDAS)
    print(f"{'base':<13}{'cohort':<14}{hdr}{'lambda*':>9}{'gain':>9}")
    for base in BASES:
        for cohort in ("seed1", "indep1"):
            c = rows[(base, cohort)]
            print(f"{base:<13}{cohort:<14}"
                  + "".join(f"{v:>9.4f}" for v in c["vals"])
                  + f"{c['lambda_star']:>9g}{c['gain']:>9.4f}")
        print()

    n_ratio = n_gain = 0
    print(f"{'base':<13}{'ratio at l=0':>14}{'gain shared':>13}"
          f"{'gain indep':>12}   ratio>=2   gain larger")
    for base in BASES:
        s, i = rows[(base, "seed1")], rows[(base, "indep1")]
        ratio = s["floor"] / i["floor"] if i["floor"] > 0 else float("inf")
        ok_r = ratio >= MIN_RATIO
        ok_g = (s["gain"] - i["gain"]) > TIE
        n_ratio += ok_r
        n_gain += ok_g
        print(f"{base:<13}{ratio:>13.2f}x{s['gain']:>13.4f}{i['gain']:>12.4f}"
              f"{'   HOLDS' if ok_r else '   fails':>11}"
              f"{'   HOLDS' if ok_g else '   fails':>13}")

    survives = n_ratio >= 3 and n_gain >= 3
    fails = n_ratio <= 2 or n_gain <= 2
    print(f"\n  ratio holds {n_ratio}/4, gain holds {n_gain}/4")
    print("=" * 100)
    if survives:
        print("R7 VERDICT: SURVIVES")
        print("  Truncation was not doing the work. The link between the")
        print("  measured kappa and the observed degradation stands, and 6.4's")
        print("  'we have not shown that' can be replaced with the measurement.")
    elif fails:
        print("R7 VERDICT: FAILS")
        print("  The conditioning story does NOT survive without truncation.")
        print("  Per the pre-registration every sentence tying kappa to the")
        print("  degradation is downgraded to a conjecture, the abstract")
        print("  included, and the paper says it was tested and did not hold.")
    else:
        print("R7 VERDICT: UNRESOLVED under the rule as written")
        print("  Reported as unresolved, not rounded to the nearer branch.")
    return 0


def report_r8():
    sub, pre = "eval_ridge_cohorts", "rdc"
    shared = {}
    for base in BASES:
        c = curve("eval_ridge_cond", "rd", base, "seed1")
        if c is None:
            print(f"INCOMPLETE: shared arm {base} missing from eval_ridge_cond")
            return 2
        shared[base] = c

    print("=" * 100)
    print("R8: the ridge sweep gets its gate (independent arm n = 3)")
    print("=" * 100)
    print(f"{'base':<13}{'gain shared':>13}{'gain indep mean':>17}"
          f"{'sd':>9}{'2xSE':>9}{'diff':>10}   verdict")
    for base in BASES:
        gains = []
        for cohort in ("indep1", "indep2", "indep3"):
            src = ("eval_ridge_cond", "rd") if cohort == "indep1" else (sub, pre)
            c = curve(src[0], src[1], base, cohort)
            if c is None:
                print(f"INCOMPLETE: {base} {cohort} missing, no verdict")
                return 2
            gains.append(c["gain"])
        mi, sd = statistics.fmean(gains), statistics.stdev(gains)
        se2 = 2 * sd / len(gains) ** 0.5
        diff = shared[base]["gain"] - mi
        gate = max(TIE, se2)
        verdict = "larger on shared" if diff > gate else (
            "larger on independent" if diff < -gate else "tie")
        print(f"{base:<13}{shared[base]['gain']:>13.4f}{mi:>17.4f}{sd:>9.4f}"
              f"{se2:>9.4f}{diff:>+10.4f}   {verdict}")

    print("\n  The shared arm is one cohort (seed1), so this SE is the")
    print("  independent arm's alone. The gate is applied symmetrically but its")
    print("  provenance is not, and that is a limitation of the design, not of")
    print("  the analysis.")
    return 0


if __name__ == "__main__":
    arm = sys.argv[1] if len(sys.argv) > 1 else "r7"
    raise SystemExit(report_r7() if arm == "r7" else report_r8())
