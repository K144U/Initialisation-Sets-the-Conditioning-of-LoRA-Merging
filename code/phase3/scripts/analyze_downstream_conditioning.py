#!/usr/bin/env python3
"""Verdict: is the conditioning collapse visible in downstream accuracy?

Rules: notes/prereg_downstream_2026-08-15.md (df832fe), amended by
notes/prereg_downstream_amendment_2026-08-15.md (4d48987). Committed before
any cell of this test landed, which is the only reason its output is worth
anything.

The registered rules, implemented and nothing else:

  Threshold   a difference in accuracy counts only if it exceeds
              max(5 points, 2 x SE), with
              SE = sqrt(p1(1-p1)/n + p2(1-p2)/n) from the actual n
              (500 on GSM8K, 164 on HumanEval per the amendment).
              One-directional: it may only downgrade a call to a tie.

  P1  on the SHARED arm, accuracy at lambda* exceeds accuracy at lambda = 0
      by more than the threshold, on >= 3 of 4 bases, on >= 1 benchmark.

  P2  the accuracy gain from lambda = 0 to lambda* is LARGER on the shared arm
      than on the independent arm, on >= 3 of 4 bases, on >= 1 benchmark.
      P2 is the one that matters: P1 alone is consistent with the ridge helping
      every merge everywhere, while P2 is the claim that it helps because of
      conditioning.

  CONFIRMED both hold.  PARTIAL exactly one.  REFUTED neither.

Reported regardless of outcome, as registered: all 32 cells, every scorer's
discard rate with any non-zero rate flagged, absolute accuracies rather than
only differences, the lambda* used per (base, cohort), and for any null the
minimum detectable effect under the gate.

Refuses a verdict while any cell is missing.

Usage:
  python code/phase3/scripts/analyze_downstream_conditioning.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"
CFG = ROOT / "code/phase3/configs"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SHARED, INDEP = "seed1", "indep1"
BENCH_N = {"gsm8k": 500, "humaneval": 164}

FLOOR = 0.05      # 5 percentage points, registered, not derived from data
N_OF_4 = 3


def load(base: str, which: str, bench: str, cohort: str):
    """Read one downstream cell.

    The harness writes the accuracy as a scalar under `metric_score`, and the
    per-item detail under `per_example` with keys task_id, score, err,
    completion_preview, gen_raw.

    The discard rate is the number of items whose generation came back EMPTY.
    It is deliberately NOT the number with a non-empty `err`: on HumanEval,
    `err` holds the unit-test traceback for a wrong solution, so a correct
    scorer produces one for every failing item. Counting those as discards
    would report a clean cell as 48% discarded and flag it as void, which is
    the opposite of the check's purpose. The fault that voided the earlier
    downstream numbers was empty completions from markdown-fenced output, and
    empty completions are what this counts.
    """
    p = RES / "eval_downstream_cond" / f"{base}__{which}_{bench}__{cohort}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    acc = d.get("metric_score")
    if acc is None:
        for k in ("metric_value", "accuracy", "pass1", "em", "score"):
            if k in d:
                acc = d[k]
                break

    pe = d.get("per_example") or []
    n_empty = sum(1 for e in pe if isinstance(e, dict)
                  and not str(e.get("gen_raw", e.get("completion_preview", ""))).strip())
    n_err = sum(1 for e in pe if isinstance(e, dict) and str(e.get("err", "")).strip())
    return {"acc": acc, "raw": d,
            "n_scored": d.get("n_eval_metric", len(pe) or None),
            "n_empty": n_empty, "n_err": n_err}


def se_diff(p1: float, p2: float, n: int) -> float:
    return math.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)


def gate(diff: float, se: float) -> tuple[bool, float]:
    thr = max(FLOOR, 2.0 * se)
    return abs(diff) > thr, thr


def main() -> int:
    lstar = {}
    p = CFG / "downstream_cond_lstar.json"
    if p.exists():
        lstar = json.loads(p.read_text())

    cells, missing = {}, []
    for base in BASES:
        for cohort in (SHARED, INDEP):
            for which in ("l0", "lstar"):
                for bench in BENCH_N:
                    v = load(base, which, bench, cohort)
                    cells[(base, cohort, which, bench)] = v
                    if v is None or v["acc"] is None:
                        missing.append(f"{base} {cohort} {which} {bench}")

    total = len(cells)
    print(f"cells present: {total - len(missing)}/{total}")
    if missing:
        print(f"MISSING {len(missing)}, first ten:")
        for m in missing[:10]:
            print("   ", m)

    # Registered report: every cell, with its discard rate.
    print("\n=== all cells (registered report 1 and 2) ===")
    print(f"{'base':<12}{'cohort':<8}{'ridge':<7}{'bench':<11}"
          f"{'acc':>8}{'n':>6}{'empty':>7}{'failed':>8}")
    flagged = 0
    for base in BASES:
        for cohort in (SHARED, INDEP):
            for which in ("l0", "lstar"):
                for bench in BENCH_N:
                    v = cells[(base, cohort, which, bench)]
                    if v is None or v["acc"] is None:
                        continue
                    ne = v["n_empty"] or 0
                    flag = "  <-- NON-ZERO DISCARD" if ne else ""
                    if ne:
                        flagged += 1
                    print(f"{base:<12}{cohort:<8}{which:<7}{bench:<11}"
                          f"{v['acc']:>8.3f}{str(v['n_scored']):>6}{ne:>7}"
                          f"{v.get('n_err', 0):>8}{flag}")
    print("  'empty' is generations that came back blank, the fault that voided "
          "the earlier numbers.\n  'failed' is items whose generated code ran "
          "and failed its tests, which is a legitimate zero.")
    if flagged:
        print(f"\n  {flagged} cells have a non-zero discard rate. The earlier "
              f"downstream numbers were void for exactly this reason; these are "
              f"flagged rather than averaged in silently.")

    if missing:
        print("\nNO VERDICT: cells are missing. The registration requires the "
              "full design before any verdict is read.")
        return 1

    print("\n=== per base and benchmark ===")
    p1_hits = {b: 0 for b in BENCH_N}
    p2_hits = {b: 0 for b in BENCH_N}
    rows = {}

    for bench, n in BENCH_N.items():
        print(f"\n--- {bench} (n = {n}) ---")
        for base in BASES:
            s0 = cells[(base, SHARED, "l0", bench)]["acc"]
            ss = cells[(base, SHARED, "lstar", bench)]["acc"]
            i0 = cells[(base, INDEP, "l0", bench)]["acc"]
            is_ = cells[(base, INDEP, "lstar", bench)]["acc"]

            gain_s, gain_i = ss - s0, is_ - i0
            se_s = se_diff(s0, ss, n)
            ok_s, thr_s = gate(gain_s, se_s)
            p1 = gain_s > 0 and ok_s

            dd = gain_s - gain_i
            se_dd = math.sqrt(se_s ** 2 + se_diff(i0, is_, n) ** 2)
            ok_dd, thr_dd = gate(dd, se_dd)
            p2 = dd > 0 and ok_dd

            p1_hits[bench] += p1
            p2_hits[bench] += p2
            rows[(bench, base)] = {
                "shared_l0": s0, "shared_lstar": ss, "gain_shared": gain_s,
                "indep_l0": i0, "indep_lstar": is_, "gain_indep": gain_i,
                "P1": p1, "P1_gate": thr_s, "P2": p2, "P2_gate": thr_dd,
                "lstar_shared": lstar.get(f"{base}|{SHARED}"),
                "lstar_indep": lstar.get(f"{base}|{INDEP}"),
            }

            print(f"{base}")
            print(f"   shared  l0 {s0:.3f} -> l* {ss:.3f}   gain {gain_s:+.3f}")
            print(f"   indep   l0 {i0:.3f} -> l* {is_:.3f}   gain {gain_i:+.3f}")
            print(f"   P1 gate {thr_s:.3f} -> {'HOLDS' if p1 else 'no'}"
                  f"    P2 diff-in-diff {dd:+.3f} gate {thr_dd:.3f} -> "
                  f"{'HOLDS' if p2 else 'no'}")
            if not p1:
                print(f"   MDE for P1 on this cell: {thr_s:.3f}")

    p1_ok = any(v >= N_OF_4 for v in p1_hits.values())
    p2_ok = any(v >= N_OF_4 for v in p2_hits.values())
    verdict = ("CONFIRMED" if p1_ok and p2_ok else
               "REFUTED" if not p1_ok and not p2_ok else "PARTIAL")

    print(f"\n=== VERDICT: {verdict} ===")
    for bench in BENCH_N:
        print(f"  {bench:<11} P1 {p1_hits[bench]}/4   P2 {p2_hits[bench]}/4")
    print(f"  P1 holds on at least one benchmark at {N_OF_4}/4: {p1_ok}")
    print(f"  P2 holds on at least one benchmark at {N_OF_4}/4: {p2_ok}")

    out = RES / "downstream_conditioning_summary.json"
    out.write_text(json.dumps(
        {"verdict": verdict, "p1_hits": p1_hits, "p2_hits": p2_hits,
         "floor": FLOOR, "bench_n": BENCH_N,
         "per_cell": {f"{b}|{k}": v for (b, k), v in rows.items()}}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
