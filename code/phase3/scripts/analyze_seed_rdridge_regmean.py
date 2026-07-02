#!/usr/bin/env python3
"""3-seed verdict: rd-encoder ridge (l=0.13) vs tuned RegMean (l=0.01) on
Qwen-2.5-7B and Yi-1.5-9B, on matched seed1/2/3 adapters.

For each base: per-seed worst-task excess for both methods, the per-seed gap
(RegMean - rd-ridge; positive => rd-ridge better), the 3-seed mean +/- std,
and a confident verdict (all 3 seeds must agree in sign of the gap).
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path("/home/sanjay.g/projects/rdmerge")
RES = ROOT / "results/phase3/eval_seed_rdridge_regmean"

BASES = ["qwen25_7b", "yi15_9b"]
SEEDS = ["seed1", "seed2", "seed3"]


def load(base: str, method: str, seed: str):
    p = RES / f"{base}__{method}__{seed}.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p)).get("worst_task_excess")
    except Exception:
        return None


def main() -> int:
    print(f"{'base':12s} {'method':9s}  seed1   seed2   seed3    mean +/- std")
    summary = {}
    for base in BASES:
        rows = {}
        for method in ("rd_ridge", "regmean"):
            vals = [load(base, method, s) for s in SEEDS]
            rows[method] = vals
            present = [v for v in vals if v is not None]
            if present:
                m = st.mean(present)
                sd = st.pstdev(present) if len(present) > 1 else 0.0
                cells = " ".join((f"{v:.4f}" if v is not None else " ....  ") for v in vals)
                print(f"{base:12s} {method:9s}  {cells}   {m:.4f} +/- {sd:.4f}")
            else:
                print(f"{base:12s} {method:9s}   (pending)")
        # per-seed gaps (RegMean - rd_ridge); +ve => rd-ridge better
        gaps = []
        for i in range(len(SEEDS)):
            r, g = rows["rd_ridge"][i], rows["regmean"][i]
            gaps.append((g - r) if (r is not None and g is not None) else None)
        pg = [x for x in gaps if x is not None]
        if len(pg) == len(SEEDS):
            mean_gap = st.mean(pg)
            signs = {"+" if x > 0 else "-" for x in pg}
            if signs == {"+"}:
                verdict = f"rd-ridge CONFIDENTLY wins (all 3 seeds, mean gap +{mean_gap:.4f})"
            elif signs == {"-"}:
                verdict = f"RegMean CONFIDENTLY wins (all 3 seeds, mean gap {mean_gap:.4f})"
            else:
                verdict = f"TIE / seed-inconsistent (gaps {[round(x,4) for x in pg]}, mean {mean_gap:+.4f})"
        else:
            verdict = "pending"
        print(f"{'':12s} -> per-seed gaps (RegMean - rd_ridge): "
              f"{[round(x,4) if x is not None else None for x in gaps]}  ::  {verdict}\n")
        summary[base] = {"rd_ridge": rows["rd_ridge"], "regmean": rows["regmean"],
                         "gaps_regmean_minus_rdridge": gaps, "verdict": verdict}
    out = ROOT / "results/phase3/seed_rdridge_regmean_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[analyze] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
