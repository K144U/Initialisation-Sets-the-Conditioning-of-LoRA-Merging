#!/usr/bin/env python3
"""Seed sweep of notes/check_solve_precision_result_2026-10-24.md: per base,
the float32 encoder at lambda 0 on the shared cohort under the default stream
(the published cell) and rng_seed 1 and 2, against the independent cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "results" / "phase3"
BASES = ("llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b")


def load(p):
    return json.loads(p.read_text())


def main() -> int:
    for b in BASES:
        pub = load(RES / "eval_ridge_cond" / f"{b}__rd_l0__seed1.json")
        ind = load(RES / "eval_ridge_cond" / f"{b}__rd_l0__indep1.json")
        draws = {"default": pub}
        for k in (1, 2):
            draws[f"s{k}"] = load(RES / "eval_precision_f64" / f"{b}__rd_l0_rerun_s{k}__seed1.json")
        for tag, d in draws.items():
            assert d["nll_tau"] == pub["nll_tau"], (b, tag)
            assert d["method_kwargs"].get("rng_seed") == (None if tag == "default" else int(tag[1:])), (b, tag)
        w = {t: d["worst_task_excess"] for t, d in draws.items()}
        a = {t: d["avg_task_excess"] for t, d in draws.items()}
        iw = ind["worst_task_excess"]
        lo, hi = min(w.values()), max(w.values())
        print(f"{b:<11} worst " + "  ".join(f"{t} {v:.4f}" for t, v in w.items())
              + f"  range {hi - lo:.4f}  indep {iw:.4f}  gap(min-indep) {lo - iw:.4f}"
              + f"  all_above {all(v > iw for v in w.values())}")
        print(f"{'':<11} avg   " + "  ".join(f"{t} {v:.4f}" for t, v in a.items()))
        for t, d in draws.items():
            print(f"{'':<11}   {t:<8}" + "  ".join(f"{k} {v:.4f}" for k, v in d["excess_per_task"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
