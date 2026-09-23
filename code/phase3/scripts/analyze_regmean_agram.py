#!/usr/bin/env python3
"""Judge the Table 4 re-run (A^T A Gram) by the rules the published table was
registered under, and report both Grams side by side.

Rules (prereg_tmlr_2026-08-14.md, as applied to Table 4): on >= 3 of 4 bases,
(a) the ridge gain L(1e-6) - L(lambda*) is larger on the shared arm, and
(b) the minimally regularised excess is worse on the shared arm by >= 2x.
Refuses to report until all 56 cells exist.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TAGS = ["1em6", "0p01", "0p03", "0p05", "0p13", "0p30", "1p00"]


def table(d: Path):
    v = {}
    for b in BASES:
        for c in ("seed1", "indep1"):
            for t in TAGS:
                p = d / f"{b}__rm_l{t}__{c}.json"
                if not p.exists():
                    return None
                v[b, c, t] = json.loads(p.read_text())["worst_task_excess"]
    out = {}
    for b in BASES:
        row = {}
        for c in ("seed1", "indep1"):
            star = min(TAGS, key=lambda t: v[b, c, t])
            row[c] = {"l_min": v[b, c, "1em6"], "lstar": star,
                      "gain": v[b, c, "1em6"] - v[b, c, star]}
        row["ratio"] = row["seed1"]["l_min"] / row["indep1"]["l_min"]
        row["a"] = row["seed1"]["gain"] > row["indep1"]["gain"]
        row["b"] = row["ratio"] >= 2
        out[b] = row
    return out


def main() -> int:
    new = table(RES / "eval_regmean_agram")
    if new is None:
        print("re-run incomplete; refusing to report")
        return 1
    old = table(RES / "eval_regmean_cond")
    rep = {"described_gram_AtA": new, "fallback_gram_DtD_published": old,
           "verdict": {"a": sum(r["a"] for r in new.values()) >= 3,
                       "b": sum(r["b"] for r in new.values()) >= 3}}
    (RES / "regmean_agram_summary.json").write_text(json.dumps(rep, indent=1))
    for b in BASES:
        n, o = new[b], old[b]
        print(f"{b:<12} A^TA: ratio {n['ratio']:7.1f} gain s/i {n['seed1']['gain']:.4f}/"
              f"{n['indep1']['gain']:.4f} | D^TD: ratio {o['ratio']:7.1f}")
    print(json.dumps(rep["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
