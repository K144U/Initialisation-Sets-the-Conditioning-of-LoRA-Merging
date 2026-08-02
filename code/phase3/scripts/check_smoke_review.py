#!/usr/bin/env python3
"""Smoke gate verdict. Bars are fixed here, before the cells run.

Exit 0 = all bars met, safe to dispatch the full campaign.
Exit 1 = at least one bar missed; do NOT dispatch, read the cell log in
         logs/orch/ first.

Usage:  python code/phase3/scripts/check_smoke_review.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"

# Published references this smoke run is checked against. Every smoke cell is
# seed1, so each reference must be a seed1 cell too. (First version of this
# file compared the seed1 b=4 cell against the 3-seed rd-ridge mean 0.0945 and
# would have reported a spurious FAIL: seed1 b=inf is 0.0822, and seed1 is the
# lowest of the three seeds.)
TA_SEED1 = 0.2132        # Table 1 TA, Llama, seed1
RD_INF_SEED1 = 0.0822    # eval_ridge_seed llama l0p05 seed1 (b -> inf)
RD_L0_B4 = 0.4051        # eval_e1 lambda=0 b=4, Llama
GSM8K_FAIL_PUB = 0.611   # published extraction-failure rate, Llama TA


def load(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    checks: list[tuple[str, bool | None, str]] = []

    # 1. CONTROL: TA at alpha = 0.25 must reproduce the published TA cell.
    d = load(RES / "eval_w1_alpha/llama31_8b__ta_alpha0p25__seed1.json")
    if d is None:
        checks.append(("ta_alpha0p25 control", None, "cell missing"))
    else:
        v = d["worst_task_excess"]
        ok = abs(v - TA_SEED1) < 0.005
        checks.append(("ta_alpha0p25 control", ok,
                       f"excess {v:.4f} vs published {TA_SEED1:.4f} "
                       f"(bar |d| < 0.005)"))

    # 2. renorm="ta" runs and actually rescaled. This is a PATH check, not a
    # performance bar: the cell answers W1's V2 and either answer is a result.
    d = load(RES / "eval_w1_alpha/llama31_8b__rd_renorm__seed1.json")
    if d is None:
        checks.append(("rd_renorm runs", None, "cell missing"))
    else:
        v = d["worst_task_excess"]
        kw = d.get("method_kwargs", {})
        ok = kw.get("renorm") == "ta" and v == v and v < 5.0
        note = ("scale is load-bearing" if v > RD_INF_SEED1 + 0.01
                else "direction carries it")
        checks.append(("rd_renorm runs", ok,
                       f"excess {v:.4f} vs unrenormed {RD_INF_SEED1:.4f} "
                       f"-> {note} (bar: path runs)"))

    # 3. KnOTS-TIES must differ from TA. Published knots-linear sits 0.00014 away.
    a = load(RES / "eval_a2_knots_ties/llama31_8b__knots_ties__seed1.json")
    b = load(RES / "eval_matrix_seeds/llama31_8b__task_arithmetic__seed1.json")
    if a is None or b is None:
        checks.append(("knots_ties differs from TA", None, "cell missing"))
    else:
        dv = abs(a["worst_task_excess"] - b["worst_task_excess"])
        ok = dv > 0.005
        checks.append(("knots_ties differs from TA", ok,
                       f"|KnOTS-TIES - TA| = {dv:.5f} "
                       f"(bar > 0.005; knots-linear was 0.00014)"))

    # 4. finite-b ridge runs and lands between b=inf and the lambda=0 b=4 cell.
    d = load(RES / "eval_w3_rate/llama31_8b__rd_ridge_b4__seed1.json")
    if d is None:
        checks.append(("rd_ridge_b4 runs", None, "cell missing"))
    else:
        v = d["worst_task_excess"]
        # Quantizing can only add distortion, so b=4 must sit at or above the
        # seed1 b=inf cell, and far below the lambda=0 b=4 collapse.
        ok = RD_INF_SEED1 - 0.002 <= v <= RD_L0_B4 * 1.5
        q = v - RD_INF_SEED1
        checks.append(("rd_ridge_b4 runs", ok,
                       f"excess {v:.4f}, quantization cost q(4) = {q:+.4f} "
                       f"(bar: >= seed1 b=inf {RD_INF_SEED1:.4f}, "
                       f"<= 1.5x lambda=0 b=4 {RD_L0_B4:.4f})"))

    # 5. fixed GSM8K scorer: extraction failures must fall sharply.
    d = load(RES / "eval_downstream_v2/llama31_8b__ta__gsm8k_em__seed1.json")
    if d is None:
        checks.append(("gsm8k extractor fixed", None, "cell missing"))
    else:
        pe = d.get("per_example", [])
        fail = sum(1 for e in pe if e.get("pred") is None) / max(1, len(pe))
        full = sum(1 for e in pe if len(e.get("gen_text") or "") > 210)
        ok = fail < 0.30 and full > 0
        checks.append(("gsm8k extractor fixed", ok,
                       f"failure {fail:.3f} vs published {GSM8K_FAIL_PUB:.3f} "
                       f"(bar < 0.30); {full}/{len(pe)} full generations stored "
                       f"(bar > 0)"))

    print("=" * 78)
    print("SMOKE GATE")
    print("=" * 78)
    n_pass = n_fail = n_missing = 0
    for name, ok, detail in checks:
        tag = "PASS" if ok else ("FAIL" if ok is False else "....")
        n_pass += ok is True
        n_fail += ok is False
        n_missing += ok is None
        print(f"  {tag}  {name:<28} {detail}")
    print()
    if n_missing:
        print(f"{n_missing} cell(s) not finished yet; re-run this check later.")
        return 1
    if n_fail:
        print(f"{n_fail} bar(s) MISSED. Do NOT dispatch the full campaign; "
              f"check logs/orch/ for the failing cell.")
        return 1
    print(f"All {n_pass} bars met. Safe to dispatch W5 + W1 + A1-train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
