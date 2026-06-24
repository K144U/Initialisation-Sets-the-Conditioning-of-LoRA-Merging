"""B4 HumanEval pass@1 sweep analysis.

Reads the 20 cells (4 models x 5 methods) from
results/phase3/eval_b4_humaneval/, builds the 4x5 pass@1 table, computes
per-model Spearman rank correlation between pass@1 and NLL excess (the
matrix's seed-1 worst-task excess), and writes a JSON summary.

Mirrors analyze_e3_gsm8k.py exactly. The added section is the
cross-metric comparison: load the existing e3_gsm8k_summary.json and
emit a side-by-side per-model "GSM8K rho vs HumanEval rho" table, which
is what §6.5 needs to declare whether the L3 NLL->accuracy inversion is
metric-specific (GSM8K only) or robust across metrics.

Decision rule (master plan §B4):
  Robust cross-metric inversion (both metrics rho < 0) on L3 -> the
    NLL->accuracy pathology is mechanism-driven, not GSM8K-specific.
    Strengthens the §6.6 sign-election story and the §6.7 R4
    recommendation.
  GSM8K-only inversion (HumanEval rho >= 0.4) on L3 -> the GSM8K result
    is metric-specific; the NLL->accuracy correspondence is broadly
    safe, with GSM8K-on-L3 the only known outlier.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
B4_DIR = PROJECT_ROOT / "results/phase3/eval_b4_humaneval"
MATRIX_DIR = PROJECT_ROOT / "results/phase3/eval_matrix_seeds"
E3_SUMMARY = PROJECT_ROOT / "results/phase3/e3_gsm8k_summary.json"
OUT_JSON = PROJECT_ROOT / "results/phase3/b4_humaneval_summary.json"

MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
METHOD_MAP = [
    ("ta",     "task_arithmetic"),
    ("ties",   "ties"),
    ("dare",   "dare"),
    ("knots",  "knots"),
    ("tvq_b2", "tvq_b2"),
]


def spearmanr(a: list[float], b: list[float]) -> float:
    def ranks(xs):
        idx = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[idx[j + 1]] == xs[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    return pearsonr(ranks(a), ranks(b))


def pearsonr(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def load_pass1(model: str, tag: str) -> float | None:
    p = B4_DIR / f"{model}__{tag}__humaneval.json"
    if not p.exists():
        return None
    return json.load(open(p))["metric_score"]


def load_nll_excess(model: str, matrix_method: str, seed: int = 1) -> float | None:
    p = MATRIX_DIR / f"{model}__{matrix_method}__seed{seed}.json"
    if not p.exists():
        return None
    return json.load(open(p)).get("worst_task_excess")


def main() -> int:
    pass1_table: dict[str, dict[str, float | None]] = {}
    nll_table: dict[str, dict[str, float | None]] = {}
    for model in MODELS:
        pass1_table[model] = {}
        nll_table[model] = {}
        for tag, mat_name in METHOD_MAP:
            pass1_table[model][tag] = load_pass1(model, tag)
            nll_table[model][tag] = load_nll_excess(model, mat_name)

    print("=" * 72)
    print(f"HumanEval pass@1 (n=164) — 4 models × 5 methods")
    print("=" * 72)
    tags = [t for t, _ in METHOD_MAP]
    print(f"{'model':<14}" + " ".join(f"{t:<8}" for t in tags))
    for model in MODELS:
        row = []
        for tag in tags:
            v = pass1_table[model][tag]
            row.append(f"{v:<8.3f}" if v is not None else f"{'--':<8}")
        print(f"{model:<14}" + " ".join(row))
    print()

    print("=" * 72)
    print("NLL worst-task excess (seed-1, from §6.1 matrix)")
    print("=" * 72)
    print(f"{'model':<14}" + " ".join(f"{t:<8}" for t in tags))
    for model in MODELS:
        row = []
        for tag in tags:
            v = nll_table[model][tag]
            row.append(f"{v:<8.3f}" if v is not None else f"{'--':<8}")
        print(f"{model:<14}" + " ".join(row))
    print()

    # Per-model Spearman: pass@1 ↔ -NLL excess
    print("=" * 72)
    print("Per-model Spearman: pass@1 ↔ -NLL excess (across 5 methods)")
    print("=" * 72)
    spearman_per_model = {}
    for model in MODELS:
        ps = [pass1_table[model][t] for t, _ in METHOD_MAP]
        nlls = [nll_table[model][t] for t, _ in METHOD_MAP]
        if any(p is None or n is None for p, n in zip(ps, nlls)):
            print(f"  {model}: incomplete data")
            spearman_per_model[model] = None
            continue
        neg_nlls = [-n for n in nlls]
        rho = spearmanr(ps, neg_nlls)
        verdict = ("STRONG ≥0.7" if rho >= 0.7
                   else "MODERATE 0.4-0.7" if rho >= 0.4
                   else "WEAK 0-0.4" if rho >= 0
                   else "INVERTED <0")
        print(f"  {model:<14} rho = {rho:+.3f}  ({verdict})")
        spearman_per_model[model] = rho

    # Best method per model
    print()
    print("=" * 72)
    print("Best method per model (by pass@1)")
    print("=" * 72)
    bests = {}
    for model in MODELS:
        valid = [(t, pass1_table[model][t]) for t, _ in METHOD_MAP
                 if pass1_table[model][t] is not None]
        if not valid:
            print(f"  {model}: no data")
            continue
        best_tag, best_p = max(valid, key=lambda x: x[1])
        bests[model] = {"tag": best_tag, "pass1": best_p}
        print(f"  {model:<14} best={best_tag:<8} pass@1={best_p:.3f}")

    # Cross-metric comparison: pull GSM8K rho per model from e3 summary
    print()
    print("=" * 72)
    print("Cross-metric comparison: GSM8K em rho vs HumanEval pass@1 rho")
    print("=" * 72)
    cross_metric = {}
    if E3_SUMMARY.exists():
        e3 = json.load(open(E3_SUMMARY))
        gsm8k_rhos = e3.get("spearman_per_model", {})
        print(f"  {'model':<14}{'GSM8K em rho':<16}{'HumanEval pass@1 rho':<22}{'verdict':<28}")
        for model in MODELS:
            r_gsm = gsm8k_rhos.get(model)
            r_he = spearman_per_model.get(model)
            if r_gsm is None or r_he is None:
                v = "incomplete"
            elif r_gsm < 0 and r_he < 0:
                v = "CROSS-METRIC INVERSION"
            elif r_gsm < 0 <= r_he or r_he < 0 <= r_gsm:
                v = "METRIC-SPECIFIC inversion"
            elif r_gsm >= 0.7 and r_he >= 0.7:
                v = "robust agreement"
            else:
                v = "mixed"
            cross_metric[model] = {"gsm8k_rho": r_gsm, "humaneval_rho": r_he,
                                   "verdict": v}
            gs = f"{r_gsm:+.3f}" if r_gsm is not None else "--"
            hs = f"{r_he:+.3f}" if r_he is not None else "--"
            print(f"  {model:<14}{gs:<16}{hs:<22}{v:<28}")
    else:
        print(f"  E3 GSM8K summary not found at {E3_SUMMARY} — run analyze_e3_gsm8k.py first.")

    # Overall verdict
    print()
    print("=" * 72)
    valid_rhos = [r for r in spearman_per_model.values() if r is not None]
    if valid_rhos:
        n_strong = sum(1 for r in valid_rhos if r >= 0.7)
        n_mod = sum(1 for r in valid_rhos if 0.4 <= r < 0.7)
        n_weak = sum(1 for r in valid_rhos if 0 <= r < 0.4)
        n_inv = sum(1 for r in valid_rhos if r < 0)
        print(f"VERDICT (HumanEval): {n_strong}/{len(valid_rhos)} STRONG, "
              f"{n_mod} MODERATE, {n_weak} WEAK, {n_inv} INVERTED")

        # The L3 cross-metric question
        l3_gsm = (json.load(open(E3_SUMMARY))["spearman_per_model"]
                  .get("llama31_8b") if E3_SUMMARY.exists() else None)
        l3_he = spearman_per_model.get("llama31_8b")
        if l3_gsm is not None and l3_he is not None:
            print()
            print("L3 cross-metric question:")
            if l3_gsm < 0 and l3_he < 0:
                print(f"  GSM8K rho = {l3_gsm:+.3f} INVERTED, "
                      f"HumanEval rho = {l3_he:+.3f} ALSO INVERTED")
                print("  -> NLL->accuracy pathology on L3 is MECHANISM-DRIVEN, not "
                      "metric-specific.")
                print("  -> §6.6 sign-election story strengthens; §6.7 R4 "
                      "recommendation lands cleanly.")
            elif l3_gsm < 0 and l3_he >= 0:
                print(f"  GSM8K rho = {l3_gsm:+.3f} INVERTED, "
                      f"HumanEval rho = {l3_he:+.3f} NOT inverted")
                print("  -> L3 NLL->accuracy inversion is GSM8K-SPECIFIC.")
                print("  -> §6.5 frames it as a single-metric outlier; §6.7 R4 "
                      "still useful but case for sign-election mechanism weakens.")
            else:
                print(f"  GSM8K rho = {l3_gsm:+.3f}, HumanEval rho = {l3_he:+.3f}")
                print("  -> §6.5 reports both rhos; no clean inversion story to make.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "pass1": pass1_table,
        "nll_excess": nll_table,
        "spearman_per_model": spearman_per_model,
        "best_per_model": bests,
        "cross_metric": cross_metric,
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
