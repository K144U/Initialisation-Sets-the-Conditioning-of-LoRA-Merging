"""Table 3 with the inference path matched between the arms.

Found while inspecting the R1 smoke cell. In eval_a1_indep, the source of the
pre-registered replication table, `rd_ridge` is the ONLY method evaluated with
`loader: plain` (vanilla transformers + PEFT). Every baseline, and rd_rank16,
uses the unsloth fast path. The two paths are not numerically identical: on the
same adapters and the same evaluation draw they differ by up to 0.0105 nats in
nll_tau, which is twice the paper's 0.005 tie threshold and comparable to three
of the four margins the table reports.

`loader: plain` is not a free choice for rd_ridge: with realize=rank_deff the
merged adapter is rank 64 while the others are rank 16, and the unsloth forward
path does not support mixed-rank adapters. So the confound is structural, not a
slip, but it is still a confound.

rd_rank16 is the same encoder realised at rank 16, and it runs on unsloth like
every baseline. That comparison is matched on both counts, rank and loader, and
needs no new compute. This script reports both, side by side, with the
pre-registered gate applied to each.

Usage:  python code/phase3/scripts/analyze_q1_loader_matched.py [results/phase3]
"""
import json
import statistics
import sys
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / "results" / "phase3"
COHORTS = ["indep1", "indep2", "indep3"]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
BASELINES = ["task_arithmetic", "ties", "dare", "tvq_b2", "knots"]
TIE = 0.005


def vals(root, base, method):
    out = []
    for c in COHORTS:
        p = root / "eval_a1_indep" / f"{base}__{method}__{c}.json"
        if not p.exists():
            return None
        out.append(json.loads(p.read_text())["worst_task_excess"])
    return out


def verdict(d, gate):
    if abs(d) <= TIE or abs(d) <= gate:
        return "ties"
    return "wins" if d > 0 else "loses"


def run(root, encoder, label):
    print("=" * 100)
    print(f"{label}  ({encoder})")
    print("=" * 100)
    print(f"{'base':<13}{'encoder':>9}{'champion':>10}{'name':<12}{'d':>9}"
          f"{'2xSE':>9}   verdict")
    counts = {}
    for base in BASES:
        e = vals(root, base, encoder)
        if e is None:
            print(f"{base:<13} missing")
            continue
        champ, cv = None, None
        for m in BASELINES:
            v = vals(root, base, m)
            if v is None:
                continue
            if cv is None or statistics.fmean(v) < statistics.fmean(cv):
                champ, cv = m, v
        d_per = [c - x for c, x in zip(cv, e)]     # positive favours encoder
        d = statistics.fmean(d_per)
        gate = 2 * statistics.stdev(d_per) / len(d_per) ** 0.5
        v = verdict(d, gate)
        counts[v] = counts.get(v, 0) + 1
        print(f"{base:<13}{statistics.fmean(e):>9.4f}{statistics.fmean(cv):>10.4f}"
              f"  {champ:<10}{d:>+9.4f}{gate:>9.4f}   {v}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print()
    return counts


def main(root):
    a = run(root, "rd_ridge",
            "AS PUBLISHED: encoder on `plain`, baselines on unsloth, rank 64 vs 16")
    b = run(root, "rd_rank16",
            "MATCHED: encoder and baselines both on unsloth, both rank 16")
    print("=" * 100)
    if a == b:
        print("Same verdict counts either way. The confound does not change the")
        print("conclusion, and the table can be reported with the matched arm.")
    else:
        print("THE VERDICT COUNTS DIFFER. The published table mixes inference")
        print("paths, and the matched comparison is the one that should be")
        print("reported. Both belong in the paper, with the reason.")
        print(f"  as published: {a}")
        print(f"  matched     : {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT))
