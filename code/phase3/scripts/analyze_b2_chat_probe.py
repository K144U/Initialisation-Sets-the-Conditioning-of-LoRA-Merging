"""B2 L3 chat-template special-token probe analysis.

Reads the 5 cells (Llama-3.1-8B × {ta, ties, dare, knots, tvq_b2}) from
results/phase3/eval_b2_l3_chat/. For each method, computes:
  - GSM8K em accuracy (score)
  - mean / max / fraction-nonzero of n_special_emitted per generation
  - correlation between special-token emission and per-example score
  - per-method special-token rate per token (n_special_emitted / n_tokens)

H1 hypothesis (§6.5): the L3 GSM8K NLL->accuracy inversion (rho=-0.60)
is explained by chat-template tokenization — the merged adapter
spuriously emits <|begin_of_text|>, <|eot_id|>, etc. which the NLL
evaluator sees as valid tokens but the generation evaluator stops on,
clobbering the score.

Falsification rules:
  H1 SUPPORTED: TIES emits significantly more special tokens than TA
    AND high special-token rate correlates with low score.
  H1 PARTIAL:   TIES emits more special tokens than TA but the
    correlation with score is weak (other causes also at play).
  H1 FALSIFIED: Methods emit special tokens at similar rates;
    NLL-best method does not emit more than NLL-worst.

Note (post-B4): the L3 inversion is GSM8K-specific (HumanEval pass@1
shows STRONG +0.894 correspondence). So even with H1 supported, the
mechanism is a generation-eval-specific (max-tokens cutoff vs early-
EOS) artifact, not a deep merge pathology.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
B2_DIR = PROJECT_ROOT / "results/phase3/eval_b2_l3_chat"
OUT_JSON = PROJECT_ROOT / "results/phase3/b2_chat_probe_summary.json"

METHODS = ["ta", "ties", "dare", "knots", "tvq_b2"]


def pearsonr(a: list[float], b: list[float]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)))
    db = math.sqrt(sum((b[i] - mb) ** 2 for i in range(n)))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def load_cell(method: str) -> dict | None:
    p = B2_DIR / f"llama31_8b__{method}__special_probe.json"
    if not p.exists():
        return None
    return json.load(open(p))


def main() -> int:
    cells = {m: load_cell(m) for m in METHODS}
    if any(c is None for c in cells.values()):
        print("INCOMPLETE: missing cells")
        return 1

    per_method = {}
    for m, cell in cells.items():
        pe = cell["per_example"]
        scores = [ex["score"] for ex in pe]
        n_spec = [ex["n_special_emitted"] for ex in pe]
        n_tok = [ex["n_tokens"] for ex in pe]
        rates = [s / t if t > 0 else 0.0 for s, t in zip(n_spec, n_tok)]
        nonzero = [1 if s > 0 else 0 for s in n_spec]
        per_method[m] = {
            "accuracy": sum(scores) / len(scores),
            "n": len(scores),
            "mean_special": sum(n_spec) / len(n_spec),
            "max_special": max(n_spec),
            "frac_nonzero_special": sum(nonzero) / len(nonzero),
            "mean_rate_per_token": sum(rates) / len(rates),
            "spearman_special_vs_score": pearsonr(
                [float(s) for s in n_spec], [float(s) for s in scores]),
        }

    # Print the table
    print("=" * 88)
    print("B2 L3 special-token probe — Llama-3.1-8B × 5 methods (n=100 GSM8K each)")
    print("=" * 88)
    print(f"{'method':<10}{'acc':<8}{'mean_special':<14}{'max':<8}"
          f"{'frac>0':<10}{'rate/tok':<10}{'r(spec,score)':<14}")
    for m in METHODS:
        pm = per_method[m]
        print(f"{m:<10}{pm['accuracy']:<8.3f}{pm['mean_special']:<14.3f}"
              f"{pm['max_special']:<8d}{pm['frac_nonzero_special']:<10.3f}"
              f"{pm['mean_rate_per_token']:<10.5f}"
              f"{pm['spearman_special_vs_score']:<+14.3f}")

    # H1 verdict
    print()
    print("=" * 88)
    print("H1 verdict: does TIES emit more special tokens than TA?")
    print("=" * 88)
    ta_spec = per_method["ta"]["mean_special"]
    ties_spec = per_method["ties"]["mean_special"]
    ta_frac = per_method["ta"]["frac_nonzero_special"]
    ties_frac = per_method["ties"]["frac_nonzero_special"]
    print(f"  TA   mean special={ta_spec:.3f}  frac>0={ta_frac:.3f}")
    print(f"  TIES mean special={ties_spec:.3f}  frac>0={ties_frac:.3f}")
    if ties_spec > 1.5 * ta_spec and (ties_frac - ta_frac) > 0.10:
        print("  -> TIES emits SUBSTANTIALLY more special tokens than TA.")
        h1_dir = "supported"
    elif ties_spec > 1.1 * ta_spec:
        print("  -> TIES emits modestly more special tokens than TA.")
        h1_dir = "partial"
    else:
        print("  -> TIES emits a comparable or lower number of special tokens.")
        h1_dir = "falsified"

    # Within-method: does emitting special tokens hurt score?
    print()
    print("=" * 88)
    print("Does special-token emission correlate with low score?")
    print("=" * 88)
    neg_correlations = 0
    for m in METHODS:
        r = per_method[m]["spearman_special_vs_score"]
        flag = "(NEG: high spec -> low score)" if r < -0.1 else ""
        print(f"  {m:<10} r(special, score) = {r:+.3f}  {flag}")
        if r < -0.1:
            neg_correlations += 1

    h1_overall = (h1_dir, neg_correlations)
    print()
    print("=" * 88)
    if h1_dir == "supported" and neg_correlations >= 3:
        verdict = "H1 SUPPORTED: chat-template emission accounts for L3 GSM8K inversion."
    elif h1_dir in ("supported", "partial") and neg_correlations >= 2:
        verdict = "H1 PARTIALLY supported: chat-template is one contributing cause."
    else:
        verdict = ("H1 FALSIFIED: chat-template emission does NOT explain the L3 "
                   "GSM8K inversion.")
    print(f"VERDICT: {verdict}")
    print()
    print("Post-B4 context: HumanEval pass@1 rho on L3 = +0.894 (NLL predicts "
          "accuracy fine on the code metric). So even if H1 holds on GSM8K, the "
          "L3 NLL->accuracy pathology is GSM8K-evaluation-specific, not a deep "
          "merge defect.")

    json.dump({
        "per_method": per_method,
        "h1_direction": h1_dir,
        "n_negative_within_method": neg_correlations,
        "verdict": verdict,
    }, open(OUT_JSON, "w"), indent=2)
    print(f"\nWrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
