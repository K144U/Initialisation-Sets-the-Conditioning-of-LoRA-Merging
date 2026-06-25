"""Bootstrap 95% CIs on the downstream metric cells (E3 GSM8K, B4
HumanEval, B2 chat-template).

For each cell with per_example score arrays, resample with replacement
N=10000 times, compute the per-resample mean, and report 2.5th /
97.5th percentile of those means.

Output: results/phase3/bootstrap_cis_summary.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path("/home/sanjay.g/projects/rdmerge")
OUT = ROOT / "results/phase3/bootstrap_cis_summary.json"
N_BOOT = 10000
RNG_SEED = 20260625

DIRS = {
    "e3_gsm8k_em":     (ROOT / "results/phase3/eval_e3_gsm8k",
                        "{model}__{tag}__gsm8k_em.json"),
    "b4_humaneval":    (ROOT / "results/phase3/eval_b4_humaneval",
                        "{model}__{tag}__humaneval.json"),
    "b2_chat_probe":   (ROOT / "results/phase3/eval_b2_l3_chat",
                        "{model}__{tag}__special_probe.json"),
}
MODELS = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TAGS = ["ta", "ties", "dare", "knots", "tvq_b2"]


def boot_mean_ci(values: list[float], n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Return (point_mean, lo, hi) where (lo, hi) is the 95% percentile CI."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = sum(values) / n
    rng = random.Random(RNG_SEED)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return point, lo, hi


def main() -> int:
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for exp, (d, tmpl) in DIRS.items():
        out[exp] = {}
        models = ["llama31_8b"] if exp == "b2_chat_probe" else MODELS
        for model in models:
            out[exp][model] = {}
            for tag in TAGS:
                p = d / tmpl.format(model=model, tag=tag)
                if not p.exists():
                    out[exp][model][tag] = {"point": None, "lo": None, "hi": None,
                                              "ci_half_width": None, "n": 0}
                    continue
                obj = json.load(open(p))
                pe = obj.get("per_example", [])
                if exp == "b2_chat_probe":
                    vals = [float(ex["score"]) for ex in pe]
                else:
                    vals = [float(ex.get("score", 0)) for ex in pe]
                pt, lo, hi = boot_mean_ci(vals)
                out[exp][model][tag] = {
                    "point": pt, "lo": lo, "hi": hi,
                    "ci_half_width": (hi - lo) / 2.0,
                    "n": len(vals),
                }
                print(f"  {exp:<16}  {model:<12}  {tag:<8}  "
                      f"n={len(vals):<4} point={pt:.4f} CI=[{lo:.4f}, {hi:.4f}]  "
                      f"half_width=±{(hi-lo)/2:.4f}")

    # Compute the largest CI half-width per experiment as a summary stat.
    for exp in out:
        widths = []
        for model in out[exp]:
            for tag in out[exp][model]:
                hw = out[exp][model][tag].get("ci_half_width")
                if hw is not None and not (hw != hw):  # not nan
                    widths.append(hw)
        if widths:
            print(f"\n  {exp} max CI half-width: ±{max(widths):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
