#!/usr/bin/env python3
"""A2 verdict: what KnOTS looks like once it is not secretly Task Arithmetic.

The published KnOTS row used inner_combination="linear", which is algebraically
TA (merging/tests/test_knots.py pins it to 3e-06). This compares the KnOTS-TIES
re-run against the published KnOTS row, TA, and plain TIES, all 3-seed means on
matched adapters.

Decision rule, fixed before the cells run:

  K1  |KnOTS-TIES - TA| > 0.005 nats on >= 3 bases
      KnOTS is a real method on our cohorts after all. Then 6.2 finding (3),
      the intro's "alignment has nothing to exploit", the related-work
      reconciliation, and App. J's "tracks TA exactly" must ALL be rewritten:
      the published agreement was an implementation artifact, not evidence
      about subspace overlap.

  K2  KnOTS-TIES still tracks TA within 0.005 on >= 3 bases
      The original conclusion survives, but the argument for it does not. The
      text must say the null was measured with a working alignment step,
      citing these cells, rather than resting on cells that could not have
      differed from TA.

Either way the four claims cannot stand as written, because the evidence they
cite was produced by a no-op.

Usage:  python code/phase3/scripts/analyze_a2_knots_ties.py
"""
from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
RES = ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SEEDS = ["seed1", "seed2", "seed3"]
TIE = 0.005


def mean3(dirname: str, pattern: str, base: str) -> tuple[float | None, int]:
    vals = []
    for s in SEEDS:
        p = RES / dirname / pattern.format(b=base, s=s)
        if p.exists():
            try:
                vals.append(float(json.loads(p.read_text())["worst_task_excess"]))
            except Exception:
                pass
    return (statistics.mean(vals) if vals else None), len(vals)


def main() -> int:
    print("=" * 84)
    print("A2  KnOTS with a working inner merge, worst-task NLL excess (3-seed)")
    print("=" * 84)
    print(f"{'base':<13}{'KnOTS(pub)':>12}{'TA':>10}{'TIES':>10}"
          f"{'KnOTS-TIES':>12}{'vs TA':>9}{'n':>4}  verdict")

    n_diff, n_have = 0, 0
    for base in BASES:
        pub, _ = mean3("eval_matrix_seeds", "{b}__knots__{s}.json", base)
        ta, _ = mean3("eval_matrix_seeds", "{b}__task_arithmetic__{s}.json", base)
        ties, _ = mean3("eval_matrix_seeds", "{b}__ties__{s}.json", base)
        kt, n = mean3("eval_a2_knots_ties", "{b}__knots_ties__{s}.json", base)

        row = f"{base:<13}"
        row += f"{pub:>12.4f}" if pub is not None else f"{'--':>12}"
        row += f"{ta:>10.4f}" if ta is not None else f"{'--':>10}"
        row += f"{ties:>10.4f}" if ties is not None else f"{'--':>10}"
        if kt is None or ta is None:
            row += f"{'--':>12}{'--':>9}{n:>4}  pending"
        else:
            n_have += 1
            d = kt - ta
            differs = abs(d) > TIE
            n_diff += differs
            row += f"{kt:>12.4f}{d:>+9.4f}{n:>4}  " \
                   f"{'differs from TA' if differs else 'still tracks TA'}"
        print(row)

    # The published KnOTS row should sit on top of TA to 3-4 decimals; that is
    # the artifact, and showing it makes the point without argument.
    print()
    for base in BASES:
        pub, _ = mean3("eval_matrix_seeds", "{b}__knots__{s}.json", base)
        ta, _ = mean3("eval_matrix_seeds", "{b}__task_arithmetic__{s}.json", base)
        if pub is not None and ta is not None:
            print(f"  published |KnOTS - TA| on {base:<13}{abs(pub - ta):.5f} nats"
                  f"   (implementation is algebraically identical)")

    if n_have:
        print(f"\n  K1/K2: KnOTS-TIES differs from TA on {n_diff}/{n_have} bases "
              f"at the {TIE} nat threshold.")
        print("  Either way, the four claims citing 'KnOTS ~ TA' as evidence for"
              "\n  the theory must be rewritten: their evidence was a no-op.")
    else:
        print("\n  pending: no eval_a2_knots_ties cells yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
