#!/usr/bin/env python3
"""Professor B4: the heuristics' lean toward the shared arm, tested with the
cohort as the unit. Post hoc, not registered; the registered sign test
(prereg_heuristics_tost, 29 of 38, p = 0.0017) stays as reported.

The sign test treats the 40 method x benchmark x base differences as
independent, but within a base all ten share the same six cohorts and the same
evaluation sets. Here the arm labels are permuted over the six cohorts of each
base (20 relabelings per base), the same relabeling applied to every method
and benchmark of that base, enumerated exactly over the four bases (20^4).
Statistic: mean over all 40 cells of d = mean_indep - mean_shared.

  python heuristics_cohort_permutation.py results/phase3/heur_tost_summary.json [out.json]
"""
from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    eq = json.loads(Path(sys.argv[1]).read_text())["equivalence"]
    bases = sorted(next(iter(eq.values()))["per_base"])
    # acc[b]: (cells, 6) with columns seed1..3, indep1..3
    acc = {b: np.array([v["per_base"][b]["shared"] + v["per_base"][b]["indep"]
                        for v in eq.values()]) for b in bases}
    splits = [s for s in itertools.combinations(range(6), 3)]          # "shared" columns

    def base_d(b, s):
        i = [c for c in range(6) if c not in s]
        return float((acc[b][:, i].mean(1) - acc[b][:, list(s)].mean(1)).mean())

    per_base = {b: [base_d(b, s) for s in splits] for b in bases}
    obs_b = {b: base_d(b, (0, 1, 2)) for b in bases}
    obs = st.mean(obs_b.values())
    null = np.array([sum(x) / len(bases) for x in itertools.product(*per_base.values())])
    p = float((np.abs(null) >= abs(obs) - 1e-12).mean())
    p_one = float((null <= obs + 1e-12).mean())                        # lean toward shared
    out = {
        "note": "post hoc (professor B4); cohort-level exact permutation, 20^4 relabelings",
        "n_cells": int(sum(a.shape[0] for a in acc.values())),
        "mean_d": obs, "per_base_mean_d": obs_b,
        "p_two_sided": p, "p_one_sided_shared_better": p_one,
        "per_base_p_two_sided": {b: float(np.mean(np.abs(per_base[b]) >= abs(obs_b[b]) - 1e-12))
                                 for b in bases},
    }
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
