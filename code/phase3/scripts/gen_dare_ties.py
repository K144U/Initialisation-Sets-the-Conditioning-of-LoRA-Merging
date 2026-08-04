#!/usr/bin/env python3
"""DARE-TIES cells: the one composition where DARE could genuinely help.

Rules fixed in notes/prereg_dare_ties_2026-08-04.md, committed at efb593f BEFORE
the dare_ties method existed. Read that first; the summary is:

  DARE + task arithmetic is an unbiased estimator of task arithmetic and so
  cannot beat it in expectation (notes/audit_dare_2026-08-04.md). That argument
  does NOT apply to DARE + TIES, because trimming and sign election are biased
  and the mask changes WHICH parameters survive. This is the test of that limit.

  4 bases x 3 cohorts (indep1/2/3) x 3 dare densities (0.5, 0.2, 0.1) = 36 cells

  ties_density is pinned at 0.2 in every cell, matching the standalone `ties`
  baseline exactly, so the ONLY difference between the arms is the DARE mask.
  The reference arm is the EXISTING eval_a1_indep/{base}__ties__{cohort}.json
  cells; they are not re-run.

  Primary verdict is at dare_density 0.2 ONLY, named in advance so this is not a
  three-shot test reported as one. 0.5 and 0.1 are the secondary monotonicity
  sweep.

The mask seed is held at 20260518, the same value every other DARE cell in the
project used, so the mask draw is not a new free parameter.

Usage:
  python code/phase3/scripts/gen_dare_ties.py
  qsub code/phase3/scripts/pbs_dare_ties.sh
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
COHORTS = ["indep1", "indep2", "indep3"]
DARE_DENSITIES = [(0.5, "0p5"), (0.2, "0p2"), (0.1, "0p1")]

TIES_DENSITY = 0.2        # pinned to the baseline; prereg binding constraint 3
MASK_SEED = 20260518      # same as every other DARE cell in the project

OUT_CFG = CFG / "eval_dare_ties"
OUT_RES = RES / "eval_dare_ties"


def main() -> int:
    problems: list[str] = []
    manifest: list[dict] = []

    for cohort in COHORTS:
        for base in BASES:
            # Start from the same template the A1 matrix cells used, so the
            # eval side (tasks, n_eval, max_seq_length, data seed) is identical
            # to the `ties` arm we compare against.
            tmpl_p = CFG / f"eval_a1_indep/{base}__ties__{cohort}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            tmpl = yaml.safe_load(tmpl_p.read_text())

            ref = RES / f"eval_a1_indep/{base}__ties__{cohort}.json"
            if not ref.exists():
                problems.append(f"missing TIES reference cell {ref}")

            for dens, tag in DARE_DENSITIES:
                name = f"{base}__dare_ties_d{tag}__{cohort}"
                cfg = dict(tmpl)
                cfg["method"] = "dare_ties"
                cfg["method_kwargs"] = {
                    "dare_density": dens,
                    "ties_density": TIES_DENSITY,
                    "majority_sign_method": "total",
                    "seed": MASK_SEED,
                }
                cfg.pop("loader", None)
                out_json = OUT_RES / f"{name}.json"
                cfg["output_path"] = out_json.as_posix()
                p = OUT_CFG / f"{name}.yaml"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                manifest.append({
                    "name": f"dt_{name}",
                    "cmd": f"python code/phase3/eval/run_eval_cell.py "
                           f"--config {p.relative_to(ROOT).as_posix()}",
                    "done": out_json.relative_to(ROOT).as_posix(),
                    "min_free_gb": 25.0,
                })

    (CFG / "dare_ties_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> dare_ties_manifest.json")
    print(f"[gen] bases {len(BASES)} x cohorts {len(COHORTS)} x densities "
          f"{len(DARE_DENSITIES)}")
    print(f"[gen] ties_density pinned {TIES_DENSITY}, mask seed {MASK_SEED}")
    print(f"[gen] primary verdict density is 0.2 (prereg efb593f)")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems)):
            print("  - " + p)
        return 1
    print("[gen] validation OK, all 12 TIES reference cells present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
