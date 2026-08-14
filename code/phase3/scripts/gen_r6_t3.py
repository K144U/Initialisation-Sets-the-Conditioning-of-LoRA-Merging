#!/usr/bin/env python3
"""R6 cells: the merge matrix with the degenerate translation adapter dropped.

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), section R6, committed before
this generator existed.

Section 6.6 and Appendix C report every geometric and conditioning quantity at
both T = 4 and T = 3, and show the contrast is unchanged. Tables 3, 4 and 5,
which carry all of the merge-performance claims, are T = 4 only, with the
untrained translation adapter inside every merged model. So the geometry is
shown robust to the defect and the performance is not. That asymmetry is what
this fixes.

  4 bases x 6 methods x indep1/2/3 = 72 cells

Six methods, not seven: the registered rule removes default-configuration
KnOTS, which is algebraically task arithmetic and matches it to four decimals,
so its 12 cells would be 12 duplicates. The repaired KnOTS-TIES can be added
later if R4 licenses it as a real arm; that is 12 more cells and needs no
change here.

Each cell is built from its OWN T = 4 template in eval_a1_indep, so every
method keeps its own method_kwargs and its own loader. That matters: rd_ridge
runs `loader: plain` because a rank-64 merged adapter cannot go through the
unsloth path, while the baselines run unsloth. Preserving each method's loader
keeps the T = 3 versus T = 4 comparison exact per method, which is the
comparison R6 is for. It also inherits the cross-method loader mismatch that
analyze_q1_loader_matched.py documents, and that is the right trade: R6 is a
robustness arm for the T = 4 table, not a replacement for it.

Dropping an adapter changes the adapter list, which changes the nll_tau cache
key, so none of the T = 4 entries apply. New entries are written under a T3
suffix, split by loader so the two paths do not overwrite each other.

Usage:
  python code/phase3/scripts/gen_r6_t3.py
  qsub -v ARM=r6 code/phase3/scripts/pbs_wave2.sh
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
# Default-config KnOTS removed by the registered R4 rule. The repaired one is
# included because R4 found it separates from task arithmetic on 4 of 4 bases
# in both arms, which is what the rule required for it to count as a method.
METHODS = ["task_arithmetic", "ties", "dare", "tvq_b2", "rd_ridge", "rd_rank16",
           "knots_ties"]
# Methods whose T = 4 template does not live in eval_a1_indep.
TEMPLATE_DIR = {"knots_ties": "eval_a2_knots_ties_indep"}
DROP_TASK = "translation"

OUT_CFG = CFG / "eval_r6_t3"
OUT_RES = RES / "eval_r6_t3"


def main() -> int:
    manifest: list[dict] = []
    problems: list[str] = []
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    for base in BASES:
        for cohort in COHORTS:
            for method in METHODS:
                tdir = TEMPLATE_DIR.get(method, "eval_a1_indep")
                tmpl_p = CFG / f"{tdir}/{base}__{method}__{cohort}.yaml"
                if not tmpl_p.exists():
                    problems.append(f"missing template {tmpl_p}")
                    continue
                cfg = yaml.safe_load(tmpl_p.read_text())

                specs = [s for s in cfg["adapter_specs"]
                         if s["name"] != DROP_TASK]
                if len(specs) != 3:
                    problems.append(
                        f"{base} {cohort} {method}: {len(specs)} adapters after "
                        f"dropping {DROP_TASK}, expected 3")
                    continue
                if any(s["name"] == DROP_TASK for s in specs):
                    problems.append(f"{base} {cohort} {method}: drop failed")
                    continue
                for s in specs:
                    if not Path(s["dir"], "adapter_model.safetensors").exists():
                        problems.append(f"missing adapter {s['dir']}")
                cfg["adapter_specs"] = specs

                # weights are absent in the templates, so run_eval_cell falls
                # back to 1/n over the REMAINING adapters. Assert rather than
                # assume: a stale 4-entry weights list would silently reweight.
                if "weights" in cfg and len(cfg["weights"]) != 3:
                    problems.append(f"{base} {cohort} {method}: stale weights")

                loader = cfg.get("loader") or "unsloth"
                name = f"{base}__{method}__{cohort}"
                out_p = OUT_RES / f"{name}.json"
                cfg["output_path"] = str(out_p)
                cfg["nll_tau_cache"] = str(
                    RES / "nll_tau_cache" / f"{base}__{cohort}__T3_{loader}.json")
                cfg_p = OUT_CFG / f"{name}.yaml"
                cfg_p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                manifest.append({
                    "name": name,
                    "cmd": ("python code/phase3/eval/run_eval_cell.py "
                            f"--config {cfg_p.relative_to(ROOT)}"),
                    "done": str(out_p.relative_to(ROOT)),
                    "min_free_gb": 25.0,
                })

    if problems:
        print("PROBLEMS, no manifest written:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2

    expect = len(BASES) * len(COHORTS) * len(METHODS)
    assert len(manifest) == expect, (len(manifest), expect)

    # Cache-aware ordering: one cell per (base, cohort, loader) first, so the
    # 24 cache entries are built by 24 cells on distinct keys and the other 48
    # are hits. Without it several workers rebuild the same entry.
    seen: set[tuple] = set()
    first, rest = [], []
    for e in manifest:
        b, m, c = e["name"].split("__")
        key = (b, c, "plain" if m == "rd_ridge" else "unsloth")
        (rest if key in seen else first).append(e)
        seen.add(key)
    manifest = first + rest

    out = CFG / "r6_t3_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(manifest)} cells "
          f"({len(BASES)} bases x {len(METHODS)} methods x {len(COHORTS)} cohorts)")
    print(f"dropped task: {DROP_TASK}; default-config knots excluded per R4")
    print(f"cache-priming cells first: {len(first)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
