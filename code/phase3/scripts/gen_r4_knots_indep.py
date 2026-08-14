#!/usr/bin/env python3
"""R4 cells: the repaired KnOTS on the independent cohorts.

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), section R4.

eval_a2_knots_ties already holds 12 cells of KnOTS with
inner_combination="ties" on the shared arm (seed1/2/3), run before that
pre-registration and not read against any rule. This completes the design with
the independent arm, so the repaired method exists in both:

  4 bases x indep1/2/3 = 12 cells

The registered rule: the repaired KnOTS enters the regime null as an
independent arm only if its worst-task excess differs from task arithmetic by
more than the tie threshold on at least 3 of 4 bases, on the shared arm.
Either way the DEFAULT-configuration KnOTS comes out of the null, because it is
task arithmetic to four decimals and was inflating a 20-cell null to 16 real
cells.

Two details that are load-bearing:

`loader` is dropped, exactly as gen_a2_knots_ties.py does it, so these cells run
the unsloth fast path like every other baseline in eval_a1_indep. Matching the
baselines matters more here than matching the rd_encoder sweeps, because the
comparison this feeds is KnOTS against task arithmetic.

The nll_tau cache path therefore carries an explicit `unsloth` suffix. The
existing entries were written by cells running `loader: plain`, and the two
paths are NOT numerically identical: they differ by up to 0.0105 nats on the
same adapters, which is what analyze_q1_loader_matched.py is about. The cache
key would catch the mismatch and recompute, but sharing one filename between
two loaders makes the two arms overwrite each other's entry in a loop, so they
get separate files.

Usage:
  python code/phase3/scripts/gen_r4_knots_indep.py
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
DENSITY = 0.2

OUT_CFG = CFG / "eval_a2_knots_ties_indep"
OUT_RES = RES / "eval_a2_knots_ties_indep"


def main() -> int:
    manifest: list[dict] = []
    problems: list[str] = []
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    for base in BASES:
        for cohort in COHORTS:
            tmpl_p = CFG / f"eval_a1_indep/{base}__knots__{cohort}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            cfg = yaml.safe_load(tmpl_p.read_text())
            for spec in cfg["adapter_specs"]:
                if not Path(spec["dir"], "adapter_model.safetensors").exists():
                    problems.append(f"missing adapter {spec['dir']}")

            # The template is the default-config KnOTS cell, i.e. the one that
            # is algebraically task arithmetic. Refuse to emit a cell that
            # silently keeps that default.
            cfg["method"] = "knots"
            cfg["method_kwargs"] = {"inner_combination": "ties",
                                    "density": DENSITY}
            cfg.pop("loader", None)
            if cfg.get("method_kwargs", {}).get("inner_combination") != "ties":
                problems.append(f"{base} {cohort}: inner_combination not set")

            name = f"{base}__knots_ties__{cohort}"
            out_p = OUT_RES / f"{name}.json"
            cfg["output_path"] = str(out_p)
            cfg["nll_tau_cache"] = str(
                RES / "nll_tau_cache" / f"{base}__{cohort}__unsloth.json")
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

    expect = len(BASES) * len(COHORTS)
    assert len(manifest) == expect, (len(manifest), expect)
    out = CFG / "r4_knots_indep_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(manifest)} cells "
          f"({len(BASES)} bases x {len(COHORTS)} cohorts)")
    print("inner_combination=ties, density=0.2, unsloth path, separate cache")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
