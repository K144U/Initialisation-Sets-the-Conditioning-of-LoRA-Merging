#!/usr/bin/env python3
"""W1 follow-up: tuned Task Arithmetic at three seeds, to settle Llama.

The alpha sweep ran at seed1 only, so the decisive Llama comparison is a
seed1 TA number (0.0839) against a 3-seed rd-ridge mean (0.0945). That is
apples-to-oranges in rd-ridge's disfavour, and Llama is exactly where seed
variance is largest: rd-ridge's own seed1 cell is 0.0823 against a 3-seed mean
of 0.0945, a spread of 0.012.

This runs TA at each base's alpha* for seeds 2 and 3, so the comparison becomes
3-seed against 3-seed. The seed1 cells already exist in eval_w1_alpha/, so only
8 cells are new.

  base        alpha*   existing            new
  llama31_8b   0.75    ta_alpha0p75 seed1  seeds 2,3
  mistral_7b   0.50    ta_alpha0p5  seed1  seeds 2,3
  qwen25_7b    0.50    ta_alpha0p5  seed1  seeds 2,3
  yi15_9b      0.50    ta_alpha0p5  seed1  seeds 2,3

Decision rule, fixed before the cells run. Let d = TA3(alpha*) - rd3, on
matched 3-seed means:
  d < -0.005  on Llama  -> a tuned scalar genuinely beats the method on the
                           flagship base; the abstract's "all four base models"
                           and Figure 2's framing must both be rewritten.
  |d| <= 0.005 on Llama -> statistical tie; claim "matches a tuned TA on Llama,
                           beats it on the other three".
  d > +0.005  on Llama  -> the seed1 tie was a seed artifact and rd-ridge wins
                           all four against tuned TA, at 30-39% rather than
                           the paper's 56-91%.
In every branch the per-base margin figures in the abstract and Section 6.3
have to change, because TA at 1/T = 0.25 is undertuned on all four bases.

Output goes to results/phase3/eval_w1_alpha3s/ so it does not disturb the
w1 stage's 52-cell completion count in the campaign keeper.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

# Measured optima from the seed1 sweep (interior optima on all four bases).
ALPHA_STAR = {"llama31_8b": 0.75, "mistral_7b": 0.50,
              "qwen25_7b": 0.50, "yi15_9b": 0.50}
SEEDS = ["seed2", "seed3"]

OUT_CFG = CFG / "eval_w1_alpha3s"
OUT_RES = RES / "eval_w1_alpha3s"


def main() -> int:
    manifest, problems = [], []
    for base, a in ALPHA_STAR.items():
        for seed in SEEDS:
            tmpl_p = CFG / f"eval_matrix_seeds/{base}__ties__{seed}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            cfg = yaml.safe_load(tmpl_p.read_text())
            for spec in cfg["adapter_specs"]:
                if not Path(spec["dir"]).exists():
                    problems.append(f"missing adapter {spec['dir']}")
            n = len(cfg["adapter_specs"])
            cfg["method"] = "task_arithmetic"
            cfg["method_kwargs"] = {}
            cfg["weights"] = [float(a)] * n
            cfg.pop("loader", None)
            name = f"{base}__ta_alphastar__{seed}"
            out_json = OUT_RES / f"{name}.json"
            cfg["output_path"] = out_json.as_posix()
            p = OUT_CFG / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({
                "name": f"w1s_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py "
                       f"--config {p.relative_to(ROOT).as_posix()}",
                "done": out_json.relative_to(ROOT).as_posix(),
                "min_free_gb": 25.0,
            })

    man_p = CFG / "w1_alphastar_3seed_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> {man_p}")
    for b, a in ALPHA_STAR.items():
        print(f"[gen]   {b:<13} alpha* = {a}")
    if problems:
        print("[gen] PROBLEMS:")
        for x in sorted(set(problems))[:10]:
            print("  - " + x)
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
