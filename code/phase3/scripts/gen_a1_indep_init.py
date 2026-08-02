#!/usr/bin/env python3
"""A1: retrain one cohort with per-task independent LoRA A initialisation.

THE FINDING (notes/audit_2026-08-02_code_and_claims.md, A1). Every task adapter
in a given (base, seed) cohort was trained from the SAME LoRA `A` init:
train_lora.py seeds from cfg["seeds"]["global"], and all four task configs for
a seed carry the same value (configs/lora_seeds/*_seed1.yaml -> global: 1).
LoRA starts B = 0, so A receives almost no gradient and stays near its init.
Measured on artifacts/lora/llama31_8b, layer 9 v_proj, same on all four bases:

    across tasks (same seed):  ||A_x - A_y|| / ||A_x|| = 0.18,
                               subspace principal cosines median 0.996
    across seeds (same task):  1.41 and 0.054

So the T task subspaces are near-identical: stacked participation ratio ~16 of
Tr = 64, sigma_max ~ sqrt(T). The paper's headline "d_eff = Tr in every layer,
so the floor is exactly zero" holds only because the rank test uses a ~1e-3
tolerance; the geometry is the MAXIMUM-overlap limit, which is the
floor-positive regime Appendix B argues cannot arise naturally.

THE TEST. Retrain the same 4 tasks x 4 bases with a DIFFERENT global seed per
task, so each adapter gets an independent A. Then re-measure the geometry and
re-run the T=4 matrix. Two outcomes, both publishable, both better than the
current text:

  (a) soft d_eff -> ~Tr. The floor-zero claim becomes true and attributable,
      but Hbar becomes well conditioned, so the sliver pathology that motivates
      the ridge should weaken and rd-encoder ridge's margin may shrink or
      vanish. That is the risk, and it is better found now than in review.

  (b) geometry unchanged. Then subspace collapse is a property of LoRA
      fine-tuning rather than of the shared init, the paper gains the control
      it currently lacks, and Appendix B's stated mechanism still needs
      rewriting (it attributes the saturation to the per-task loss geometry).

Emits three stages. Run them in order; each has its own manifest.

  stage 1  train   16 cells  4 bases x 4 tasks, per-task seeds 101/102/103/104
  stage 2  geom     CPU      re-measure hard/soft d_eff, principal angles
  stage 3  matrix  24 cells  5 baselines + rd-ridge at lambda*, T=4, indep cohort

Usage:
  python code/phase3/scripts/gen_a1_indep_init.py
  qsub code/phase3/scripts/pbs_a1_indep_train.sh      # stage 1
  python code/phase3/scripts/measure_subspace_geometry.py --cohort indep1  # stage 2
  qsub code/phase3/scripts/pbs_a1_indep_matrix.sh     # stage 3
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"
ART = ROOT / "artifacts/lora"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
# (task name, adapter subdirectory used by the existing cohorts)
TASKS = [("gsm8k", "gsm8k"), ("alpaca", "alpaca"),
         ("magicoder", "magicoder"), ("translation", "flores")]

# The whole point: a DIFFERENT global seed per task, so each adapter draws an
# independent LoRA A. Kept away from 1/2/3 so the artifact dirs never collide
# with the existing seed1/2/3 cohorts.
TASK_SEED = {"gsm8k": 101, "alpaca": 102, "magicoder": 103, "translation": 104}

# The data shuffle is held FIXED across tasks, decoupled from the init seed via
# seeds.data (train_lora.py). Two reasons:
#   1. If the data subset moved with the init, a change in subspace geometry
#      could not be attributed to the init, which is the whole experiment.
#   2. It equals the eval cells' shuffle seed, so train/eval stay disjoint for
#      alpaca and magicoder, whose eval split IS the train split. The published
#      cohorts violate this and overlap by ~14.5% / ~10% (audit B1).
DATA_SEED = 20260518

COHORT = "indep1"          # artifacts/lora/<base>/<task>/indep1
LAMBDA_STAR = {"llama31_8b": 0.05, "mistral_7b": 0.13,
               "qwen25_7b": 0.13, "yi15_9b": 0.13}

MATRIX_METHODS = {
    "task_arithmetic": ({}, None),
    "ties": ({"density": 0.2, "majority_sign_method": "total"}, None),
    "dare": ({"density": 0.2, "seed": 20260518}, None),
    "knots": ({"inner_combination": "linear"}, None),
    "tvq_b2": ({"rate_bits": 2}, None),
}

OUT_TRAIN_CFG = CFG / "lora_indep"
OUT_EVAL_CFG = CFG / "eval_a1_indep"
OUT_EVAL_RES = RES / "eval_a1_indep"


def main() -> int:
    problems: list[str] = []
    train_manifest: list[dict] = []
    eval_manifest: list[dict] = []

    # ---- stage 1: training configs -------------------------------------
    for base in BASES:
        for task, adir in TASKS:
            # Template: the existing seed1 training config for this (base, task).
            tmpl_p = CFG / f"lora_seeds/{base}_{adir}_seed1.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing train template {tmpl_p}")
                continue
            cfg = yaml.safe_load(tmpl_p.read_text())
            s = TASK_SEED[task]
            # global/train drive the LoRA A init (the variable under test);
            # data is held fixed and matched to the eval cells' seed.
            cfg["seeds"] = {"global": s, "train": s, "data": DATA_SEED}
            cfg["output"] = {
                "adapter_dir": f"artifacts/lora/{base}/{adir}/{COHORT}",
                "results_path": f"results/phase3/lora_train/"
                                f"{base}_{adir}_{COHORT}.json",
            }
            p = OUT_TRAIN_CFG / f"{base}_{adir}_{COHORT}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            train_manifest.append({
                "name": f"a1tr_{base}_{adir}",
                "cmd": f"python code/phase3/training/train_lora.py "
                       f"--config {p.relative_to(ROOT).as_posix()}",
                "done": f"results/phase3/lora_train/{base}_{adir}_{COHORT}.json",
                "min_free_gb": 40.0,
            })

    # ---- stage 3: T=4 matrix on the independent cohort ------------------
    for base in BASES:
        tmpl_p = CFG / f"eval_matrix_seeds/{base}__ties__seed1.yaml"
        if not tmpl_p.exists():
            problems.append(f"missing eval template {tmpl_p}")
            continue
        tmpl = yaml.safe_load(tmpl_p.read_text())

        def with_indep_adapters(c: dict) -> dict:
            c = dict(c)
            specs = []
            for spec in c["adapter_specs"]:
                spec = dict(spec)
                # swap the trailing /seed1 for /indep1, keeping everything else
                spec["dir"] = str(Path(spec["dir"]).parent / COHORT).replace("\\", "/")
                specs.append(spec)
            c["adapter_specs"] = specs
            return c

        for mname, (mkw, loader) in MATRIX_METHODS.items():
            cfg = with_indep_adapters(tmpl)
            cfg["method"] = "tvq" if mname == "tvq_b2" else mname
            cfg["method_kwargs"] = dict(mkw)
            cfg.pop("loader", None)
            name = f"{base}__{mname}__{COHORT}"
            out_json = OUT_EVAL_RES / f"{name}.json"
            cfg["output_path"] = out_json.as_posix()
            p = OUT_EVAL_CFG / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            eval_manifest.append({
                "name": f"a1ev_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py "
                       f"--config {p.relative_to(ROOT).as_posix()}",
                "done": out_json.relative_to(ROOT).as_posix(),
                "min_free_gb": 25.0,
            })

        # rd-encoder ridge at the base-optimal lambda, both realizations, so the
        # comparison against the published cohort is like for like.
        for tag, realize, loader in [("rd_ridge", "rank_deff", "plain"),
                                     ("rd_rank16", "rank_r", None)]:
            cfg = with_indep_adapters(tmpl)
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": 32, "seed": 20260611,
                                    "realize": realize,
                                    "ridge_lambda": LAMBDA_STAR[base]}
            if loader:
                cfg["loader"] = loader
            else:
                cfg.pop("loader", None)
            name = f"{base}__{tag}__{COHORT}"
            out_json = OUT_EVAL_RES / f"{name}.json"
            cfg["output_path"] = out_json.as_posix()
            p = OUT_EVAL_CFG / f"{name}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            eval_manifest.append({
                "name": f"a1ev_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py "
                       f"--config {p.relative_to(ROOT).as_posix()}",
                "done": out_json.relative_to(ROOT).as_posix(),
                "min_free_gb": 25.0,
            })

    (CFG / "a1_indep_train_manifest.json").write_text(
        json.dumps(train_manifest, indent=2))
    (CFG / "a1_indep_matrix_manifest.json").write_text(
        json.dumps(eval_manifest, indent=2))
    print(f"[gen] stage 1 train : {len(train_manifest):>3} cells -> "
          f"a1_indep_train_manifest.json")
    print(f"[gen] stage 3 matrix: {len(eval_manifest):>3} cells -> "
          f"a1_indep_matrix_manifest.json")
    print(f"[gen] per-task seeds: {TASK_SEED}")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems)):
            print("  - " + p)
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
