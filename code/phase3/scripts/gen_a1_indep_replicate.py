#!/usr/bin/env python3
"""A1 replication: two more independent-init cohorts, so the matrix has 3 seeds.

WHY. The step-0 result (notes/campaign_results_2026-08-03.md) rests on 28 cells
at ONE seed per cell. Its three verdicts are:

  Q1 WEAKENED                   1 win, 2 ties, 1 loss vs the best baseline
  Q2 RANKINGS CHANGE MATERIALLY one decisive case (Yi) + one coin flip (Mistral)
  Q3 RANK IS IMMATERIAL         within threshold on 3 of 4

Two of the four Q1 gaps (Mistral -0.0032, Qwen +0.0001) are smaller than the
per-seed sd measured on the shared cohort (Llama 0.0064, Mistral 0.0029, Qwen
0.0005, Yi 0.0009), and the Mistral top-1 swap in Q2 is inside the 0.005
threshold in BOTH cohorts. Nothing above can be claimed in a paper until it is
replicated.

WHAT THIS RUNS. Two more cohorts, indep2 and indep3, built exactly like indep1
but with a different independent per-task LoRA A draw:

  indep1  gsm8k 101  alpaca 102  magicoder 103  translation 104   (done)
  indep2  gsm8k 201  alpaca 202  magicoder 203  translation 204
  indep3  gsm8k 301  alpaca 302  magicoder 303  translation 304

  stage 1  train   32 cells  2 cohorts x 4 bases x 4 tasks
  stage 2  matrix  56 cells  2 cohorts x 4 bases x 7 methods

DATA_SEED is held at 20260518 for both, matching indep1 and the eval cells. The
init draw is therefore the ONLY thing that varies across the three cohorts,
which is what makes this a clean estimate of init-draw variance. It does not
estimate data-shuffle variance; that would need a separate sweep and is not what
the step-0 claims depend on.

Nothing here changes a threshold. The step-0 verdicts stand as recorded; this
only attaches error bars to them.

Usage:
  python code/phase3/scripts/gen_a1_indep_replicate.py
  qsub code/phase3/scripts/pbs_a1_rep_train.sh     # stage 1
  qsub code/phase3/scripts/pbs_a1_rep_matrix.sh    # stage 2, gated on 32 adapters
  python code/phase3/scripts/measure_subspace_geometry.py --cohort indep2
  python code/phase3/scripts/measure_subspace_geometry.py --cohort indep3
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
TASKS = [("gsm8k", "gsm8k"), ("alpaca", "alpaca"),
         ("magicoder", "magicoder"), ("translation", "flores")]

COHORT_SEEDS = {
    "indep2": {"gsm8k": 201, "alpaca": 202, "magicoder": 203, "translation": 204},
    "indep3": {"gsm8k": 301, "alpaca": 302, "magicoder": 303, "translation": 304},
}
DATA_SEED = 20260518

LAMBDA_STAR = {"llama31_8b": 0.05, "mistral_7b": 0.13,
               "qwen25_7b": 0.13, "yi15_9b": 0.13}

MATRIX_METHODS = {
    "task_arithmetic": {},
    "ties": {"density": 0.2, "majority_sign_method": "total"},
    "dare": {"density": 0.2, "seed": 20260518},
    "knots": {"inner_combination": "linear"},
    "tvq_b2": {"rate_bits": 2},
}

OUT_TRAIN_CFG = CFG / "lora_indep"
OUT_EVAL_CFG = CFG / "eval_a1_indep"
OUT_EVAL_RES = RES / "eval_a1_indep"


def main() -> int:
    problems: list[str] = []
    train_manifest: list[dict] = []
    eval_manifest: list[dict] = []

    for cohort, task_seed in COHORT_SEEDS.items():
        # ---- stage 1: training ------------------------------------------
        for base in BASES:
            for task, adir in TASKS:
                tmpl_p = CFG / f"lora_seeds/{base}_{adir}_seed1.yaml"
                if not tmpl_p.exists():
                    problems.append(f"missing train template {tmpl_p}")
                    continue
                cfg = yaml.safe_load(tmpl_p.read_text())
                s = task_seed[task]
                cfg["seeds"] = {"global": s, "train": s, "data": DATA_SEED}
                cfg["output"] = {
                    "adapter_dir": f"artifacts/lora/{base}/{adir}/{cohort}",
                    "results_path": f"results/phase3/lora_train/"
                                    f"{base}_{adir}_{cohort}.json",
                }
                p = OUT_TRAIN_CFG / f"{base}_{adir}_{cohort}.yaml"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                train_manifest.append({
                    "name": f"a1rt_{base}_{adir}_{cohort}",
                    "cmd": f"python code/phase3/training/train_lora.py "
                           f"--config {p.relative_to(ROOT).as_posix()}",
                    "done": f"results/phase3/lora_train/"
                            f"{base}_{adir}_{cohort}.json",
                    "min_free_gb": 40.0,
                })

        # ---- stage 2: the T=4 matrix -------------------------------------
        for base in BASES:
            tmpl_p = CFG / f"eval_matrix_seeds/{base}__ties__seed1.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing eval template {tmpl_p}")
                continue
            tmpl = yaml.safe_load(tmpl_p.read_text())

            def with_cohort(c: dict) -> dict:
                c = dict(c)
                specs = []
                for spec in c["adapter_specs"]:
                    spec = dict(spec)
                    spec["dir"] = str(
                        Path(spec["dir"]).parent / cohort).replace("\\", "/")
                    specs.append(spec)
                c["adapter_specs"] = specs
                return c

            def emit(name: str, cfg: dict) -> None:
                out_json = OUT_EVAL_RES / f"{name}.json"
                cfg["output_path"] = out_json.as_posix()
                p = OUT_EVAL_CFG / f"{name}.yaml"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                eval_manifest.append({
                    "name": f"a1rm_{name}",
                    "cmd": f"python code/phase3/eval/run_eval_cell.py "
                           f"--config {p.relative_to(ROOT).as_posix()}",
                    "done": out_json.relative_to(ROOT).as_posix(),
                    "min_free_gb": 25.0,
                })

            for mname, mkw in MATRIX_METHODS.items():
                cfg = with_cohort(tmpl)
                cfg["method"] = "tvq" if mname == "tvq_b2" else mname
                cfg["method_kwargs"] = dict(mkw)
                cfg.pop("loader", None)
                emit(f"{base}__{mname}__{cohort}", cfg)

            for tag, realize, loader in [("rd_ridge", "rank_deff", "plain"),
                                         ("rd_rank16", "rank_r", None)]:
                cfg = with_cohort(tmpl)
                cfg["method"] = "rd_encoder"
                cfg["method_kwargs"] = {"bits": 32, "seed": 20260611,
                                        "realize": realize,
                                        "ridge_lambda": LAMBDA_STAR[base]}
                if loader:
                    cfg["loader"] = loader
                else:
                    cfg.pop("loader", None)
                emit(f"{base}__{tag}__{cohort}", cfg)

    (CFG / "a1_rep_train_manifest.json").write_text(
        json.dumps(train_manifest, indent=2))
    (CFG / "a1_rep_matrix_manifest.json").write_text(
        json.dumps(eval_manifest, indent=2))
    print(f"[gen] stage 1 train : {len(train_manifest):>3} cells -> "
          f"a1_rep_train_manifest.json")
    print(f"[gen] stage 2 matrix: {len(eval_manifest):>3} cells -> "
          f"a1_rep_matrix_manifest.json")
    for c, s in COHORT_SEEDS.items():
        print(f"[gen]   {c}: {s}")
    print(f"[gen] data seed held at {DATA_SEED} for both (init is the only variable)")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems)):
            print("  - " + p)
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
