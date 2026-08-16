#!/usr/bin/env python3
"""Training configs for the drift cohort.

Rules: notes/prereg_drift_2026-08-16.md (9202d3b), committed before this
generator existed and before any adapter of this arm is trained.

8 runs: one base (Llama-3.1-8B-Instruct), four tasks, two initialisation arms.
Each writes A_0 before the first optimiser step and keeps checkpoints at
25/50/75/100% of training, which is what the original cohorts cannot provide
and why E3 has stood undischarged.

Everything except those two additions is copied from the existing configs, so
this cohort is comparable to the ones the paper already reports. In particular
the translation task keeps its unshuffled streaming prefix: this run measures
drift, and fixing the translation adapter at the same time would leave the
result comparable to nothing.

Usage:
  python code/phase3/scripts/gen_drift_cohort.py
  qsub code/phase3/scripts/pbs_drift_cohort.sh
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
OUT_CFG = CFG / "lora_drift"
RES = ROOT / "results/phase3"

BASE = "llama31_8b"
TASKS = ["alpaca", "gsm8k", "magicoder", "flores"]

# Arms. The shared arm gives every task one seed, which is the condition
# section 5.6 shows produces one shared A; the independent arm gives each task
# its own. Seeds are kept away from 1/2/3 and 101-104 so the artifact
# directories can never collide with an existing cohort.
ARMS = {
    "drift_shared": {t: 7001 for t in TASKS},
    "drift_indep": dict(zip(TASKS, [7101, 7102, 7103, 7104])),
}

CHECKPOINT_FRACTION = 0.25          # -> 25 / 50 / 75 / 100 %


def build(problems: list[str]) -> list[dict]:
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for arm, seeds in ARMS.items():
        for task in TASKS:
            src = CFG / "lora_seeds" / f"{BASE}_{task}_seed1.yaml"
            if not src.exists():
                problems.append(f"missing source config {src}")
                continue
            cfg = yaml.safe_load(src.read_text())

            cfg["seeds"] = {
                "global": seeds[task],
                "train": seeds[task],
                # The data shuffle is held fixed across tasks and arms, exactly
                # as in the A1 cohorts, so a change in geometry can only be
                # attributed to the initialisation draw.
                "data": 20260518,
            }
            cfg["save_init_weights"] = True
            cfg["train"]["checkpoint_fraction"] = CHECKPOINT_FRACTION

            adir = f"artifacts/lora/{BASE}/{task}/{arm}"
            cfg["output"] = {
                "adapter_dir": adir,
                "results_path": f"results/phase3/lora_train/{BASE}_{task}_{arm}.json",
            }

            name = f"{BASE}_{task}_{arm}"
            (OUT_CFG / f"{name}.yaml").write_text(
                yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({
                "name": name,
                "cmd": ("python code/phase3/training/train_lora.py --config "
                        f"{(OUT_CFG / (name + '.yaml')).relative_to(ROOT)}"),
                # The trained adapter, NOT adapter_A0: A_0 lands ~90s
                # into a run, and the orchestrator skips any cell whose
                # done-marker exists at dispatch, so marking on A_0
                # would make every requeue skip unfinished training.
                "done": f"{adir}/adapter_model.safetensors",
                "min_free_gb": 40.0,
            })

    # The arms must differ in the initialisation seed and in nothing else.
    for task in TASKS:
        a = OUT_CFG / f"{BASE}_{task}_drift_shared.yaml"
        b = OUT_CFG / f"{BASE}_{task}_drift_indep.yaml"
        if not (a.exists() and b.exists()):
            continue
        ca, cb = yaml.safe_load(a.read_text()), yaml.safe_load(b.read_text())
        for k in ("lora", "train", "task", "base_model"):
            if ca.get(k) != cb.get(k):
                problems.append(f"{task}: arms differ in {k}, they must not")
        if ca["seeds"]["global"] == cb["seeds"]["global"]:
            problems.append(f"{task}: arms share an initialisation seed")

    # Within the shared arm, all four tasks must carry one seed, or it is not
    # a shared-initialisation cohort at all.
    shared = {yaml.safe_load((OUT_CFG / f"{BASE}_{t}_drift_shared.yaml")
                             .read_text())["seeds"]["global"] for t in TASKS
              if (OUT_CFG / f"{BASE}_{t}_drift_shared.yaml").exists()}
    if len(shared) > 1:
        problems.append(f"shared arm carries {len(shared)} distinct seeds")

    return manifest


def main() -> int:
    problems: list[str] = []
    manifest = build(problems)
    if problems:
        print("PROBLEMS, nothing dispatched:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2
    if len(manifest) != len(TASKS) * len(ARMS):
        print(f"PROBLEM: {len(manifest)} runs, expected "
              f"{len(TASKS) * len(ARMS)}")
        return 2

    mpath = CFG / "drift_cohort_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} training runs -> {mpath.relative_to(ROOT)}")
    print(f"  base {BASE}, {len(TASKS)} tasks, arms: {', '.join(ARMS)}")
    print(f"  A_0 saved before the first step; checkpoints every "
          f"{CHECKPOINT_FRACTION:.0%} of training")
    done = sum(1 for e in manifest if (ROOT / e["done"]).exists())
    print(f"  already complete: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
