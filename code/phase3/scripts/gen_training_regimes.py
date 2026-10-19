#!/usr/bin/env python3
"""Training configs and cells of notes/prereg_training_regimes_2026-10-19.md.

  python gen_training_regimes.py train   -> configs/lora_regimes/*.yaml + manifest (60)
  python gen_training_regimes.py eval    -> NLL cells (17) once the adapters exist

Template: configs/lora_seeds/llama31_8b_<task>_seed1.yaml. Every run holds the
data split at 20260518, saves A_0, and differs from the template only in the
fields its cohort names.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

from _c1019 import CACHE, CFG, INVARIANT, RES, ROOT, Manifest, template

BASE = "llama31_8b"
TASKS = [("gsm8k", "gsm8k"), ("alpaca", "alpaca"), ("magicoder", "magicoder"),
         ("translation", "flores")]
TASK_SEED = {"gsm8k": 101, "alpaca": 102, "magicoder": 103, "translation": 104}
CONT_SEED = {"gsm8k": 201, "alpaca": 202, "magicoder": 203, "translation": 204}
DATA_SEED = 20260518
PARENT = ROOT / "artifacts_new" / "lora" / BASE / "dolly" / "indep1fx"
LR = 5e-5


def shared(_t):
    return 1


def indep(t):
    return TASK_SEED[t]


# name -> (init seed fn, trainer seed fn, overrides)
COHORTS = {
    # Part S
    "s_sast": (shared, shared, {}),
    "s_sait": (shared, indep, {}),
    "s_iast": (indep, shared, {}),
    # Part K (baseline is s_sast)
    "k_lr033": (shared, shared, {"train.lr": LR / 3}),
    "k_lr3": (shared, shared, {"train.lr": LR * 3}),
    "k_ep1": (shared, shared, {"train.epochs": 1}),
    "k_ep4": (shared, shared, {"train.epochs": 4}),
    "k_amult4": (shared, shared, {"train.lr_A_mult": 4.0}),
    "k_loraplus16": (shared, shared, {"train.lr_B_mult": 16.0}),
    "k_scale025": (shared, shared, {"lora.init_scale": 0.25}),
    "ki_lr3": (indep, indep, {"train.lr": LR * 3}),
    "ki_amult4": (indep, indep, {"train.lr_A_mult": 4.0}),
    # Part C
    "c_cont": (lambda t: CONT_SEED[t], lambda t: CONT_SEED[t], {"lora.init_from": str(PARENT)}),
    # Part D
    "d_shared": (shared, shared, {"lora.use_dora": True}),
    "d_indep": (indep, indep, {"lora.use_dora": True}),
}
SHARED_K = ["s_sast", "k_lr033", "k_lr3", "k_ep1", "k_ep4", "k_amult4", "k_loraplus16", "k_scale025"]


def train() -> int:
    out_dir = CFG / "lora_regimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for cohort, (fi, ft, ov) in COHORTS.items():
        for task, adir in TASKS:
            cfg = yaml.safe_load((CFG / "lora_seeds" / f"{BASE}_{adir}_seed1.yaml").read_text())
            cfg = copy.deepcopy(cfg)
            cfg["base_model"] = str(ROOT / "models" / "Llama-3.1-8B-Instruct")
            cfg["seeds"] = {"global": ft(task), "train": ft(task), "init": fi(task), "data": DATA_SEED}
            for k, v in ov.items():
                sec, key = k.split(".")
                cfg[sec][key] = v
            cfg["save_init_weights"] = True
            cfg["output"] = {
                "adapter_dir": f"artifacts/lora_regimes/{BASE}/{adir}/{cohort}",
                "results_path": f"results/phase3/lora_train_regimes/{BASE}_{adir}_{cohort}.json"}
            p = out_dir / f"{BASE}_{adir}_{cohort}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            cells.append({"name": f"train_{cohort}_{adir}",
                          "cmd": f"python code/phase3/training/train_lora.py --config {p.relative_to(ROOT)}",
                          "done": cfg["output"]["results_path"], "min_free_gb": 40.0})
    if not PARENT.exists():
        raise SystemExit(f"missing parent adapter {PARENT}")
    if len(cells) != 60:
        raise SystemExit(f"expected 60 training cells, built {len(cells)}")
    import json
    (CFG / "regimes_train_manifest.json").write_text(json.dumps(cells, indent=1))
    print(f"wrote {len(cells)} training cells")
    return 0


def evaluate() -> int:
    t = template(RES / "eval_regmean_real" / f"{BASE}__rmdf_l0p01__seed1.json")
    m = Manifest()

    def cell(cohort, tag, method, kw):
        specs = []
        for s in t["adapter_specs"]:
            adir = dict(TASKS)[s["name"]]
            d = ROOT / "artifacts" / "lora_regimes" / BASE / adir / cohort
            if not (d / "adapter_model.safetensors").exists():
                raise SystemExit(f"missing {d}")
            specs.append(dict(s, dir=str(d)))
        name = f"{BASE}__{tag}__{cohort}"
        cfg = {k: v for k, v in t.items() if k != "output_path"}
        cfg.update(method=method, method_kwargs=kw, adapter_specs=specs,
                   nll_tau_cache=str(CACHE / f"{BASE}__{cohort}.json"),
                   output_path=str(RES / "eval_regimes" / f"{name}.json"))
        for k in INVARIANT:
            assert cfg.get(k) == t.get(k), (name, k)
        m.add(name, cfg, "eval_regimes", "run_eval_cell.py")

    for c in ["s_sast", "s_sait", "s_iast", "c_cont"]:
        cell(c, "rm1em6", "regmean", {"ridge_lambda": 1e-6})
        cell(c, "ta", "task_arithmetic", {})
    for c in ["k_lr033", "k_lr3", "k_ep1", "k_ep4", "k_amult4", "k_loraplus16", "k_scale025",
              "ki_lr3", "ki_amult4"]:
        cell(c, "rm1em6", "regmean", {"ridge_lambda": 1e-6})
    m.write("regimes_eval_manifest.json", expect=17)
    return 0


if __name__ == "__main__":
    sys.exit(train() if sys.argv[1:] == ["train"] else evaluate() if sys.argv[1:] == ["eval"]
             else (print(__doc__) or 2))
