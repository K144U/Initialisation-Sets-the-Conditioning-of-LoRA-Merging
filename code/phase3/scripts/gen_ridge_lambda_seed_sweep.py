#!/usr/bin/env python3
"""Re-run the Llama-3.1 rd-encoder ridge lambda-sweep on matched seed1/2/3
adapters, so paper Table tab:rd-ridge-sweep (the salvage arc), the held-out
lambda table, and the salvage figure are all on the same adapters as the
baselines (seed1/2/3) instead of the legacy `v1` set.

Llama lambda grid = {0.001,0.01,0.05,0.07,0.1,0.13,0.17,0.2,0.3,1.0} x 3 seeds
= 30 cells. Output -> results/phase3/eval_ridge_seed/.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path("/home/sanjay.g/projects/rdmerge")
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

BASE = "llama31_8b"
LAMBDAS = [0.001, 0.01, 0.05, 0.07, 0.1, 0.13, 0.17, 0.2, 0.3, 1.0]
SEEDS = ["seed1", "seed2", "seed3"]
OUT_CFG = CFG / "eval_ridge_seed"
OUT_RES = RES / "eval_ridge_seed"


def ltag(l: float) -> str:
    if l == 1.0:
        return "l1"
    return "l" + ("%g" % l).replace("0.", "0p").replace(".", "p")


def main() -> int:
    man = []
    prob = []
    for seed in SEEDS:
        tp = CFG / f"eval_matrix_seeds/{BASE}__ties__{seed}.yaml"
        if not tp.exists():
            prob.append(f"missing template {tp}")
            continue
        t = yaml.safe_load(tp.read_text())
        for spec in t["adapter_specs"]:
            if not Path(spec["dir"]).exists():
                prob.append(f"missing adapter {spec['dir']}")
        for l in LAMBDAS:
            cfg = dict(t)
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": 32, "seed": 20260611,
                                    "realize": "rank_deff", "ridge_lambda": l}
            cfg["loader"] = "plain"
            name = f"{BASE}__ridge_{ltag(l)}__{seed}"
            oj = OUT_RES / f"{name}.json"
            cfg["output_path"] = str(oj)
            cp = OUT_CFG / f"{name}.yaml"
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
            man.append({
                "name": f"rls_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py --config {cp.relative_to(ROOT)}",
                "done": str(oj.relative_to(ROOT)),
                "min_free_gb": 25.0,
            })
    (CFG / "ridge_lambda_seed_sweep_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[gen] {len(man)} cells -> ridge_lambda_seed_sweep_manifest.json")
    if prob:
        print("[gen] PROBLEMS:", *prob, sep="\n  - ")
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
