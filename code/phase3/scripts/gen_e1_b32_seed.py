#!/usr/bin/env python3
"""Re-run the E1 rd-encoder b=32, lambda=0 (exact construction) Llama cell --
the "collapse" bar (0.497 on v1) in the salvage figure -- on matched seed1/2/3
adapters, so the salvage arc (disease 0.497 -> cure 0.094) is on ONE adapter
set instead of v1-vs-seed. 3 cells -> results/phase3/eval_e1_seed/."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path("/home/sanjay.g/projects/rdmerge")
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"
SRC = CFG / "eval_e1/llama31_8b__rd_b32.yaml"
SEEDS = ["seed1", "seed2", "seed3"]
OUT_CFG = CFG / "eval_e1_seed"
OUT_RES = RES / "eval_e1_seed"


def main() -> int:
    base = yaml.safe_load(SRC.read_text())
    man = []
    prob = []
    for seed in SEEDS:
        cfg = copy.deepcopy(base)
        for spec in cfg["adapter_specs"]:
            spec["dir"] = str(Path(spec["dir"]).parent / seed)
            if not Path(spec["dir"]).exists():
                prob.append(spec["dir"])
        name = f"llama31_8b__rd_b32__{seed}"
        oj = OUT_RES / f"{name}.json"
        cfg["output_path"] = str(oj)
        cp = OUT_CFG / f"{name}.yaml"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(yaml.safe_dump(cfg, sort_keys=False))
        man.append({
            "name": f"e1s_{name}",
            "cmd": f"python code/phase3/eval/run_eval_cell.py --config {cp.relative_to(ROOT)}",
            "done": str(oj.relative_to(ROOT)),
            "min_free_gb": 25.0,
        })
    (CFG / "e1_b32_seed_manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[gen] {len(man)} cells -> e1_b32_seed_manifest.json")
    if prob:
        print("[gen] MISSING:", *sorted(set(prob)), sep="\n  - ")
        return 1
    print("[gen] validation OK: all seedN adapters present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
