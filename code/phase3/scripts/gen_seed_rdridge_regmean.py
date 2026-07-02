#!/usr/bin/env python3
"""Generate apples-to-apples 3-seed cells: rd-encoder ridge (lambda=0.13) vs
tuned RegMean (ridge_lambda=0.01) on Qwen-2.5-7B and Yi-1.5-9B, evaluated on
the SAME seed1/2/3 adapters the TIES/DARE baselines use.

Motivation: the published rd-ridge cells were run on the `v1` adapter set,
which is a DIFFERENT training run from `seed1/2/3` (md5 mismatch). To decide
the rd-ridge-vs-RegMean ties on Qwen/Yi honestly, both methods must run on the
matched seedN adapters. Templates = the matrix `*__ties__seedN.yaml` configs
(correct base_model + seedN adapter_specs); we only swap method/method_kwargs.

2 bases x 3 seeds x 2 methods = 12 cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path("/home/sanjay.g/projects/rdmerge")
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

BASES = ["qwen25_7b", "yi15_9b"]
SEEDS = ["seed1", "seed2", "seed3"]
METHODS = {
    # rd-encoder ridge at the base-optimal lambda (0.13 for Qwen/Yi), matching
    # the published eval_ridge_xmodel config exactly (loader: plain).
    "rd_ridge": {
        "method": "rd_encoder",
        "method_kwargs": {"bits": 32, "seed": 20260611,
                          "realize": "rank_deff", "ridge_lambda": 0.13},
        "loader": "plain",
    },
    # RegMean at its swept-best lambda (0.01 on Qwen/Yi).
    "regmean": {
        "method": "regmean",
        "method_kwargs": {"ridge_lambda": 0.01},
    },
}

OUT_CFG = CFG / "eval_seed_rdridge_regmean"
OUT_RES = RES / "eval_seed_rdridge_regmean"


def main() -> int:
    manifest = []
    problems = []
    for base in BASES:
        for seed in SEEDS:
            tmpl_p = CFG / f"eval_matrix_seeds/{base}__ties__{seed}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            tmpl = yaml.safe_load(tmpl_p.read_text())
            # validate seedN adapters exist
            for spec in tmpl["adapter_specs"]:
                if not Path(spec["dir"]).exists():
                    problems.append(f"missing adapter {spec['dir']}")
            for mkey, mspec in METHODS.items():
                cfg = dict(tmpl)
                cfg["method"] = mspec["method"]
                cfg["method_kwargs"] = dict(mspec["method_kwargs"])
                if "loader" in mspec:
                    cfg["loader"] = mspec["loader"]
                else:
                    cfg.pop("loader", None)
                name = f"{base}__{mkey}__{seed}"
                out_json = OUT_RES / f"{name}.json"
                cfg["output_path"] = str(out_json)
                cfg_p = OUT_CFG / f"{name}.yaml"
                cfg_p.parent.mkdir(parents=True, exist_ok=True)
                cfg_p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                manifest.append({
                    "name": f"s3_{name}",
                    "cmd": f"python code/phase3/eval/run_eval_cell.py "
                           f"--config {cfg_p.relative_to(ROOT)}",
                    "done": str(out_json.relative_to(ROOT)),
                    "min_free_gb": 25.0,
                })
    (CFG / "seed_rdridge_regmean_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> {CFG / 'seed_rdridge_regmean_manifest.json'}")
    if problems:
        print("[gen] PROBLEMS:")
        for p in problems:
            print("  - " + p)
        return 1
    print("[gen] validation OK: all seedN adapters present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
