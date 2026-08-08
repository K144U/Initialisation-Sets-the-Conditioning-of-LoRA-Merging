#!/usr/bin/env python3
"""Fix the W3 rate-sweep asymptote. Three cells, no new method.

BUG. analyze_w3_rate.py fits log2(excess(b) - excess(inf)) against b, but
excess_inf() reads "the published 3-seed rd-ridge mean" from a DIFFERENT
directory with a DIFFERENT ridge:

    eval_seed_rdridge_regmean/{base}__rd_ridge__{seed}.json   ridge_lambda 0.05
    eval_ridge_seed/{base}__ridge_l0p05__{seed}.json          ridge_lambda 0.05

while the sweep cells are eval_w3_rate/{base}__rd_ridge_b{b}__seed1.json at
per-base lambda* (0.05 Llama, 0.13 Mistral/Qwen/Yi) on seed1 alone. So the
asymptote differs from the curve in TWO ways:

  1. lambda mismatch on 3 of 4 bases: the reference is a different method.
  2. a 3-seed mean compared against a single-seed curve.

That is why Llama and Yi produced finite-rate excess BELOW the b=infinity
value, which the paper reported as "impossible under the model". It is not
impossible; the reference was wrong. This is the same class of error as the
W1s "best of three against a 3-seed mean" mismatch already recorded in
decisions.md.

FIX. Generate the matched asymptote: bits=32, realize=rank_deff, seed1, at
each base's own lambda*. Llama already has one (lambda* = 0.05 in
eval_seed_rdridge_regmean, seed1). Mistral, Qwen and Yi need lambda = 0.13
and do not have it: eval_ridge_seed carries l0p13 for Llama only.

Each config is copied from that base's own W3 sweep cell with bits set to 32
and nothing else touched, so the asymptote differs from the curve in rate
alone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

# Llama's matched asymptote already exists at lambda* = 0.05.
NEEDED = ["mistral_7b", "qwen25_7b", "yi15_9b"]
OUT_CFG = CFG / "eval_w3_asymptote"
OUT_RES = RES / "eval_w3_asymptote"


def main() -> int:
    problems, manifest = [], []
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    for base in NEEDED:
        src = CFG / f"eval_w3_rate/{base}__rd_ridge_b2__seed1.yaml"
        if not src.exists():
            problems.append(f"missing {src}")
            continue
        cfg = yaml.safe_load(src.read_text())
        kw = dict(cfg["method_kwargs"])
        lam = kw.get("ridge_lambda")
        if lam != 0.13:
            problems.append(f"{base}: expected lambda* 0.13, found {lam}")
        # the ONE change: infinite rate. lambda*, realize and cohort inherited.
        kw["bits"] = 32
        cfg["method_kwargs"] = kw

        name = f"{base}__rd_ridge_binf__seed1"
        out_p = OUT_RES / f"{name}.json"
        cfg["output_path"] = str(out_p)
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
        print("PROBLEMS:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2

    out = CFG / "w3_asymptote_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(manifest)} cells")
    for m in manifest:
        print("  ", m["name"])
    print("each differs from its own W3 sweep cell in bits only (2 -> 32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
