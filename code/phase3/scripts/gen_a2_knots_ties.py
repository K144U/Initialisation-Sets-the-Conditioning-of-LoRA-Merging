#!/usr/bin/env python3
"""A2: re-run KnOTS with an inner merge that is not algebraically Task Arithmetic.

As shipped, every KnOTS cell used the registry default
inner_combination="linear", which makes the method exactly TA:
Delta_t V V^T = Delta_t, so the projection round trip cancels and the linear
inner merge reduces to sum_t w_t Delta_t. Verified to 3e-06 in
merging/tests/test_knots.py.

The paper cites the resulting "KnOTS ~ TA" agreement as evidence FOR its theory
in four places:
  - intro, "subspace-alignment merging (KnOTS) has nothing to exploit and is
    statistically indistinguishable from naive averaging"
  - 6.2 finding (3), "exactly as the theory predicts at zero subspace overlap"
  - related work, "we measure KnOTS indistinguishable from Task Arithmetic"
  - App. J, "tracks TA exactly at T = 7 on Llama"
None of those survive: the implementation could not have differed from TA.

This generates the KnOTS-TIES variant (the one the KnOTS paper headlines) on
the T = 4 matrix, matched seed1/2/3, so Table 1's KnOTS row and 6.2 finding (3)
can be restated against a method that is actually a subspace-alignment merge.

  4 bases x 3 seeds = 12 cells, ~12-18 min each.

The density is pinned to the TIES default (0.2) so the KnOTS-TIES row is
comparable to the plain TIES row on the same page.

Follow-up not generated here: the T-scaling pool also carries KnOTS cells
(App. D, "tracks TA exactly at T = 7"). Those need the same treatment, 3 bases
x 3 T-values x subsets, before that sentence can be restated.
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
SEEDS = ["seed1", "seed2", "seed3"]
DENSITY = 0.2

OUT_CFG = CFG / "eval_a2_knots_ties"
OUT_RES = RES / "eval_a2_knots_ties"


def main() -> int:
    manifest: list[dict] = []
    problems: list[str] = []

    for base in BASES:
        for seed in SEEDS:
            tmpl_p = CFG / f"eval_matrix_seeds/{base}__knots__{seed}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            cfg = yaml.safe_load(tmpl_p.read_text())
            for spec in cfg["adapter_specs"]:
                if not Path(spec["dir"]).exists():
                    problems.append(f"missing adapter {spec['dir']}")
            cfg["method"] = "knots"
            cfg["method_kwargs"] = {"inner_combination": "ties",
                                    "density": DENSITY}
            cfg.pop("loader", None)
            name = f"{base}__knots_ties__{seed}"
            out_json = OUT_RES / f"{name}.json"
            cfg["output_path"] = out_json.as_posix()
            p = OUT_CFG / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({
                "name": f"a2_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py "
                       f"--config {p.relative_to(ROOT).as_posix()}",
                "done": out_json.relative_to(ROOT).as_posix(),
                "min_free_gb": 25.0,
            })

    man_p = CFG / "a2_knots_ties_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> {man_p}")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems))[:10]:
            print("  - " + p)
        return 1
    print("[gen] validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
