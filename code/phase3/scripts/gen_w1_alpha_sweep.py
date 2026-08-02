#!/usr/bin/env python3
"""W1: the merge-coefficient control the review asks for.

The reviewer's objection: as lambda grows, (Hbar + lambda I)^{-1} -> lambda^{-1} I,
so rd-encoder ridge tends toward a scaled-down projected Task Arithmetic, and the
paper never sweeps TA's own merge coefficient. Until Table 1 has a scaling-tuned
TA row, the contribution cannot be distinguished from rediscovering shrinkage.

Offline reconstruction of W*(lambda) (notes/review_response_2026-08-02.md) says
the mechanism reading is wrong at the operating point: at lambda* the method
AMPLIFIES (||W*||/||TA|| = 1.3x to 2.4x, best alpha 1.02 to 1.31) and 62 to 84
percent of its mass is orthogonal to every rescaling of TA. But the control is
still mandatory, because TA is pinned at alpha = 1/T with no sweep while rd-ridge
gets a tuned lambda. This script generates it.

Three cell groups, all on matched seed1/2/3 adapters, all templated off the
existing eval_matrix_seeds configs so base_model and adapter_specs are identical:

  A. ta_alpha   TA with weights [alpha]*T, alpha over 7 values, 4 bases, seed1.
                alpha = 0.25 reproduces the paper's default and is the control
                that must match Table 1's TA row.                    28 cells
  B. rd_renorm  rd-encoder ridge at lambda*, renorm="ta": same direction, TA's
                Frobenius norm per layer. Isolates scale from direction.
                4 bases x 3 seeds.                                   12 cells
  C. rd_rank16  rd-encoder ridge at lambda*, realize="rank_r": rank 16 like
                every baseline, instead of the published rank_deff (= Tr = 64).
                Answers the separate storage-parity objection (audit A3).
                Llama and Yi seed1 already exist in eval_e11_quadbridge at
                alpha=1.0, so only Mistral and Qwen are strictly new, but all
                four are regenerated here for a clean matched set.
                4 bases x 3 seeds.                                   12 cells

Total 52 cells, ~12 to 18 min each.

Interpretation fixed in advance, so this is not a fishing expedition:
  - If best-alpha TA reaches rd-ridge's worst-task excess on any base, W1 lands
    and the contribution needs restating.
  - If renorm="ta" leaves rd-ridge's excess essentially unchanged, the win is
    about direction, not scale, and W1 is answered.
  - If rank16 rd-ridge stays below every baseline, the storage objection is
    answered too.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

# Defaults to the cluster path. Override to dry-run the generator anywhere
# (e.g. RDMERGE_ROOT=/tmp/scratch) without a cluster checkout; the adapter
# paths inside the emitted configs come from the templates either way.
ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SEEDS = ["seed1", "seed2", "seed3"]

# Per-task TA coefficient. The paper's default is 1/T = 0.25; the standard
# Task Arithmetic sweep range in the literature is roughly 0.1 to 1.0.
ALPHAS = [0.10, 0.15, 0.25, 0.35, 0.50, 0.75, 1.00]

# Base-optimal ridge from the published sweep.
LAMBDA_STAR = {"llama31_8b": 0.05, "mistral_7b": 0.13,
               "qwen25_7b": 0.13, "yi15_9b": 0.13}

OUT_CFG = CFG / "eval_w1_alpha"
OUT_RES = RES / "eval_w1_alpha"


def _tag(x: float) -> str:
    return str(x).replace(".", "p").rstrip("0").rstrip("p") or "0"


def main() -> int:
    manifest: list[dict] = []
    problems: list[str] = []

    def emit(name: str, cfg: dict) -> None:
        out_json = OUT_RES / f"{name}.json"
        cfg["output_path"] = out_json.as_posix()
        cfg_p = OUT_CFG / f"{name}.yaml"
        cfg_p.parent.mkdir(parents=True, exist_ok=True)
        cfg_p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        # as_posix(): the generator may be run from Windows, but the manifest
        # is consumed by the orchestrator on the Linux cluster.
        manifest.append({
            "name": f"w1_{name}",
            "cmd": f"python code/phase3/eval/run_eval_cell.py "
                   f"--config {cfg_p.relative_to(ROOT).as_posix()}",
            "done": out_json.relative_to(ROOT).as_posix(),
            "min_free_gb": 25.0,
        })

    for base in BASES:
        for seed in SEEDS:
            tmpl_p = CFG / f"eval_matrix_seeds/{base}__ties__{seed}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            tmpl = yaml.safe_load(tmpl_p.read_text())
            for spec in tmpl["adapter_specs"]:
                if not Path(spec["dir"]).exists():
                    problems.append(f"missing adapter {spec['dir']}")
            n_tasks = len(tmpl["adapter_specs"])

            # A. TA merge-coefficient sweep (seed1 only; matches Table 1's basis)
            if seed == "seed1":
                for a in ALPHAS:
                    cfg = dict(tmpl)
                    cfg["method"] = "task_arithmetic"
                    cfg["method_kwargs"] = {}
                    cfg["weights"] = [float(a)] * n_tasks
                    cfg.pop("loader", None)
                    emit(f"{base}__ta_alpha{_tag(a)}__{seed}", cfg)

            lam = LAMBDA_STAR[base]

            # B. norm-matched rd-encoder ridge
            cfg = dict(tmpl)
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": 32, "seed": 20260611,
                                    "realize": "rank_deff",
                                    "ridge_lambda": lam, "renorm": "ta"}
            cfg["loader"] = "plain"
            cfg.pop("weights", None)
            emit(f"{base}__rd_renorm__{seed}", cfg)

            # C. rank-16 rd-encoder ridge (storage parity with the baselines)
            cfg = dict(tmpl)
            cfg["method"] = "rd_encoder"
            cfg["method_kwargs"] = {"bits": 32, "seed": 20260611,
                                    "realize": "rank_r", "ridge_lambda": lam}
            cfg.pop("loader", None)      # uniform rank 16 -> unsloth path is fine
            cfg.pop("weights", None)
            emit(f"{base}__rd_rank16__{seed}", cfg)

    man_p = CFG / "w1_alpha_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2))
    print(f"[gen] {len(manifest)} cells -> {man_p}")
    n_ta = sum(1 for m in manifest if "ta_alpha" in m["name"])
    print(f"[gen]   {n_ta} TA alpha cells, "
          f"{sum(1 for m in manifest if 'rd_renorm' in m['name'])} renorm cells, "
          f"{sum(1 for m in manifest if 'rd_rank16' in m['name'])} rank16 cells")
    if problems:
        print("[gen] PROBLEMS:")
        for p in sorted(set(problems)):
            print("  - " + p)
        return 1
    print("[gen] validation OK: all templates and adapters present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
