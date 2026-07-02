#!/usr/bin/env python3
"""Generate eval configs + manifests for two robustness sweeps.

  A) TIES/DARE density sweep   -- fairness check vs the fixed density=0.2
                                  used in the main matrix.
  B) RegMean ridge_lambda sweep -- find RegMean's *best* regularizer and
                                  test whether rd-encoder ridge still wins
                                  (defends or retracts the centroid claim).

Reuses existing matrix / e12 configs as templates; only overrides
method_kwargs + output_path. Idempotent (rewrites each run). Validates
that every referenced base model + adapter dir exists before emitting a
manifest, so a job never fails mid-queue on a missing artifact.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/sanjay.g/projects/rdmerge")
CFG = ROOT / "code/phase3/configs"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]

# ---- Experiment A: density sweep (0.2 reused from existing matrix cells) ----
DENS_METHODS = ["ties", "dare"]
DENSITIES = [0.05, 0.1, 0.3, 0.5]

# ---- Experiment B: regmean lambda sweep (1e-3 reused from existing e12) ----
LAMBDAS = [1e-4, 1e-2, 1e-1, 1.0, 10.0]


def dtag(d: float) -> str:
    return "d" + str(d).replace(".", "p")


def ltag(l: float) -> str:
    e = round(math.log10(l))
    return f"lam1e{'m' if e < 0 else ''}{abs(e)}"


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def dump_yaml(obj: dict, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False))


def check_artifacts(cfg: dict, problems: list[str], where: str) -> None:
    bm = Path(cfg["base_model"])
    if not bm.exists():
        problems.append(f"{where}: missing base_model {bm}")
    for spec in cfg.get("adapter_specs", []):
        d = Path(spec["dir"])
        if not d.exists():
            problems.append(f"{where}: missing adapter dir {d}")


def main() -> int:
    problems: list[str] = []

    # ===== Experiment A =====
    out_dir_A = CFG / "eval_density_sweep"
    res_dir_A = RES / "eval_density_sweep"
    manifest_A = []
    for base in BASES:
        for method in DENS_METHODS:
            tmpl_p = CFG / "eval_matrix_seeds" / f"{base}__{method}__seed1.yaml"
            if not tmpl_p.exists():
                problems.append(f"A: missing template {tmpl_p}")
                continue
            tmpl = load_yaml(tmpl_p)
            for d in DENSITIES:
                cfg = dict(tmpl)
                cfg["method_kwargs"] = {"density": float(d)}
                name = f"{base}__{method}__{dtag(d)}__seed1"
                out_json = res_dir_A / f"{name}.json"
                cfg["output_path"] = str(out_json)
                cfg_p = out_dir_A / f"{name}.yaml"
                dump_yaml(cfg, cfg_p)
                check_artifacts(cfg, problems, name)
                manifest_A.append({
                    "name": f"dens_{name}",
                    "cmd": f"python code/phase3/eval/run_eval_cell.py "
                           f"--config {cfg_p.relative_to(ROOT)}",
                    "done": str(out_json.relative_to(ROOT)),
                    "min_free_gb": 25.0,
                })
    (CFG / "density_sweep_manifest.json").write_text(json.dumps(manifest_A, indent=2))

    # ===== Experiment B =====
    out_dir_B = CFG / "eval_regmean_lambda"
    res_dir_B = RES / "eval_regmean_lambda"
    manifest_B = []
    for base in BASES:
        tmpl_p = CFG / "eval_e12_regmean_adamerging" / f"{base}__regmean__seed1.yaml"
        if not tmpl_p.exists():
            problems.append(f"B: missing template {tmpl_p}")
            continue
        tmpl = load_yaml(tmpl_p)
        for l in LAMBDAS:
            cfg = dict(tmpl)
            cfg["method_kwargs"] = {"ridge_lambda": float(l)}
            name = f"{base}__regmean__{ltag(l)}__seed1"
            out_json = res_dir_B / f"{name}.json"
            cfg["output_path"] = str(out_json)
            cfg_p = out_dir_B / f"{name}.yaml"
            dump_yaml(cfg, cfg_p)
            check_artifacts(cfg, problems, name)
            manifest_B.append({
                "name": f"rgm_{name}",
                "cmd": f"python code/phase3/eval/run_eval_cell.py "
                       f"--config {cfg_p.relative_to(ROOT)}",
                "done": str(out_json.relative_to(ROOT)),
                "min_free_gb": 25.0,
            })
    (CFG / "regmean_lambda_sweep_manifest.json").write_text(json.dumps(manifest_B, indent=2))

    print(f"[gen] Experiment A density sweep: {len(manifest_A)} cells "
          f"-> {CFG / 'density_sweep_manifest.json'}")
    print(f"[gen] Experiment B regmean-lambda sweep: {len(manifest_B)} cells "
          f"-> {CFG / 'regmean_lambda_sweep_manifest.json'}")
    print(f"[gen] densities={DENSITIES} (+0.2 reused)  lambdas={LAMBDAS} (+1e-3 reused)")

    if problems:
        print("\n[gen] VALIDATION PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 1
    print("[gen] validation OK: all base models + adapter dirs exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
