#!/usr/bin/env python3
"""Cells for notes/prereg_third_solver_2026-09-23.md.

72 NLL cells (TSV-M, MaTS, diagonal Fisher x 4 bases x 6 cohorts) and 64
downstream cells (TSV-M on all six cohorts, MaTS on seed1/indep1). Templates
and invariance checks are those of gen_regmean_real.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_regmean_real import (BASES, CACHE, CFG, COHORTS, INVARIANT, MIN_FREE,  # noqa: E402
                              RES, ROOT, load_cfg, retarget)

NLL_DIR, DS_DIR = "eval_third_solver", "eval_third_solver_ds"
METHODS = {"tsv": ("tsv", {"alpha": 1.0}),
           "mats": ("mats", {"kfac": True, "num_iterations": 100, "model_lambda": 1.0,
                             "n_gram_samples": 256, "gram_max_len": 512}),
           "fisherdiag": ("fisher_avg", {})}
DS_COHORTS = {"tsv": COHORTS, "mats": ["seed1", "indep1"]}


def main() -> int:
    for d in (CFG / NLL_DIR, CFG / DS_DIR, RES / NLL_DIR, RES / DS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    manifest = []

    def add(name, cfg, cfg_dir, runner):
        p = CFG / cfg_dir / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest.append({"name": name,
                         "cmd": f"python code/phase3/eval/{runner} --config {p.relative_to(ROOT)}",
                         "done": str(Path(cfg["output_path"]).relative_to(ROOT)),
                         "min_free_gb": 45.0 if cfg["method"] == "mats" else MIN_FREE})

    for base in BASES:
        nll_t = load_cfg(RES / "eval_regmean_cond" / f"{base}__rm_l0p01__seed1.json")
        for cohort in COHORTS:
            specs = retarget(nll_t["adapter_specs"], "seed1", cohort)
            for tag, (method, kw) in METHODS.items():
                name = f"{base}__{tag}__{cohort}"
                cfg = {k: v for k, v in nll_t.items() if k != "output_path"}
                cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs,
                           min_free_gb=MIN_FREE,
                           nll_tau_cache=str(CACHE / f"{base}__{cohort}.json"),
                           output_path=str(RES / NLL_DIR / f"{name}.json"))
                for k in INVARIANT:
                    assert cfg.get(k) == nll_t.get(k), (name, k)
                add(name, cfg, NLL_DIR, "run_eval_cell.py")
        for bench in ("gsm8k", "humaneval"):
            ds_t = load_cfg(RES / "eval_heur_downstream"
                            / f"{base}__task_arithmetic_{bench}__seed1.json")
            for tag in ("tsv", "mats"):
                method, kw = METHODS[tag]
                for cohort in DS_COHORTS[tag]:
                    specs = retarget(ds_t["adapter_specs"], "seed1", cohort)
                    name = f"{base}__{tag}_{bench}__{cohort}"
                    cfg = {k: v for k, v in ds_t.items() if k != "output_path"}
                    cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs,
                               min_free_gb=MIN_FREE,
                               output_path=str(RES / DS_DIR / f"{name}.json"))
                    if tag == "mats":
                        cfg["loader"] = "plain"      # the Fisher needs a backward pass
                    add(name, cfg, DS_DIR, "run_downstream_cell.py")

    n_nll = sum(NLL_DIR + "/" in m["done"] for m in manifest)
    n_ds = sum(DS_DIR + "/" in m["done"] for m in manifest)
    if (n_nll, n_ds) != (72, 64):
        raise SystemExit(f"expected 72 + 64 cells, built {n_nll} + {n_ds}")
    manifest.sort(key=lambda m: (DS_DIR + "/" in m["done"], m["name"]))
    out = CFG / "third_solver_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}: {n_nll} NLL + {n_ds} downstream cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
