#!/usr/bin/env python3
"""Generate the cells of notes/prereg_regmean_real_2026-09-23.md.

72 NLL cells + 144 downstream cells:
  methods  M1 regmean_act (alpha 0.9, activation Grams)
           M2 regmean, adapter-only, alpha 0.9, no ridge (the shipped default)
           M3 regmean, adapter-only, ridge 0.01 (the audit's starting ridge)
  bases    llama31_8b mistral_7b qwen25_7b yi15_9b
  cohorts  seed1 seed2 seed3 | indep1 indep2 indep3
  bench    NLL; GSM8K (n=500); HumanEval (n=164)

Templates are the configs embedded in already-published result JSONs, so the
new cells inherit base_model, max_seq_length, seed, loader and task configs
exactly: eval_regmean_cond for NLL (plain loader, Table 4) and
eval_heur_downstream for accuracy (Table 5). Only method, kwargs, cohort and
output path change, and that is checked before anything is written.

Runs on the student account against a read-only sanjay.g tree, so the nll_tau
cache is a local copy (nll_tau_cache_v2), seeded from the published one.

Usage:
  RDMERGE_ROOT=~/projects/rdmerge_v2 python code/phase3/scripts/gen_regmean_real.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
CFG = ROOT / "code" / "phase3" / "configs"
NLL_DIR, DS_DIR = "eval_regmean_real", "eval_regmean_real_ds"
CACHE = RES / "nll_tau_cache_v2"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
SHARED = ["seed1", "seed2", "seed3"]
INDEP = ["indep1", "indep2", "indep3"]
COHORTS = SHARED + INDEP
METHODS = {
    "rma": ("regmean_act", {"alpha": 0.9, "ridge_lambda": 0.0, "n_gram_samples": 256}),
    "rmdf_a09": ("regmean", {"alpha": 0.9, "ridge_lambda": 0.0}),
    "rmdf_l0p01": ("regmean", {"ridge_lambda": 0.01}),
}
BENCH = ["gsm8k", "humaneval"]
MIN_FREE = 30.0
INVARIANT = ["base_model", "max_seq_length", "seed", "loader"]


def load_cfg(p: Path) -> dict:
    return json.loads(p.read_text())["meta"]["config"]


def retarget(specs: list[dict], src: str, dst: str) -> list[dict]:
    out = []
    for s in specs:
        s = dict(s)
        d = s["dir"].rstrip("/")
        if not d.endswith("/" + src):
            raise SystemExit(f"template dir {d} does not end in {src}")
        s["dir"] = d[: -len(src)] + dst
        if not Path(s["dir"], "adapter_model.safetensors").exists():
            raise SystemExit(f"missing adapter {s['dir']}")
        out.append(s)
    return out


def main() -> int:
    (CFG / NLL_DIR).mkdir(parents=True, exist_ok=True)
    (CFG / DS_DIR).mkdir(parents=True, exist_ok=True)
    (RES / NLL_DIR).mkdir(parents=True, exist_ok=True)
    (RES / DS_DIR).mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    manifest = []

    def add(name, cfg, cfg_dir, runner):
        p = CFG / cfg_dir / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest.append({
            "name": name,
            "cmd": f"python code/phase3/eval/{runner} --config {p.relative_to(ROOT)}",
            "done": str(Path(cfg["output_path"]).relative_to(ROOT)),
            "min_free_gb": MIN_FREE,
        })

    for base in BASES:
        nll_t = load_cfg(RES / "eval_regmean_cond" / f"{base}__rm_l0p01__seed1.json")
        for cohort in COHORTS:
            # seed the local nll_tau cache from the published one when it exists
            src_cache = RES / "nll_tau_cache" / f"{base}__{cohort}.json"
            dst_cache = CACHE / f"{base}__{cohort}.json"
            if src_cache.exists() and not dst_cache.exists():
                shutil.copy(src_cache, dst_cache)
            specs = retarget(nll_t["adapter_specs"], "seed1", cohort)
            for tag, (method, kw) in METHODS.items():
                name = f"{base}__{tag}__{cohort}"
                cfg = {k: v for k, v in nll_t.items() if k != "output_path"}
                cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs,
                           min_free_gb=MIN_FREE, nll_tau_cache=str(dst_cache),
                           output_path=str(RES / NLL_DIR / f"{name}.json"))
                for k in INVARIANT:
                    assert cfg.get(k) == nll_t.get(k), (name, k)
                add(name, cfg, NLL_DIR, "run_eval_cell.py")

        for bench in BENCH:
            ds_t = load_cfg(RES / "eval_heur_downstream"
                            / f"{base}__task_arithmetic_{bench}__seed1.json")
            for cohort in COHORTS:
                specs = retarget(ds_t["adapter_specs"], "seed1", cohort)
                for tag, (method, kw) in METHODS.items():
                    name = f"{base}__{tag}_{bench}__{cohort}"
                    cfg = {k: v for k, v in ds_t.items() if k != "output_path"}
                    cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs,
                               min_free_gb=MIN_FREE,
                               output_path=str(RES / DS_DIR / f"{name}.json"))
                    for k in INVARIANT + ["metric", "n_eval_metric", "metric_kwargs"]:
                        assert cfg.get(k) == ds_t.get(k), (name, k)
                    add(name, cfg, DS_DIR, "run_downstream_cell.py")

    n_nll = sum(1 for m in manifest if NLL_DIR + "/" in m["done"])
    n_ds = sum(1 for m in manifest if DS_DIR + "/" in m["done"])
    if (n_nll, n_ds) != (72, 144):
        raise SystemExit(f"expected 72 + 144 cells, built {n_nll} + {n_ds}")
    # NLL first: it is the cheap half and decides Q1-Q3.
    manifest.sort(key=lambda m: (DS_DIR + "/" in m["done"], m["name"]))
    out = CFG / "regmean_real_manifest.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}: {n_nll} NLL + {n_ds} downstream cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
