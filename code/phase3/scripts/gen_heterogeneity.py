#!/usr/bin/env python3
"""Cells of notes/prereg_heterogeneity_2026-10-19.md: Part G (48 + 12 references)
and Part M (16 + 8 references).

Part G uses the copies written by make_perturbed_cohorts.py. Perturbations that
leave every update unchanged (gauge, canon) reuse the unperturbed nll_tau cache,
since the individual adapters' NLLs are then identical; the others get their own.
Part M builds mixed cohorts from existing seed1 and indep1 adapters.
"""
from __future__ import annotations

import sys
from pathlib import Path

from _c1019 import (BASES, CACHE, INVARIANT, RES, ROOT, Manifest, retarget,
                    template)

G_DIR, M_DIR = "eval_hetero_g", "eval_hetero_m"
G_BASES = ["llama31_8b", "qwen25_7b"]
G_COHORTS = ["indep1", "indep2", "indep3"]
PERT_RM = ["gauge", "scale", "canon", "rank", "module"]
PERT_TA = ["scale", "rank", "module"]
SAME_UPDATE = {"gauge", "canon"}
RM = ("regmean", {"ridge_lambda": 1e-6})
TA = ("task_arithmetic", {})
MIXES = {"M1": {"gsm8k": "seed1", "magicoder": "seed1", "alpaca": "indep1", "translation": "indep1"},
         "M2": {"gsm8k": "seed1", "magicoder": "seed1", "alpaca": "seed1", "translation": "indep1"}}


def cell(m, t, name, method, kw, specs, cache, out_dir):
    cfg = {k: v for k, v in t.items() if k != "output_path"}
    cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs,
               nll_tau_cache=str(cache), output_path=str(RES / out_dir / f"{name}.json"))
    for k in INVARIANT:
        assert cfg.get(k) == t.get(k), (name, k)
    m.add(name, cfg, out_dir, "run_eval_cell.py")


def perturbed_specs(specs, cohort, pert):
    out = []
    for s in specs:
        s = dict(s)
        d = Path(s["dir"])
        task_dir = d.parent.name
        s["dir"] = str(ROOT / "artifacts" / "lora_perturbed" / d.parent.parent.name
                       / task_dir / f"{cohort}__{pert}")
        if not Path(s["dir"], "adapter_model.safetensors").exists():
            raise SystemExit(f"missing perturbed adapter {s['dir']}")
        out.append(s)
    return out


def main() -> int:
    m = Manifest()
    for base in G_BASES:
        t = template(RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json")
        for cohort in G_COHORTS:
            base_specs = retarget(t["adapter_specs"], "seed1", cohort)
            # unperturbed references, recomputed on the re-staged models (the
            # registration assumed they existed; only indep1 RegMean(1e-6) did)
            ref_cache = CACHE / f"{base}__{cohort}.json"
            cell(m, t, f"{base}__rm1em6_ref__{cohort}", *RM, base_specs, ref_cache, G_DIR)
            cell(m, t, f"{base}__ta_ref__{cohort}", *TA, base_specs, ref_cache, G_DIR)
            for pert in PERT_RM:
                specs = perturbed_specs(base_specs, cohort, pert)
                cache = (CACHE / f"{base}__{cohort}.json" if pert in SAME_UPDATE
                         else CACHE / f"{base}__{cohort}__{pert}.json")
                cell(m, t, f"{base}__rm1em6_{pert}__{cohort}", *RM, specs, cache, G_DIR)
                if pert in PERT_TA:
                    cell(m, t, f"{base}__ta_{pert}__{cohort}", *TA, specs, cache, G_DIR)
    for base in BASES:
        t = template(RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json")
        for mix, plan in MIXES.items():
            specs = []
            for s in t["adapter_specs"]:
                specs += retarget([s], "seed1", plan[s["name"]])
            cache = CACHE / f"{base}__{mix}.json"
            cell(m, t, f"{base}__rm1em6__{mix}", *RM, specs, cache, M_DIR)
            cell(m, t, f"{base}__ta__{mix}", *TA, specs, cache, M_DIR)
        ref = retarget(t["adapter_specs"], "seed1", "indep1")
        cell(m, t, f"{base}__rm1em6__indep1", *RM, ref, CACHE / f"{base}__indep1.json", M_DIR)
        cell(m, t, f"{base}__ta__indep1", *TA, ref, CACHE / f"{base}__indep1.json", M_DIR)
    m.write("heterogeneity_manifest.json", expect=48 + 12 + 16 + 8)
    return 0


if __name__ == "__main__":
    sys.exit(main())
