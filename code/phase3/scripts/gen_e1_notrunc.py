#!/usr/bin/env python3
"""E1: is the flat rate curve a rank-truncation artifact?

Rules fixed in notes/prereg_conditioning_2026-08-07.md (f9d230e). The registered
decision rule is unchanged and is reproduced here:

  FALSIFICATION VOID   if the untruncated slope enters [-2.4, -1.6] on >= 3 of 4
  FALSIFICATION STANDS if |slope| < 0.5 on >= 3 of 4
  UNRESOLVED           otherwise

DESIGN DEFECT IN THE PRE-REGISTRATION, CORRECTED HERE BY ADDING CELLS ONLY.
The prereg said "repeat the sweep on indep1 with truncation disabled, the
existing truncated sweep is the comparison arm". But the existing W3 sweep ran
on SEED1 (shared init), not indep1. Comparing indep1-untruncated against
seed1-truncated varies cohort AND truncation at once and cannot attribute the
flatness to either. No threshold or rule is being changed; we generate the
missing arm so the registered question is answerable:

  seed1  untruncated (24)  -> pairs with the EXISTING seed1 truncated sweep,
                              giving a controlled truncation contrast
  indep1 untruncated (24)  -> the arm the pre-registration actually named

48 cells. The existing seed1-truncated cells are not re-run (prereg constraint
on not re-running reference arms).

Truncation is disabled via realize="rank_deff", which the rd_encoder docstring
describes as carrying W* EXACTLY via its natural factorization, "no SVD, no
truncation". The default "rank_r" truncates to rank 16 and, per the same
docstring, "discards ~30% of solution mass on real adapters".

Usage:
  python code/phase3/scripts/gen_e1_notrunc.py
  qsub code/phase3/scripts/pbs_e1_notrunc.sh
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
BITS = [1, 2, 3, 4, 8, 16]
COHORTS = ["seed1", "indep1"]
TEMPLATE_COHORT = "seed1"

OUT_CFG = CFG / "eval_e1_notrunc"
OUT_RES = RES / "eval_e1_notrunc"


def retarget(spec: dict, cohort: str) -> dict:
    s = dict(spec)
    d = s["dir"].rstrip("/")
    assert d.endswith(TEMPLATE_COHORT), d
    s["dir"] = d[: -len(TEMPLATE_COHORT)] + cohort
    return s


def main() -> int:
    problems: list[str] = []
    manifest: list[dict] = []
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    for base in BASES:
        for b in BITS:
            tmpl_p = CFG / f"eval_w3_rate/{base}__rd_ridge_b{b}__{TEMPLATE_COHORT}.yaml"
            if not tmpl_p.exists():
                problems.append(f"missing template {tmpl_p}")
                continue
            tmpl = yaml.safe_load(tmpl_p.read_text())

            for cohort in COHORTS:
                specs = [retarget(s, cohort) for s in tmpl["adapter_specs"]]
                for s in specs:
                    if not Path(s["dir"], "adapter_model.safetensors").exists():
                        problems.append(f"missing adapter {s['dir']}")

                name = f"{base}__rd_ridge_b{b}_notrunc__{cohort}"
                cfg = dict(tmpl)
                kw = dict(tmpl.get("method_kwargs", {}))
                # the ONE change from the W3 arm: no post-merge truncation.
                # Every other kwarg (bits, c, seed, ridge_lambda) is inherited
                # from the template so the arms differ in truncation alone.
                kw["realize"] = "rank_deff"
                cfg["method_kwargs"] = kw
                cfg["adapter_specs"] = specs

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

    expect = len(BASES) * len(BITS) * len(COHORTS)
    assert len(manifest) == expect, (len(manifest), expect)
    out = CFG / "e1_notrunc_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(manifest)} cells "
          f"({len(BASES)} bases x {len(BITS)} bits x {len(COHORTS)} cohorts)")
    print("truncation disabled via realize=rank_deff; all other kwargs inherited")
    print("existing eval_w3_rate seed1 cells are the truncated arm and are NOT re-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
