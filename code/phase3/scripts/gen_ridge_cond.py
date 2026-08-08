#!/usr/bin/env python3
"""E2 cells: does the optimal ridge track the conditioning?

Rules fixed in notes/prereg_conditioning_2026-08-07.md, committed at f9d230e
BEFORE any of these cells were generated or dispatched. Summary:

  The exact instance floor is ZERO in both regimes (2fdb223), so the paper's
  floor story is dead. What actually differs is the conditioning of Hbar:
  cond 17.6k-83k under shared init against 1.5-1.6 under independent init, and
  an exact-interpolator norm amplification of 43-64 against exactly 4.0 = T.

  A Tikhonov ridge is a conditioning fix. So the registered prediction is that
  lambda* is large on the shared arm and collapses toward zero on the
  independent one, which would explain the otherwise unexplained Q1' result.

  4 bases x 7 lambdas x 2 cohorts (seed1 shared, indep1 independent) = 56 cells

BOTH ARMS ARE BUILT FROM THE SAME TEMPLATE, the indep1 rd_ridge config, with
only the adapter cohort directory swapped. That is deliberate: it makes the two
arms byte-identical on the eval side (tasks, n_eval, max_seq_length, data seed,
min_free_gb), so the only difference is the cohort and the ridge value.

`realize` is pinned to "rank_r" in EVERY cell, overriding the template's
"rank_deff". The published rd_ridge cells were rank 64 while the baselines were
rank 16 (audit finding A3), and this sweep must not inherit that confound.

Usage:
  python code/phase3/scripts/gen_ridge_cond.py
  qsub code/phase3/scripts/pbs_ridge_cond.sh
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
# (value, filename tag). Grid fixed in the pre-registration.
LAMBDAS = [(0.0, "0"), (0.01, "0p01"), (0.03, "0p03"), (0.05, "0p05"),
           (0.13, "0p13"), (0.30, "0p30"), (1.00, "1p00")]
COHORTS = ["seed1", "indep1"]      # shared, independent
TEMPLATE_COHORT = "indep1"

OUT_CFG = CFG / "eval_ridge_cond"
OUT_RES = RES / "eval_ridge_cond"


def retarget(spec: dict, cohort: str) -> dict:
    """Point an adapter spec at `cohort`, leaving everything else alone."""
    s = dict(spec)
    d = s["dir"]
    assert d.rstrip("/").endswith(TEMPLATE_COHORT), d
    s["dir"] = d.rstrip("/")[: -len(TEMPLATE_COHORT)] + cohort
    return s


def main() -> int:
    problems: list[str] = []
    manifest: list[dict] = []
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    for base in BASES:
        tmpl_p = CFG / f"eval_a1_indep/{base}__rd_ridge__{TEMPLATE_COHORT}.yaml"
        if not tmpl_p.exists():
            problems.append(f"missing template {tmpl_p}")
            continue
        tmpl = yaml.safe_load(tmpl_p.read_text())

        for cohort in COHORTS:
            specs = [retarget(s, cohort) for s in tmpl["adapter_specs"]]
            for s in specs:
                if not Path(s["dir"], "adapter_model.safetensors").exists():
                    problems.append(f"missing adapter {s['dir']}")

            for lam, tag in LAMBDAS:
                name = f"{base}__rd_l{tag}__{cohort}"
                cfg = dict(tmpl)
                cfg["method"] = "rd_encoder"
                kw = dict(tmpl.get("method_kwargs", {}))
                kw["ridge_lambda"] = lam
                kw["realize"] = "rank_r"     # prereg: pinned in BOTH arms
                cfg["method_kwargs"] = kw
                cfg["adapter_specs"] = specs

                # Three things here are load-bearing and the first version of
                # this generator got all three wrong; the smoke caught it.
                #  1. run_eval_cell.py takes ONLY --config. There is no --out.
                #     The destination is read from `output_path` in the YAML.
                #  2. orchestrator.py runs cmd with shell=True, so cmd must be
                #     a STRING. A list under shell=True executes only its first
                #     element, i.e. a bare `python`, which reads EOF from stdin
                #     and exits 0 having written nothing. That shows up as
                #     "FAIL ... rc=0 (0 min)" with an empty per-cell log.
                #  3. `done` is relative to ROOT, and the orchestrator's VRAM
                #     gate needs min_free_gb on the manifest entry.
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

    expect = len(BASES) * len(LAMBDAS) * len(COHORTS)
    assert len(manifest) == expect, (len(manifest), expect)
    out = CFG / "ridge_cond_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out} with {len(manifest)} cells "
          f"({len(BASES)} bases x {len(LAMBDAS)} lambdas x {len(COHORTS)} cohorts)")
    print("lambdas:", [l for l, _ in LAMBDAS])
    print("realize pinned to rank_r in every cell (prereg, avoids audit A3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
