#!/usr/bin/env python3
"""R1 cells: does cohort conditioning affect a solver we did NOT build?

Rules fixed in notes/prereg_tmlr_2026-08-14.md (acebd1a) and amended in
notes/prereg_tmlr_amendment_2026-08-14.md (7ce15b1), both committed BEFORE this
generator existed.

The paper currently claims that merging methods which solve a linear system are
severely affected by cohort conditioning, and names RegMean, LoRM and RegMean++
as that family. The only solver swept is ours. This sweeps RegMean, in the
data-free adapter-only form already in the tree, on the same grid and the same
two cohorts.

  4 bases x 7 lambdas x 2 cohorts (seed1 shared, indep1 independent) = 56 cells

The grid starts at 1e-6, not 0: RegMean solves in the full input dimension
against sum_t A_t^T A_t, whose rank is at most T*r = 64 against in_dim 4096, so
lambda = 0 is singular rather than ill-conditioned and returns null-space noise
without raising. See the amendment and probe_regmean_lambda0.py.

BOTH ARMS ARE BUILT FROM ONE TEMPLATE, the indep1 rd_ridge config, with only
the adapter cohort directory swapped, exactly as gen_ridge_cond.py does it. That
keeps tasks, n_eval, max_seq_length, evaluation seed and the VRAM gate byte
identical across arms, so the only difference is the cohort and the ridge.

method_kwargs is replaced wholesale rather than updated: the template's kwargs
are rd_encoder's (bits, c, seed, realize) and mean nothing to RegMean. RegMean
SVD-truncates to rank r internally, so rank parity with the baselines holds by
construction and needs no `realize` pin.

Usage:
  python code/phase3/scripts/gen_regmean_cond.py
  qsub code/phase3/scripts/pbs_regmean_cond.sh
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
# (value, filename tag). Grid fixed in the pre-registration as amended.
LAMBDAS = [(1e-6, "1em6"), (0.01, "0p01"), (0.03, "0p03"), (0.05, "0p05"),
           (0.13, "0p13"), (0.30, "0p30"), (1.00, "1p00")]
COHORTS = ["seed1", "indep1"]      # shared, independent
TEMPLATE_COHORT = "indep1"

OUT_CFG = CFG / "eval_regmean_cond"
OUT_RES = RES / "eval_regmean_cond"

# Keys that must be identical between the two arms. Protocol item 1 of the
# pre-registration: the generator asserts the arms differ only in the intended
# variable and refuses to write cells otherwise. The task definitions are
# checked separately, since they live inside adapter_specs alongside the one
# field that is SUPPOSED to differ.
ARM_INVARIANT = ["base_model", "max_seq_length", "seed", "min_free_gb",
                 "loader"]


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
    smoke: list[dict] = []
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    for base in BASES:
        tmpl_p = CFG / f"eval_a1_indep/{base}__rd_ridge__{TEMPLATE_COHORT}.yaml"
        if not tmpl_p.exists():
            problems.append(f"missing template {tmpl_p}")
            continue
        tmpl = yaml.safe_load(tmpl_p.read_text())

        written: dict[str, dict] = {}
        for cohort in COHORTS:
            specs = [retarget(s, cohort) for s in tmpl["adapter_specs"]]
            for s in specs:
                if not Path(s["dir"], "adapter_model.safetensors").exists():
                    problems.append(f"missing adapter {s['dir']}")

            for lam, tag in LAMBDAS:
                name = f"{base}__rm_l{tag}__{cohort}"
                cfg = dict(tmpl)
                cfg["method"] = "regmean"
                cfg["method_kwargs"] = {"ridge_lambda": lam}
                cfg["adapter_specs"] = specs

                out_p = OUT_RES / f"{name}.json"
                cfg["output_path"] = str(out_p)
                # nll_tau depends on (base, cohort) and nothing else in this
                # sweep, so seven lambda cells share one cache entry. The key
                # inside the file is what actually gates reuse; this path only
                # decides which cells can possibly share.
                cfg["nll_tau_cache"] = str(
                    RES / "nll_tau_cache" / f"{base}__{cohort}.json")
                cfg_p = OUT_CFG / f"{name}.yaml"
                cfg_p.write_text(yaml.safe_dump(cfg, sort_keys=False))
                written[f"{cohort}|{tag}"] = cfg

                entry = {
                    "name": name,
                    "cmd": ("python code/phase3/eval/run_eval_cell.py "
                            f"--config {cfg_p.relative_to(ROOT)}"),
                    "done": str(out_p.relative_to(ROOT)),
                    "min_free_gb": 25.0,
                }
                manifest.append(entry)
                # Smoke cell: one cell, at a lambda that is NOT the minimally
                # regularised reference, so the smoke does not sit on the point
                # P2 is read at. Prereg protocol item 2.
                if base == BASES[0] and cohort == "indep1" and tag == "0p30":
                    smoke.append(entry)

        # Protocol item 1: the arms must differ ONLY in the cohort.
        for _, tag in LAMBDAS:
            a = written.get(f"{COHORTS[0]}|{tag}")
            b = written.get(f"{COHORTS[1]}|{tag}")
            if not a or not b:
                continue
            for key in ARM_INVARIANT:
                if a.get(key) != b.get(key):
                    problems.append(
                        f"{base} lambda {tag}: arms differ in {key!r}: "
                        f"{a.get(key)!r} vs {b.get(key)!r}")
            if a["method_kwargs"] != b["method_kwargs"]:
                problems.append(f"{base} lambda {tag}: method_kwargs differ")

            # adapter_specs must agree on everything except the cohort dir:
            # same task order, same task_cfg (so same eval data and n_eval).
            sa, sb = a["adapter_specs"], b["adapter_specs"]
            if [s["name"] for s in sa] != [s["name"] for s in sb]:
                problems.append(f"{base} lambda {tag}: task order differs")
            for x, y in zip(sa, sb):
                if x.get("task_cfg") != y.get("task_cfg"):
                    problems.append(
                        f"{base} lambda {tag}: task_cfg differs for {x['name']}")
            dirs_a = {s["dir"] for s in sa}
            dirs_b = {s["dir"] for s in sb}
            if dirs_a == dirs_b:
                problems.append(f"{base} lambda {tag}: both arms share a cohort")

    if problems:
        print("PROBLEMS, no manifest written:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2

    expect = len(BASES) * len(LAMBDAS) * len(COHORTS)
    assert len(manifest) == expect, (len(manifest), expect)
    assert len(smoke) == 1, len(smoke)

    # Order matters now that nll_tau is cached. Put one cell per (base, cohort)
    # first, so the eight cache entries are populated by eight cells running on
    # different cohorts, and every later cell is a hit. Without this the five
    # workers can start two cells of the same cohort at once and both pay the
    # full twenty evaluations. The orchestrator pulls in manifest order.
    seen: set[str] = set()
    first, rest = [], []
    for entry in manifest:
        cohort_key = entry["name"].split("__")[0] + entry["name"].split("__")[2]
        (rest if cohort_key in seen else first).append(entry)
        seen.add(cohort_key)
    assert len(first) == len(BASES) * len(COHORTS), len(first)
    manifest = first + rest

    out = CFG / "regmean_cond_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    smoke_out = CFG / "regmean_cond_smoke_manifest.json"
    smoke_out.write_text(json.dumps(smoke, indent=2))

    print(f"wrote {out} with {len(manifest)} cells "
          f"({len(BASES)} bases x {len(LAMBDAS)} lambdas x {len(COHORTS)} cohorts)")
    print(f"wrote {smoke_out} with 1 cell: {smoke[0]['name']}")
    print("lambdas:", [l for l, _ in LAMBDAS])
    print("arm-invariance asserted on:", ", ".join(ARM_INVARIANT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
