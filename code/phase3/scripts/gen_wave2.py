#!/usr/bin/env python3
"""Wave 2 cells: R7 (untruncated) and R8 (the ridge sweep gets its gate).

Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), sections R7 and R8, committed
before this generator existed.

R7. Section 6.4 admits that theory, geometry and experiment constrain three
different objects, and that we expect the conditioning story to survive rank
truncation but have not shown it. This is the E2 sweep with `realize` set to
rank_deff instead of rank_r, so the truncated sweep already on disk is the
comparison arm and is not re-run. 4 bases x 7 lambdas x 2 cohorts = 56 cells.

R8. E2 has one cohort per arm, so the pre-registered 2 x SE gate could not be
computed and only the tie threshold was applied. This extends it to indep2 and
indep3, giving n = 3 on the independent arm. Same method, same grid, same
rank_r pin as E2. 4 bases x 7 lambdas x 2 cohorts = 56 cells.

Both reuse the nll_tau cache. R7 shares seed1 and indep1 with the sweeps
already run, so every one of its cells is a cache hit and costs about 10
minutes instead of 57. R8's cohorts are new, so its first eight cells build
their cache entries and the remaining 48 hit.

Usage:
  python code/phase3/scripts/gen_wave2.py
  qsub -v ARM=r7 code/phase3/scripts/pbs_wave2.sh
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
LAMBDAS = [(0.0, "0"), (0.01, "0p01"), (0.03, "0p03"), (0.05, "0p05"),
           (0.13, "0p13"), (0.30, "0p30"), (1.00, "1p00")]
TEMPLATE_COHORT = "indep1"

# arm -> (cohorts, realize, output dir, prefix)
ARMS = {
    "r7": (["seed1", "indep1"], "rank_deff", "eval_ridge_untrunc", "rdu"),
    "r8": (["indep2", "indep3"], "rank_r", "eval_ridge_cohorts", "rdc"),
}
ARM_INVARIANT = ["base_model", "max_seq_length", "seed", "min_free_gb",
                 "loader"]


def retarget(spec: dict, cohort: str) -> dict:
    s = dict(spec)
    d = s["dir"].rstrip("/")
    assert d.endswith(TEMPLATE_COHORT), d
    s["dir"] = d[: -len(TEMPLATE_COHORT)] + cohort
    return s


def build(arm: str, problems: list[str]) -> list[dict]:
    cohorts, realize, out_dir, prefix = ARMS[arm]
    out_cfg, out_res = CFG / out_dir, RES / out_dir
    out_cfg.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for base in BASES:
        tmpl_p = CFG / f"eval_a1_indep/{base}__rd_ridge__{TEMPLATE_COHORT}.yaml"
        if not tmpl_p.exists():
            problems.append(f"missing template {tmpl_p}")
            continue
        tmpl = yaml.safe_load(tmpl_p.read_text())
        written: dict[str, dict] = {}

        for cohort in cohorts:
            specs = [retarget(s, cohort) for s in tmpl["adapter_specs"]]
            for s in specs:
                if not Path(s["dir"], "adapter_model.safetensors").exists():
                    problems.append(f"missing adapter {s['dir']}")

            for lam, tag in LAMBDAS:
                name = f"{base}__{prefix}_l{tag}__{cohort}"
                cfg = dict(tmpl)
                cfg["method"] = "rd_encoder"
                kw = dict(tmpl.get("method_kwargs", {}))
                kw["ridge_lambda"] = lam
                kw["realize"] = realize
                cfg["method_kwargs"] = kw
                cfg["adapter_specs"] = specs
                out_p = out_res / f"{name}.json"
                cfg["output_path"] = str(out_p)
                cfg["nll_tau_cache"] = str(
                    RES / "nll_tau_cache" / f"{base}__{cohort}.json")
                (out_cfg / f"{name}.yaml").write_text(
                    yaml.safe_dump(cfg, sort_keys=False))
                written[f"{cohort}|{tag}"] = cfg
                manifest.append({
                    "name": name,
                    "cmd": ("python code/phase3/eval/run_eval_cell.py "
                            f"--config {(out_cfg / (name + '.yaml')).relative_to(ROOT)}"),
                    "done": str(out_p.relative_to(ROOT)),
                    "min_free_gb": 25.0,
                })

        # Protocol item 1: the arms differ only in the cohort.
        for _, tag in LAMBDAS:
            a = written.get(f"{cohorts[0]}|{tag}")
            b = written.get(f"{cohorts[1]}|{tag}")
            if not a or not b:
                continue
            for key in ARM_INVARIANT:
                if a.get(key) != b.get(key):
                    problems.append(f"{arm} {base} {tag}: arms differ in {key}")
            if a["method_kwargs"] != b["method_kwargs"]:
                problems.append(f"{arm} {base} {tag}: method_kwargs differ")
            if {s["dir"] for s in a["adapter_specs"]} == \
               {s["dir"] for s in b["adapter_specs"]}:
                problems.append(f"{arm} {base} {tag}: both arms share a cohort")
            for x, y in zip(a["adapter_specs"], b["adapter_specs"]):
                if x.get("task_cfg") != y.get("task_cfg"):
                    problems.append(f"{arm} {base} {tag}: task_cfg differs")

    # Cache-aware ordering: one cell per (base, cohort) first.
    seen: set[str] = set()
    first, rest = [], []
    for e in manifest:
        b, _, c = e["name"].split("__")
        (rest if (b, c) in seen else first).append(e)
        seen.add((b, c))
    return first + rest


def main() -> int:
    problems: list[str] = []
    for arm in ARMS:
        manifest = build(arm, problems)
        if problems:
            continue
        expect = len(BASES) * len(LAMBDAS) * 2
        assert len(manifest) == expect, (arm, len(manifest), expect)
        out = CFG / f"wave2_{arm}_manifest.json"
        out.write_text(json.dumps(manifest, indent=2))
        cohorts, realize, _, _ = ARMS[arm]
        print(f"{arm}: wrote {out.name} with {len(manifest)} cells, "
              f"cohorts {cohorts}, realize={realize}")
        print(f"    first four: {[e['name'] for e in manifest[:4]]}")

    if problems:
        print("PROBLEMS, nothing written:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
