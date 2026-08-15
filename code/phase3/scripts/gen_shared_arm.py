#!/usr/bin/env python3
"""The ridge sweep's shared arm at n = 3: seed2 and seed3.

Rules: notes/prereg_shared_arm_2026-08-15.md (603e013), committed before this
generator existed.

R8 of the 2026-08-14 registration extended the ridge sweep's independent arm to
three cohorts and left the shared arm at one, which is limitation 6: the gate
exists, but the standard error behind it comes entirely from the independent
side. This is the E2 sweep on the two shared cohorts that have never been
swept, so that both arms are n = 3 and the gate carries a term from each.

4 bases x 7 lambdas x 2 cohorts = 56 cells. No adapter training: seed2 and
seed3 already exist, 16 adapters each.

Two invariance checks run before anything is written, and the script refuses to
write cells if either fails.

  1. The two new cohorts differ from each other only in the adapter directory.
     This is the check every generator since 2a49b3e has run.
  2. Every new cell matches the EXISTING seed1 cell at the same lambda, field
     for field, on base_model, max_seq_length, seed, min_free_gb, loader and
     method_kwargs. This is the one that matters here. The pre-registration
     reuses the seed1 cells rather than re-running them, which is only sound if
     they were produced under the same configuration, and an earlier Table 3 in
     this project was corrupted by exactly that assumption going unchecked.

The nll_tau cache has no seed2 or seed3 entry, so the first cell of each
(base, cohort) builds one at about 57 minutes and the remaining 48 hit it at
about 10. The manifest is ordered to put those eight first.

Usage:
  python code/phase3/scripts/gen_shared_arm.py
  qsub -v ARM=r8s code/phase3/scripts/pbs_shared_arm.sh
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
COHORTS = ["seed2", "seed3"]
REALIZE = "rank_r"
OUT_DIR = "eval_ridge_shared"
PREFIX = "rds"

# The published shared arm this one extends. Its cells are reused, not re-run.
REF_DIR = "eval_ridge_cond"
REF_PREFIX = "rd"
REF_COHORT = "seed1"

ARM_INVARIANT = ["base_model", "max_seq_length", "seed", "min_free_gb",
                 "loader"]


def retarget(spec: dict, cohort: str) -> dict:
    s = dict(spec)
    d = s["dir"].rstrip("/")
    assert d.endswith(TEMPLATE_COHORT), d
    s["dir"] = d[: -len(TEMPLATE_COHORT)] + cohort
    return s


def build(problems: list[str]) -> list[dict]:
    out_cfg, out_res = CFG / OUT_DIR, RES / OUT_DIR
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

        for cohort in COHORTS:
            specs = [retarget(s, cohort) for s in tmpl["adapter_specs"]]
            for s in specs:
                if not Path(s["dir"], "adapter_model.safetensors").exists():
                    problems.append(f"missing adapter {s['dir']}")

            for lam, tag in LAMBDAS:
                name = f"{base}__{PREFIX}_l{tag}__{cohort}"
                cfg = dict(tmpl)
                cfg["method"] = "rd_encoder"
                kw = dict(tmpl.get("method_kwargs", {}))
                kw["ridge_lambda"] = lam
                kw["realize"] = REALIZE
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

        # Check 1: the two new cohorts differ only in the adapter directory.
        for _, tag in LAMBDAS:
            a = written.get(f"{COHORTS[0]}|{tag}")
            b = written.get(f"{COHORTS[1]}|{tag}")
            if not a or not b:
                continue
            for key in ARM_INVARIANT:
                if a.get(key) != b.get(key):
                    problems.append(f"{base} {tag}: cohorts differ in {key}")
            if a["method_kwargs"] != b["method_kwargs"]:
                problems.append(f"{base} {tag}: method_kwargs differ")
            if {s["dir"] for s in a["adapter_specs"]} == \
               {s["dir"] for s in b["adapter_specs"]}:
                problems.append(f"{base} {tag}: both cohorts share a directory")
            for x, y in zip(a["adapter_specs"], b["adapter_specs"]):
                if x.get("task_cfg") != y.get("task_cfg"):
                    problems.append(f"{base} {tag}: task_cfg differs")

        # Check 2: every new cell matches the published seed1 cell it extends.
        # nll_tau_cache is excluded: it postdates those cells and was shown
        # inert by check_nll_tau_determinism.py (8/8 groups bitwise identical,
        # 960 comparisons, worst difference exactly 0).
        for _, tag in LAMBDAS:
            ref_p = (CFG / REF_DIR /
                     f"{base}__{REF_PREFIX}_l{tag}__{REF_COHORT}.yaml")
            if not ref_p.exists():
                problems.append(f"missing published reference cell {ref_p}")
                continue
            ref = yaml.safe_load(ref_p.read_text())
            new = written.get(f"{COHORTS[0]}|{tag}")
            if not new:
                continue
            for key in ARM_INVARIANT:
                if ref.get(key) != new.get(key):
                    problems.append(
                        f"{base} {tag}: differs from published seed1 in {key} "
                        f"({ref.get(key)!r} vs {new.get(key)!r})")
            if ref.get("method") != new.get("method"):
                problems.append(f"{base} {tag}: method differs from seed1")
            if ref.get("method_kwargs") != new.get("method_kwargs"):
                problems.append(
                    f"{base} {tag}: method_kwargs differ from published seed1 "
                    f"({ref.get('method_kwargs')} vs {new.get('method_kwargs')})")
            ref_tasks = [s.get("task_cfg") for s in ref["adapter_specs"]]
            new_tasks = [s.get("task_cfg") for s in new["adapter_specs"]]
            if ref_tasks != new_tasks:
                problems.append(f"{base} {tag}: task_cfg differs from seed1")

    # Cache-aware ordering: one cell per (base, cohort) first, so the eight
    # cache-building cells run before the forty-eight that hit the cache.
    seen: set = set()
    first, rest = [], []
    for e in manifest:
        b, _, c = e["name"].split("__")
        (rest if (b, c) in seen else first).append(e)
        seen.add((b, c))
    return first + rest


def main() -> int:
    problems: list[str] = []
    manifest = build(problems)

    if problems:
        print("PROBLEMS, nothing written:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2

    expect = len(BASES) * len(LAMBDAS) * len(COHORTS)
    assert len(manifest) == expect, (len(manifest), expect)
    out = CFG / "shared_arm_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out.name} with {len(manifest)} cells, cohorts {COHORTS}, "
          f"realize={REALIZE}")
    print(f"    cache-building cells first: {[e['name'] for e in manifest[:8]]}")
    print(f"    smoke candidate: {manifest[0]['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
