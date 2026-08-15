#!/usr/bin/env python3
"""Downstream accuracy on the lambda = 0 conditioning collapse.

Rules: notes/prereg_downstream_2026-08-15.md (df832fe), amended by
notes/prereg_downstream_amendment_2026-08-15.md (4d48987) for HumanEval's
164-problem ceiling. Both committed before this generator existed.

32 cells: 4 bases x 2 cohorts x {lambda 0, lambda*} x {GSM8K, HumanEval}.

The merges are ones the paper already reports; only the evaluation is new. The
question is whether an effect of 0.13 to 9.63 nats, which is what the ridge
recovers on a shared-initialisation cohort, is visible in accuracy at all. If
it is not, limitation 2 stops being about fine-grained method comparisons and
becomes a statement about the metric as a whole.

lambda* is READ per (base, cohort) from the committed E2 sweep in
eval_ridge_cond and is never re-selected here. Binding constraint 2 of the
registration forbids re-selecting it on accuracy under any outcome, so this
script recomputes it from those cells rather than accepting a literal, and
prints what it read.

Three invariance checks, all before anything is written:

  1. the two cohorts of a pair differ only in the adapter directory;
  2. every cell matches the NLL cell it is meant to be compared against on
     base_model, max_seq_length, seed, realize and loader, so the comparison
     is not confounded the way an earlier Table 3 was by a loader mismatch;
  3. n_eval_metric is the amended value for its benchmark.

Usage:
  python code/phase3/scripts/gen_downstream_conditioning.py
  qsub code/phase3/scripts/pbs_downstream_cond.sh
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
COHORTS = ["seed1", "indep1"]
TAGS = ["0", "0p01", "0p03", "0p05", "0p13", "0p30", "1p00"]
LAM = {"0": 0.0, "0p01": 0.01, "0p03": 0.03, "0p05": 0.05,
       "0p13": 0.13, "0p30": 0.30, "1p00": 1.00}

OUT_DIR = "eval_downstream_cond"
TEMPLATE_COHORT = "indep1"
REALIZE = "rank_r"

# Amended per 4d48987: HumanEval's whole test split is 164 problems.
BENCH = {
    "gsm8k": {
        "metric": "gsm8k_em", "metric_task_spec": "gsm8k",
        "n_eval_metric": 500, "max_new_tokens": 256,
    },
    "humaneval": {
        "metric": "humaneval_pass1", "metric_task_spec": "humaneval",
        "n_eval_metric": 164, "max_new_tokens": 512,
        "metric_task_cfg": {
            "name": "humaneval", "dataset": "openai/openai_humaneval",
            "config": None, "split_train": "test", "split_eval": "test",
            "n_train": 0, "n_eval": 164,
            "prompt_field": "prompt", "answer_field": "canonical_solution",
        },
    },
}

INVARIANT = ["base_model", "max_seq_length", "seed", "min_free_gb", "loader"]


def read_lstar(base: str, cohort: str, problems: list[str]):
    """lambda* = arg-min over the committed E2 grid. Read, never fitted."""
    v = {}
    for t in TAGS:
        p = RES / "eval_ridge_cond" / f"{base}__rd_l{t}__{cohort}.json"
        if p.exists():
            v[t] = json.loads(p.read_text())["worst_task_excess"]
    if len(v) != len(TAGS):
        problems.append(f"E2 sweep incomplete for {base}/{cohort}: "
                        f"{len(v)}/{len(TAGS)} cells")
        return None
    return min(v, key=lambda k: v[k])


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
    lstars: dict[str, str] = {}

    for base in BASES:
        tmpl_p = CFG / f"eval_a1_indep/{base}__rd_ridge__{TEMPLATE_COHORT}.yaml"
        if not tmpl_p.exists():
            problems.append(f"missing template {tmpl_p}")
            continue
        tmpl = yaml.safe_load(tmpl_p.read_text())
        written: dict[str, dict] = {}

        for cohort in COHORTS:
            star_tag = read_lstar(base, cohort, problems)
            if star_tag is None:
                continue
            lstars[f"{base}|{cohort}"] = star_tag
            if star_tag == "0":
                print(f"  NOTE {base}/{cohort}: lambda* is 0, so the two ridge "
                      f"cells are the same merge and the gain is exactly zero "
                      f"by construction. Both are still generated so the table "
                      f"is complete and a reader does not have to infer it.")

            specs = [retarget(s, cohort) for s in tmpl["adapter_specs"]]
            for s in specs:
                if not Path(s["dir"], "adapter_model.safetensors").exists():
                    problems.append(f"missing adapter {s['dir']}")

            for which, tag in (("l0", "0"), ("lstar", star_tag)):
                for bench, bspec in BENCH.items():
                    name = f"{base}__{which}_{bench}__{cohort}"
                    cfg = dict(tmpl)
                    cfg["method"] = "rd_encoder"
                    kw = dict(tmpl.get("method_kwargs", {}))
                    kw["ridge_lambda"] = LAM[tag]
                    kw["realize"] = REALIZE
                    cfg["method_kwargs"] = kw
                    cfg["adapter_specs"] = specs
                    cfg["metric"] = bspec["metric"]
                    cfg["metric_task_spec"] = bspec["metric_task_spec"]
                    cfg["metric_kwargs"] = {
                        "max_new_tokens": bspec["max_new_tokens"]}
                    cfg["n_eval_metric"] = bspec["n_eval_metric"]
                    if "metric_task_cfg" in bspec:
                        cfg["metric_task_cfg"] = bspec["metric_task_cfg"]
                    out_p = out_res / f"{name}.json"
                    cfg["output_path"] = str(out_p)
                    (out_cfg / f"{name}.yaml").write_text(
                        yaml.safe_dump(cfg, sort_keys=False))
                    written[f"{cohort}|{which}|{bench}"] = cfg
                    manifest.append({
                        "name": name,
                        "cmd": ("python code/phase3/eval/run_downstream_cell.py "
                                f"--config {(out_cfg / (name + '.yaml')).relative_to(ROOT)}"),
                        "done": str(out_p.relative_to(ROOT)),
                        "min_free_gb": 25.0,
                    })

        # Check 1: the two cohorts differ only in the adapter directory.
        for which in ("l0", "lstar"):
            for bench in BENCH:
                a = written.get(f"{COHORTS[0]}|{which}|{bench}")
                b = written.get(f"{COHORTS[1]}|{which}|{bench}")
                if not a or not b:
                    continue
                for k in INVARIANT + ["metric", "n_eval_metric"]:
                    if a.get(k) != b.get(k):
                        problems.append(
                            f"{base} {which} {bench}: cohorts differ in {k}")
                if {s["dir"] for s in a["adapter_specs"]} == \
                   {s["dir"] for s in b["adapter_specs"]}:
                    problems.append(
                        f"{base} {which} {bench}: both cohorts share a directory")

        # Check 2: every cell matches the NLL cell it will be compared against.
        for cohort in COHORTS:
            ref_p = CFG / "eval_ridge_cond" / f"{base}__rd_l0__{cohort}.yaml"
            if not ref_p.exists():
                problems.append(f"missing NLL reference {ref_p}")
                continue
            ref = yaml.safe_load(ref_p.read_text())
            for bench in BENCH:
                new = written.get(f"{cohort}|l0|{bench}")
                if not new:
                    continue
                for k in INVARIANT:
                    if ref.get(k) != new.get(k):
                        problems.append(
                            f"{base} {cohort} {bench}: differs from the NLL cell "
                            f"in {k} ({ref.get(k)!r} vs {new.get(k)!r})")
                if ref["method_kwargs"].get("realize") != REALIZE:
                    problems.append(f"{base} {cohort}: NLL reference is not "
                                    f"{REALIZE}")

        # Check 3: the amended evaluation sizes.
        for bench, bspec in BENCH.items():
            for cohort in COHORTS:
                for which in ("l0", "lstar"):
                    c = written.get(f"{cohort}|{which}|{bench}")
                    if c and c["n_eval_metric"] != bspec["n_eval_metric"]:
                        problems.append(f"{base} {bench}: wrong n_eval_metric")

    # Cheapest benchmark first within each base, so a smoke cell is quick.
    manifest.sort(key=lambda e: (0 if "humaneval" in e["name"] else 1, e["name"]))
    (CFG / "downstream_cond_lstar.json").write_text(json.dumps(lstars, indent=2))
    return manifest


def main() -> int:
    problems: list[str] = []
    manifest = build(problems)

    if problems:
        print("PROBLEMS, nothing written:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2

    expect = len(BASES) * len(COHORTS) * 2 * len(BENCH)
    assert len(manifest) == expect, (len(manifest), expect)
    out = CFG / "downstream_cond_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {out.name} with {len(manifest)} cells")
    print(f"    smoke candidate: {manifest[0]['name']}")
    print(f"    lambda* read from the committed E2 sweep, see "
          f"downstream_cond_lstar.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
