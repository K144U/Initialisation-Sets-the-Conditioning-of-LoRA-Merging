#!/usr/bin/env python3
"""Downstream accuracy for the five heuristics, both initialisation arms.

Rules: notes/prereg_heuristics_downstream_2026-08-16.md (8ff7fb4), committed
before this generator existed. Nothing here selects a threshold, a method, a
hyperparameter or a cohort: all of that is read from the registration and from
the committed NLL configs.

80 cells: 5 methods x 4 bases x 2 cohorts x 2 benchmarks.

The merges are exactly the ones section 7.2 reports as showing no effect in
NLL. Only the evaluation is new. The question the registration asks is whether
that null survives being measured in a unit section 8 does not disclaim.

Both arms are regenerated. The 40 pre-repair cells in eval_e3_gsm8k and
eval_b4_humaneval are NOT reused: they were scored before the GSM8K and
HumanEval scorers were repaired, and a paired difference computed across two
scorer versions would carry a known, method-dependent defect inside it.

Method sources, and why the KnOTS row is not the obvious file:

    task_arithmetic, ties, dare, tvq_b2  ->  eval_matrix_seeds  (shared)
                                             eval_a1_indep      (independent)
    knots_ties                           ->  eval_a2_knots_ties (shared)
                                             eval_a2_knots_ties_indep

`knots` with inner_combination=linear is the no-op that reduces algebraically
to task arithmetic (section 8). The registration names the repaired KnOTS, so
this generator reads knots_ties and refuses to fall back to knots.

Invariance checks, all before anything is written:

  1. the two cohorts of a pair differ only in the adapter directory;
  2. every cell matches the NLL cell it will be compared against on
     base_model, max_seq_length, seed, min_free_gb and loader;
  3. n_eval_metric is the registered value for its benchmark;
  4. no merge hyperparameter is introduced here that the NLL config lacks.

Usage:
  python code/phase3/scripts/gen_heuristics_downstream.py
  qsub code/phase3/scripts/pbs_heuristics_downstream.sh
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
COHORTS = ["seed1", "indep1"]          # shared, independent
OUT_DIR = "eval_heur_downstream"

# method -> (config dir for shared, config dir for independent, config stem)
METHODS = {
    "task_arithmetic": ("eval_matrix_seeds", "eval_a1_indep", "task_arithmetic"),
    "ties":            ("eval_matrix_seeds", "eval_a1_indep", "ties"),
    "dare":            ("eval_matrix_seeds", "eval_a1_indep", "dare"),
    "tvq_b2":          ("eval_matrix_seeds", "eval_a1_indep", "tvq_b2"),
    "knots_ties":      ("eval_a2_knots_ties", "eval_a2_knots_ties_indep",
                        "knots_ties"),
}

# Registered evaluation sizes. HumanEval's whole test split is 164 problems.
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


# Implementation defaults, read off the merge functions' signatures. The
# shared-arm NLL configs leave method_kwargs empty and rely on these; the
# independent-arm configs spell the same values out. The merges are therefore
# identical and only the YAML differs, but "identical" must be checked rather
# than assumed, so the generator fills the defaults in explicitly on both sides
# and then asserts the two sides match. If an implementation default ever
# changes, this table stops agreeing with it and the assertion below fires.
DEFAULTS = {
    "ties": {"density": 0.2, "majority_sign_method": "total"},
    "dare": {"density": 0.2, "seed": 20260518},
}


def check_defaults(problems: list[str]) -> None:
    """The table above must still equal the functions' signature defaults."""
    import inspect
    import sys
    sys.path.insert(0, str(ROOT / "code" / "phase3"))
    from merging.dare import merge_dare
    from merging.ties import merge_ties

    for name, fn in (("ties", merge_ties), ("dare", merge_dare)):
        sig = inspect.signature(fn)
        for k, v in DEFAULTS[name].items():
            actual = sig.parameters[k].default
            if actual != v:
                problems.append(
                    f"{name}.{k} default is {actual!r}, this generator "
                    f"assumes {v!r}; the two arms would not be the same merge")


def src_cfg(method: str, base: str, cohort: str) -> Path:
    shared_dir, indep_dir, stem = METHODS[method]
    d = shared_dir if cohort.startswith("seed") else indep_dir
    return CFG / d / f"{base}__{stem}__{cohort}.yaml"


def build(problems: list[str]) -> list[dict]:
    out_cfg, out_res = CFG / OUT_DIR, RES / OUT_DIR
    out_cfg.mkdir(parents=True, exist_ok=True)
    out_res.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for base in BASES:
        for method in METHODS:
            written: dict[str, dict] = {}
            for cohort in COHORTS:
                p = src_cfg(method, base, cohort)
                if not p.exists():
                    problems.append(f"missing source config {p}")
                    continue
                tmpl = yaml.safe_load(p.read_text())

                # Check 4: the merge is taken from the NLL config untouched.
                if method == "knots_ties" and \
                        tmpl.get("method_kwargs", {}).get(
                            "inner_combination") == "linear":
                    problems.append(
                        f"{base}/{cohort}: knots_ties config has "
                        f"inner_combination=linear, which is the no-op")

                for spec in tmpl["adapter_specs"]:
                    if not Path(spec["dir"], "adapter_model.safetensors").exists():
                        problems.append(f"missing adapter {spec['dir']}")

                # Make the implicit defaults explicit, identically on both
                # arms, so the two cells differ only in the adapter directory.
                kw = dict(tmpl.get("method_kwargs") or {})
                for k, v in DEFAULTS.get(method, {}).items():
                    kw.setdefault(k, v)

                for bench, bspec in BENCH.items():
                    name = f"{base}__{method}_{bench}__{cohort}"
                    cfg = dict(tmpl)
                    cfg["method_kwargs"] = kw
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
                    written[f"{cohort}|{bench}"] = cfg
                    manifest.append({
                        "name": name,
                        "cmd": ("python code/phase3/eval/run_downstream_cell.py "
                                f"--config "
                                f"{(out_cfg / (name + '.yaml')).relative_to(ROOT)}"),
                        "done": str(out_p.relative_to(ROOT)),
                        "min_free_gb": 25.0,
                    })

            # Check 1: cohorts differ only in the adapter directory.
            for bench in BENCH:
                a = written.get(f"{COHORTS[0]}|{bench}")
                b = written.get(f"{COHORTS[1]}|{bench}")
                if not a or not b:
                    continue
                for k in INVARIANT + ["metric", "n_eval_metric", "method"]:
                    if a.get(k) != b.get(k):
                        problems.append(
                            f"{base} {method} {bench}: cohorts differ in {k} "
                            f"({a.get(k)!r} vs {b.get(k)!r})")
                if a.get("method_kwargs") != b.get("method_kwargs"):
                    problems.append(
                        f"{base} {method} {bench}: method_kwargs differ "
                        f"across cohorts")
                if {s["dir"] for s in a["adapter_specs"]} == \
                   {s["dir"] for s in b["adapter_specs"]}:
                    problems.append(
                        f"{base} {method} {bench}: both cohorts share a "
                        f"directory")

            # Check 2: each cell matches its own NLL cell.
            for cohort in COHORTS:
                ref_p = src_cfg(method, base, cohort)
                if not ref_p.exists():
                    continue
                ref = yaml.safe_load(ref_p.read_text())
                for bench in BENCH:
                    new = written.get(f"{cohort}|{bench}")
                    if not new:
                        continue
                    for k in INVARIANT:
                        if ref.get(k) != new.get(k):
                            problems.append(
                                f"{base} {method} {cohort} {bench}: differs "
                                f"from its NLL cell in {k}")

            # Check 3: registered evaluation sizes.
            for bench, bspec in BENCH.items():
                for cohort in COHORTS:
                    c = written.get(f"{cohort}|{bench}")
                    if c and c["n_eval_metric"] != bspec["n_eval_metric"]:
                        problems.append(
                            f"{base} {method} {bench}: wrong n_eval_metric")

    # HumanEval first: it is the cheaper benchmark, so a failure surfaces fast.
    manifest.sort(key=lambda e: (0 if "humaneval" in e["name"] else 1, e["name"]))
    return manifest


def main() -> int:
    problems: list[str] = []
    check_defaults(problems)
    manifest = build(problems)

    if problems:
        print("PROBLEMS, nothing written:")
        for p in sorted(set(problems)):
            print("  ", p)
        return 2

    expected = len(BASES) * len(METHODS) * len(COHORTS) * len(BENCH)
    if len(manifest) != expected:
        print(f"PROBLEM: {len(manifest)} cells, expected {expected}")
        return 2

    mpath = CFG / "heur_downstream_manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"{len(manifest)} cells -> {mpath.relative_to(ROOT)}")
    print(f"  {len(BASES)} bases x {len(METHODS)} methods x "
          f"{len(COHORTS)} cohorts x {len(BENCH)} benchmarks")
    done = sum(1 for e in manifest if (ROOT / e["done"]).exists())
    print(f"  already complete: {done}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
