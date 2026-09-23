#!/usr/bin/env python3
"""Evaluation cells for the new cohorts (stage eval).

Rules: notes/prereg_new_cohorts_2026-09-23.md (Parts T and R) and
notes/prereg_lambda_rule_2026-09-23.md. Run after training; refuses to write
cells for a cohort whose adapters do not all exist.

Every config is cloned from a published one, changing only the adapter
directories, the task list (Part R's T = 2 / T = 8), the method and its
kwargs, and the output path:
  NLL        eval_ridge_cond (encoder, plain loader) and eval_regmean_cond
  downstream eval_heur_downstream (unsloth loader)

Part T  (llama, mistral) x (seed1fx, indep1fx):
        NLL  encoder lambda 0 and lambda*, RegMean 1e-6 and 0.01, 5 heuristics
        DS   encoder lambda 0 and lambda*, 5 heuristics, GSM8K and HumanEval
Part R  (llama) T in {2, 8} at r16, r in {8, 64} at T4, both arms:
        NLL  RegMean 1e-6 and 0.01, task arithmetic
Rule    Part T cohorts and Part R cohorts: RegMean kappa-target and alpha 0.9
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
CFG = ROOT / "code" / "phase3" / "configs" / "eval_new_cohorts"
NLL_OUT, DS_OUT = RES / "eval_new_cohorts", RES / "eval_new_cohorts_ds"
CACHE = RES / "nll_tau_cache_v2"
NEW_ART = ROOT / "artifacts_new" / "lora"
OLD_ART = Path("/home/sanjay.g/projects/rdmerge/artifacts/lora")
TAGS = ["0", "0p01", "0p03", "0p05", "0p13", "0p30", "1p00"]
LAM = {"0": 0.0, "0p01": 0.01, "0p03": 0.03, "0p05": 0.05, "0p13": 0.13,
       "0p30": 0.30, "1p00": 1.00}
HEUR = {"ta": ("task_arithmetic", {}),
        "ties": ("ties", {"density": 0.2, "majority_sign_method": "total"}),
        "dare": ("dare", {"density": 0.2, "seed": 20260518}),
        "tvq_b2": ("tvq", {"rate_bits": 2}),
        "knots_ties": ("knots", {"inner_combination": "ties", "density": 0.2})}
RM = {"rm_1em6": ("regmean", {"ridge_lambda": 1e-6}),
      "rm_0p01": ("regmean", {"ridge_lambda": 0.01})}
RULE = {"rm_rule": ("regmean", {"ridge_mode": "kappa_target", "kappa_target": 100.0}),
        "rm_a09": ("regmean", {"alpha": 0.9, "ridge_lambda": 0.0})}
NEW_TASK_EVAL = {  # task configs for the T = 8 tasks, n_eval as the paper's eval cells
    "dolly": {"name": "dolly", "format": "dolly", "dataset": "databricks/databricks-dolly-15k",
              "config": None, "split_train": "train", "split_eval": "train",
              "n_train": 7500, "n_eval": 1000},
    "xsum": {"name": "xsum", "dataset": "EdinburghNLP/xsum", "config": None,
             "split_train": "train", "split_eval": "validation", "n_train": 7500,
             "n_eval": 1000, "prompt_field": "document", "answer_field": "summary"},
    "codealpaca": {"name": "codealpaca", "format": "alpaca",
                   "dataset": "sahil2801/CodeAlpaca-20k", "config": None,
                   "split_train": "train", "split_eval": "train", "n_train": 7500,
                   "n_eval": 1000},
    "translation_fr": {"name": "translation_fr", "format": "translation",
                       "dataset": "wmt/wmt14", "config": "fr-en", "split_train": "train",
                       "split_eval": "validation", "n_train": 7500, "n_eval": 1000,
                       "src_lang": "en", "tgt_lang": "fr", "streaming": True,
                       "stream_shuffle_buffer": 50000},
}
TASK_DIR = {"gsm8k": "gsm8k", "alpaca": "alpaca", "magicoder": "magicoder",
            "translation": "flores"}


def tmpl(path: Path) -> dict:
    return json.loads(path.read_text())["meta"]["config"]


def specs_for(template_specs, dirs: dict[str, Path], extra: list[str] = ()) -> list[dict]:
    out = []
    by_name = {s["name"]: s for s in template_specs}
    for name, d in dirs.items():
        s = copy.deepcopy(by_name[name]) if name in by_name else {
            "name": name, "task_cfg": NEW_TASK_EVAL[name]}
        s["dir"] = str(d)
        if not Path(d, "adapter_model.safetensors").exists():
            raise FileNotFoundError(d)
        out.append(s)
    return out


def main() -> int:
    for p in (CFG, NLL_OUT, DS_OUT, CACHE):
        p.mkdir(parents=True, exist_ok=True)
    manifest, missing = [], []

    def add(name, cfg, ds: bool):
        out = (DS_OUT if ds else NLL_OUT) / f"{name}.json"
        cfg["output_path"] = str(out)
        cfg["min_free_gb"] = 30.0
        p = CFG / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        runner = "run_downstream_cell.py" if ds else "run_eval_cell.py"
        manifest.append({"name": name, "cmd": f"python code/phase3/eval/{runner} --config {p.relative_to(ROOT)}",
                         "done": str(out.relative_to(ROOT)), "min_free_gb": 30.0})

    def nll_cells(base, cohort, dirs, methods, rd_tags=()):
        rm_t = tmpl(RES / "eval_regmean_cond" / f"{base}__rm_l0p01__seed1.json")
        specs = specs_for(rm_t["adapter_specs"], dirs)
        for tag, (method, kw) in methods.items():
            cfg = {k: v for k, v in rm_t.items() if k not in ("output_path",)}
            cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs,
                       nll_tau_cache=str(CACHE / f"{base}__{cohort}.json"))
            add(f"{base}__{tag}__{cohort}", cfg, ds=False)
        for label, tag in rd_tags:
            rd_t = tmpl(RES / "eval_ridge_cond" / f"{base}__rd_l0__seed1.json")
            cfg = {k: v for k, v in rd_t.items() if k != "output_path"}
            kw = dict(rd_t["method_kwargs"], ridge_lambda=LAM[tag])
            cfg.update(method_kwargs=kw, adapter_specs=specs_for(rd_t["adapter_specs"], dirs),
                       nll_tau_cache=str(CACHE / f"{base}__{cohort}.json"))
            add(f"{base}__rd_{label}__{cohort}", cfg, ds=False)

    def ds_cells(base, cohort, dirs, rd_tags):
        for bench in ("gsm8k", "humaneval"):
            ds_t = tmpl(RES / "eval_heur_downstream" / f"{base}__task_arithmetic_{bench}__seed1.json")
            specs = specs_for(ds_t["adapter_specs"], dirs)
            for tag, (method, kw) in HEUR.items():
                cfg = {k: v for k, v in ds_t.items() if k != "output_path"}
                cfg.update(method=method, method_kwargs=dict(kw), adapter_specs=specs)
                add(f"{base}__{tag}_{bench}__{cohort}", cfg, ds=True)
            rd_t = tmpl(RES / "eval_ridge_cond" / f"{base}__rd_l0__seed1.json")
            for label, tag in rd_tags:
                cfg = {k: v for k, v in ds_t.items() if k != "output_path"}
                cfg.update(method="rd_encoder",
                           method_kwargs=dict(rd_t["method_kwargs"], ridge_lambda=LAM[tag]),
                           adapter_specs=specs, loader=rd_t.get("loader"))
                add(f"{base}__rd_{label}_{bench}__{cohort}", cfg, ds=True)

    def lstar(base, old_cohort):
        v = {t: json.loads((RES / "eval_ridge_cond" / f"{base}__rd_l{t}__{old_cohort}.json")
                           .read_text())["worst_task_excess"] for t in TAGS}
        return min(v, key=v.get)

    # Part T and its rule cells
    for base in ("llama31_8b", "mistral_7b"):
        for old, new in (("seed1", "seed1fx"), ("indep1", "indep1fx")):
            dirs = {n: (NEW_ART / base / "flores" / new if n == "translation"
                        else OLD_ART / base / TASK_DIR[n] / old) for n in TASK_DIR}
            try:
                star = lstar(base, old)
                rd_tags = [("l0", "0"), ("lstar", star)]
                nll_cells(base, new, dirs, {**RM, **{k: v for k, v in HEUR.items()}, **RULE}, rd_tags)
                ds_cells(base, new, dirs, rd_tags)
            except FileNotFoundError as e:
                missing.append(f"{base}/{new}: {e}")

    # Part R (llama only)
    base = "llama31_8b"
    for arm, old in (("seed1fx", "seed1"), ("indep1fx", "indep1")):
        four = {n: (NEW_ART / base / "flores" / arm if n == "translation"
                    else OLD_ART / base / TASK_DIR[n] / old) for n in TASK_DIR}
        configs = {
            f"{arm}_T2": {n: four[n] for n in ("gsm8k", "alpaca")},
            f"{arm}_T8": {**four, **{n: NEW_ART / base / n / arm for n in NEW_TASK_EVAL}},
        }
        for r in (8, 64):
            configs[f"{arm}_r{r}"] = {n: NEW_ART / base / TASK_DIR[n] / f"{arm}_r{r}"
                                      for n in TASK_DIR}
        for cohort, dirs in configs.items():
            try:
                nll_cells(base, cohort, dirs, {**RM, "ta": HEUR["ta"], **RULE})
            except FileNotFoundError as e:
                missing.append(f"{base}/{cohort}: {e}")

    if missing:
        print("adapters missing, no cells written for:\n  " + "\n  ".join(missing))
    out = ROOT / "code" / "phase3" / "configs" / "new_cohorts_eval_manifest.json"
    manifest.sort(key=lambda m: ("_ds/" in m["done"], m["name"]))
    out.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}: {len(manifest)} cells")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
