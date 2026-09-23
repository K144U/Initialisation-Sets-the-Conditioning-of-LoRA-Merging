#!/usr/bin/env python3
"""Select and configure the public cohorts to merge.

Rules: notes/prereg_public_merge_2026-09-23.md. Selection walks the 45 scored
cohorts of the prevalence audit in their listed order and applies criteria 1-5
using metadata only (base config.json, safetensors index, adapter configs, HF
gating). Every decision, pass or fail, is written with its reason, so the list
can be audited without re-running anything.

Usage (login node, needs network):
  RDMERGE_ROOT=~/projects/rdmerge_v2 python code/phase3/scripts/gen_public_merge.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
PREV = Path("/home/sanjay.g/projects/rdmerge/results/phase3/prevalence")
CFG = ROOT / "code" / "phase3" / "configs" / "public_merge"
OUT = ROOT / "results" / "phase3" / "public_merge"
MAX_PARAMS = 15e9
NOT_TEXT = ("vl", "vision", "llava", "whisper", "vit", "t5", "bert", "clip",
            "omni", "speech", "audio", "siglip", "mllama", "idefics")
BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
TOKEN = os.environ.get("HF_TOKEN") or (
    Path.home() / ".cache/huggingface/token").read_text().strip() \
    if (Path.home() / ".cache/huggingface/token").exists() else None


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "rdmerge-audit"})
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:  # 401/403/404 are all "cannot use"
        return {"__error__": str(e)}


def base_params(repo: str, cfg: dict):
    idx = get_json(f"https://huggingface.co/{repo}/resolve/main/model.safetensors.index.json")
    total = (idx.get("metadata") or {}).get("total_size") if "__error__" not in idx else None
    if total is None:
        info = get_json(f"https://huggingface.co/api/models/{repo}?blobs=true")
        if "__error__" in info:
            return None
        total = sum(s.get("size", 0) for s in info.get("siblings", [])
                    if s.get("rfilename", "").endswith(".safetensors"))
        if not total:
            return None
    dt = str(cfg.get("torch_dtype") or cfg.get("dtype") or "bfloat16").replace("torch.", "")
    return total / BYTES.get(dt, 2)


def check(cohort: dict, members: list[dict]) -> tuple[bool, str, dict]:
    base = cohort["base_model"]
    if base.startswith(("/", "checkpoints/")) or base.count("/") != 1:
        return False, "1: base is a local path, not a hub model", {}
    low = base.lower()
    if any(k in low for k in NOT_TEXT) or "bnb-4bit" in low:
        return False, "1/5: vision, speech, encoder, embedding or 4-bit-only base", {}
    cfg = get_json(f"https://huggingface.co/{base}/resolve/main/config.json")
    if "__error__" in cfg:
        return False, f"3: base config not downloadable ({cfg['__error__'][:60]})", {}
    arch = (cfg.get("architectures") or [""])[0]
    if not arch.endswith("ForCausalLM") or cfg.get("auto_map"):
        return False, f"1: architecture {arch!r} is not a plain causal LM", {}
    try:
        from transformers import CONFIG_MAPPING
        if cfg.get("model_type") not in CONFIG_MAPPING:
            return False, f"1: model_type {cfg.get('model_type')!r} unknown to transformers", {}
    except Exception:
        pass
    n = base_params(base, cfg)
    if n is None:
        return False, "3: base weights not listable", {}
    if n > MAX_PARAMS:
        return False, f"2: {n / 1e9:.1f}B parameters", {}
    ranks = {m.get("rank") for m in members}
    tms = {tuple(sorted(m.get("target_modules") or [])) for m in members}
    if len(ranks) != 1 or len(tms) != 1:
        return False, f"4: members differ in rank {sorted(ranks)} or target modules", {}
    for m in members:
        a = get_json(f"https://huggingface.co/{m['repo']}/resolve/main/adapter_config.json")
        if "__error__" in a:
            return False, f"4: adapter {m['repo']} config unreadable", {}
    return True, "ok", {"params": n, "rank": ranks.pop(), "arch": arch}


def main() -> int:
    CFG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((PREV / "prevalence_summary.json").read_text())["cohorts"]
    phase1 = {(c["namespace"], c["base_model"]): c
              for c in json.loads((PREV / "cohorts.json").read_text())["cohorts"]}
    decisions, manifest = [], []
    for row in summary:
        if "median_principal_cosine" not in row:
            continue                                  # the 5 unscored cohorts
        cos = row["median_principal_cosine"]
        regime = "collapsed" if cos > 0.9 else "separated" if cos < 0.5 else "intermediate"
        members = [m for m in phase1[(row["namespace"], row["base_model"])]["members"]
                   if m["repo"] in row["repos"]]
        ok, why, meta = check(row, members)
        key = f"{row['namespace']}__{row['base_model'].replace('/', '--')}"
        decisions.append({"cohort": key, "regime": regime, "cos": cos,
                          "selected": ok and regime != "intermediate",
                          "reason": why if ok or regime != "intermediate" else "intermediate",
                          **meta})
        print(f"{'SELECT' if ok else 'reject'} {regime:<12} {key:<70} {why}", flush=True)
        if not ok or regime == "intermediate":
            continue
        cfg = {"base_model": row["base_model"], "adapters": row["repos"],
               "rank": meta["rank"], "regime": regime, "audit_cos": cos,
               "n_prompts": 100, "max_new_tokens": 128, "seed": 20260518,
               "methods": {"ta": ["task_arithmetic", {}],
                           "rm_1em6": ["regmean", {"ridge_lambda": 1e-6}],
                           "rm_a09": ["regmean", {"alpha": 0.9, "ridge_lambda": 0.0}],
                           "rm_0p01": ["regmean", {"ridge_lambda": 0.01}],
                           # notes/prereg_lambda_rule_2026-09-23.md, V4
                           "rm_rule": ["regmean", {"ridge_mode": "kappa_target",
                                                   "kappa_target": 100.0}]},
               "min_free_gb": 30.0 if meta["params"] < 9e9 else 45.0,
               "output_path": str(OUT / f"{key}.json")}
        p = CFG / f"{key}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        manifest.append({"name": f"pub_{key}"[:120],
                         "cmd": f"python code/phase3/eval/run_behaviour_cell.py --config {p.relative_to(ROOT)}",
                         "done": str((OUT / f"{key}.json").relative_to(ROOT)),
                         "min_free_gb": cfg["min_free_gb"]})
    (OUT / "selection.json").write_text(json.dumps(decisions, indent=1))
    (ROOT / "code/phase3/configs/public_merge_manifest.json").write_text(
        json.dumps(manifest, indent=1))
    sel = [d for d in decisions if d["selected"]]
    print(f"\nselected {len(sel)}: "
          f"{sum(d['regime'] == 'collapsed' for d in sel)} collapsed, "
          f"{sum(d['regime'] == 'separated' for d in sel)} separated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
