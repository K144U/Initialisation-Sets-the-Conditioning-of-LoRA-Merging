#!/usr/bin/env python3
"""Professor B3: metadata of the public cohorts the audit calls collapsed.

For each cohort COLLAPSED under the maximum pairwise cosine
(prevalence_maxpair.json), or stored COLLAPSED and unreadable now, fetch from
the HuggingFace Hub, for up to MAX_PER members: the adapter config, the model
card's YAML header and opening text, tags, library, creation date and the
names of any checkpoint folders. No weights. Descriptive; classification is
done by reading the output against these categories, fixed here:

  checkpoints   members are checkpoints or variants of one training run
  derived       members were trained starting from another adapter
  library       a stack that seeds before adapter creation (Unsloth,
                LLaMA-Factory) with no seed change recorded
  init          a non-random initialisation (PiSSA, OLoRA, LoftQ, ...)
  by design     A shared or frozen on purpose (LoRAHub-style, FFA-LoRA, ...)
  unknown       nothing in the metadata says

  python collapsed_cohort_metadata.py <cohorts.json> <prevalence_maxpair.json> \
      <prevalence_summary.json> <out.json>
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests

API = "https://huggingface.co/api/models/"
RAW = "https://huggingface.co/{}/raw/main/{}"
MAX_PER = 6
CARD_CHARS = 1500


def get(url, as_json=True):
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                return r.json() if as_json else r.text
            if r.status_code in (401, 403, 404):
                return {"_status": r.status_code} if as_json else None
        except requests.RequestException:
            pass
        time.sleep(2)
    return None


def member(repo):
    info = get(API + repo) or {}
    files = [s.get("rfilename", "") for s in info.get("siblings", [])] if isinstance(info, dict) else []
    cfg = get(RAW.format(repo, "adapter_config.json"))
    card = get(RAW.format(repo, "README.md"), as_json=False) or ""
    keep = ("init_lora_weights", "r", "lora_alpha", "lora_dropout", "target_modules",
            "base_model_name_or_path", "use_dora", "use_rslora", "revision")
    return {
        "repo": repo,
        "status": info.get("_status") if isinstance(info, dict) else None,
        "created": info.get("createdAt") if isinstance(info, dict) else None,
        "library": info.get("library_name") if isinstance(info, dict) else None,
        "tags": info.get("tags", []) if isinstance(info, dict) else [],
        "checkpoint_dirs": sorted({f.split("/")[0] for f in files if re.match(r"checkpoint-\d+/", f)})[:10],
        "adapter_files": [f for f in files if f.endswith("adapter_config.json")][:10],
        "config": {k: cfg.get(k) for k in keep} if isinstance(cfg, dict) and "_status" not in cfg else cfg,
        "card": card[:CARD_CHARS],
    }


def main() -> int:
    spec, prev, stored, out = map(Path, sys.argv[1:5])
    members = {(c["namespace"], c["base_model"]): [m["repo"] for m in c["members"]]
               for c in json.loads(spec.read_text())["cohorts"]}
    old = {(c["namespace"], c["base_model"]): c.get("status")
           for c in json.loads(stored.read_text())["cohorts"]}
    rows = []
    for c in json.loads(prev.read_text())["cohorts"]:
        key = (c["namespace"], c["base_model"])
        collapsed = c.get("status_max_pair") == "COLLAPSED" or (
            "max_pair_cos" not in c and old.get(key) == "COLLAPSED")
        if not collapsed:
            continue
        repos = members[(c["namespace"], c["base_model"])]
        print(f"{c['namespace']} / {c['base_model']}: {len(repos)} members", flush=True)
        rows.append({"namespace": c["namespace"], "base_model": c["base_model"],
                     "n_members": len(repos), "max_pair_cos": c.get("max_pair_cos"),
                     "members": [member(r) for r in repos[:MAX_PER]]})
    out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} cohorts -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
