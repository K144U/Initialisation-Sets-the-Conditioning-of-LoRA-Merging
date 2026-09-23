#!/usr/bin/env python3
"""Table 4 re-run with the Gram the paper describes (A^T A).

notes/prereg_regmean_real_amendment_2026-09-23.md. Clones the 56 published
eval_regmean_cond configs (7 lambdas x 4 bases x seed1/indep1), changing only
the output path and the nll_tau cache location (the published cache dir is
read-only from this account). The merge itself changes because
PeftModelView.get_layer now exists; that is the point.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
SRC = RES / "eval_regmean_cond"
OUT = RES / "eval_regmean_agram"
CFG = ROOT / "code" / "phase3" / "configs" / "eval_regmean_agram"
CACHE = RES / "nll_tau_cache_v2"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CFG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = []
    srcs = sorted(SRC.glob("*.json"))
    if len(srcs) != 56:
        raise SystemExit(f"expected 56 published cells, found {len(srcs)}")
    for p in srcs:
        cfg = json.loads(p.read_text())["meta"]["config"]
        # Older cells predate the nll_tau cache; give them the cohort's cache,
        # which run_eval_cell only uses on an exact key match.
        base, _, cohort = p.stem.split("__")
        old_cache = Path(cfg.get("nll_tau_cache") or RES / "nll_tau_cache" / f"{base}__{cohort}.json")
        new_cache = CACHE / old_cache.name
        if old_cache.exists() and not new_cache.exists():
            shutil.copy(old_cache, new_cache)
        new = dict(cfg, output_path=str(OUT / p.name), nll_tau_cache=str(new_cache),
                   min_free_gb=30.0)
        changed = {k for k in set(cfg) | set(new) if cfg.get(k) != new.get(k)}
        assert changed <= {"output_path", "nll_tau_cache", "min_free_gb"}, changed
        y = CFG / f"{p.stem}.yaml"
        y.write_text(yaml.safe_dump(new, sort_keys=False))
        manifest.append({"name": f"agram_{p.stem}",
                         "cmd": f"python code/phase3/eval/run_eval_cell.py --config {y.relative_to(ROOT)}",
                         "done": str((OUT / p.name).relative_to(ROOT)), "min_free_gb": 30.0})
    (ROOT / "code/phase3/configs/regmean_agram_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(manifest)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
