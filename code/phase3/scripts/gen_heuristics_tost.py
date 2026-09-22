#!/usr/bin/env python3
"""Cells for notes/prereg_heuristics_tost_2026-09-23.md.

160 downstream cells: the 40 (method, base, benchmark) configs of
eval_heur_downstream, cloned onto seed2, seed3 (shared) and indep2, indep3
(independent). The existing seed1/indep1 cells are the third cohort per arm.
Each clone differs from its template only in adapter directory and output
path, which is asserted before anything is written.

Usage:
  RDMERGE_ROOT=~/projects/rdmerge_v2 python code/phase3/scripts/gen_heuristics_tost.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
RES = ROOT / "results" / "phase3"
CFG = ROOT / "code" / "phase3" / "configs" / "eval_heur_tost"
OUT = RES / "eval_heur_tost"
SRC = RES / "eval_heur_downstream"
NEW = {"seed1": ["seed2", "seed3"], "indep1": ["indep2", "indep3"]}


def main() -> int:
    CFG.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    templates = sorted(SRC.glob("*__seed1.json")) + sorted(SRC.glob("*__indep1.json"))
    if len(templates) != 80:
        raise SystemExit(f"expected 80 template cells, found {len(templates)}")
    for tp in templates:
        stem, src_cohort = tp.stem.rsplit("__", 1)
        t = json.loads(tp.read_text())["meta"]["config"]
        for cohort in NEW[src_cohort]:
            cfg = json.loads(json.dumps(t))
            for s in cfg["adapter_specs"]:
                d = s["dir"].rstrip("/")
                assert d.endswith("/" + src_cohort), d
                s["dir"] = d[: -len(src_cohort)] + cohort
                assert Path(s["dir"], "adapter_model.safetensors").exists(), s["dir"]
            name = f"{stem}__{cohort}"
            cfg["output_path"] = str(OUT / f"{name}.json")
            diff = {k for k in set(t) | set(cfg) if t.get(k) != cfg.get(k)}
            assert diff == {"adapter_specs", "output_path"}, diff
            p = CFG / f"{name}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False))
            manifest.append({"name": name,
                             "cmd": f"python code/phase3/eval/run_downstream_cell.py --config {p.relative_to(ROOT)}",
                             "done": str((OUT / f"{name}.json").relative_to(ROOT)),
                             "min_free_gb": 30.0})
    if len(manifest) != 160:
        raise SystemExit(f"built {len(manifest)} cells, expected 160")
    m = ROOT / "code" / "phase3" / "configs" / "heur_tost_manifest.json"
    m.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {m}: {len(manifest)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
