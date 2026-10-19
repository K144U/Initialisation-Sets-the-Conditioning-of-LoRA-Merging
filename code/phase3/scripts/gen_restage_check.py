#!/usr/bin/env python3
"""Restage gate (notes/campaign_2026-10-19_redteam.md): before any registered
cell, re-run four published NLL cells against the re-staged base models and
adapters, with a FRESH nll_tau cache, and compare with the published JSONs.

Cells: adapter-only RegMean at lambda = 0.01, seed1, one per base (published in
eval_regmean_real). check_restage.py compares worst-task excess and every
nll_tau entry; tolerance 0.002 nats.
"""
from __future__ import annotations

import sys

from _c1019 import BASES, RES, Manifest, template

OUT = "restage_check"


def main() -> int:
    m = Manifest()
    for base in BASES:
        src = RES / "eval_regmean_real" / f"{base}__rmdf_l0p01__seed1.json"
        cfg = template(src)
        cfg.pop("output_path", None)
        cfg["nll_tau_cache"] = str(RES / "nll_tau_cache_v3" / f"{base}__seed1.json")
        cfg["output_path"] = str(RES / OUT / f"{base}__rmdf_l0p01__seed1.json")
        m.add(f"{base}__restage", cfg, OUT, "run_eval_cell.py")
    m.write("restage_check_manifest.json", expect=4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
