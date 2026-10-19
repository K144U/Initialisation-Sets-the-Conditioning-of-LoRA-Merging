#!/usr/bin/env python3
"""Re-key nll_tau_cache_v2 to the re-staged paths, per base, after the gate.

The cache key embeds the base-model path and adapter dirs, which moved from
the sanjay.g tree to the student tree on 2026-10-19. A base is re-keyed only
if its restage cell (gen_restage_check.py) reproduced the published nll_tau
within 0.002 nats everywhere and the cached file holds the same values; then
only the path strings in the key change, never a number. The original key is
kept under "key_before_restage".
"""
from __future__ import annotations

import json
import sys

from _c1019 import BASES, RES, restage

TOL = 0.002
CACHE = RES / "nll_tau_cache_v2"


def gate_ok(base):
    new = RES / "restage_check" / f"{base}__rmdf_l0p01__seed1.json"
    if not new.exists():
        return False, "restage cell missing"
    a = json.loads(new.read_text())
    c = json.loads((CACHE / f"{base}__seed1.json").read_text())["nll_tau"]
    d = max(abs(a["nll_tau"][s][t] - c[s][t]) for s in c for t in c[s])
    return d <= TOL, f"max |d nll_tau| = {d:.5f}"


def main() -> int:
    bases = sys.argv[1:] or BASES
    for base in bases:
        ok, why = gate_ok(base)
        print(f"{base}: {why} -> {'re-key' if ok else 'SKIP'}")
        if not ok:
            continue
        for p in sorted(CACHE.glob(f"{base}__*.json")):
            blob = json.loads(p.read_text())
            k = blob["key"]
            if "key_before_restage" in blob:
                continue
            new = dict(k, base_model=restage(k["base_model"]),
                       adapters=[[n, restage(d)] for n, d in k["adapters"]])
            if new == k:
                continue
            blob["key_before_restage"] = k
            blob["key"] = new
            p.write_text(json.dumps(blob))
            print(f"  re-keyed {p.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
