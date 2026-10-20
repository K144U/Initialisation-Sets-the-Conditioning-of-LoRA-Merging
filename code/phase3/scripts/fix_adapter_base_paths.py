#!/usr/bin/env python3
"""Point restaged adapters at the restaged base models.

The adapters copied into ~/projects/rdmerge_v2 on 2026-10-19 still record
base_model_name_or_path = /home/sanjay.g/projects/rdmerge/models/<name>, a tree
emptied on 2026-10-10. Cells with `loader: plain` take the base from their
config and never read that field; the Unsloth loader (every downstream cell)
reads it and fails. This rewrites only that field, to the same model name
under rdmerge_v2/models (whose weights the restage gate verified: nll_tau
identical to 5 decimals on all four bases), keeping the original file as
adapter_config.json.sanjay_orig. Weights are not touched.

  python fix_adapter_base_paths.py [--dry-run]
"""
import json
import sys
from pathlib import Path

ROOT = Path.home() / "projects" / "rdmerge_v2"
OLD = "/home/sanjay.g/projects/rdmerge/models/"
NEW = str(ROOT / "models") + "/"


def main():
    dry = "--dry-run" in sys.argv
    fixed, missing = 0, set()
    for tree in sorted(ROOT.glob("artifacts*")):
        for cfg in tree.rglob("adapter_config.json"):
            d = json.loads(cfg.read_text())
            old = d.get("base_model_name_or_path", "")
            if not old.startswith(OLD):
                continue
            new = NEW + old[len(OLD):]
            if not Path(new).is_dir():
                missing.add(new)
                continue
            if not dry:
                bak = cfg.with_name("adapter_config.json.sanjay_orig")
                if not bak.exists():
                    bak.write_text(cfg.read_text())
                d["base_model_name_or_path"] = new
                cfg.write_text(json.dumps(d, indent=2))
            fixed += 1
    print(f"{'would fix' if dry else 'fixed'} {fixed} adapter configs")
    for m in sorted(missing):
        print("  no restaged model for", m)


if __name__ == "__main__":
    main()
