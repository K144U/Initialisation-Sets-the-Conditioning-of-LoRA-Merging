#!/usr/bin/env python3
"""Manifests for notes/prereg_vision_2026-10-19.md: 48 training cells, then 6
evaluation cells (one per cohort, every registered merge inside)."""
from __future__ import annotations

import json
import sys

from vision_lora import NAMES, ROOT

COHORTS = ["shared1", "shared2", "shared3", "indep1", "indep2", "indep3"]
CFG = ROOT / "code" / "phase3" / "configs"


def main() -> int:
    train = [{"name": f"vis_train_{t}_{c}",
              "cmd": f"python code/phase3/vision/vision_lora.py train --task {t} --cohort {c}",
              "done": f"results/phase3/vision/train__{t}__{c}.json", "min_free_gb": 12.0}
             for c in COHORTS for t in NAMES]
    ev = [{"name": f"vis_eval_{c}",
           "cmd": f"python code/phase3/vision/vision_lora.py eval --cohort {c}",
           "done": f"results/phase3/vision/eval__{c}__DONE", "min_free_gb": 12.0}
          for c in COHORTS]
    assert len(train) == 48 and len(ev) == 6
    (CFG / "vision_train_manifest.json").write_text(json.dumps(train, indent=1))
    (CFG / "vision_eval_manifest.json").write_text(json.dumps(ev, indent=1))
    print("wrote 48 training + 6 evaluation cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
