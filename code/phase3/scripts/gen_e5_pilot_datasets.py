"""Pre-bake 12 mixed-data JSONLs for the E5 Arm 2 pilot.

For target_task in {gsm8k, alpaca, magicoder, translation}
and alpha in {0.0, 0.5, 0.9}:
  - builds {n_total=7500} mixed training examples via build_mixed_train
  - writes JSONL to artifacts/e5_pilot_data/qwen25_7b/<task>__alpha<X>.jsonl

The shared pool is built ONCE (seed 20260614) and reused across all 12
cells. The pool itself is also persisted at
artifacts/e5_pilot_data/_shared_pool.jsonl for auditing.

Usage:
    PYTHONNOUSERSITE=1 python code/phase3/scripts/gen_e5_pilot_datasets.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Set PYTHONPATH so training.* imports work
PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
sys.path.insert(0, str(PROJECT_ROOT / "code/phase3"))

from training.e5_data_mixing import (
    SHARED_TASKS, build_shared_pool, build_mixed_train, save_jsonl,
)


# Task configs match the v1 LoRA training (artifacts/lora/qwen25_7b/*/v1)
TASK_CFGS = {
    "gsm8k": {
        "name": "gsm8k", "dataset": "openai/gsm8k", "config": "main",
        "split_train": "train", "split_eval": "test",
        "n_train": 7500, "n_eval": 1000,
        "prompt_field": "question", "answer_field": "answer",
    },
    "alpaca": {
        "name": "alpaca", "dataset": "yahma/alpaca-cleaned", "config": None,
        "split_train": "train", "split_eval": "train",
        "n_train": 7500, "n_eval": 1000,
        "prompt_field": "instruction", "answer_field": "output",
    },
    "magicoder": {
        "name": "magicoder", "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "config": None, "split_train": "train", "split_eval": "train",
        "n_train": 7500, "n_eval": 1000,
        "prompt_field": "problem", "answer_field": "solution",
    },
    "translation": {
        "name": "translation", "dataset": "wmt/wmt19", "config": "de-en",
        "split_train": "train", "split_eval": "validation",
        "n_train": 7500, "n_eval": 1000,
        "src_lang": "en", "tgt_lang": "de", "streaming": True,
    },
}


def main() -> int:
    out_dir = PROJECT_ROOT / "artifacts/e5_pilot_data/qwen25_7b"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[gen_e5] building shared pool (n=7500, seed=20260614) ...",
          flush=True)
    shared_pool = build_shared_pool(TASK_CFGS, n_total=7500, seed=20260614)
    pool_path = out_dir.parent / "_shared_pool.jsonl"
    save_jsonl(shared_pool, pool_path)
    src_counts = {t: sum(1 for r in shared_pool if r.get("_src") == t)
                  for t in SHARED_TASKS}
    print(f"[gen_e5] pool size={len(shared_pool)}  per-task={src_counts}",
          flush=True)
    print(f"[gen_e5] wrote {pool_path}", flush=True)

    for alpha in (0.0, 0.5, 0.9):
        alpha_str = f"alpha{int(alpha * 100):02d}"
        for task_name in SHARED_TASKS:
            target_cfg = TASK_CFGS[task_name]
            print(f"[gen_e5] mixing {task_name} at alpha={alpha} ...",
                  flush=True)
            mixed = build_mixed_train(target_cfg, shared_pool,
                                       alpha=alpha, n_total=7500,
                                       seed=20260614)
            out_path = out_dir / f"{task_name}__{alpha_str}.jsonl"
            save_jsonl(mixed, out_path)
            print(f"[gen_e5] wrote {out_path}  (n={len(mixed)})", flush=True)
    print("[gen_e5] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
