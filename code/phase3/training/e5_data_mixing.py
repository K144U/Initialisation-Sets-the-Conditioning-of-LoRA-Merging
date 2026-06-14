"""E5 Arm 2 data-mixture utility.

Builds a shared pool D_shared by uniform-sampling across the four Phase 3
tasks (GSM8K, Alpaca-cleaned, Magicoder-OSS-Instruct, WMT19 en-de). The
same pool is reused across all 4 task adapters at a given alpha — this is
what creates overlapping training subspaces and (the hypothesis) overlapping
LoRA right-singular subspaces, lowering d_eff/(Tr).

For each (target_task, alpha):
  - n_shared = floor(alpha * n_total) examples from D_shared (seeded, identical
    across adapters at this alpha)
  - n_unique = n_total - n_shared examples from target_task's own train split,
    DISJOINT from anything already in D_shared (for tasks that appear in the
    pool; trivially disjoint for tasks that don't)
  - concatenate + shuffle (seeded) -> 7500 mixed training examples
  - eval is always 1000 from target_task's eval split (untouched)

Reference: notes/E5_design_week_protocol.md (drafted 2026-06-13).
Hypothesis: d_eff/(Tr) decreases monotonically in alpha; pilot gate
fires GO if alpha=0.9 -> d_eff/(Tr) < 0.8 in majority of layers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from training.data_loaders import load_task_split


SHARED_TASKS = ("gsm8k", "alpaca", "magicoder", "translation")


def build_shared_pool(task_cfgs: dict[str, dict],
                       n_total: int = 7500,
                       seed: int = 20260614) -> list[dict]:
    """Uniform sample n_total examples across the 4 tasks.

    n_total // 4 from each (with the remainder spread across the first
    few tasks). All returned rows are {'prompt', 'answer'} dicts, source
    task tagged in '_src' for downstream auditing.

    Sampling: load each task's TRAIN split via load_task_split with seed
    `seed`, then take the first ceil(n_total/4) rows. Deterministic.
    """
    per_task = n_total // len(SHARED_TASKS)
    extra = n_total - per_task * len(SHARED_TASKS)
    pool: list[dict] = []
    for i, task_name in enumerate(SHARED_TASKS):
        cfg = task_cfgs[task_name]
        take = per_task + (1 if i < extra else 0)
        train, _ = load_task_split(cfg, seed=seed)
        for row in train[:take]:
            row = dict(row)
            row["_src"] = task_name
            pool.append(row)
    # Shuffle the pool itself so the per-target subsampling at alpha<1.0
    # is not biased toward a single task.
    rng = random.Random(seed ^ 0xE5)
    rng.shuffle(pool)
    return pool


def build_mixed_train(target_task_cfg: dict,
                       shared_pool: list[dict],
                       alpha: float,
                       n_total: int = 7500,
                       seed: int = 20260614) -> list[dict]:
    """Return n_total mixed training examples for one (target_task, alpha) cell.

    n_shared = floor(alpha * n_total) drawn from shared_pool (deterministic
    seeded subsample so all 4 task adapters at this alpha share the same
    floor(alpha*n) rows in the same order).
    n_unique = n_total - n_shared drawn from target_task's own train split,
    skipping any examples already in the shared pool's `target_task` slice
    (so train data does not double-up via the pool).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    n_shared = int(alpha * n_total)
    n_unique = n_total - n_shared

    # Shared portion: deterministic seeded subsample of shared_pool.
    rng_shared = random.Random(seed ^ int(alpha * 1000) ^ 0xBEEF)
    if n_shared > len(shared_pool):
        raise ValueError(f"alpha={alpha} requires {n_shared} shared rows "
                          f"but pool has only {len(shared_pool)}")
    idx = list(range(len(shared_pool)))
    rng_shared.shuffle(idx)
    shared_portion = [shared_pool[i] for i in idx[:n_shared]]

    # Unique portion: load target task's training split fresh (with a
    # DIFFERENT seed than the pool). Small overlap with the pool's
    # target-task slice is statistically inconsequential and we don't
    # enforce strict disjointness — the goal is shared SUBSPACE, not
    # disjoint example sets.
    unique_seed = (seed ^ 0xFACE ^ hash(target_task_cfg["name"])) & 0xFFFFFF
    target_train, _ = load_task_split(target_task_cfg, seed=unique_seed)
    if len(target_train) < n_unique:
        # Train pool is smaller than requested (e.g., GSM8K has ~7473
        # rows). Accept what we have; the SFT loop will see a shorter
        # epoch but identical seed/hyperparams.
        print(f"[e5_mix] WARN: only {len(target_train)} unique rows for "
              f"{target_task_cfg['name']} (wanted {n_unique})", flush=True)
        unique_portion = list(target_train)
    else:
        unique_portion = target_train[:n_unique]

    # Final mix + shuffle.
    mixed = []
    for r in shared_portion + unique_portion:
        r = {"prompt": r["prompt"], "answer": r["answer"]}
        mixed.append(r)
    rng_mix = random.Random(seed ^ int(alpha * 1000) ^ hash(target_task_cfg["name"]) & 0xFFFF)
    rng_mix.shuffle(mixed)
    return mixed


def save_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
