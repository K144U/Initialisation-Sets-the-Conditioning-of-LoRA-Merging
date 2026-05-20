"""Generate + submit all Phase 3 eval-matrix cells.

Cells:
  - 4 non-TVQ methods (TA, TIES, DARE, KnOTS) × 2 models = 8 cells
  - TVQ × 6 rates ({1,2,4,8,16,32} bits) × 2 models = 12 cells
  Total: 20 cells.

  Skips cells whose result JSON already exists (e.g. the TA+Llama cell from
  the Day-4 pilot run).

Respects the 3-concurrent-job cap on the gpu queue.
"""

import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
# Overridable via env vars (added 2026-05-19 for v3 per-example re-eval):
#   RDMERGE_EVAL_RESULTS_DIR  → results/phase3/eval_matrix_n1k_v2 by default
#   RDMERGE_EVAL_CONFIGS_DIR  → code/phase3/configs/eval_n1k_v2 by default
#   RDMERGE_EVAL_LAUNCH_LOG   → logs/launch_eval_matrix_n1k_v2.log by default
import os as _os
CONFIGS_DIR = Path(_os.environ.get(
    "RDMERGE_EVAL_CONFIGS_DIR",
    str(PROJECT_ROOT / "code/phase3/configs/eval_n1k_v2"),
))
RESULTS_DIR = Path(_os.environ.get(
    "RDMERGE_EVAL_RESULTS_DIR",
    str(PROJECT_ROOT / "results/phase3/eval_matrix_n1k_v2"),
))
LAUNCH_LOG = Path(_os.environ.get(
    "RDMERGE_EVAL_LAUNCH_LOG",
    str(PROJECT_ROOT / "logs/launch_eval_matrix_n1k_v2.log"),
))
PBS_SH = PROJECT_ROOT / "code/phase3/scripts/pbs_eval_cell.sh"
MAX_CONCURRENT = 3


MODELS = {
    "llama31_8b": "/home/sanjay.g/projects/rdmerge/models/Llama-3.1-8B-Instruct",
    "qwen25_7b":  "/home/sanjay.g/projects/rdmerge/models/Qwen2.5-7B-Instruct",
    # T1.D — added 2026-05-19 (robustness panel: 4 architecture families):
    "mistral_7b": "/home/sanjay.g/projects/rdmerge/models/Mistral-7B-Instruct-v0.3",
    "yi15_9b":    "/home/sanjay.g/projects/rdmerge/models/Yi-1.5-9B-Chat",
}

ADAPTER_DIRS = {
    ("llama31_8b", "gsm8k"):       "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/gsm8k/v1",
    ("llama31_8b", "alpaca"):      "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/alpaca/v1",
    ("llama31_8b", "magicoder"):   "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/magicoder/v1",
    ("llama31_8b", "translation"): "/home/sanjay.g/projects/rdmerge/artifacts/lora/llama31_8b/flores/v1",
    ("qwen25_7b",  "gsm8k"):       "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/gsm8k/v1",
    ("qwen25_7b",  "alpaca"):      "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/alpaca/v1",
    ("qwen25_7b",  "magicoder"):   "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/magicoder/v1",
    ("qwen25_7b",  "translation"): "/home/sanjay.g/projects/rdmerge/artifacts/lora/qwen25_7b/flores/v1",
    ("mistral_7b", "gsm8k"):       "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/gsm8k/v1",
    ("mistral_7b", "alpaca"):      "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/alpaca/v1",
    ("mistral_7b", "magicoder"):   "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/magicoder/v1",
    ("mistral_7b", "translation"): "/home/sanjay.g/projects/rdmerge/artifacts/lora/mistral_7b/flores/v1",
    ("yi15_9b",    "gsm8k"):       "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/gsm8k/v1",
    ("yi15_9b",    "alpaca"):      "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/alpaca/v1",
    ("yi15_9b",    "magicoder"):   "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/magicoder/v1",
    ("yi15_9b",    "translation"): "/home/sanjay.g/projects/rdmerge/artifacts/lora/yi15_9b/flores/v1",
}

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
        "name": "magicoder", "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K", "config": None,
        "split_train": "train", "split_eval": "train",
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

TASKS_IN_ORDER = ["gsm8k", "alpaca", "magicoder", "translation"]


def build_config(model_key: str, method: str, rate_bits: int | None = None) -> tuple[str, dict]:
    """Return (cell_name, config_dict)."""
    if method == "tvq" and rate_bits is None:
        raise ValueError("tvq needs rate_bits")
    method_kwargs: dict = {}
    if method == "tvq":
        method_kwargs["rate_bits"] = rate_bits
        cell_name = f"{model_key}__tvq_b{rate_bits}"
    else:
        cell_name = f"{model_key}__{method}"
    cfg = {
        "base_model": MODELS[model_key],
        "method": method,
        "method_kwargs": method_kwargs,
        "max_seq_length": 1024,
        "seed": 20260518,
        "min_free_gb": 25,
        "adapter_specs": [
            {
                "name": task,
                "dir": ADAPTER_DIRS[(model_key, task)],
                "task_cfg": TASK_CFGS[task],
            }
            for task in TASKS_IN_ORDER
        ],
        "output_path": str(RESULTS_DIR / f"{cell_name}.json"),
    }
    return cell_name, cfg


def n_running() -> int:
    out = subprocess.run(
        ["qstat", "-u", os.environ.get("USER", "sanjay.g")],
        capture_output=True, text=True,
    ).stdout
    return sum(1 for line in out.splitlines() if "rdm_eval" in line)


def main() -> int:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCH_LOG.parent.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[str, dict]] = []
    # Non-TVQ: 4 methods × 2 models
    for model in MODELS:
        for method in ["task_arithmetic", "ties", "dare", "knots"]:
            cells.append(build_config(model, method))
    # TVQ rate sweep
    for model in MODELS:
        for b in [1, 2, 4, 8, 16, 32]:
            cells.append(build_config(model, "tvq", rate_bits=b))

    with LAUNCH_LOG.open("a") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launch_eval_matrix start — {len(cells)} cells\n")

    submitted = 0
    skipped = 0
    for cell_name, cfg in cells:
        out_json = PROJECT_ROOT / cfg["output_path"]
        if out_json.exists():
            print(f"SKIP  {cell_name} — JSON already exists", flush=True)
            skipped += 1
            with LAUNCH_LOG.open("a") as f:
                f.write(f"  skip {cell_name} (exists)\n")
            continue

        cfg_path = CONFIGS_DIR / f"{cell_name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

        while n_running() >= MAX_CONCURRENT:
            time.sleep(60)

        result = subprocess.run(
            ["qsub", "-v", f"CONFIG={cfg_path}", str(PBS_SH)],
            capture_output=True, text=True,
        )
        jobid = result.stdout.strip()
        print(f"QSUB  {cell_name} -> {jobid}", flush=True)
        submitted += 1
        with LAUNCH_LOG.open("a") as f:
            f.write(f"  submitted {cell_name} -> {jobid}\n")

    print(f"\n[launch] {submitted} submitted, {skipped} skipped (already done)", flush=True)

    # Block until all our eval jobs drain
    while n_running() > 0:
        time.sleep(120)
    print("ALL_EVAL_CELLS_DONE", flush=True)
    with LAUNCH_LOG.open("a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ALL_EVAL_CELLS_DONE\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
