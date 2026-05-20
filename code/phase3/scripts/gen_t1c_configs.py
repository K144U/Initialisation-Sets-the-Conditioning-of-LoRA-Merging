"""Generate the 16 T1.C LoRA training configs (4 new models × 4 tasks).

Mirrors the shape of the existing configs/lora/{llama31_8b,qwen25_7b}_<task>.yaml
exactly — same LoRA hyperparams (rank 16, Q/K/V/O, lr 5e-5, 7500 train × 2 ep,
bs 8 ga 4, seq 2048) and same per-task dataset/split spec. Only base_model,
adapter_dir, results_path, and min_free_gb vary by model.

Run once; idempotent (overwrites).
"""

from pathlib import Path

import yaml


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
OUT_DIR = PROJECT_ROOT / "code/phase3/configs/lora"


# (model_key, hf_dir_name, min_free_gb for headroom over base + activations + adam state)
# Honest thresholds: ~Base bf16 (N*2 GB) + 9-11 GB activations/LoRA/Adam.
# Dropped 2026-05-19: gemma3_12b (Gemma3ForConditionalGeneration architecture risk)
# and qwen25_14b (cluster share-load makes 14B slot-finding unreliable).
MODELS = [
    ("mistral_7b",  "Mistral-7B-Instruct-v0.3",  30),
    ("yi15_9b",     "Yi-1.5-9B-Chat",            35),
]


TASKS = {
    "gsm8k": {
        "task": {
            "name": "gsm8k",
            "dataset": "openai/gsm8k",
            "config": "main",
            "split_train": "train",
            "split_eval": "test",
            "n_train": 7500,
            "n_eval": 200,
            "prompt_field": "question",
            "answer_field": "answer",
        },
        "adapter_subdir": "gsm8k",
    },
    "alpaca": {
        "task": {
            "name": "alpaca",
            "dataset": "yahma/alpaca-cleaned",
            "config": None,
            "split_train": "train",
            "split_eval": "train",
            "n_train": 7500,
            "n_eval": 200,
            "prompt_field": "instruction",
            "answer_field": "output",
        },
        "adapter_subdir": "alpaca",
    },
    "magicoder": {
        "task": {
            "name": "magicoder",
            "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "config": None,
            "split_train": "train",
            "split_eval": "train",
            "n_train": 7500,
            "n_eval": 200,
            "prompt_field": "problem",
            "answer_field": "solution",
        },
        "adapter_subdir": "magicoder",
    },
    "flores": {
        "task": {
            "name": "translation",
            "dataset": "wmt/wmt19",
            "config": "de-en",
            "split_train": "train",
            "split_eval": "validation",
            "n_train": 7500,
            "n_eval": 200,
            "src_lang": "en",
            "tgt_lang": "de",
            "streaming": True,
        },
        "adapter_subdir": "flores",
    },
}


LORA = {
    "r": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def train_block(min_free_gb: int) -> dict:
    return {
        "epochs": 2,
        "lr": 5.0e-5,
        "batch_size": 8,
        "grad_accum": 4,
        "max_seq_length": 2048,
        "warmup_ratio": 0.03,
        "optim": "adamw_torch",
        "logging_steps": 10,
        "dataset_num_proc": 4,
        "min_free_gb": min_free_gb,
    }


SEEDS = {"global": 20260519, "train": 20260519}


def build_one(model_key: str, hf_dir: str, min_free_gb: int,
              task_key: str, task_spec: dict) -> tuple[Path, dict]:
    adapter_subdir = task_spec["adapter_subdir"]
    cfg = {
        "base_model": f"/home/sanjay.g/projects/rdmerge/models/{hf_dir}",
        "task": task_spec["task"],
        "lora": LORA,
        "train": train_block(min_free_gb),
        "output": {
            "adapter_dir": f"artifacts/lora/{model_key}/{adapter_subdir}/v1",
            "results_path": f"results/phase3/lora_train/{model_key}_{task_key}_v1.json",
        },
        "seeds": SEEDS,
    }
    out_path = OUT_DIR / f"{model_key}_{task_key}.yaml"
    return out_path, cfg


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for model_key, hf_dir, min_free_gb in MODELS:
        for task_key, task_spec in TASKS.items():
            out_path, cfg = build_one(model_key, hf_dir, min_free_gb,
                                      task_key, task_spec)
            out_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            written.append(str(out_path))
    for p in written:
        print(p)
    print(f"\nwrote {len(written)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
