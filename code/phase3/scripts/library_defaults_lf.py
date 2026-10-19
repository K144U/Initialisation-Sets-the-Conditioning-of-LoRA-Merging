#!/usr/bin/env python3
"""Part L of notes/prereg_training_regimes_2026-10-19.md: LLaMA-Factory at defaults.

Runs `llamafactory-cli train` from the isolated venv .venv_lf with only the
fields a run cannot omit (model, stage, dataset, template, output, steps);
every LoRA setting and the seed are the library's defaults. A at creation is
read from a 1-step run at learning rate 0; A after training from a 50-step run.

  .venv_lf/bin/python code/phase3/scripts/library_defaults_lf.py run --task gsm8k --run 1
  python code/phase3/scripts/library_defaults.py compare   (reads both libraries)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
OUT = ROOT / "artifacts" / "library_defaults" / "llamafactory"
DATA = ROOT / "cache" / "lf_data"
BASE = "unsloth/Llama-3.2-1B-Instruct"
TASKS = {"gsm8k": ("openai/gsm8k", "main", "question", "answer"),
         "alpaca": ("yahma/alpaca-cleaned", None, "instruction", "output")}


def build_data():
    from datasets import load_dataset
    DATA.mkdir(parents=True, exist_ok=True)
    info = {}
    for t, (repo, cfg, q, a) in TASKS.items():
        p = DATA / f"{t}_small.json"
        if not p.exists():
            ds = load_dataset(repo, cfg, split="train").select(range(400))
            p.write_text(json.dumps([{"instruction": r[q], "input": "", "output": r[a]} for r in ds]))
        info[f"{t}_small"] = {"file_name": p.name}
    (DATA / "dataset_info.json").write_text(json.dumps(info, indent=1))


def train(task, out, steps, lr):
    cfg = {"model_name_or_path": BASE, "stage": "sft", "do_train": True,
           "finetuning_type": "lora", "dataset": f"{task}_small", "dataset_dir": str(DATA),
           "template": "llama3", "cutoff_len": 512, "output_dir": str(out),
           "max_steps": steps, "per_device_train_batch_size": 4, "report_to": "none",
           "overwrite_output_dir": True, "save_strategy": "no"}
    if lr is not None:
        cfg["learning_rate"] = lr
    out.mkdir(parents=True, exist_ok=True)
    y = out / "train.yaml"
    import yaml
    y.write_text(yaml.safe_dump(cfg))
    cli = Path(sys.executable).parent / "llamafactory-cli"
    subprocess.run([str(cli), "train", str(y)], check=True)


def a_only(adapter_dir, dst):
    from safetensors.torch import load_file, save_file
    sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
    save_file({k: v.float().contiguous() for k, v in sd.items() if "lora_A" in k}, str(dst))


def cmd_run(a):
    build_data()
    out = OUT / f"{a.task}_run{a.run}"
    train(a.task, out / "creation", 1, 0.0)
    a_only(out / "creation", out / "A_creation.safetensors")
    train(a.task, out / "step50", 50, None)
    a_only(out / "step50", out / "A_step50.safetensors")
    import llamafactory
    (out / "meta.json").write_text(json.dumps({
        "llamafactory": llamafactory.__version__,
        "adapter_config": json.loads((out / "step50" / "adapter_config.json").read_text())}, indent=1))
    print("LIBDEF_LF_DONE", a.task, a.run, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--task")
    ap.add_argument("--run", type=int, default=1)
    cmd_run(ap.parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
