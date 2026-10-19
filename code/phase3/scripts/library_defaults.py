#!/usr/bin/env python3
"""Part L of notes/prereg_training_regimes_2026-10-19.md: Unsloth at defaults.

One process per (task, run), as a practitioner's separate fine-tuning runs
would be: load unsloth/Llama-3.2-1B-Instruct, FastLanguageModel.get_peft_model
with NO arguments beyond the model (every LoRA setting and random_state at the
library default), save A at creation, train 50 steps with SFTTrainer at its
default seed, save A again.

  python library_defaults.py run --task gsm8k --run 1
  python library_defaults.py compare      -> results/phase3/library_defaults.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
OUT = ROOT / "artifacts" / "library_defaults" / "unsloth"
BASE = "unsloth/Llama-3.2-1B-Instruct"
TASKS = {"gsm8k": ("openai/gsm8k", "main", "question", "answer"),
         "alpaca": ("yahma/alpaca-cleaned", None, "instruction", "output")}


def snapshot(model):
    import torch
    return {n: p.detach().float().cpu().clone() for n, p in model.named_parameters() if "lora_A" in n}


def cmd_run(a):
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch
    from datasets import load_dataset
    from safetensors.torch import save_file
    from trl import SFTConfig, SFTTrainer
    out = OUT / f"{a.task}_run{a.run}"
    out.mkdir(parents=True, exist_ok=True)
    model, tok = FastLanguageModel.from_pretrained(model_name=BASE, max_seq_length=512,
                                                   dtype=torch.bfloat16, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(model)          # library defaults only
    save_file(snapshot(model), str(out / "A_creation.safetensors"))
    repo, cfg, q, ans = TASKS[a.task]
    ds = load_dataset(repo, cfg, split="train").select(range(400))
    ds = ds.map(lambda r: {"text": f"{r[q]}\n{r[ans]}"}, remove_columns=ds.column_names)
    tr = SFTTrainer(model=model, train_dataset=ds, processing_class=tok,
                    args=SFTConfig(output_dir=str(out / "tmp"), max_steps=50,
                                   per_device_train_batch_size=4, report_to="none",
                                   dataset_text_field="text", max_length=512))
    tr.train()
    save_file(snapshot(model), str(out / "A_step50.safetensors"))
    import unsloth as U
    (out / "meta.json").write_text(json.dumps({"unsloth": getattr(U, "__version__", "?"),
                                               "peft_config": str(model.peft_config)}, indent=1))
    print("LIBDEF_DONE", a.task, a.run, flush=True)


def median_cos(sa, sb):
    import torch
    cos = []
    for k in sorted(set(sa) & set(sb)):
        qa = torch.linalg.qr(sa[k].double().T)[0]
        qb = torch.linalg.qr(sb[k].double().T)[0]
        cos.append(float(torch.linalg.svdvals(qa.T @ qb).clamp(max=1).median()))
    cos.sort()
    return cos[len(cos) // 2]


def cmd_compare(_a):
    from safetensors.torch import load_file
    res = {}
    for stage in ("A_creation", "A_step50"):
        a = load_file(str(OUT / "gsm8k_run1" / f"{stage}.safetensors"))
        b = load_file(str(OUT / "alpaca_run1" / f"{stage}.safetensors"))
        c = load_file(str(OUT / "gsm8k_run2" / f"{stage}.safetensors"))
        res[stage] = {"gsm8k_vs_alpaca": median_cos(a, b), "gsm8k_run1_vs_run2": median_cos(a, c)}
    res["meta"] = json.loads((OUT / "gsm8k_run1" / "meta.json").read_text())
    p = ROOT / "results" / "phase3" / "library_defaults.json"
    p.write_text(json.dumps({"unsloth": res}, indent=1))
    print(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "compare"])
    ap.add_argument("--task")
    ap.add_argument("--run", type=int, default=1)
    a = ap.parse_args()
    (cmd_run if a.cmd == "run" else cmd_compare)(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
