"""Train a single LoRA adapter on one task. Driver for Phase 3 Day 1+.

Usage:
    python train_lora.py --config configs/lora/<model>_<task>.yaml

Reads YAML config, loads base model via Unsloth at full bf16, trains
a LoRA on the configured task, saves the adapter + a JSON metrics blob.
"""

# unsloth_zoo (>=2026-06 env) introspects torch._inductor.config at import
# time, which is lazy in this torch build — import it explicitly first.
import torch._inductor.config  # noqa: F401

# Unsloth must import before transformers.
import unsloth  # noqa: F401  pylint: disable=unused-import

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from training.data_loaders import load_task_split
from utils.io_helpers import write_result
from utils.pbs_env import assert_env
from utils.seeds import seed_everything


def _build_sft_dataset(train_data: list[dict], tokenizer) -> Dataset:
    eos = tokenizer.eos_token or "<|eot_id|>"
    return Dataset.from_list([
        {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": ex["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            "completion": ex["answer"] + eos,
        }
        for ex in train_data
    ])


def train_one_lora(cfg: dict, project_root: Path) -> dict:
    """Train one LoRA per the config. Returns metrics dict."""
    task = cfg["task"]
    task_name = task["name"]
    base = cfg["base_model"]
    out_dir = project_root / cfg["output"]["adapter_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg["seeds"]["global"])

    # lora.init selects a non-default initialisation (e.g. "pissa"). Unsloth
    # has no path for these, so they go through plain transformers + PEFT.
    # Every config without lora.init takes the Unsloth path exactly as before.
    lora_init = cfg["lora"].get("init")
    # notes/prereg_training_regimes_2026-10-19.md options. All default to the
    # historical behaviour, so every existing config trains byte-identically.
    use_dora = bool(cfg["lora"].get("use_dora", False))
    init_seed = int(cfg["seeds"].get("init", cfg["seeds"]["train"]))
    t_load = time.time()
    if lora_init or use_dora:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[train:{task_name}] loading {base} via plain PEFT "
              f"(init_lora_weights={lora_init!r})", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="cuda:0",
            attn_implementation="sdpa")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        print(f"[train:{task_name}] base loaded in {time.time()-t_load:.1f}s", flush=True)
        torch.manual_seed(init_seed)
        model = get_peft_model(model, LoraConfig(
            r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            init_lora_weights=lora_init if lora_init else True,
            use_dora=use_dora, task_type="CAUSAL_LM"))
    else:
        print(f"[train:{task_name}] loading {base} via Unsloth", flush=True)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base,
            max_seq_length=cfg["train"]["max_seq_length"],
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )
        print(f"[train:{task_name}] base loaded in {time.time()-t_load:.1f}s", flush=True)

        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["alpha"],
            lora_dropout=cfg["lora"]["dropout"],
            target_modules=cfg["lora"]["target_modules"],
            use_gradient_checkpointing="unsloth",
            random_state=init_seed,
        )

    # Continue from a parent adapter (Part C): copy its A and B into the new
    # LoRA before the first step.
    if cfg["lora"].get("init_from"):
        from safetensors.torch import load_file
        parent = load_file(str(Path(cfg["lora"]["init_from"]) / "adapter_model.safetensors"))
        parent = {k.replace(".default.weight", ".weight"): v for k, v in parent.items()}
        n_copied = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                key = name.replace(".default.weight", ".weight")
                if ("lora_A" in key or "lora_B" in key) and key in parent:
                    param.copy_(parent[key].to(param.dtype))
                    n_copied += 1
        if n_copied == 0:
            raise RuntimeError("init_from matched no LoRA tensors")
        print(f"[train:{task_name}] initialised {n_copied} LoRA tensors from "
              f"{cfg['lora']['init_from']}", flush=True)

    # Scale A_0 (Part K): applied before the A_0 snapshot below.
    if cfg["lora"].get("init_scale") not in (None, 1, 1.0):
        c = float(cfg["lora"]["init_scale"])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A" in name:
                    param.mul_(c)
        print(f"[train:{task_name}] scaled A_0 by {c}", flush=True)

    # Snapshot the LoRA A factors BEFORE any optimiser step, for the drift
    # measurement registered in notes/prereg_drift_2026-08-16.md. Off by
    # default so every existing config trains byte-identically; the
    # registration forbids reconstructing A_0 from a seed after the fact,
    # since that would be a claim about RNG determinism and not a measurement.
    if cfg.get("save_init_weights"):
        from safetensors.torch import save_file
        init = {}
        for name, param in model.named_parameters():
            if "lora_A" in name:
                init[name] = param.detach().clone().to(torch.float32).cpu()
        if not init:
            raise RuntimeError(
                "save_init_weights set but no lora_A parameters found; "
                "the drift measurement would be undefined")
        a0_path = out_dir / "adapter_A0.safetensors"
        save_file(init, str(a0_path))
        print(f"[train:{task_name}] wrote {len(init)} lora_A tensors to "
              f"{a0_path.name} before the first step", flush=True)

    # Data-shuffle seed, decoupled from the LoRA-init seed. Defaults to
    # seeds.global so every existing config behaves exactly as before.
    #
    # Two reasons it exists (2026-08-02):
    #  1. The A1 independent-init experiment varies the LoRA init per task. If
    #     that also moved the data shuffle, a change in subspace geometry could
    #     not be attributed to the init, which is the whole point of the test.
    #  2. load_task_split only guarantees train/eval disjointness when the
    #     TRAINING run and the EVAL cell shuffle with the SAME seed (it slices
    #     one shuffled split into [0:n_train] and [n_train:n_train+n_eval]).
    #     Training used seeds.global (1/2/3) while eval cells use seed 20260518,
    #     so for alpaca and magicoder, whose eval split IS the train split, the
    #     "held-out" set overlapped training by roughly 14.5% and 10%. Setting
    #     seeds.data to the eval cell's seed closes that.
    data_seed = int(cfg["seeds"].get("data", cfg["seeds"]["global"]))
    if data_seed != cfg["seeds"]["global"]:
        print(f"[train:{task_name}] data-shuffle seed {data_seed} "
              f"(init seed {cfg['seeds']['global']})", flush=True)

    # E5 hook: if task.train_jsonl is set, load mixed training data from disk
    # (built by gen_e5_pilot_datasets.py). Eval ALWAYS comes from
    # load_task_split so all alphas share the same target-task held-out set.
    train_jsonl = task.get("train_jsonl")
    if train_jsonl:
        import json
        from pathlib import Path as _P
        jp = _P(str(project_root) + "/" + train_jsonl) if not _P(train_jsonl).is_absolute() else _P(train_jsonl)
        print(f"[train:{task_name}] loading TRAIN from JSONL: {jp}", flush=True)
        train_data = [json.loads(line) for line in open(jp) if line.strip()]
        _, eval_data = load_task_split(task, seed=data_seed)
    else:
        print(f"[train:{task_name}] loading {task['dataset']} ({task.get('config') or '-'})", flush=True)
        train_data, eval_data = load_task_split(task, seed=data_seed)
    print(f"[train:{task_name}] train={len(train_data)}  eval={len(eval_data)}", flush=True)

    ds_train = _build_sft_dataset(train_data, tokenizer)

    sft_cfg = SFTConfig(
        output_dir=str(out_dir / "trainer_tmp"),
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        optim=cfg["train"]["optim"],
        bf16=True,
        logging_steps=max(1, cfg["train"].get("logging_steps", 10)),
        # Drift runs need intermediate checkpoints; everything else keeps the
        # historical behaviour of one final checkpoint. A float save_steps in
        # (0, 1) is read by HF as a fraction of total training steps, so 0.25
        # gives exactly the 25/50/75/100% grid the drift registration asks for.
        # save_total_limit must be lifted too, or the earlier ones are deleted
        # as the later ones are written, which is what left the original
        # cohorts unable to answer E3.
        **({"save_strategy": "steps",
            "save_steps": float(cfg["train"]["checkpoint_fraction"]),
            "save_total_limit": None}
           if cfg["train"].get("checkpoint_fraction")
           else {"save_strategy": "epoch", "save_total_limit": 1}),
        report_to="none",
        seed=cfg["seeds"]["train"],
        max_length=cfg["train"]["max_seq_length"],
        completion_only_loss=True,
        dataset_num_proc=cfg["train"].get("dataset_num_proc", 4),
    )

    # Pass tokenizer explicitly via processing_class — Unsloth's patched
    # SFTTrainer.__init__ calls fix_untrained_tokens(model, tokenizer, ...)
    # which crashes if tokenizer is None. Loading from a local path
    # (as we are here) doesn't set the implicit tokenizer slot that the
    # HF-repo path sets, so we must pass it explicitly.
    # Separate learning rates for A and B (Part K: A-lr x4; LoRA+ ratio 16).
    # Same AdamW defaults as optim=adamw_torch; the scheduler is still built by
    # the trainer on this optimizer.
    optimizers = (None, None)
    mult_A = float(cfg["train"].get("lr_A_mult", 1.0))
    mult_B = float(cfg["train"].get("lr_B_mult", 1.0))
    if (mult_A, mult_B) != (1.0, 1.0):
        lr = float(cfg["train"]["lr"])
        pA = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and "lora_A" in n_]
        pB = [p_ for n_, p_ in model.named_parameters() if p_.requires_grad and "lora_A" not in n_]
        opt = torch.optim.AdamW([{"params": pA, "lr": lr * mult_A},
                                 {"params": pB, "lr": lr * mult_B}],
                                lr=lr, weight_decay=sft_cfg.weight_decay)
        optimizers = (opt, None)
        print(f"[train:{task_name}] lr A {lr*mult_A:g}, lr B {lr*mult_B:g}", flush=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds_train,
        args=sft_cfg,
        processing_class=tokenizer,
        optimizers=optimizers,
    )

    print(f"[train:{task_name}] training starts; {cfg['train']['epochs']}ep "
          f"x bs{cfg['train']['batch_size']}*ga{cfg['train']['grad_accum']} "
          f"(eff {cfg['train']['batch_size']*cfg['train']['grad_accum']})", flush=True)
    t_train = time.time()
    train_result = trainer.train()
    train_runtime_s = time.time() - t_train

    loss_history = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    print(f"[train:{task_name}] training done in {train_runtime_s:.1f}s, "
          f"{len(loss_history)} log entries", flush=True)

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[train:{task_name}] adapter saved to {out_dir}", flush=True)

    adapter_path = out_dir / "adapter_model.safetensors"
    adapter_size_mb = adapter_path.stat().st_size / (1024 ** 2) if adapter_path.exists() else 0.0

    # In-process NLL eval on this LoRA over its own held-out (sanity check).
    print(f"[train:{task_name}] running NLL eval on held-out (n={len(eval_data)})", flush=True)
    if not (lora_init or use_dora):
        FastLanguageModel.for_inference(model)
    nll_tau = _compute_nll(model, tokenizer, eval_data, cfg["train"]["max_seq_length"])
    print(f"[train:{task_name}] NLL_tau on held-out = {nll_tau:.4f}", flush=True)

    metrics = {
        "task": task_name,
        "base_model": base,
        "train_runtime_s": train_runtime_s,
        "first_loss": float(loss_history[0]) if loss_history else None,
        "last_loss": float(loss_history[-1]) if loss_history else None,
        "n_log_steps": len(loss_history),
        "loss_history": [float(x) for x in loss_history],
        "adapter_size_mb": adapter_size_mb,
        "nll_tau_on_eval": nll_tau,
        "n_train": len(train_data),
        "n_eval": len(eval_data),
        "adapter_dir": str(out_dir),
    }

    # Free memory before exit
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return metrics


def _compute_nll(model, tokenizer, eval_data: list[dict], max_len: int) -> float:
    """Mean NLL nats per assistant-response token. Template-agnostic."""
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for ex in eval_data:
            user_msgs = [{"role": "user", "content": ex["prompt"]}]
            full_msgs = user_msgs + [{"role": "assistant", "content": ex["answer"]}]

            # Prompt-only with generation prompt -> defines the mask boundary.
            prompt_text = tokenizer.apply_chat_template(
                user_msgs, tokenize=False, add_generation_prompt=True,
            )
            prompt_ids = tokenizer(
                prompt_text, return_tensors="pt", add_special_tokens=False,
            ).input_ids

            # Full conversation -> what we score NLL over.
            full_text = tokenizer.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False,
            )
            ids = tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=max_len, add_special_tokens=False,
            ).input_ids.to(device)

            prompt_len = prompt_ids.shape[1]
            if prompt_len >= ids.shape[1]:
                continue

            labels = ids.clone()
            labels[:, :prompt_len] = -100
            n_answer = (labels != -100).sum().item()
            if n_answer == 0:
                continue

            out = model(input_ids=ids, labels=labels)
            total_nll += out.loss.item() * n_answer
            total_tokens += n_answer

    return total_nll / total_tokens if total_tokens else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    env_info = assert_env(min_free_gb=cfg.get("train", {}).get("min_free_gb", 40.0))
    print(f"[train] env OK: {env_info}", flush=True)

    project_root = Path(os.environ["PROJECT_ROOT"])

    t0 = time.time()
    metrics = train_one_lora(cfg, project_root)
    wallclock_s = time.time() - t0

    results_path = project_root / cfg["output"]["results_path"]
    payload = {
        "metrics": metrics,
        "wallclock_seconds": wallclock_s,
        "env": env_info,
    }
    write_result(results_path, payload, config=cfg)
    print(f"[train] wrote {results_path}", flush=True)
    print(f"[train] wallclock={wallclock_s:.1f}s", flush=True)
    print(f"TRAIN_DONE: task={metrics['task']} base={metrics['base_model']} "
          f"nll_tau={metrics['nll_tau_on_eval']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
