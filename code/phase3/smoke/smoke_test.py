"""Smoke test — gates all Day 1+ Phase 3 work.

Loads Llama-3.2-1B-Instruct via Unsloth at full bf16, trains two LoRAs
(GSM8K, Alpaca) on 100 examples each, merges via Task Arithmetic
(0.5/0.5), evaluates NLL on 50 held-out per task for each of
{base, tau_gsm8k, tau_alpaca, merged}, and verifies the 9 pass
criteria from the Phase 3 bootstrap plan.

Prints SMOKE_TEST_GREEN as the final stdout line iff all pass.
Writes JSON to results/phase3/smoke/smoke_<timestamp>.json.

Usage: python smoke_test.py --config smoke_config.yaml
"""

# IMPORTANT: Unsloth must import before transformers so it can patch the kernels.
import unsloth  # noqa: F401  pylint: disable=unused-import

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

from utils.io_helpers import write_result
from utils.pbs_env import assert_env
from utils.seeds import seed_everything


# ----------------------------- data helpers -----------------------------

def _format_alpaca_prompt(ex: dict) -> str:
    instr = ex.get("instruction", "")
    inp = ex.get("input", "") or ""
    if inp.strip():
        return f"{instr}\n\nInput: {inp}"
    return instr


def load_task_split(task_cfg: dict, n_train: int, n_eval: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Returns (train, eval) lists of {'prompt': ..., 'answer': ...} dicts."""
    name = task_cfg["dataset"]
    cfg = task_cfg.get("config")
    ds = load_dataset(name, cfg) if cfg else load_dataset(name)

    train_split_name = task_cfg["split_train"]
    eval_split_name = task_cfg["split_eval"]

    train_raw = ds[train_split_name].shuffle(seed=seed).select(range(n_train))

    if eval_split_name == train_split_name:
        # take a disjoint slice for eval (e.g. Alpaca only has 'train')
        eval_raw = ds[train_split_name].shuffle(seed=seed + 1).select(range(n_train, n_train + n_eval))
    else:
        eval_raw = ds[eval_split_name].shuffle(seed=seed + 1).select(range(n_eval))

    p_field = task_cfg["prompt_field"]
    a_field = task_cfg["answer_field"]

    def _row_to_dict(row: dict) -> dict:
        if p_field == "instruction":
            return {"prompt": _format_alpaca_prompt(row), "answer": row[a_field]}
        return {"prompt": row[p_field], "answer": row[a_field]}

    return [_row_to_dict(r) for r in train_raw], [_row_to_dict(r) for r in eval_raw]


def make_messages(prompt: str, answer: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]


def llama3_response_template_ids(tokenizer) -> list[int]:
    """The token id span that marks the start of the assistant response."""
    tmpl = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return tokenizer.encode(tmpl, add_special_tokens=False)


# ----------------------------- training step -----------------------------

def train_one_lora(
    task_name: str,
    train_data: list[dict],
    base_model_name: str,
    out_dir: Path,
    cfg: dict,
) -> dict:
    """Trains one LoRA adapter via Unsloth + SFT. Returns metrics dict."""
    print(f"[train:{task_name}] loading base via Unsloth", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=cfg["train"]["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seeds"]["train"],
    )

    # Use prompt/completion dataset format with completion_only_loss.
    # This avoids the chat template needing {% generation %} markers,
    # which Llama-3.2-Instruct's stock template lacks.
    from datasets import Dataset
    eos = tokenizer.eos_token or "<|eot_id|>"
    ds = Dataset.from_list([
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

    sft_cfg = SFTConfig(
        output_dir=str(out_dir / "trainer_tmp"),
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        optim=cfg["train"]["optim"],
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=cfg["seeds"]["train"],
        max_length=cfg["train"]["max_seq_length"],
        completion_only_loss=True,
        dataset_num_proc=cfg["train"].get("dataset_num_proc", 2),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=sft_cfg,
    )

    print(f"[train:{task_name}] starting; n_examples={len(ds)}", flush=True)
    train_result = trainer.train()
    loss_history = [r["loss"] for r in trainer.state.log_history if "loss" in r]
    print(f"[train:{task_name}] losses: first={loss_history[0]:.4f} "
          f"last={loss_history[-1]:.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[train:{task_name}] saved adapter to {out_dir}", flush=True)

    # Adapter size on disk (in MB)
    adapter_file = out_dir / "adapter_model.safetensors"
    size_mb = adapter_file.stat().st_size / (1024 ** 2) if adapter_file.exists() else 0.0

    # Free Unsloth model before next call
    del trainer, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "first_loss": float(loss_history[0]),
        "last_loss": float(loss_history[-1]),
        "n_steps": len(loss_history),
        "adapter_size_mb": float(size_mb),
        "train_runtime_s": float(train_result.metrics.get("train_runtime", 0.0)),
    }


# ----------------------------- eval step -----------------------------

def compute_nll(model, tokenizer, eval_data: list[dict], max_len: int, device: torch.device) -> float:
    """Mean NLL nats per assistant-response token."""
    total_nll, total_tokens = 0.0, 0
    model.eval()
    response_ids = llama3_response_template_ids(tokenizer)
    n_resp = len(response_ids)

    with torch.no_grad():
        for ex in eval_data:
            msgs = make_messages(ex["prompt"], ex["answer"])
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len, add_special_tokens=False)
            ids = enc.input_ids.to(device)

            # find response template span in ids[0]
            ids_list = ids[0].tolist()
            response_start = None
            for i in range(len(ids_list) - n_resp + 1):
                if ids_list[i : i + n_resp] == response_ids:
                    response_start = i + n_resp
                    break
            if response_start is None or response_start >= ids.shape[1]:
                continue  # malformed; skip

            labels = ids.clone()
            labels[:, :response_start] = -100
            n_answer = (labels != -100).sum().item()
            if n_answer == 0:
                continue

            out = model(input_ids=ids, labels=labels)
            total_nll += out.loss.item() * n_answer
            total_tokens += n_answer

    if total_tokens == 0:
        return float("nan")
    return total_nll / total_tokens


def evaluate_all_states(
    base_model_name: str,
    adapter_dirs: dict[str, Path],
    eval_data: dict[str, list[dict]],
    cfg: dict,
) -> dict[str, dict[str, float]]:
    """Returns nll[state][task] for state in {base, gsm8k, alpaca, merged}.

    Unsloth's import patches LlamaForCausalLM.forward globally — that patched
    forward expects `apply_qkv` which is only set up via FastLanguageModel.
    So we load the eval model via Unsloth (passing an adapter dir to
    from_pretrained brings up base + adapter + apply_qkv in one call).
    """
    print("[eval] loading base+gsm8k via Unsloth (adapter name 'default')", flush=True)
    pmodel, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_dirs["gsm8k"]),
        max_seq_length=cfg["train"]["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(pmodel)
    device = next(pmodel.parameters()).device

    print("[eval] loading alpaca adapter", flush=True)
    pmodel.load_adapter(str(adapter_dirs["alpaca"]), adapter_name="alpaca")

    print("[eval] building merged adapter via Task Arithmetic 0.5/0.5", flush=True)
    # PeftModel forwards add_weighted_adapter to its inner LoraModel.
    # Unsloth named the first adapter "default" — keep that name internally.
    target = pmodel.base_model if hasattr(pmodel, "base_model") else pmodel
    target.add_weighted_adapter(
        adapters=["default", "alpaca"],
        weights=cfg["merge"]["weights"],
        adapter_name="merged",
        combination_type="linear",
    )

    # state name -> PEFT adapter name (None for disabled-adapter base)
    state_to_adapter = {
        "base": None,
        "gsm8k": "default",
        "alpaca": "alpaca",
        "merged": "merged",
    }

    nll: dict[str, dict[str, float]] = {}
    max_len = cfg["train"]["max_seq_length"]

    for state, adapter in state_to_adapter.items():
        if adapter is None:
            ctx = pmodel.disable_adapter()
        else:
            pmodel.set_adapter(adapter)
            ctx = _NullCtx()
        with ctx:
            nll[state] = {}
            for task in ["gsm8k", "alpaca"]:
                v = compute_nll(pmodel, tokenizer, eval_data[task], max_len, device)
                nll[state][task] = v
                print(f"[eval] state={state:<8} task={task:<8} nll={v:.4f}", flush=True)

    return nll


class _NullCtx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# ----------------------------- pass-criteria check -----------------------------

def check_pass_criteria(
    train_metrics: dict[str, dict],
    nll: dict[str, dict[str, float]],
    wallclock_s: float,
    cfg: dict,
) -> tuple[bool, list[str]]:
    """Returns (passed, list of failed-criterion messages)."""
    p = cfg["pass"]
    fails: list[str] = []

    # 1: adapter sizes
    for task, m in train_metrics.items():
        sz = m["adapter_size_mb"]
        if not (p["adapter_size_mb_min"] <= sz <= p["adapter_size_mb_max"]):
            fails.append(f"[1] adapter {task} size {sz:.1f} MB outside "
                         f"[{p['adapter_size_mb_min']}, {p['adapter_size_mb_max']}]")

    # 2: loss decrease
    for task, m in train_metrics.items():
        if m["last_loss"] >= p["loss_decrease_ratio"] * m["first_loss"]:
            fails.append(f"[2] {task} last_loss={m['last_loss']:.4f} >= "
                         f"{p['loss_decrease_ratio']}*first={p['loss_decrease_ratio']*m['first_loss']:.4f}")

    # 3: NLL finite + in range
    for state, td in nll.items():
        for task, v in td.items():
            if not (v == v) or v <= 0:  # nan / non-positive
                fails.append(f"[3] nll[{state}][{task}]={v} not finite-positive")
            elif not (p["min_nll_nats"] <= v <= p["max_nll_nats"]):
                fails.append(f"[3] nll[{state}][{task}]={v:.4f} outside "
                             f"[{p['min_nll_nats']}, {p['max_nll_nats']}]")

    # 4: per-task improvement (LoRA helped its own task)
    for task in ["gsm8k", "alpaca"]:
        base_nll = nll["base"][task]
        tau_nll = nll[task][task]
        if not (base_nll > tau_nll):
            fails.append(f"[4] {task}: NLL_base={base_nll:.4f} not > NLL_tau={tau_nll:.4f}")

    # 5: merged excess non-negative (with tolerance)
    for task in ["gsm8k", "alpaca"]:
        excess = nll["merged"][task] - nll[task][task]
        if excess < p["excess_floor"]:
            fails.append(f"[5] merged excess on {task} = {excess:.4f} < {p['excess_floor']}")

    # 8: wallclock
    if wallclock_s > p["wallclock_seconds_max"]:
        fails.append(f"[8] wallclock {wallclock_s:.1f}s > {p['wallclock_seconds_max']}s")

    return len(fails) == 0, fails


# ----------------------------- main -----------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    env_info = assert_env(min_free_gb=12.0)
    print(f"[smoke] env OK: {env_info}", flush=True)

    seed_everything(cfg["seeds"]["global"])

    project_root = Path(os.environ["PROJECT_ROOT"])
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_path = project_root / cfg["output"]["results_path"].format(timestamp=ts)

    t0 = time.time()
    base = cfg["base_model"]

    # 1. Load datasets
    print("[smoke] loading task data", flush=True)
    eval_data = {}
    train_data = {}
    for task_name, task_cfg in cfg["tasks"].items():
        tr, ev = load_task_split(
            task_cfg,
            n_train=task_cfg["n_train"],
            n_eval=task_cfg["n_eval"],
            seed=cfg["seeds"]["global"],
        )
        train_data[task_name] = tr
        eval_data[task_name] = ev
        print(f"[smoke] task={task_name}: train={len(tr)} eval={len(ev)}", flush=True)

    # 2. Train LoRAs (one at a time to free memory between)
    adapter_dirs: dict[str, Path] = {}
    train_metrics: dict[str, dict] = {}
    for task_name in cfg["tasks"]:
        adir = project_root / cfg["output"]["adapter_dir_template"].format(task=task_name)
        adapter_dirs[task_name] = adir
        train_metrics[task_name] = train_one_lora(
            task_name=task_name,
            train_data=train_data[task_name],
            base_model_name=base,
            out_dir=adir,
            cfg=cfg,
        )

    # 3. Eval all 4 states × 2 tasks
    nll = evaluate_all_states(base, adapter_dirs, eval_data, cfg)

    wallclock_s = time.time() - t0

    # 4. Check pass criteria
    passed, fails = check_pass_criteria(train_metrics, nll, wallclock_s, cfg)

    payload = {
        "train_metrics": train_metrics,
        "nll": nll,
        "excess_merged": {
            task: nll["merged"][task] - nll[task][task]
            for task in cfg["tasks"]
        },
        "wallclock_seconds": wallclock_s,
        "passed": passed,
        "fails": fails,
        "env": env_info,
    }
    write_result(results_path, payload, config=cfg)
    print(f"[smoke] wrote {results_path}", flush=True)
    print(f"[smoke] wallclock={wallclock_s:.1f}s", flush=True)

    if not passed:
        print("[smoke] FAILED criteria:", flush=True)
        for f in fails:
            print(f"  {f}", flush=True)
        return 1

    print("SMOKE_TEST_GREEN", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
