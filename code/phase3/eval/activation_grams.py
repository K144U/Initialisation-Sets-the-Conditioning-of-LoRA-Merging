"""Per-task input-activation Grams for RegMean with real data.

RegMean (Jin et al., 2023) solves, per linear layer,

    W* = (sum_t G_t)^{-1} (sum_t G_t W_t),     G_t = X_t^T X_t

with X_t the inputs that layer sees when fine-tuned model t runs on task t's
own training data. The data-free variant in merging/regmean.py replaces G_t by
A_t^T A_t, which is where the initialisation collapse lives; the real Grams do
not depend on the adapters' initialisation at all.

For each task the adapter is switched on (model t is base + adapter t, as in
the reference implementation), n_samples training examples are forwarded, and
the input to every LoRA-wrapped module is accumulated. q_proj, k_proj and v_proj
read the same tensor, so they share one Gram; this is exact, not an
approximation, and it cuts memory by half.

Grams are normalised by token count, so every task enters the sum with the
same weight regardless of how long its examples are, and returned on CPU in
float32.
"""
from __future__ import annotations

import torch

SHARED_INPUT = ("q_proj", "k_proj", "v_proj")


def gram_key(layer: str) -> str:
    """Modules reading the same input tensor map to the same key."""
    head, _, leaf = layer.rpartition(".")
    if leaf in SHARED_INPUT:
        return head + ".qkv_in"
    return layer


def compute_activation_grams(model, peft_view, adapter_name: str, tokenizer,
                             train_data: list[dict], max_len: int,
                             n_samples: int) -> tuple[dict, int]:
    """Return ({gram_key: G (in x in, fp32, CPU)}, tokens_seen) for one task."""
    device = next(model.parameters()).device
    model.set_adapter(adapter_name)
    model.eval()

    owners: dict[str, str] = {}          # gram key -> the one layer that records it
    for name in peft_view.layer_names():
        owners.setdefault(gram_key(name), name)
    acc = {k: torch.zeros(peft_view.layer_topology[n][1], peft_view.layer_topology[n][1],
                          dtype=torch.float32, device=device)
           for k, n in owners.items()}
    tokens = {"n": 0}

    def make_hook(key):
        def hook(module, inputs):
            x = inputs[0].reshape(-1, inputs[0].shape[-1]).to(torch.float32)
            acc[key].addmm_(x.T, x)
        return hook

    handles = []
    for key, name in owners.items():
        handles.append(peft_view._layers[name].register_forward_pre_hook(make_hook(key)))
    try:
        with torch.no_grad():
            for ex in train_data[:n_samples]:
                msgs = [{"role": "user", "content": ex["prompt"]},
                        {"role": "assistant", "content": ex["answer"]}]
                text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                                     add_generation_prompt=False)
                ids = tokenizer(text, return_tensors="pt", truncation=True,
                                max_length=max_len,
                                add_special_tokens=False).input_ids.to(device)
                if ids.shape[1] < 2:
                    continue
                tokens["n"] += ids.shape[1]
                model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    # Fused inference kernels can bypass a module's forward, and then its hook
    # never fires. A Gram that saw nothing must stop the cell, not merge.
    dead = [k for k, v in acc.items() if float(v.diagonal().sum()) == 0.0]
    if dead or tokens["n"] == 0:
        raise RuntimeError(f"activation hooks recorded nothing for {len(dead)} of "
                           f"{len(acc)} Grams (e.g. {dead[:2]}); use loader: plain")
    n = max(tokens["n"], 1)
    return {k: (v / n).cpu() for k, v in acc.items()}, tokens["n"]


def attach_grams(method_kwargs: dict, cfg: dict, model, view, adapter_names,
                 tokenizer, load_task_split, max_len: int) -> None:
    """If the method needs real Grams, compute them and put them in kwargs.

    n_gram_samples is consumed here and never forwarded to the merge.
    """
    n = int(method_kwargs.pop("n_gram_samples", 256))
    grams, counts = {}, {}
    for spec, peft_name in zip(cfg["adapter_specs"], adapter_names):
        train_split, _ = load_task_split(spec["task_cfg"], seed=cfg.get("seed", 20260518))
        grams[peft_name], counts[spec["name"]] = compute_activation_grams(
            model, view, peft_name, tokenizer, train_split, max_len, n)
        print(f"  grams for {spec['name']:<14}: {counts[spec['name']]} tokens "
              f"from {min(n, len(train_split))} examples", flush=True)
    method_kwargs["grams"] = grams
    method_kwargs["_gram_tokens"] = counts
