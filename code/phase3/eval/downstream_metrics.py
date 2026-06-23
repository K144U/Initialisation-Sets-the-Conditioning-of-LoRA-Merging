"""E3 — Downstream task metrics.

GSM8K exact-match is fully implemented. HumanEval pass@1, COMET-22,
IFEval strict are stubbed pending dependency installation (sandbox-exec
for HumanEval, unbabel-comet for COMET, google IFEval verifier).

Per master plan §E3: NLL-based conclusions need to survive on task
metrics. Decision rule: Spearman across methods per model.
  r_s > 0.7   -> cheap NLL conclusions hold (paper §6.1/6.2 untouched)
  0.4-0.7     -> noisier; report alongside NLL
  < 0.4       -> NLL not predictive; new design work

Metric specs frozen 2026-06-14:
  GSM8K        : exact-match on final numeric answer, greedy decoding,
                 max 256 new tokens, 0-shot CoT prompt.
  HumanEval    : pass@1, greedy decoding, 512 new tokens, sandboxed
                 exec with 5s timeout per test.
  COMET-22     : unbabel-comet wmt22-comet-da reference-based.
  IFEval       : strict-mode rubric verification on a 100-prompt subset
                 from the IFEval public release (seed 20260518).
"""

from __future__ import annotations

import re
from typing import Callable, Iterator


# --- GSM8K extraction + scoring -----------------------------------------


_GSM8K_FINAL_PATTERNS = [
    re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)"),         # "#### 123"
    re.compile(r"answer\s+is\s+\$?([-+]?\d[\d,]*(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"=\s*\$?([-+]?\d[\d,]*(?:\.\d+)?)\s*[.,!?]?\s*$"),  # ending equation
    re.compile(r"\$?([-+]?\d[\d,]*(?:\.\d+)?)\s*[.,!?]?\s*$"),       # last number
]


def gsm8k_extract_answer(text: str) -> str | None:
    """Extract a normalized final numeric answer from generated text.

    GSM8K's gold-answer convention is `#### <number>`. We accept that, plus
    several common chat-output variants (`The answer is X`, `= X` at end,
    or just the last number in the response). Returns a normalized string
    (commas stripped, trailing `.0` stripped) or None if extraction failed.
    """
    text = text.strip()
    for pat in _GSM8K_FINAL_PATTERNS:
        m = pat.search(text)
        if m:
            s = m.group(1).replace(",", "").strip()
            # Normalize floating point: "3" == "3.0" == "3.00"
            try:
                v = float(s)
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return str(v)
            except ValueError:
                return s
    return None


def gsm8k_score(pred_text: str, gold_text: str) -> int:
    """Returns 1 if extracted predictions match gold, else 0."""
    pred = gsm8k_extract_answer(pred_text)
    gold = gsm8k_extract_answer(gold_text)
    if pred is None or gold is None:
        return 0
    try:
        return int(abs(float(pred) - float(gold)) < 1e-6)
    except ValueError:
        return int(pred == gold)


# --- Greedy generation helper -------------------------------------------


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy decode (do_sample=False, temperature ignored), return only the
    newly generated text."""
    import torch
    device = next(model.parameters()).device
    chat_msgs = [{"role": "user", "content": prompt}]
    in_text = tokenizer.apply_chat_template(
        chat_msgs, tokenize=False, add_generation_prompt=True)
    in_ids = tokenizer(in_text, return_tensors="pt",
                       add_special_tokens=False).input_ids.to(device)
    in_len = in_ids.shape[1]
    with torch.no_grad():
        out = model.generate(
            in_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    gen_ids = out[0, in_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


# --- Per-task evaluators ------------------------------------------------


def evaluate_gsm8k_em(model, tokenizer, eval_data: list[dict],
                     max_new_tokens: int = 256,
                     progress_every: int = 50,
                     ) -> tuple[float, list[dict]]:
    """eval_data: list of {prompt, answer} where answer is gold GSM8K
    response containing `#### N`. Returns (accuracy, per_example_list).
    """
    correct = 0
    per_example: list[dict] = []
    model.eval()
    for i, ex in enumerate(eval_data):
        gen = greedy_generate(model, tokenizer, ex["prompt"], max_new_tokens)
        score = gsm8k_score(gen, ex["answer"])
        correct += score
        per_example.append({
            "score": score,
            "pred": gsm8k_extract_answer(gen),
            "gold": gsm8k_extract_answer(ex["answer"]),
            "gen_text": gen[:200],   # truncated for storage
        })
        if (i + 1) % progress_every == 0:
            acc_so_far = correct / (i + 1)
            print(f"  [gsm8k_em] {i+1}/{len(eval_data)}  "
                  f"running acc={acc_so_far:.3f}", flush=True)
    accuracy = correct / len(eval_data) if eval_data else float("nan")
    return accuracy, per_example


def evaluate_humaneval_pass1(model, tokenizer, eval_data: list[dict],
                              max_new_tokens: int = 512,
                              ) -> tuple[float, list[dict]]:
    """Stub: needs the openai/human-eval sandbox + test harness, deferred."""
    raise NotImplementedError("HumanEval pass@1 not yet implemented")


def evaluate_comet22(model, tokenizer, eval_data: list[dict],
                     max_new_tokens: int = 256,
                     ) -> tuple[float, list[dict]]:
    """Stub: needs unbabel-comet pip install + model download, deferred."""
    raise NotImplementedError("COMET-22 not yet implemented")


def evaluate_ifeval_strict(model, tokenizer, eval_data: list[dict],
                            max_new_tokens: int = 512,
                            ) -> tuple[float, list[dict]]:
    """Stub: needs google-research IFEval verifier port, deferred."""
    raise NotImplementedError("IFEval strict not yet implemented")


# L3 special token IDs — used by B2 (chat-template H1 probe).
# These are the Llama-3.1 chat-template literals; same ID range is reserved
# in other Llama-3 family bases.
L3_SPECIAL_TOKEN_LITERALS = (
    "<|begin_of_text|>", "<|end_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>",
    "<|eot_id|>",
    "<|reserved_special_token_0|>", "<|reserved_special_token_1|>",
)


def evaluate_gsm8k_special_token_probe(model, tokenizer,
                                        eval_data: list[dict],
                                        max_new_tokens: int = 256,
                                        progress_every: int = 25,
                                        ) -> tuple[float, list[dict]]:
    """B2 — same greedy CoT as gsm8k_em, but decode with
    skip_special_tokens=False so chat-template specials are preserved,
    and record per-example raw generated token IDs plus a per-example
    special-token emission count. Returns (mean emission count per
    generation, per_example_list).

    Tests the §6.5 L3 H1 hypothesis: aggressive merge methods (TIES at
    low density, TVQ at low bits, KnOTS, DARE) on Llama-3.1 may emit
    chat-template special tokens unpredictably, breaking the regex-based
    gold-answer extraction.
    """
    import torch
    device = next(model.parameters()).device
    # Map our special-token literals to their token IDs in this tokenizer.
    special_ids: set[int] = set()
    for lit in L3_SPECIAL_TOKEN_LITERALS:
        tid = tokenizer.convert_tokens_to_ids(lit)
        if tid is not None and tid != tokenizer.unk_token_id:
            special_ids.add(int(tid))
    print(f"  [special_probe] special_ids={sorted(special_ids)}", flush=True)

    per_example: list[dict] = []
    total_emit = 0
    model.eval()
    for i, ex in enumerate(eval_data):
        chat_msgs = [{"role": "user", "content": ex["prompt"]}]
        in_text = tokenizer.apply_chat_template(
            chat_msgs, tokenize=False, add_generation_prompt=True)
        in_ids = tokenizer(in_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
        in_len = in_ids.shape[1]
        with torch.no_grad():
            out = model.generate(
                in_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False, num_beams=1, temperature=1.0, top_p=1.0,
                pad_token_id=tokenizer.eos_token_id, use_cache=True,
            )
        gen_ids = out[0, in_len:].tolist()
        emit = sum(1 for t in gen_ids if t in special_ids)
        gen_text_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        gen_text_clean = tokenizer.decode(gen_ids, skip_special_tokens=True)
        score = gsm8k_score(gen_text_clean, ex["answer"])
        per_example.append({
            "score": score,
            "pred": gsm8k_extract_answer(gen_text_clean),
            "gold": gsm8k_extract_answer(ex["answer"]),
            "n_special_emitted": emit,
            "n_tokens": len(gen_ids),
            "gen_text_raw": gen_text_raw[:300],  # truncated; specials preserved
            "gen_text_clean": gen_text_clean[:200],
        })
        total_emit += emit
        if (i + 1) % progress_every == 0:
            mean_emit = total_emit / (i + 1)
            print(f"  [special_probe] {i+1}/{len(eval_data)}  "
                  f"running mean specials/gen={mean_emit:.3f}", flush=True)
    mean_emit = total_emit / max(1, len(eval_data))
    return mean_emit, per_example


METRIC_FNS: dict[str, Callable] = {
    "gsm8k_em": evaluate_gsm8k_em,
    "gsm8k_special_token_probe": evaluate_gsm8k_special_token_probe,
    "humaneval_pass1": evaluate_humaneval_pass1,
    "comet22": evaluate_comet22,
    "ifeval_strict": evaluate_ifeval_strict,
}


# --- Quick unit tests for GSM8K extraction ------------------------------


def _self_test() -> None:
    cases = [
        ("Let's see... 5+3 = 8. #### 8", "8"),
        ("The answer is 42", "42"),
        ("After computing: 2 + 2 = 4.", "4"),
        ("So the result is 1.5\n#### 1.5", "1.5"),
        ("I have no idea.", None),
        ("Answer: $3,200", "3200"),
    ]
    for text, want in cases:
        got = gsm8k_extract_answer(text)
        assert got == want, f"extract({text!r}) = {got!r}, want {want!r}"
    print("downstream_metrics._self_test: OK")


if __name__ == "__main__":
    _self_test()
