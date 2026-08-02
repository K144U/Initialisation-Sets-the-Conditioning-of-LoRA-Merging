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
    re.compile(r"\$?([-+]?\d[\d,]*(?:\.\d+)?)\s*[.,!?]?\s*$"),       # number at end
]

# Fallback: the last number appearing ANYWHERE in the text, the standard
# GSM8K harness convention. Deliberately last, so the structured patterns
# above still win when present.
#
# Why this exists (2026-08-02): the four patterns above are all anchored at
# end-of-string, so any answer closing with words ("Thus, 12 students are
# good at math.") extracted None and scored 0. Measured failure rates on the
# shipped runs were 60% to 81% on three of four bases and strongly
# method-dependent (Llama rd-ridge 0.693 vs Mistral rd-ridge 0.107), which
# made EM a measure of output formatting rather than arithmetic. Do not
# remove this fallback.
_GSM8K_ANY_NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _normalize_number(s: str) -> str:
    """Commas stripped, integral floats collapsed: "3" == "3.0" == "3.00"."""
    s = s.replace(",", "").strip()
    try:
        v = float(s)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return str(v)
    except ValueError:
        return s


def gsm8k_extract_answer(text: str) -> str | None:
    """Extract a normalized final numeric answer from generated text.

    GSM8K's gold-answer convention is `#### <number>`. We accept that, plus
    common chat-output variants (`The answer is X`, `= X` at end, a number at
    end), and finally the last number anywhere in the text. Returns None only
    when the text contains no number at all.
    """
    if not text:
        return None
    text = text.strip()
    for pat in _GSM8K_FINAL_PATTERNS:
        m = pat.search(text)
        if m:
            return _normalize_number(m.group(1))
    matches = _GSM8K_ANY_NUMBER.findall(text)
    if matches:
        return _normalize_number(matches[-1])
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
            # Full text, not a preview. The old 200-char cap made offline
            # re-scoring impossible, so a scorer bug cost a whole GPU re-run.
            # About 1 KB per example, 0.5 MB per cell.
            "gen_text": gen,
        })
        if (i + 1) % progress_every == 0:
            acc_so_far = correct / (i + 1)
            print(f"  [gsm8k_em] {i+1}/{len(eval_data)}  "
                  f"running acc={acc_so_far:.3f}", flush=True)
    accuracy = correct / len(eval_data) if eval_data else float("nan")
    return accuracy, per_example


def _strip_humaneval_completion(text: str) -> str:
    """Reduce a generation to the completion body to append to the prompt.

    Bug fixed 2026-08-02: the old version tested
    `if out and line.startswith(("def ", ...))`, so a generation beginning
    with a blank line followed by a top-level `def` broke on line 2 with
    `out == [""]` and returned "". That is every markdown-fenced answer,
    because the fence strip in the caller leaves a leading newline. The
    candidate file was then the bare prompt, a body-less function, which
    fails every assertion. It silently discarded 122 to 130 of 164
    completions for TA/DARE/KnOTS versus 0 to 11 for TIES/TVQ2/rd-ridge,
    making pass@1 a measure of markdown habits rather than code quality.

    Handles the four shapes models actually emit:
      1. body only, indented          -> kept as-is
      2. full function, leading `def` -> kept (redefines the prompt's stub;
                                         Python resolves to the second)
      3. prose, then a full function  -> prose dropped, function kept
      4. nothing usable               -> ""
    """
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    # A second def/class ends the completion, but the FIRST one does not: an
    # import or decorator preamble legitimately precedes it. Test scaffolding
    # ends the completion whether or not a def has been seen.
    DEF_BOUNDARY = ("def ", "class ")
    TEST_BOUNDARY = ("if __name__", "print(", "assert ", "# Test")

    # Shape 3: a top-level `def` appears but not on the first line, and what
    # precedes it is not code. That preamble is prose; drop it.
    first_def = next((i for i, l in enumerate(lines) if l.startswith("def ")), None)
    if first_def is not None and first_def > 0:
        preamble_is_code = any(
            l.strip() and l.startswith((" ", "\t", "@", "import ", "from "))
            for l in lines[:first_def]
        )
        if not preamble_is_code:
            lines = lines[first_def:]

    out: list[str] = []
    started = False
    seen_def = False
    for i, line in enumerate(lines):
        if i > 0 and started and line.startswith(TEST_BOUNDARY):
            break
        if i > 0 and seen_def and line.startswith(DEF_BOUNDARY):
            break
        out.append(line)
        if line.strip():
            started = True
        if line.startswith(DEF_BOUNDARY):
            seen_def = True

    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _run_humaneval_check(source: str, test_code: str, entry_point: str,
                         timeout_s: float = 5.0) -> tuple[bool, str]:
    """Execute the candidate function and its tests in a subprocess.
    Returns (passed, error_msg). Uses subprocess + signal-based timeout
    for sandbox-lite isolation."""
    import subprocess
    import tempfile
    full_source = (
        source
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})\n"
    )
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False) as f:
        f.write(full_source)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname],
            capture_output=True, text=True,
            timeout=timeout_s,
        )
        if result.returncode == 0:
            return True, ""
        # First 200 chars of stderr for diagnostics
        return False, (result.stderr or result.stdout)[:200]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"SUBPROC_ERR: {e!s}"[:200]


def evaluate_humaneval_pass1(model, tokenizer, eval_data: list[dict],
                              max_new_tokens: int = 512,
                              progress_every: int = 25,
                              timeout_s: float = 5.0,
                              ) -> tuple[float, list[dict]]:
    """eval_data: list of {prompt, canonical_solution, test, entry_point}
    rows from openai/openai_humaneval. Greedy decode the completion,
    strip trailing top-level definitions, exec prompt+completion+test,
    count pass@1."""
    import torch
    device = next(model.parameters()).device
    correct = 0
    per_example: list[dict] = []
    model.eval()
    for i, ex in enumerate(eval_data):
        prompt = ex["prompt"]
        # We feed the function signature + docstring as a code-completion
        # task via chat template; the model returns the function body.
        # Some bases (esp. instruct-tuned) emit markdown ```python ...```
        # wrappers which we strip below.
        chat_msgs = [{"role": "user",
                      "content": "Complete this Python function:\n\n" + prompt}]
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
        gen = tokenizer.decode(out[0, in_len:], skip_special_tokens=True)
        # Strip markdown wrappers if present
        if "```python" in gen:
            gen = gen.split("```python", 1)[1]
        if "```" in gen:
            gen = gen.split("```", 1)[0]
        body = _strip_humaneval_completion(gen)
        source = prompt + body
        passed, err = _run_humaneval_check(
            source, ex["test"], ex["entry_point"], timeout_s)
        if passed:
            correct += 1
        per_example.append({
            "task_id": ex.get("task_id", f"hu{i}"),
            "score": int(passed),
            "err": err if not passed else "",
            "completion_preview": body,   # full; see the gen_text note above
            "gen_raw": gen,               # pre-strip, to debug the stripper
        })
        if (i + 1) % progress_every == 0:
            acc = correct / (i + 1)
            print(f"  [humaneval] {i+1}/{len(eval_data)}  "
                  f"running pass@1={acc:.3f}", flush=True)
    accuracy = correct / max(1, len(eval_data))
    return accuracy, per_example


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
            "gen_text_raw": gen_text_raw,     # full; specials preserved
            "gen_text_clean": gen_text_clean,
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


# --- Unit tests for answer extraction and completion stripping ----------


def _self_test() -> None:
    fails: list[str] = []

    def chk(label, got, want):
        if got != want:
            fails.append(f"{label}: got {got!r}, want {want!r}")

    extraction_cases = [
        ("Let's see... 5+3 = 8. #### 8", "8"),
        ("The answer is 42", "42"),
        ("After computing: 2 + 2 = 4.", "4"),
        ("So the result is 1.5\n#### 1.5", "1.5"),
        ("I have no idea.", None),
        ("", None),
        ("Answer: $3,200", "3200"),
        # Regression 2026-08-02: answer stated mid-sentence, text ending on a
        # word. The end-anchored patterns returned None, scoring 0, which
        # turned EM into a formatting metric.
        ("There are 20 - 5 - 8 = 7 students.\n"
         "So, there are 5 + 7 = 12 students good at math.", "12"),
        ("Thus, 12 students are good at math.", "12"),
        ("The number of pounds not used is 200 - 80 = 120 pounds", "120"),
        # Structured patterns must still beat the last-number fallback.
        ("#### 7\nsome trailing chatter 999 words", "7"),
    ]
    for text, want in extraction_cases:
        chk(f"extract({text[:30]!r})", gsm8k_extract_answer(text), want)

    chk("score match", gsm8k_score("Thus, 12 students are good at math.",
                                   "#### 12"), 1)
    chk("score mismatch", gsm8k_score("Thus, 11 students.", "#### 12"), 0)

    strip_cases = [
        ("body only", "    return x + 1", "    return x + 1"),
        ("full func", "def add(x, y):\n    return x + y",
         "def add(x, y):\n    return x + y"),
        # Regression 2026-08-02: the caller's markdown-fence strip leaves a
        # leading newline, which used to yield an empty completion and a
        # guaranteed fail, discarding up to 79% of some methods' generations.
        ("leading blank", "\ndef add(x, y):\n    return x + y",
         "def add(x, y):\n    return x + y"),
        ("prose then func",
         "Here is the function:\n\ndef add(x, y):\n    return x + y",
         "def add(x, y):\n    return x + y"),
        ("cuts second def",
         "def add(x, y):\n    return x + y\n\ndef other():\n    pass",
         "def add(x, y):\n    return x + y"),
        ("cuts trailing tests", "    return x + 1\n\nassert add(1,2) == 3",
         "    return x + 1"),
        ("keeps import preamble",
         "import math\n\ndef f(x):\n    return math.sqrt(x)",
         "import math\n\ndef f(x):\n    return math.sqrt(x)"),
        ("empty", "", ""),
        ("blank only", "\n\n  \n", ""),
    ]
    for label, text, want in strip_cases:
        chk(f"strip[{label}]", _strip_humaneval_completion(text), want)

    if fails:
        print("downstream_metrics._self_test: FAILURES")
        for f in fails:
            print("  " + f)
        raise SystemExit(1)
    print(f"downstream_metrics._self_test: OK "
          f"({len(extraction_cases) + len(strip_cases) + 2} cases)")


if __name__ == "__main__":
    _self_test()
