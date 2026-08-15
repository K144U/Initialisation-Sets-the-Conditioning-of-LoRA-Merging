"""Per-task dataset loading -> (train, eval) lists of {prompt, answer}.

Mirrors smoke_test.py's load_task_split but generalized across the four
Phase 3 tasks (GSM8K, Magicoder-HumanEval, Alpaca, FLORES).
"""

from typing import Tuple

from datasets import load_dataset


def _format_alpaca_prompt(ex: dict) -> str:
    instr = ex.get("instruction", "")
    inp = (ex.get("input") or "").strip()
    return f"{instr}\n\nInput: {inp}" if inp else instr


def _translation_prompt(src_text: str, src_label: str, tgt_label: str) -> str:
    return f"Translate the following from {src_label} to {tgt_label}:\n\n{src_text}"


# Human-readable language labels for the prompt (separate from dataset key codes).
_LANG_LABEL = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "eng_Latn": "English", "deu_Latn": "German", "fra_Latn": "French", "spa_Latn": "Spanish",
}


def _row_to_dict(row: dict, task_cfg: dict) -> dict:
    name = task_cfg["name"]
    if name == "alpaca":
        return {"prompt": _format_alpaca_prompt(row), "answer": row["output"]}
    if name == "translation":
        # WMT (wmt19, wmt14): row['translation'] is a dict {src_lang: ..., tgt_lang: ...}.
        # FLORES (legacy): row has 'sentence_<lang>' columns.
        src = task_cfg.get("src_lang", "en")
        tgt = task_cfg.get("tgt_lang", "de")
        src_label = _LANG_LABEL.get(src, src)
        tgt_label = _LANG_LABEL.get(tgt, tgt)
        if "translation" in row and isinstance(row["translation"], dict):
            src_text = row["translation"][src]
            tgt_text = row["translation"][tgt]
        else:
            src_text = row[f"sentence_{src}"]
            tgt_text = row[f"sentence_{tgt}"]
        return {"prompt": _translation_prompt(src_text, src_label, tgt_label), "answer": tgt_text}
    if name == "humaneval":
        # HumanEval rows have prompt + canonical_solution + test + entry_point.
        # The downstream metric needs all four; training (NLL on canonical
        # solution) needs prompt + answer = canonical_solution.
        return {
            "prompt": row["prompt"],
            "answer": row["canonical_solution"],
            "canonical_solution": row["canonical_solution"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "task_id": row.get("task_id", ""),
        }
    # default: prompt_field + answer_field
    p_field = task_cfg["prompt_field"]
    a_field = task_cfg["answer_field"]
    return {"prompt": row[p_field], "answer": row[a_field]}


def load_task_split(task_cfg: dict, seed: int) -> Tuple[list[dict], list[dict]]:
    """Returns (train, eval) lists of {'prompt': ..., 'answer': ...}.

    task_cfg fields:
      - name: str (used for special-case formatters)
      - dataset: HF repo id
      - config: optional subset config
      - split_train, split_eval: HF split names
      - n_train, n_eval: how many examples to take
      - prompt_field, answer_field: column names (ignored for special tasks)
      - streaming: bool (default false) — set true for huge datasets like wmt19
        where loading the full split would be GBs of download
      - stream_shuffle_buffer: int (default 0) — size of the reservoir used to
        shuffle a streamed split before taking the prefix. 0 keeps the
        original take-the-first-n behaviour.

    NOTE on the streaming branch, which had a defect worth recording rather
    than quietly fixing. With stream_shuffle_buffer unset it takes the FIRST
    n_train examples of the split in corpus order, while the non-streaming
    branch below shuffles first. Translation was the only task configured to
    stream, and WMT19 de-en is a concatenation of sub-corpora of roughly 35M
    pairs, so its 7500-example prefix was 7500 consecutive sentences of
    European Parliament proceedings rather than a sample of the corpus:
    17.4 parliamentary terms per 1000 tokens against 0.6 in a shuffled sample,
    evaluated against a news-domain split. That is why the translation adapter
    never learned its task; see code/phase3/scripts/diagnose_translation_adapter.py.

    The default is left at 0 deliberately. Every committed config predates this
    fix, and changing what they load would mean a re-run no longer reproduces
    the cell it claims to. New training should set stream_shuffle_buffer.
    """
    name = task_cfg["dataset"]
    cfg = task_cfg.get("config")
    streaming = task_cfg.get("streaming", False)
    train_split = task_cfg["split_train"]
    eval_split = task_cfg["split_eval"]
    n_train = task_cfg["n_train"]
    n_eval = task_cfg["n_eval"]

    if streaming:
        # Iterate parquet shards rather than downloading the full split.
        import itertools
        buf = int(task_cfg.get("stream_shuffle_buffer", 0))
        train_iter = (
            load_dataset(name, cfg, split=train_split, streaming=True) if cfg
            else load_dataset(name, split=train_split, streaming=True)
        )
        eval_iter = (
            load_dataset(name, cfg, split=eval_split, streaming=True) if cfg
            else load_dataset(name, split=eval_split, streaming=True)
        )
        if buf > 0:
            # Reservoir-shuffle first, so the prefix samples the corpus rather
            # than whichever sub-corpus happens to be written first.
            train_iter = train_iter.shuffle(seed=seed, buffer_size=buf)
            eval_iter = eval_iter.shuffle(seed=seed, buffer_size=buf)
            print(f"[data] streaming with shuffle buffer {buf}", flush=True)
        else:
            print("[data] streaming UNSHUFFLED prefix: this samples the head "
                  "of the corpus, not the corpus. See the note in "
                  "load_task_split.", flush=True)
        train_raw = list(itertools.islice(train_iter, n_train))
        # eval skips the first n_train if same split, else just takes first n_eval
        if eval_split == train_split:
            eval_raw = list(itertools.islice(eval_iter, n_train, n_train + n_eval))
        else:
            eval_raw = list(itertools.islice(eval_iter, n_eval))
    else:
        ds = load_dataset(name, cfg) if cfg else load_dataset(name)
        if eval_split == train_split:
            # CRITICAL: use ONE shuffle so train/eval are disjoint slices.
            # The earlier code used two independent shuffles (seed vs seed+1)
            # which silently allowed 7-13% overlap between train and eval.
            # Fixed 2026-05-18 after audit confirmed the bug for alpaca + magicoder.
            shuffled = ds[train_split].shuffle(seed=seed)
            n_avail = len(shuffled)
            train_raw = shuffled.select(range(min(n_train, n_avail)))
            eval_raw = shuffled.select(range(n_train, min(n_train + n_eval, n_avail)))
        else:
            train_raw = ds[train_split].shuffle(seed=seed).select(range(min(n_train, len(ds[train_split]))))
            eval_raw = ds[eval_split].shuffle(seed=seed + 1).select(range(min(n_eval, len(ds[eval_split]))))

    return (
        [_row_to_dict(r, task_cfg) for r in train_raw],
        [_row_to_dict(r, task_cfg) for r in eval_raw],
    )
