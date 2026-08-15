#!/usr/bin/env python3
"""Why did the translation adapter never learn its task?

It failed on four bases in both initialisation arms. That pattern rules out a
random training failure: something deterministic and shared is wrong, and it is
diagnosable from the data pipeline without retraining anything.

The candidate is in `training/data_loaders.py`. Translation is the only task
configured with `streaming: true`, and the streaming branch does

    train_raw = list(itertools.islice(train_iter, n_train))

which takes the **first** `n_train` examples of the split in corpus order. The
non-streaming branch, which every other task takes, shuffles first. WMT19 de-en
is a concatenation of sub-corpora of roughly 38M pairs, so an unshuffled prefix
of 7500 is a sample of whichever sub-corpus happens to be written first, not a
sample of the corpus. The eval split is a different distribution again.

This script measures that rather than asserting it: it pulls the prefix the
trainer actually saw and the eval split it was actually scored on, and reports
how far apart they are on surface statistics that need no model. It downloads a
few thousand rows and runs on CPU.

Usage:
  python code/phase3/scripts/diagnose_translation_adapter.py [--n 2000]
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "results" / "phase3" / "translation_diagnosis.json"

DATASET, CONFIG = "wmt/wmt19", "de-en"


def surface(rows: list[dict]) -> dict:
    """Statistics that separate one text domain from another without a model."""
    en = [r["translation"]["en"] for r in rows]
    de = [r["translation"]["de"] for r in rows]
    toks = [w.lower() for s in en for w in re.findall(r"[a-zA-Z']+", s)]
    counts = Counter(toks)
    n_tok = max(len(toks), 1)
    return {
        "n_rows": len(rows),
        "mean_chars_en": round(sum(map(len, en)) / max(len(en), 1), 1),
        "mean_words_en": round(
            sum(len(s.split()) for s in en) / max(len(en), 1), 2),
        "mean_chars_de": round(sum(map(len, de)) / max(len(de), 1), 1),
        "type_token_ratio": round(len(counts) / n_tok, 4),
        "top_content_words": [w for w, _ in counts.most_common(400)
                              if len(w) > 5][:12],
        # A domain fingerprint: parliamentary proceedings are saturated with
        # these and news text is not.
        "parliamentary_rate_per_1k": round(1000 * sum(
            counts[w] for w in ("president", "commission", "parliament",
                                "council", "madam", "rapporteur", "debate",
                                "vote", "directive", "member")) / n_tok, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000,
                    help="rows to sample from each source")
    args = ap.parse_args()
    from datasets import load_dataset

    report: dict = {"dataset": DATASET, "config": CONFIG, "n_sampled": args.n}

    # 1. Exactly what the trainer saw: an unshuffled prefix of the train split.
    it = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    prefix = list(itertools.islice(it, args.n))
    report["train_prefix_as_run"] = surface(prefix)

    # 2. What a shuffled sample of the same split looks like, which is what
    #    every other task in the matrix got.
    it = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    shuffled = list(itertools.islice(
        it.shuffle(seed=20260517, buffer_size=50_000), args.n))
    report["train_shuffled_control"] = surface(shuffled)

    # 3. What it was evaluated on.
    ev = load_dataset(DATASET, CONFIG, split="validation")
    report["eval_split"] = surface(list(ev)[: args.n])

    a = report["train_prefix_as_run"]
    b = report["train_shuffled_control"]
    c = report["eval_split"]
    report["summary"] = {
        "parliamentary_rate_prefix_vs_eval":
            [a["parliamentary_rate_per_1k"], c["parliamentary_rate_per_1k"]],
        "mean_words_prefix_vs_eval": [a["mean_words_en"], c["mean_words_en"]],
        "prefix_differs_from_shuffled_control":
            a["top_content_words"] != b["top_content_words"],
    }

    for k in ("train_prefix_as_run", "train_shuffled_control", "eval_split"):
        r = report[k]
        print(f"\n{k}")
        print(f"  mean words (en)        {r['mean_words_en']}")
        print(f"  type/token ratio       {r['type_token_ratio']}")
        print(f"  parliamentary per 1k   {r['parliamentary_rate_per_1k']}")
        print(f"  top content words      {', '.join(r['top_content_words'][:8])}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
