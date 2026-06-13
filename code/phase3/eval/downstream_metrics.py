"""E3 — Downstream task metrics harness (SKELETON).

Per master_plan §E3: NLL-based conclusions need to survive on task
metrics. Adds four generation/extraction metrics to the eval pipeline,
to be run on the same merge matrix as E2.

Metrics (frozen choices — do NOT change after first eval):
  - GSM8K          : exact-match accuracy on the final numerical answer,
                     0-shot CoT, greedy decoding, max 256 new tokens.
  - HumanEval/MBPP : pass@1 with greedy decoding, 256 new tokens, the
                     standard sanitization pipeline (strip trailing
                     comments, drop after first def/class, etc.).
  - WMT19 en-de    : COMET-22 reference-based scoring,
                     unbabel-comet's wmt22-comet-da model.
  - Alpaca-cleaned : IFEval strict (instruction following), via the
                     IFEval verifier set (a fixed subset of 100 prompts
                     from the original eval set).

Decision rule (per master_plan §E3):
  Test that excess NLL and excess metric drop are rank-correlated
  across the merge matrix (Spearman across methods, per model).
  If r_s > 0.7 across all 4 models -> the cheap NLL conclusions hold.
  If 0.4 < r_s < 0.7 -> noisier; report alongside NLL.
  If r_s < 0.4 -> NLL is not predictive; do new design work.

Status: SKELETON. Implementation deferred until cross-model sweep
verdict is in (~2026-06-13 evening). Then prioritize on whether E5
or E3 owns the next GPU-hour slot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Metric harness skeleton ---------------------------------------------


class Metric:
    """Base class. Subclasses implement evaluate(model, tokenizer, eval_data)
    and return a scalar in [0, 1] (higher is better)."""

    name: str = "metric"

    def evaluate(self, model, tokenizer, eval_data) -> tuple[float, list[dict]]:
        raise NotImplementedError


class GSM8KExactMatch(Metric):
    """0-shot CoT, greedy decoding, max 256 new tokens. Extracts the
    final numerical answer after '####' or after 'answer is' (the two
    formats covered by GSM8K convention)."""
    name = "gsm8k_em"

    def evaluate(self, model, tokenizer, eval_data):
        # TODO: implement greedy generation loop + numeric extraction
        # See training/data_loaders.py for the GSM8K prompt format.
        raise NotImplementedError


class HumanEvalPass1(Metric):
    """Pass@1 with greedy decoding. Sanitization: strip prompt prefix,
    truncate at first 'def ' or 'class ' or '#' after the function
    signature, exec inside a sandboxed subprocess with timeout 5s per
    test."""
    name = "humaneval_pass1"

    def evaluate(self, model, tokenizer, eval_data):
        # TODO: standard openai/human-eval harness. Use the sandboxed
        # exec to avoid side effects.
        raise NotImplementedError


class COMET22(Metric):
    """Reference-based COMET score via unbabel-comet wmt22-comet-da.
    Downloads the model on first run; cache at $HF_HOME."""
    name = "comet22"

    def evaluate(self, model, tokenizer, eval_data):
        # TODO: pip install unbabel-comet (needs verification in conda env)
        # then COMET(model='wmt22-comet-da').predict([...])
        raise NotImplementedError


class IFEvalStrict(Metric):
    """Instruction-following: strict-mode verification against the
    IFEval rubric. Fixed subset of 100 prompts (seed 20260518) from
    the original release."""
    name = "ifeval_strict"

    def evaluate(self, model, tokenizer, eval_data):
        # TODO: pull google-research/instruction_following_eval verifier
        # and adapt to our prompt format.
        raise NotImplementedError


METRICS_BY_TASK = {
    "gsm8k": GSM8KExactMatch(),
    "magicoder": HumanEvalPass1(),  # use HumanEval as the held-out code metric
    "translation": COMET22(),
    "alpaca": IFEvalStrict(),
}


# --- Analysis harness skeleton -------------------------------------------


def rank_correlation_nll_vs_metric(nll_table: dict, metric_table: dict,
                                   model: str) -> float:
    """Spearman correlation between excess NLL and excess metric drop
    across methods for a given model."""
    # TODO: import scipy.stats.spearmanr; align method order; return r_s.
    raise NotImplementedError


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path,
                   help="YAML config naming (model, merged-adapter ckpt, task)")
    args = p.parse_args()
    # TODO: load model+adapter, run METRICS_BY_TASK[task], write JSON.
    raise NotImplementedError


if __name__ == "__main__":
    sys.exit(main())
