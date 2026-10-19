"""Method-name -> merge-function registry.

Every merge function has the same call signature:

    merge_fn(
        model,                      # FakePeftModel (tests) or PeftModelView (eval)
        adapter_names: list[str],
        weights: list[float],
        merged_adapter_name: str,
        **method_kwargs,            # density, rate_bits, inner_combination, etc.
    ) -> None

The function adds the merged adapter to `model` in place.
"""

from __future__ import annotations

from .adamerging import merge_adamerging
from .dare import merge_dare
from .dare_ties import merge_dare_ties
from .della import merge_della
from .fisher_avg import merge_fisher_avg
from .knots import merge_knots
from .magnitude_prune import merge_magnitude_prune
from .rd_encoder import merge_rd_encoder
from .regmean import merge_regmean
from .regmean_act import merge_regmean_act
from .mats_kfac import merge_mats
from .tsv_merge import merge_tsv
from .task_arithmetic import merge_task_arithmetic
from .ties import merge_ties
from .tvq import merge_tvq
from .wudi import merge_wudi


REGISTRY = {
    "task_arithmetic": merge_task_arithmetic,
    "ties": merge_ties,
    "dare": merge_dare,
    "dare_ties": merge_dare_ties,
    "knots": merge_knots,
    "tvq": merge_tvq,
    "rd_encoder": merge_rd_encoder,
    "magnitude_prune": merge_magnitude_prune,
    "fisher_avg": merge_fisher_avg,
    "della": merge_della,
    "regmean": merge_regmean,
    "regmean_act": merge_regmean_act,
    "mats": merge_mats,
    "tsv": merge_tsv,
    "adamerging": merge_adamerging,
    "wudi": merge_wudi,
}

# Methods whose module exposes a DIAGNOSTICS dict filled by the merge; the
# eval cells record it in the result JSON and the merge cache.
DIAG_MODULES = {"mats": "merging.mats_kfac", "regmean_act": "merging.regmean_act",
                "wudi": "merging.wudi", "regmean": "merging.regmean"}

# Methods whose merge needs per-task input-activation Grams computed in the
# eval cell (eval/activation_grams.py) before the merge is called.
NEEDS_GRAMS = {"regmean_act", "mats"}


# Sensible per-method defaults (Phase 3 v1)
DEFAULT_KWARGS = {
    "task_arithmetic": {},
    "ties": {"density": 0.2, "majority_sign_method": "total"},
    "dare": {"density": 0.2, "seed": 20260518},
    # dare_density is DARE's keep fraction; ties_density is held at TIES's own
    # 0.2 so the only difference from the `ties` baseline is the mask. See
    # notes/prereg_dare_ties_2026-08-04.md.
    "dare_ties": {"dare_density": 0.2, "ties_density": 0.2,
                  "majority_sign_method": "total", "seed": 20260518},
    # WARNING: "linear" is algebraically Task Arithmetic (see knots.py and
    # tests/test_knots.py). Kept as the default only so historical cells stay
    # reproducible. For a KnOTS that differs from TA, pass
    # inner_combination="ties" explicitly.
    "knots": {"inner_combination": "linear"},
    "tvq": {"rate_bits": 32},
    "rd_encoder": {"bits": 32, "c": 5.0, "seed": 20260611},
    "magnitude_prune": {"density": 0.5},
    "fisher_avg": {},
    "della": {"drop_p": 0.2, "density": 0.2,
              "majority_sign_method": "total", "seed": 20260624},
    "regmean": {"ridge_lambda": 1e-3},
    # RegMean's published defaults: alpha = 0.9 off-diagonal shrinkage and no
    # ridge; n_gram_samples is consumed by the eval cell, not the merge.
    "regmean_act": {"alpha": 0.9, "ridge_lambda": 0.0, "n_gram_samples": 256},
    # MaTS, official defaults: KFAC Fisher, 100 CG iterations from the average,
    # model_lambda 1.0. Gram length capped at 512 tokens to fit the backward pass.
    "mats": {"kfac": True, "num_iterations": 100, "model_lambda": 1.0,
             "n_gram_samples": 256, "gram_max_len": 512},
    "tsv": {"alpha": 1.0},
    "adamerging": {},
}
