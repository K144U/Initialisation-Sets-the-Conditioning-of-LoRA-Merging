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

from .dare import merge_dare
from .knots import merge_knots
from .magnitude_prune import merge_magnitude_prune
from .rd_encoder import merge_rd_encoder
from .task_arithmetic import merge_task_arithmetic
from .ties import merge_ties
from .tvq import merge_tvq


REGISTRY = {
    "task_arithmetic": merge_task_arithmetic,
    "ties": merge_ties,
    "dare": merge_dare,
    "knots": merge_knots,
    "tvq": merge_tvq,
    "rd_encoder": merge_rd_encoder,
    "magnitude_prune": merge_magnitude_prune,
}


# Sensible per-method defaults (Phase 3 v1)
DEFAULT_KWARGS = {
    "task_arithmetic": {},
    "ties": {"density": 0.2, "majority_sign_method": "total"},
    "dare": {"density": 0.2, "seed": 20260518},
    "knots": {"inner_combination": "linear"},
    "tvq": {"rate_bits": 32},
    "rd_encoder": {"bits": 32, "c": 5.0, "seed": 20260611},
    "magnitude_prune": {"density": 0.5},
}
