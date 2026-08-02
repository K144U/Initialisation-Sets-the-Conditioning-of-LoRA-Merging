"""Synthetic-tau unit tests for the 5 merging methods.

Properties tested per method:
  P1 — zero idempotence: all tau_t = 0 -> merged delta = 0.
  P2 — identity collapse: all tau_t = tau -> merged delta = tau.
       (TIES/DARE need density=1.0; pruning otherwise destroys identity.)
  P3 — rank-r preservation: merged adapter's B @ A has rank <= r.

Plus TVQ-specific extras:
  P4 — TVQ at b=32 equals Task Arithmetic on the same input.
  P5 — TVQ at b=1 differs measurably from b=32.

All tests CPU-only via FakePeftModel. Total runtime <2 sec.

Run as:
    python -m merging.tests.test_synthetic
or
    PYTHONPATH=code/phase3 python code/phase3/merging/tests/test_synthetic.py
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable

import torch

# allow running both as module and as script
try:
    from merging.registry import REGISTRY
    from merging.tests._fake_model import (
        FakePeftModel,
        make_identical_fake_model,
        make_random_fake_model,
        make_zero_fake_model,
    )
except ImportError:
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from merging.registry import REGISTRY
    from merging.tests._fake_model import (
        FakePeftModel,
        make_identical_fake_model,
        make_random_fake_model,
        make_zero_fake_model,
    )


# Small test dims keep all 15+ tests under 2 sec total.
LAYER_SPECS = {
    "layer.0.q_proj": (64, 32, 4),
    "layer.0.v_proj": (64, 32, 4),
}
ADAPTERS = ["a0", "a1"]
WEIGHTS = [0.5, 0.5]
MERGED = "merged"


# ----- helpers ------------------------------------------------------

def merged_delta(model: FakePeftModel, layer: str) -> torch.Tensor:
    return model.get_delta(MERGED, layer)


def merged_rank(model: FakePeftModel, layer: str) -> int:
    f = model.adapters[MERGED][layer]
    return torch.linalg.matrix_rank(f.B @ f.A).item()


# ----- the 3 standard properties -----------------------------------

def check_zero_idempotence(method_name: str, merge_fn: Callable, **kwargs) -> None:
    """P1: all tau_t = 0 implies merged delta = 0."""
    model = make_zero_fake_model(LAYER_SPECS, ADAPTERS)
    merge_fn(model, ADAPTERS, WEIGHTS, MERGED, **kwargs)
    for layer in LAYER_SPECS:
        d = merged_delta(model, layer)
        assert torch.all(d.abs() < 1e-6), (
            f"[{method_name}/P1] merged delta on {layer} is not ~0 "
            f"(max abs = {d.abs().max().item():.2e})"
        )


def check_identity_collapse(
    method_name: str, merge_fn: Callable, atol: float = 1e-4, **kwargs,
) -> None:
    """P2: all tau_t = tau (identical) implies merged delta ~= tau."""
    rng = torch.Generator().manual_seed(42)
    model = make_identical_fake_model(LAYER_SPECS, ADAPTERS, rng=rng, scale=0.01)
    # capture tau before merging
    target = {layer: model.get_delta(ADAPTERS[0], layer).clone() for layer in LAYER_SPECS}
    merge_fn(model, ADAPTERS, WEIGHTS, MERGED, **kwargs)
    for layer in LAYER_SPECS:
        d = merged_delta(model, layer)
        diff = (d - target[layer]).abs().max().item()
        assert diff < atol, (
            f"[{method_name}/P2] merged delta on {layer} differs from tau by {diff:.2e} (atol={atol})"
        )


def check_rank_r_preservation(method_name: str, merge_fn: Callable, **kwargs) -> None:
    """P3: merged adapter has rank <= r per layer."""
    rng = torch.Generator().manual_seed(43)
    model = make_random_fake_model(LAYER_SPECS, ADAPTERS, rng=rng, scale=0.01)
    merge_fn(model, ADAPTERS, WEIGHTS, MERGED, **kwargs)
    for layer, (out, ind, r) in LAYER_SPECS.items():
        # also assert the LoRA factors have the right shape
        f = model.adapters[MERGED][layer]
        assert f.A.shape == (r, ind), f"[{method_name}/P3] A shape {f.A.shape} != (r={r}, in={ind})"
        assert f.B.shape == (out, r), f"[{method_name}/P3] B shape {f.B.shape} != (out={out}, r={r})"
        eff_rank = merged_rank(model, layer)
        assert eff_rank <= r, f"[{method_name}/P3] rank {eff_rank} > r={r} on {layer}"


# ----- TVQ extras --------------------------------------------------

def check_tvq_b32_equals_task_arith() -> None:
    """P4: TVQ at b=32 must equal Task Arithmetic exactly (no-op quantization)."""
    rng_ta = torch.Generator().manual_seed(44)
    rng_tvq = torch.Generator().manual_seed(44)  # same seed = same adapters
    model_ta = make_random_fake_model(LAYER_SPECS, ADAPTERS, rng=rng_ta, scale=0.01)
    model_tvq = make_random_fake_model(LAYER_SPECS, ADAPTERS, rng=rng_tvq, scale=0.01)

    REGISTRY["task_arithmetic"](model_ta, ADAPTERS, WEIGHTS, MERGED)
    REGISTRY["tvq"](model_tvq, ADAPTERS, WEIGHTS, MERGED, rate_bits=32)

    for layer in LAYER_SPECS:
        diff = (model_ta.get_delta(MERGED, layer) - model_tvq.get_delta(MERGED, layer)).abs().max().item()
        assert diff < 1e-5, f"[tvq/P4] b=32 differs from TA on {layer} by {diff:.2e}"


def check_tvq_b1_differs() -> None:
    """P5: TVQ at b=1 must DIFFER measurably from b=32 (sanity check the quantizer fires)."""
    rng_a = torch.Generator().manual_seed(45)
    rng_b = torch.Generator().manual_seed(45)
    model_b32 = make_random_fake_model(LAYER_SPECS, ADAPTERS, rng=rng_a, scale=0.01)
    model_b1 = make_random_fake_model(LAYER_SPECS, ADAPTERS, rng=rng_b, scale=0.01)
    REGISTRY["tvq"](model_b32, ADAPTERS, WEIGHTS, MERGED, rate_bits=32)
    REGISTRY["tvq"](model_b1, ADAPTERS, WEIGHTS, MERGED, rate_bits=1)
    diffs = [
        (model_b32.get_delta(MERGED, layer) - model_b1.get_delta(MERGED, layer)).abs().max().item()
        for layer in LAYER_SPECS
    ]
    assert max(diffs) > 1e-5, f"[tvq/P5] b=1 did not differ from b=32 (max diff {max(diffs):.2e})"


# ----- test runner -------------------------------------------------

def main() -> int:
    # density=1.0 for TIES/DARE so identity collapse holds (pruning is a separate concern)
    method_kwargs_for_identity = {
        "task_arithmetic": {},
        "ties": {"density": 1.0, "majority_sign_method": "total"},
        "dare": {"density": 1.0, "seed": 20260518},
        "knots": {"inner_combination": "linear"},
        "tvq": {"rate_bits": 32},
    }
    # P1 and P3 use the registry defaults (with pruning enabled)
    method_kwargs_default = {
        "task_arithmetic": {},
        "ties": {"density": 0.5, "majority_sign_method": "total"},
        "dare": {"density": 0.5, "seed": 20260518},
        "knots": {"inner_combination": "linear"},
        "tvq": {"rate_bits": 4},
    }

    n_pass = 0
    n_fail = 0
    failures: list[tuple[str, str]] = []

    # The registry grew (fisher_avg, della, regmean, adamerging, rd_encoder,
    # magnitude_prune) without these kwarg tables growing with it, so the loop
    # died on the first unlisted method with a KeyError and every property
    # check after it silently stopped running. Cover what is declared and name
    # what is not, instead of crashing.
    covered = [m for m in REGISTRY
               if m in method_kwargs_default and m in method_kwargs_for_identity]
    uncovered = [m for m in REGISTRY if m not in covered]
    if uncovered:
        print(f"NOTE: no kwargs declared, not property-checked: "
              f"{', '.join(sorted(uncovered))}")
        print()

    for method, fn in [(m, REGISTRY[m]) for m in covered]:
        for prop_name, check_fn, kwargs in [
            ("P1_zero_idempotence",
             lambda fn=fn, m=method: check_zero_idempotence(m, fn, **method_kwargs_default[m]),
             None),
            ("P2_identity_collapse",
             lambda fn=fn, m=method: check_identity_collapse(m, fn, atol=1e-3, **method_kwargs_for_identity[m]),
             None),
            ("P3_rank_r_preservation",
             lambda fn=fn, m=method: check_rank_r_preservation(m, fn, **method_kwargs_default[m]),
             None),
        ]:
            label = f"{method}/{prop_name}"
            try:
                check_fn()
                print(f"PASS  {label}")
                n_pass += 1
            except Exception as e:
                print(f"FAIL  {label}: {e}")
                failures.append((label, traceback.format_exc()))
                n_fail += 1

    # TVQ extras
    for label, fn in [
        ("tvq/P4_b32_equals_task_arith", check_tvq_b32_equals_task_arith),
        ("tvq/P5_b1_differs_from_b32", check_tvq_b1_differs),
    ]:
        try:
            fn()
            print(f"PASS  {label}")
            n_pass += 1
        except Exception as e:
            print(f"FAIL  {label}: {e}")
            failures.append((label, traceback.format_exc()))
            n_fail += 1

    print()
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed")
    if failures:
        print()
        for label, tb in failures:
            print(f"--- TRACEBACK {label} ---")
            print(tb)
        return 1
    print("ALL_TESTS_GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
