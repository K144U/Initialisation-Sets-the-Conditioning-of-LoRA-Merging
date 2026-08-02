"""CPU tests for KnOTS, pinning the identity that invalidated four claims.

`merge_knots` projects each task delta into the right-singular basis V of the
row-concatenated stack, merges there, then reconstructs with V^T. Because
`torch.linalg.svd(..., full_matrices=False)` returns a V whose columns span
(at least) the row space of every Delta_t, the round trip Delta_t V V^T is the
identity on that row space. With the registry default inner_combination
="linear" the inner merge is itself linear, so the whole method collapses to

    Delta_merged = (sum_t w_t Delta_t) V V^T = sum_t w_t Delta_t = Task Arithmetic

exactly. That is why every shipped KnOTS cell matched TA to 3-4 decimals, and
the paper read that agreement as evidence for its theory ("subspace alignment
has nothing to exploit at zero overlap") in four places. It is not evidence of
anything: the implementation cannot differ from TA.

The TIES inner merge is a genuinely different method, because sign election in
the rotated basis is not sign election in the original basis. That is also the
variant the KnOTS paper headlines (KnOTS-TIES).

Run as a script; there is no pytest in the conda env.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from merging.knots import merge_knots
from merging.task_arithmetic import merge_task_arithmetic
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel

OUT, IN, R, T = 48, 64, 4, 3


def _make_model(seed=0):
    g = torch.Generator().manual_seed(seed)
    topo = {"l0": (OUT, IN, R), "l1": (OUT, IN, R)}
    m = FakePeftModel(layer_topology=topo)
    for t in range(T):
        factors = {}
        for layer in topo:
            A = torch.randn(R, IN, generator=g) / math.sqrt(IN)
            B = torch.randn(OUT, R, generator=g) / math.sqrt(R)
            factors[layer] = FakeLoraLayer(A=A, B=B, scaling=1.0)
        m.add_adapter(f"task{t}", factors)
    return m


def _names():
    return [f"task{t}" for t in range(T)]


def test_knots_linear_is_exactly_task_arithmetic():
    """The registry default is algebraically TA. Pinned, not aspirational."""
    m = _make_model()
    w = [1 / T] * T
    merge_task_arithmetic(m, _names(), w, "ta")
    merge_knots(m, _names(), w, "knots_linear", inner_combination="linear")
    for layer in m.layer_topology:
        ta = m.get_delta("ta", layer).to(torch.float32)
        kn = m.get_delta("knots_linear", layer).to(torch.float32)
        rel = float((ta - kn).norm() / ta.norm())
        assert rel < 1e-4, (
            f"{layer}: KnOTS-linear should be identical to TA, rel err {rel}")


def test_knots_ties_differs_from_task_arithmetic():
    """The TIES inner merge is a real method; it must NOT collapse to TA."""
    m = _make_model()
    w = [1 / T] * T
    merge_task_arithmetic(m, _names(), w, "ta")
    merge_knots(m, _names(), w, "knots_ties", inner_combination="ties",
                density=0.2)
    seen = []
    for layer in m.layer_topology:
        ta = m.get_delta("ta", layer).to(torch.float32)
        kn = m.get_delta("knots_ties", layer).to(torch.float32)
        seen.append(float((ta - kn).norm() / ta.norm()))
    assert min(seen) > 0.05, (
        f"KnOTS-TIES collapsed toward TA (rel diffs {seen}); the inner merge "
        f"is not taking effect")


def test_knots_ties_differs_from_plain_ties():
    """Rotating before the sign election must change the outcome, otherwise
    the alignment step is doing nothing at all."""
    from merging.ties import merge_ties
    m = _make_model()
    w = [1 / T] * T
    merge_ties(m, _names(), w, "ties", density=0.2,
               majority_sign_method="total")
    merge_knots(m, _names(), w, "knots_ties", inner_combination="ties",
                density=0.2)
    diffs = []
    for layer in m.layer_topology:
        a = m.get_delta("ties", layer).to(torch.float32)
        b = m.get_delta("knots_ties", layer).to(torch.float32)
        diffs.append(float((a - b).norm() / a.norm()))
    assert min(diffs) > 0.01, (
        f"KnOTS-TIES is indistinguishable from plain TIES (rel diffs {diffs})")


def test_inner_merge_ties_creates_no_off_device_tensors():
    """Device-safety check that works on a CPU-only box.

    The GPU failure of 2026-08-02 was `w_t = torch.tensor(weights, ...)` with
    no device=, which lands on cpu while `stacked` is cuda. CPU unit tests
    cannot see that: on CPU the mismatch is a no-op. So instead of needing a
    GPU, patch torch.tensor and assert every tensor the function constructs
    was given an explicit device.
    """
    from merging import knots as knots_mod

    real_tensor = torch.tensor
    offenders = []

    def spy(*args, **kwargs):
        if "device" not in kwargs:
            offenders.append(args[0] if args else None)
        return real_tensor(*args, **kwargs)

    coeffs = [torch.randn(8, 6) for _ in range(3)]
    torch.tensor = spy
    try:
        knots_mod._inner_merge_ties(coeffs, [1 / 3] * 3, density=0.5)
    finally:
        torch.tensor = real_tensor

    assert not offenders, (
        f"_inner_merge_ties built {len(offenders)} tensor(s) without an "
        f"explicit device=; that raises on GPU. Offending values: {offenders}")


def test_knots_rejects_unknown_inner_combination():
    m = _make_model()
    try:
        merge_knots(m, _names(), [1 / T] * T, "x", inner_combination="nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def main() -> int:
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    n_pass, failures = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            n_pass += 1
        except Exception:
            print(f"  FAIL  {name}")
            failures.append((name, traceback.format_exc()))
    print(f"\nSUMMARY: {n_pass} passed, {len(failures)} failed")
    for name, tb in failures:
        print(f"--- TRACEBACK {name} ---\n{tb}")
    if failures:
        return 1
    print("ALL_TESTS_GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
