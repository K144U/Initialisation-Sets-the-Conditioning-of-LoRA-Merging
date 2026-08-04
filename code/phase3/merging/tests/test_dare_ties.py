"""Tests for dare_ties, plus a regression pin on the ties.py refactor.

The refactor extracted merge_ties' per-layer body into ties_merge_deltas so that
dare_ties could reuse it. ties.py had no test file before 2026-08-04, so the
first test here freezes a verbatim copy of the ORIGINAL body and asserts the
extracted function still matches it bit for bit. That is what makes the refactor
safe for the already-published TIES numbers.
"""

from __future__ import annotations

import torch

from .._adapter_utils import svd_truncate_to_rank
from ..dare_ties import _dare_mask, merge_dare_ties
from ..registry import DEFAULT_KWARGS, REGISTRY
from ..ties import _elect_sign, _trim_topk, merge_ties, ties_merge_deltas
from ._fake_model import make_random_fake_model

LAYERS = {"l0": (64, 48, 8), "l1": (32, 32, 4)}
ADAPTERS = ["t0", "t1", "t2", "t3"]
W = [0.25, 0.25, 0.25, 0.25]


def _frozen_original_ties_body(deltas, weights, density, majority_sign_method):
    """Verbatim copy of merge_ties' per-layer body as it stood before the
    2026-08-04 extraction. Do not refactor this; its whole purpose is to be
    stale."""
    trimmed = [_trim_topk(d, density) for d in deltas]
    stacked = torch.stack(trimmed, dim=0)
    device, dtype = stacked.device, stacked.dtype
    elected = _elect_sign(stacked, majority_sign_method)
    sign_t = torch.sign(stacked)
    match = ((sign_t == elected.unsqueeze(0)) & (elected.unsqueeze(0) != 0)).to(dtype)
    w_t = torch.tensor(weights, dtype=dtype, device=device).view(-1, 1, 1)
    weighted = stacked * match * w_t
    denom_sum = (match * w_t).sum(dim=0).abs()
    return torch.where(
        denom_sum > 1e-12,
        weighted.sum(dim=0) / denom_sum.clamp_min(1e-12),
        torch.zeros_like(denom_sum),
    )


def test_refactor_matches_frozen_reference():
    """ties_merge_deltas is numerically identical to the pre-refactor body."""
    g = torch.Generator().manual_seed(7)
    for density in (0.05, 0.2, 0.5, 1.0):
        for msm in ("total", "frequency"):
            deltas = [torch.randn(40, 30, generator=g) for _ in range(4)]
            got = ties_merge_deltas(deltas, W, density, msm)
            want = _frozen_original_ties_body(deltas, W, density, msm)
            assert torch.equal(got, want), f"drift at density={density} msm={msm}"


def test_registered():
    assert REGISTRY["dare_ties"] is merge_dare_ties
    d = DEFAULT_KWARGS["dare_ties"]
    assert d["dare_density"] == 0.2 and d["ties_density"] == 0.2


def test_mask_is_unbiased_and_rescaled():
    """The mask keeps ~density of entries and divides survivors by density."""
    g = torch.Generator().manual_seed(11)
    x = torch.ones(400, 400)
    for dens in (0.1, 0.2, 0.5):
        y = _dare_mask(x, dens, g)
        kept = (y != 0).float().mean().item()
        assert abs(kept - dens) < 0.01, f"kept {kept} vs density {dens}"
        # survivors are exactly 1/density, so the mean is ~1 (unbiased)
        assert torch.allclose(y[y != 0], torch.tensor(1.0 / dens), atol=1e-5)
        assert abs(y.mean().item() - 1.0) < 0.02


def test_density_one_reduces_to_plain_ties():
    """dare_density=1.0 makes the mask the identity, so dare_ties == ties."""
    m = make_random_fake_model(LAYERS, ADAPTERS)
    merge_ties(m, ADAPTERS, W, "ties_out", density=0.2)
    merge_dare_ties(m, ADAPTERS, W, "dt_out", dare_density=1.0, ties_density=0.2)
    for layer in LAYERS:
        a = m.adapters["ties_out"][layer]
        b = m.adapters["dt_out"][layer]
        assert torch.allclose(a.B @ a.A, b.B @ b.A, atol=1e-5)


def test_differs_from_plain_ties_when_masking():
    """At dare_density=0.2 the mask must actually change the merge."""
    m = make_random_fake_model(LAYERS, ADAPTERS)
    merge_ties(m, ADAPTERS, W, "ties_out", density=0.2)
    merge_dare_ties(m, ADAPTERS, W, "dt_out", dare_density=0.2, ties_density=0.2)
    diffs = []
    for layer in LAYERS:
        a = m.adapters["ties_out"][layer]
        b = m.adapters["dt_out"][layer]
        da, db = a.B @ a.A, b.B @ b.A
        diffs.append((db - da).norm().item() / da.norm().item())
    assert min(diffs) > 0.05, f"mask had ~no effect: {diffs}"


def test_trim_goes_inert_below_ties_density():
    """dare_density < ties_density: TIES' trim keeps every surviving entry.

    This is the structural claim recorded in the pre-registration, so it is
    asserted rather than assumed.
    """
    g = torch.Generator().manual_seed(3)
    x = torch.randn(200, 200, generator=g)
    masked = _dare_mask(x, 0.10, g)
    nz_before = (masked != 0).sum().item()
    trimmed = _trim_topk(masked, 0.20)
    nz_after = (trimmed != 0).sum().item()
    assert nz_after == nz_before, (nz_before, nz_after)


def test_trim_binds_above_ties_density():
    """dare_density > ties_density: the trim really does cut further."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(200, 200, generator=g)
    masked = _dare_mask(x, 0.50, g)
    nz_before = (masked != 0).sum().item()
    nz_after = (_trim_topk(masked, 0.20) != 0).sum().item()
    assert nz_after < nz_before * 0.55, (nz_before, nz_after)


def test_masks_independent_across_tasks():
    """Each adapter must get its own mask draw, not a shared one."""
    g = torch.Generator().manual_seed(5)
    x = torch.randn(300, 300, generator=g)
    m1 = _dare_mask(x, 0.2, g) != 0
    m2 = _dare_mask(x, 0.2, g) != 0
    overlap = (m1 & m2).sum().item() / max(1, m1.sum().item())
    # independent draws overlap at ~density (0.2), identical draws at 1.0
    assert 0.15 < overlap < 0.25, overlap


def test_deterministic_given_seed():
    m1 = make_random_fake_model(LAYERS, ADAPTERS)
    m2 = make_random_fake_model(LAYERS, ADAPTERS)
    merge_dare_ties(m1, ADAPTERS, W, "o", dare_density=0.2, seed=123)
    merge_dare_ties(m2, ADAPTERS, W, "o", dare_density=0.2, seed=123)
    for layer in LAYERS:
        a, b = m1.adapters["o"][layer], m2.adapters["o"][layer]
        assert torch.equal(a.B @ a.A, b.B @ b.A)


def test_output_rank_is_r():
    m = make_random_fake_model(LAYERS, ADAPTERS)
    merge_dare_ties(m, ADAPTERS, W, "o", dare_density=0.2)
    for layer, (out_dim, in_dim, r) in LAYERS.items():
        f = m.adapters["o"][layer]
        assert f.A.shape == (r, in_dim) and f.B.shape == (out_dim, r)
        assert torch.linalg.matrix_rank(f.B @ f.A).item() <= r
