"""CPU tests for the RD encoder merge method (E1)."""

import math

import torch

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from merging.rd_encoder import _fwht, _next_pow2, merge_rd_encoder
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


def test_fwht_orthogonal():
    x = torch.randn(256)
    y = _fwht(x) / math.sqrt(256)
    assert abs(float(y.norm() - x.norm())) < 1e-4          # isometry
    back = _fwht(y) / math.sqrt(256)
    assert torch.allclose(back, x, atol=1e-4)              # involution


def test_b_inf_is_weighted_centroid():
    m = _make_model()
    merge_rd_encoder(m, _names(), [1 / T] * T, "rd_inf", bits=32)
    for layer in m.layer_topology:
        W = m.get_delta("rd_inf", layer)
        # reference centroid computed independently
        deltas = [m.get_delta(f"task{t}", layer).to(torch.float32)
                  for t in range(T)]
        Vs = []
        for d in deltas:
            _, _, vh = torch.linalg.svd(d, full_matrices=False)
            Vs.append(vh[:R, :].T)
        Mw = torch.cat([V / math.sqrt(T) for V in Vs], dim=1)
        Hbar = Mw @ Mw.T
        N = sum(deltas) / T
        tau = N @ torch.linalg.pinv(Hbar)
        # rd_inf result was rank-truncated to R; compare after truncating ref
        U, S, Vh = torch.linalg.svd(tau, full_matrices=False)
        tau_r = U[:, :R] @ torch.diag(S[:R]) @ Vh[:R, :]
        rel = float((W.to(torch.float32) - tau_r).norm() / tau_r.norm())
        assert rel < 5e-2, f"{layer}: rel err {rel}"


def test_quantization_monotone_in_bits():
    m = _make_model()
    merge_rd_encoder(m, _names(), [1 / T] * T, "rd_inf", bits=32)
    errs = {}
    for b in (2, 8):
        merge_rd_encoder(m, _names(), [1 / T] * T, f"rd_b{b}", bits=b)
        e = 0.0
        for layer in m.layer_topology:
            ref = m.get_delta("rd_inf", layer)
            got = m.get_delta(f"rd_b{b}", layer)
            e += float((got - ref).norm() ** 2)
        errs[b] = e
    assert errs[8] < errs[2], f"b=8 should beat b=2: {errs}"
    assert errs[8] < 1e-2 * errs[2] + 1e-8 or errs[8] < errs[2] * 0.2


def test_deterministic_under_seed():
    m1, m2 = _make_model(), _make_model()
    merge_rd_encoder(m1, _names(), [1 / T] * T, "rd", bits=4, seed=7)
    merge_rd_encoder(m2, _names(), [1 / T] * T, "rd", bits=4, seed=7)
    for layer in m1.layer_topology:
        assert torch.equal(m1.get_delta("rd", layer),
                           m2.get_delta("rd", layer))


def test_next_pow2():
    assert _next_pow2(1) == 1
    assert _next_pow2(12) == 16
    assert _next_pow2(64) == 64


def test_fisher_diag_mode_matches_per_column_formula():
    """h_t_mode='fisher_diag' must implement
    W*[i,j] = sum_t (w_t * d_t[j] * Δ_t[i,j]) / (sum_t w_t * d_t[j] + λ)
    per layer, per column. Verify against an independent reconstruction
    and that ridge_lambda=0 recovers the diagonal-Hbar pseudoinverse.
    """
    m = _make_model(seed=11)
    g = torch.Generator().manual_seed(2026)
    fisher_data = {
        f"task{t}": {
            layer: 0.1 + torch.rand(IN, generator=g, dtype=torch.float32)
            for layer in m.layer_topology
        }
        for t in range(T)
    }
    weights = [1.0 / T] * T
    merge_rd_encoder(m, _names(), weights, "rd_fisher", bits=32,
                     h_t_mode="fisher_diag", fisher_data=fisher_data,
                     ridge_lambda=0.0)
    for layer in m.layer_topology:
        deltas = [m.get_delta(f"task{t}", layer).to(torch.float32)
                  for t in range(T)]
        d_ts = [fisher_data[f"task{t}"][layer] for t in range(T)]
        d_bar = sum(w * d for w, d in zip(weights, d_ts))
        N = sum(w * d_t[None, :] * delt
                for w, d_t, delt in zip(weights, d_ts, deltas))
        ref_W = N / d_bar[None, :]
        # rd_fisher result is rank-R truncated. Compare after truncating ref.
        U, S, Vh = torch.linalg.svd(ref_W, full_matrices=False)
        ref_W_r = U[:, :R] @ torch.diag(S[:R]) @ Vh[:R, :]
        W = m.get_delta("rd_fisher", layer).to(torch.float32)
        rel = float((W - ref_W_r).norm() / ref_W_r.norm())
        assert rel < 5e-3, f"{layer}: rel err {rel}"


def test_fisher_diag_ridge_interpolates_to_uniform_ta():
    """With ridge_lambda >> max(d_bar), the per-column denom is ridge-dominated
    and W* approaches lambda^{-1} * sum_t (w_t * d_t * Δ_t), which scales the
    contributions but preserves direction. For sanity, check that a moderate
    ridge produces a finite, smaller-magnitude W* than ridge=0 (the
    well-conditioned interior of the ridge curve)."""
    m_a, m_b = _make_model(seed=5), _make_model(seed=5)
    g = torch.Generator().manual_seed(13)
    fisher_data = {
        f"task{t}": {
            layer: 1e-3 + torch.rand(IN, generator=g, dtype=torch.float32)
            for layer in m_a.layer_topology
        }
        for t in range(T)
    }
    merge_rd_encoder(m_a, _names(), [1 / T] * T, "rd_l0", bits=32,
                     h_t_mode="fisher_diag", fisher_data=fisher_data,
                     ridge_lambda=0.0)
    merge_rd_encoder(m_b, _names(), [1 / T] * T, "rd_lhi", bits=32,
                     h_t_mode="fisher_diag", fisher_data=fisher_data,
                     ridge_lambda=1.0)
    for layer in m_a.layer_topology:
        W0 = m_a.get_delta("rd_l0", layer).to(torch.float32)
        Wh = m_b.get_delta("rd_lhi", layer).to(torch.float32)
        assert torch.isfinite(W0).all() and torch.isfinite(Wh).all()
        assert Wh.norm() < W0.norm(), f"{layer}: ridge should shrink"


def test_fisher_diag_rejects_unsupported_combos():
    """fisher_diag mode currently supports bits>=32, realize='rank_r', no
    full_rank_patch — assertions enforce this."""
    m = _make_model(seed=9)
    fisher_data = {
        f"task{t}": {layer: torch.ones(IN) for layer in m.layer_topology}
        for t in range(T)
    }
    # Missing fisher_data -> ValueError
    try:
        merge_rd_encoder(m, _names(), [1 / T] * T, "x",
                         h_t_mode="fisher_diag")
        assert False, "expected ValueError"
    except ValueError:
        pass
    # bits < 32 -> AssertionError
    try:
        merge_rd_encoder(m, _names(), [1 / T] * T, "x", bits=4,
                         h_t_mode="fisher_diag", fisher_data=fisher_data)
        assert False, "expected AssertionError for bits<32"
    except AssertionError:
        pass
    # realize='rank_deff' -> AssertionError
    try:
        merge_rd_encoder(m, _names(), [1 / T] * T, "x", bits=32,
                         realize="rank_deff",
                         h_t_mode="fisher_diag", fisher_data=fisher_data)
        assert False, "expected AssertionError for rank_deff"
    except AssertionError:
        pass


def test_renorm_ta_matches_task_arithmetic_norm():
    """renorm='ta' must leave direction alone and set ||W*||_F == ||TA||_F."""
    m = _make_model()
    w = [1 / T] * T
    merge_rd_encoder(m, _names(), w, "plain", bits=32, ridge_lambda=0.05,
                     realize="rank_deff")
    merge_rd_encoder(m, _names(), w, "renormed", bits=32, ridge_lambda=0.05,
                     realize="rank_deff", renorm="ta")
    for layer in m.layer_topology:
        ta = sum(wt * m.get_delta(f"task{t}", layer).to(torch.float32)
                 for t, wt in enumerate(w))
        plain = m.get_delta("plain", layer).to(torch.float32)
        renormed = m.get_delta("renormed", layer).to(torch.float32)
        # norm matches TA (rank_deff carries W* exactly, so this is tight)
        rel = abs(float(renormed.norm()) - float(ta.norm())) / float(ta.norm())
        assert rel < 1e-4, f"{layer}: renormed norm off by {rel}"
        # direction is unchanged: renormed is a positive multiple of plain
        cos = float((plain * renormed).sum()
                    / (plain.norm() * renormed.norm()))
        assert cos > 1 - 1e-5, f"{layer}: renorm rotated the solution, cos={cos}"


def test_renorm_rejects_unknown_value():
    m = _make_model()
    try:
        merge_rd_encoder(m, _names(), [1 / T] * T, "x", bits=32, renorm="nope")
        assert False, "expected ValueError for unknown renorm"
    except ValueError:
        pass


def main() -> int:
    """No pytest in the conda env; run the module as a script."""
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
