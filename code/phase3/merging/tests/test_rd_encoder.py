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
