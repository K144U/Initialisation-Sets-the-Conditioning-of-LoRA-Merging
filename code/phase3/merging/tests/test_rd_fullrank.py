"""CPU test for the rd_encoder full-rank residual patch (v2)."""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from merging.rd_encoder import merge_rd_encoder
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel

OUT, IN, R, T = 48, 64, 4, 3


def _make_model(seed=0):
    g = torch.Generator().manual_seed(seed)
    topo = {"l0": (OUT, IN, R)}
    m = FakePeftModel(layer_topology=topo)
    for t in range(T):
        factors = {}
        for layer in topo:
            A = torch.randn(R, IN, generator=g) / math.sqrt(IN)
            B = torch.randn(OUT, R, generator=g) / math.sqrt(R)
            factors[layer] = FakeLoraLayer(A=A, B=B, scaling=1.0)
        m.add_adapter(f"task{t}", factors)
    return m


def test_fullrank_patch_realizes_w_star():
    names = [f"task{t}" for t in range(T)]
    # reference: plain (truncated) merge on an identical model
    m_ref = _make_model()
    merge_rd_encoder(m_ref, names, [1 / T] * T, "rd", bits=32)
    # patched merge
    m_fr = _make_model()
    merge_rd_encoder(m_fr, names, [1 / T] * T, "rd", bits=32,
                     full_rank_patch=True)
    # active model = base (residual) + truncated adapter delta
    realized = m_fr.base_weight("l0") + m_fr.get_delta("rd", "l0")
    truncated_only = m_ref.get_delta("rd", "l0")
    # the realized matrix must differ from the truncated one (residual
    # is nonzero: trunc_mass ~0.3 on random tensors)...
    assert float((realized - truncated_only).norm()) > 1e-3
    # ...and must have rank > R (full-rank W*), while truncated has rank <= R
    s_real = torch.linalg.svdvals(realized)
    s_trunc = torch.linalg.svdvals(truncated_only)
    rank_real = int((s_real > 1e-6 * s_real[0]).sum())
    rank_trunc = int((s_trunc > 1e-6 * s_trunc[0]).sum())
    assert rank_trunc <= R
    assert rank_real > R, f"rank_real={rank_real} should exceed {R}"
    print("PASS test_fullrank_patch_realizes_w_star")


if __name__ == "__main__":
    test_fullrank_patch_realizes_w_star()
