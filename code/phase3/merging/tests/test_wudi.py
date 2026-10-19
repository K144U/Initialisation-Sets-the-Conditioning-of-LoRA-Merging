"""WUDI checks required by notes/prereg_wudi_2026-10-19.md before any cell:
the closed form lowers the loss below its value at the sum, and Adam from the
sum, run long enough, approaches the omega -> 0 closed form."""
import torch

from merging.wudi import wudi_adam, wudi_closed, _wudi_loss, merge_wudi
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel


def _cohort(T=4, r=3, d_in=24, d_out=20, tilt=0.2, seed=0):
    g = torch.Generator().manual_seed(seed)
    A0 = torch.randn(r, d_in, generator=g)
    deltas, layers = [], []
    for _ in range(T):
        A = A0 + tilt * torch.randn(r, d_in, generator=g)
        B = torch.randn(d_out, r, generator=g)
        layers.append(FakeLoraLayer(A=A, B=B, scaling=1.0))
        deltas.append(B @ A)
    return deltas, layers


def test_closed_form_lowers_loss():
    deltas, _ = _cohort()
    tau, l0, l1 = wudi_closed(deltas, omega=1e-6)
    assert l1 < l0


def test_adam_long_run_approaches_closed_form():
    torch.manual_seed(0)
    deltas, _ = _cohort(tilt=0.5)
    tau_cf, _, l_cf = wudi_closed(deltas, omega=1e-9)
    tau_ad, l0, l_ad = wudi_adam(deltas, iters=20000, lr=1e-2)
    assert (l_ad - l_cf) / (l0 - l_cf) < 1e-4
    # The loss only sees tau on the span of the update rows, so agreement is
    # checked there; Adam's per-coordinate step also leaves mass off the span.
    U, S, _ = torch.linalg.svd(torch.cat([d.T for d in deltas], 1), full_matrices=False)
    P = U[:, :12] @ U[:, :12].T
    assert float((tau_ad @ P - tau_cf @ P).norm() / (tau_cf @ P).norm()) < 1e-3


def test_published_steps_stay_near_the_sum_at_small_lr():
    deltas, _ = _cohort()
    tau, _, _ = wudi_adam(deltas, iters=300, lr=1e-5)
    s = torch.stack(deltas).sum(0)
    assert float((tau - s).norm() / s.norm()) < 0.05


def test_factored_loss_matches_dense():
    # the factored evaluation must be the same function of tau, gradient and
    # Adam trajectory included, as the official dense one
    from merging.wudi import _wudi_loss_factored
    torch.manual_seed(1)
    _, layers = _cohort(tilt=0.5)
    factors = [(lo.A, 0.7 * lo.B) for lo in layers]
    deltas = [b @ a for a, b in factors]
    vectors = torch.stack(deltas)
    l2 = torch.square(torch.norm(vectors.reshape(4, -1), dim=-1))
    A = torch.stack([a for a, _ in factors])
    Bs = torch.stack([b for _, b in factors])
    AAt, BtB = A @ A.transpose(1, 2), Bs.transpose(1, 2) @ Bs
    tau = (vectors.sum(0) + 0.3 * torch.randn(20, 24)).requires_grad_()
    ld = _wudi_loss(tau, vectors, l2)
    gd, = torch.autograd.grad(ld, tau)
    lf = _wudi_loss_factored(tau, A, Bs, AAt, BtB, l2)
    gf, = torch.autograd.grad(lf, tau)
    assert abs(float(ld - lf)) / float(ld) < 1e-5
    assert float((gd - gf).norm() / gd.norm()) < 1e-5
    td, _, _ = wudi_adam(deltas, iters=50, lr=1e-3)
    tf, _, _ = wudi_adam(deltas, iters=50, lr=1e-3, factors=factors)
    assert float((td - tf).norm() / td.norm()) < 1e-5


def test_merge_records_diagnostics():
    _, layers = _cohort()
    m = FakePeftModel(layer_topology={"layers.0.q_proj": (20, 24, 3)})
    for i, lo in enumerate(layers):
        m.add_adapter(f"t{i}", {"layers.0.q_proj": lo})
    merge_wudi(m, [f"t{i}" for i in range(4)], [0.25] * 4, "merged", mode="closed")
    from merging.wudi import DIAGNOSTICS
    d = DIAGNOSTICS["layers.0.q_proj"]
    assert {"amp", "tail_frac", "kappa_hbar", "off_range_frac", "loss_final"} <= set(d)
    assert "merged" in m.adapters
