"""The filter-factor formulas in select_ridge agree with a direct ridge solve,
and both rules return a lambda inside the registered grid."""
import torch

from merging.regmean import SELECT_GRID, select_ridge


def _cohort(T=4, r=3, d_in=30, d_out=12, tilt=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    A0 = torch.randn(r, d_in, generator=g) / d_in ** 0.5
    A, D = [], []
    for _ in range(T):
        a = A0 + tilt * torch.randn(r, d_in, generator=g) / d_in ** 0.5
        b = torch.randn(d_out, r, generator=g)
        A.append(a)
        D.append(b @ a)
    return A, D


def test_formulas_match_direct_solve():
    A, D = _cohort()
    w = [0.25] * 4
    X = torch.cat([0.5 * a.double().T for a in A], 1)
    Y = torch.cat([0.5 * (d.double() @ a.double().T) for a, d in zip(A, D)], 1)
    lam = 1e-3
    W = Y @ X.T @ torch.linalg.inv(X @ X.T + lam * torch.eye(X.shape[0], dtype=torch.float64))
    R = Y - W @ X
    H = X.T @ torch.linalg.inv(X @ X.T + lam * torch.eye(X.shape[0], dtype=torch.float64)) @ X
    n = X.shape[1]
    gcv_direct = n * float((R ** 2).sum()) / (n - float(torch.trace(H))) ** 2
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    yn2 = torch.sum((Y @ Vh.T) ** 2, 0)
    f = S ** 2 / (S ** 2 + lam)
    gcv_filter = n * float(torch.sum((1 - f) ** 2 * yn2)) / (n - float(f.sum())) ** 2
    assert abs(gcv_direct - gcv_filter) / gcv_direct < 1e-8
    assert abs(float((W ** 2).sum()) - float(torch.sum(yn2 * S ** 2 / (S ** 2 + lam) ** 2))) < 1e-8


def test_rules_return_grid_values():
    A, D = _cohort()
    for mode in ("gcv", "lcurve"):
        lam, info = select_ridge(A, D, [0.25] * 4, mode)
        assert float(SELECT_GRID[0]) <= lam <= float(SELECT_GRID[-1])
        assert info["lambda"] == lam
