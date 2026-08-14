"""Is RegMean's lambda = 0 cell defined at all?

RegMean solves  (sum_t A_t^T A_t + lambda I) X^T = (sum_t A_t^T A_t Delta_t)^T
in the FULL input dimension. Each A_t is (r, in_dim), so each Gram has rank <= r
and the sum has rank <= T*r = 64, against in_dim in the thousands. If that is
right, lambda = 0 is not "ill-conditioned", it is singular, and the cell has no
value to report rather than a large one.

Checked here on the production A-branch shapes before anything is dispatched.
"""
import torch

torch.manual_seed(20260814)

T, R, IN_DIM, OUT_DIM = 4, 16, 512, 512
A = [0.01 * torch.randn(R, IN_DIM) for _ in range(T)]
B = [0.01 * torch.randn(OUT_DIM, R) for _ in range(T)]
deltas = [b @ a for a, b in zip(A, B)]

gram_sum = torch.zeros(IN_DIM, IN_DIM, dtype=torch.float64)
numer = torch.zeros(OUT_DIM, IN_DIM, dtype=torch.float64)
for a, d in zip(A, deltas):
    g = (a.T.double() @ a.double()) / T
    gram_sum += g
    numer += d.double() @ g

rank = torch.linalg.matrix_rank(gram_sum).item()
eigs = torch.linalg.eigvalsh(gram_sum)
print(f"in_dim={IN_DIM}  T*r={T * R}  rank(sum A^T A) = {rank}")
print(f"eigenvalue range: {eigs.min():.3e} to {eigs.max():.3e}")
print(f"eigenvalue #{T * R} (largest that should be nonzero): {eigs[-T * R]:.3e}")
print(f"eigenvalue #{T * R + 1} (first that should be zero):  {eigs[-T * R - 1]:.3e}")

for lam in (0.0, 1e-8, 1e-6, 1e-3, 0.01, 0.13):
    m = gram_sum + lam * torch.eye(IN_DIM, dtype=torch.float64)
    cond = torch.linalg.cond(m).item()
    try:
        x = torch.linalg.solve(m, numer.T).T
        finite = bool(torch.isfinite(x).all())
        norm = x.norm().item()
        status = f"solved, ||X||={norm:.4g}, finite={finite}"
    except Exception as exc:  # noqa: BLE001
        status = f"RAISED {type(exc).__name__}: {str(exc).splitlines()[0][:60]}"
    print(f"  lambda={lam:<8g} cond={cond:.3e}  {status}")

print("\nreference: ||mean delta|| =",
      f"{torch.stack(deltas).mean(0).norm().item():.4g}")
