"""CPU checks for TSV-M (merging/tsv_merge.py) and MaTS (merging/mats_kfac.py)."""
import torch

from merging.mats_kfac import merge_mats
from merging.tsv_merge import tsv_merge_matrix
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel
from eval.activation_grams import gram_key


def test_tsv_is_the_sum_on_orthogonal_tasks():
    # Two full-rank 4x4 updates on disjoint blocks of an 8x8 layer. TSV-M keeps
    # int(8 / 2) = 4 directions per task, exactly each block's rank, so U and V
    # are already orthogonal, the whitening is the identity, and it returns the sum.
    torch.manual_seed(0)
    d = 8
    D1 = torch.zeros(d, d); D1[:4, :4] = torch.randn(4, 4)
    D2 = torch.zeros(d, d); D2[4:, 4:] = torch.randn(4, 4)
    got = tsv_merge_matrix([D1, D2])
    assert torch.allclose(got, D1 + D2, atol=1e-5)


def test_tsv_whitening_inflates_near_collinear_inputs():
    # Same output directions, nearly identical input directions: the whitening
    # pushes the small difference between the V's to full length.
    torch.manual_seed(1)
    d = 16
    u = torch.linalg.qr(torch.randn(d, 2))[0]
    v = torch.linalg.qr(torch.randn(d, 2))[0]
    e = 1e-3 * torch.randn(d, 2)
    D1 = u @ torch.diag(torch.tensor([3.0, 2.0])) @ v.T
    D2 = torch.linalg.qr(torch.randn(d, 2))[0] @ torch.diag(torch.tensor([3.0, 2.0])) @ (v + e).T
    merged = tsv_merge_matrix([D1, D2])
    # energy outside span(v) should be O(1), not O(1e-3)
    P = v @ v.T
    off = (merged @ (torch.eye(d) - P)).norm() / merged.norm()
    assert off > 0.2


def _model(T=3, out=6, d=5, r=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = FakePeftModel(layer_topology={"b.0.self_attn.o_proj": (out, d, r)})
    for t in range(T):
        m.add_adapter(f"t{t}", {k: FakeLoraLayer(A=torch.randn(r, d, generator=g),
                                                 B=torch.randn(out, r, generator=g))
                                for k in m.layer_topology})
    return m, [f"t{t}" for t in range(T)]


def _spd(n, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(3 * n, n, generator=g)
    return X.T @ X / (3 * n) + 0.1 * torch.eye(n)


def test_mats_solves_the_kfac_normal_equations():
    m, names = _model()
    layer = "b.0.self_attn.o_proj"
    out, d, r = m.layer_topology[layer]
    gi = {n: {gram_key(layer): _spd(d, i)} for i, n in enumerate(names)}
    go = {n: {layer: _spd(out, 10 + i)} for i, n in enumerate(names)}
    D = [m.get_delta(n, layer) for n in names]
    lhs = lambda X: sum(go[n][layer] @ X @ gi[n][gram_key(layer)] for n in names)
    b = sum(go[n][layer] @ Dt @ gi[n][gram_key(layer)] for n, Dt in zip(names, D))
    # exact solution by vectorisation
    K = sum(torch.kron(go[n][layer], gi[n][gram_key(layer)].T.contiguous()) for n in names)
    exact = torch.linalg.solve(K, b.reshape(-1)).reshape(out, d)
    assert torch.allclose(lhs(exact), b, atol=1e-4)
    merge_mats(m, names, [1 / 3] * 3, "merged", grams=gi, grams_out=go,
               num_iterations=200)
    U, S, Vh = torch.linalg.svd(exact)
    assert torch.allclose(m.get_delta("merged", layer),
                          U[:, :r] @ torch.diag(S[:r]) @ Vh[:r], atol=1e-3)
