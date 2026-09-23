"""CPU checks for RegMean with real Grams (merging/regmean_act.py)."""
import torch

from merging.regmean_act import merge_regmean_act
from merging.regmean import merge_regmean
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel
from eval.activation_grams import gram_key


def _model(T=3, out=12, d=10, r=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = FakePeftModel(layer_topology={"blk.0.self_attn.q_proj": (out, d, r),
                                      "blk.0.self_attn.o_proj": (out, d, r)})
    for t in range(T):
        m.add_adapter(f"t{t}", {k: FakeLoraLayer(A=torch.randn(r, d, generator=g),
                                                 B=torch.randn(out, r, generator=g))
                                for k in m.layer_topology})
    return m, [f"t{t}" for t in range(T)], d, r


def _spd(d, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * d, d, generator=g)
    return X.T @ X / (4 * d)


def test_gram_key_shares_qkv():
    assert gram_key("m.layers.3.self_attn.q_proj") == gram_key("m.layers.3.self_attn.v_proj")
    assert gram_key("m.layers.3.self_attn.o_proj") != gram_key("m.layers.3.self_attn.q_proj")


def test_identical_grams_give_the_mean_before_truncation():
    m, names, d, r = _model(r=6, d=10, T=2)
    G = _spd(d, 1)
    grams = {n: {gram_key(k): G for k in m.layer_topology} for n in names}
    merge_regmean_act(m, names, [0.5, 0.5], "merged", grams=grams, alpha=0.9)
    from merging.regmean_act import DIAGNOSTICS
    for layer in m.layer_topology:
        mean = sum(m.get_delta(n, layer) for n in names) / 2
        # amp is |merged|/|mean|; with identical Grams merged == mean exactly
        assert abs(DIAGNOSTICS[layer]["amp_pre_trunc"] - 1.0) < 1e-5


def test_solves_regmean_normal_equations():
    m, names, d, r = _model(T=3)
    Gs = {n: _spd(d, i + 10) for i, n in enumerate(names)}
    grams = {n: {gram_key(k): Gs[n] for k in m.layer_topology} for n in names}
    w = [1 / 3] * 3
    layer = "blk.0.self_attn.q_proj"
    Gt = {n: 0.9 * Gs[n] + 0.1 * torch.diag(torch.diagonal(Gs[n])) for n in names}
    expect = (sum(m.get_delta(n, layer) @ Gt[n] for n in names)
              @ torch.linalg.inv(sum(Gt[n] for n in names)))
    merge_regmean_act(m, names, w, "merged", grams=grams, alpha=0.9)
    got = m.get_delta("merged", layer)
    # merged is rank-r truncated; compare against the truncation of expect
    U, S, Vh = torch.linalg.svd(expect)
    expect_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
    assert torch.allclose(got, expect_r, atol=1e-4)


def test_dataless_alpha_none_is_unchanged():
    m1, names, d, r = _model(T=2, seed=3)
    m2, _, _, _ = _model(T=2, seed=3)
    merge_regmean(m1, names, [0.5, 0.5], "a", ridge_lambda=1e-3)
    merge_regmean(m2, names, [0.5, 0.5], "a", ridge_lambda=1e-3, alpha=None)
    for layer in m1.layer_topology:
        assert torch.equal(m1.get_delta("a", layer), m2.get_delta("a", layer))


def test_kappa_target_rule_caps_the_condition_number():
    from merging.regmean import RULE_LAMBDAS

    class View(FakePeftModel):
        def get_layer(self, adapter, layer):
            return self.adapters[adapter][layer]

    g = torch.Generator().manual_seed(5)
    d, r = 12, 2
    base = torch.randn(r, d, generator=g)
    m = View(layer_topology={"L": (6, d, r)})
    for t in range(3):   # near-collinear A's: an ill-conditioned Gram
        A = base + 1e-3 * torch.randn(r, d, generator=g)
        m.add_adapter(f"t{t}", {"L": FakeLoraLayer(A=A, B=torch.randn(6, r, generator=g))})
    names = ["t0", "t1", "t2"]
    merge_regmean(m, names, [1 / 3] * 3, "merged", ridge_mode="kappa_target", kappa_target=100)
    lam = RULE_LAMBDAS["L"]
    G = sum(m.get_layer(n, "L").A.T @ m.get_layer(n, "L").A for n in names) / 3
    ev = torch.linalg.eigvalsh(G)
    nz = ev[ev > 1e-10 * ev[-1]]
    assert lam > 0
    assert abs((nz[-1] + lam) / (nz[0] + lam) - 100) < 1e-2
