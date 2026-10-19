"""WUDI-Merging (Cheng et al., arXiv 2503.08099), as published and in closed form.

Registered in notes/prereg_wudi_2026-10-19.md. Per LoRA-wrapped layer, with the
task updates D_t = scaling_t B_t A_t (out x in), WUDI minimises

    L(tau) = sum_t ||(tau - D_t) D_t^T||_F^2 / ||D_t||_F^2.

mode="adam" is the official procedure (github.com/nathanielyvo/WUDI-Merging,
vit/wudi_main.py, get_redundant_task_vector): tau initialised at sum_t D_t,
Adam(lr=1e-5, weight_decay=0), `iters` steps (300 published), float32.

mode="closed" is the paper's Eq. 20 with ridge omega toward the task updates:

    tau = [sum_t w_t D_t (D_t^T D_t + omega I)] [sum_t w_t (D_t^T D_t + omega I)]^{-1},
    w_t = 1 / ||D_t||_F^2,

solved in float64. As published there are no task weights and no rescaling
coefficient, so `weights` is ignored. The result is SVD-truncated to rank r
for parity with every other merge in this pipeline; amplification and the
tail fraction of the pre-truncation solution are recorded on 8 sampled layers.

2026-10-19, before any WUDI cell finished: the official code evaluates the
loss densely, (tau - D_t) D_t^T is out x out, which on an 8B base made one
3000-step cell cost about an exaflop (the first two cells ran 4.7 h without
finishing a merge). With D_t = B'_t A_t (B'_t = scaling_t B_t) the same loss is

    ||(tau - D_t) D_t^T||^2 = tr(X_t^T X_t B'_t^T B'_t),
    X_t = tau A_t^T - B'_t (A_t A_t^T)   (out x r),

the identical function of tau with the identical gradient, evaluated with out x r
and r x r products. Same init, optimiser, lr, steps and float32, so the
registered procedure is unchanged; tests/test_wudi.py checks loss, gradient
and a 50-step Adam trajectory against the dense form. When factors are
available D_t is formed from them in float32 (the exact update, not its bf16
product), and the closed form, the init and the diagnostics use the same D_t.
"""
from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from ._diag import diag_layers, mechanism_diag
from .tests._fake_model import FakeLoraLayer

DIAGNOSTICS: dict = {}


def _wudi_loss(tau, vectors, l2):
    inner = torch.matmul(tau.unsqueeze(0) - vectors, vectors.transpose(1, 2))
    return torch.sum(torch.square(inner) / l2.view(-1, 1, 1))


def _wudi_loss_factored(tau, A, Bs, AAt, BtB, l2):
    """The same loss from factors: A (T, r, in), Bs (T, out, r) = scaling * B."""
    X = torch.einsum("oi,tri->tor", tau, A) - torch.matmul(Bs, AAt)
    XtX = torch.matmul(X.transpose(1, 2), X)
    return torch.sum(torch.einsum("trs,tsr->t", XtX, BtB) / l2)


def wudi_adam(deltas, iters=300, lr=1e-5, factors=None):
    """factors: optional [(A_t, B'_t)] with D_t = B'_t A_t, for the factored loss."""
    vectors = torch.stack([d.float() for d in deltas])
    l2 = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
    tau = torch.nn.Parameter(vectors.sum(dim=0).clone())
    opt = torch.optim.Adam([tau], lr=lr, weight_decay=0)
    if factors is None:
        loss_fn = lambda t: _wudi_loss(t, vectors, l2)
    else:
        A = torch.stack([a.float() for a, _ in factors])
        Bs = torch.stack([b.float() for _, b in factors])
        AAt = torch.matmul(A, A.transpose(1, 2))
        BtB = torch.matmul(Bs.transpose(1, 2), Bs)
        del vectors
        loss_fn = lambda t: _wudi_loss_factored(t, A, Bs, AAt, BtB, l2)
    with torch.no_grad():
        loss0 = float(loss_fn(tau))
    for _ in range(int(iters)):
        loss = loss_fn(tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss1 = float(loss_fn(tau))
    return tau.detach(), loss0, loss1


def wudi_closed(deltas, omega=1e-6):
    D = [d.double() for d in deltas]
    in_dim = D[0].shape[1]
    eye = torch.eye(in_dim, dtype=torch.float64, device=D[0].device)
    denom = torch.zeros(in_dim, in_dim, dtype=torch.float64, device=D[0].device)
    numer = torch.zeros_like(D[0])
    for d in D:
        w = 1.0 / float(torch.sum(d * d))
        G = d.T @ d
        denom += w * (G + omega * eye)
        numer += w * (d @ G + omega * d)
    tau = torch.linalg.solve(denom, numer.T).T
    vectors = torch.stack([d.float() for d in deltas])
    l2 = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
    with torch.no_grad():
        loss0 = float(_wudi_loss(vectors.sum(0), vectors, l2))
        loss1 = float(_wudi_loss(tau.float(), vectors, l2))
    return tau.float(), loss0, loss1


def _factors(model, adapter, layer):
    """(A, scaling * B) in float32, or None when the model exposes no factors."""
    if hasattr(model, "get_layer"):
        f = model.get_layer(adapter, layer)
    elif hasattr(model, "adapters"):
        f = model.adapters[adapter][layer]
    else:
        return None
    return f.A.float(), f.scaling * f.B.float()


def merge_wudi(model, adapter_names, weights, merged_adapter_name,
               mode: str = "adam", iters: int = 300, lr: float = 1e-5,
               omega: float = 1e-6, **_kwargs) -> None:
    DIAGNOSTICS.clear()
    sampled = set(diag_layers(model.layer_names()))
    new_factors = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        out_dtype = model.get_delta(adapter_names[0], layer).dtype
        factors = [_factors(model, a, layer) for a in adapter_names]
        if any(f is None for f in factors):
            factors = None
            deltas = [model.get_delta(a, layer) for a in adapter_names]
        else:
            deltas = [b @ a for a, b in factors]
        if mode == "adam":
            with torch.enable_grad():
                tau, l0, l1 = wudi_adam(deltas, iters=iters, lr=lr, factors=factors)
        elif mode == "closed":
            tau, l0, l1 = wudi_closed(deltas, omega=omega)
        else:
            raise ValueError(f"unknown WUDI mode {mode!r}")
        entry = {"loss_at_sum": l0, "loss_final": l1}
        if layer in sampled:
            entry.update(mechanism_diag(model, adapter_names, layer, tau, deltas))
        DIAGNOSTICS[layer] = entry
        A_new, B_new = svd_truncate_to_rank(tau.to(out_dtype), r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
        del tau, deltas, factors
    model.add_adapter(merged_adapter_name, new_factors)
