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


def wudi_adam(deltas, iters=300, lr=1e-5):
    vectors = torch.stack([d.float() for d in deltas])
    l2 = torch.square(torch.norm(vectors.reshape(vectors.shape[0], -1), p=2, dim=-1))
    tau = torch.nn.Parameter(vectors.sum(dim=0).clone())
    opt = torch.optim.Adam([tau], lr=lr, weight_decay=0)
    with torch.no_grad():
        loss0 = float(_wudi_loss(tau, vectors, l2))
    for _ in range(int(iters)):
        loss = _wudi_loss(tau, vectors, l2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        loss1 = float(_wudi_loss(tau, vectors, l2))
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


def merge_wudi(model, adapter_names, weights, merged_adapter_name,
               mode: str = "adam", iters: int = 300, lr: float = 1e-5,
               omega: float = 1e-6, **_kwargs) -> None:
    DIAGNOSTICS.clear()
    sampled = set(diag_layers(model.layer_names()))
    new_factors = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [model.get_delta(a, layer) for a in adapter_names]
        if mode == "adam":
            with torch.enable_grad():
                tau, l0, l1 = wudi_adam(deltas, iters=iters, lr=lr)
        elif mode == "closed":
            tau, l0, l1 = wudi_closed(deltas, omega=omega)
        else:
            raise ValueError(f"unknown WUDI mode {mode!r}")
        entry = {"loss_at_sum": l0, "loss_final": l1}
        if layer in sampled:
            entry.update(mechanism_diag(model, adapter_names, layer, tau, deltas))
        DIAGNOSTICS[layer] = entry
        A_new, B_new = svd_truncate_to_rank(tau.to(deltas[0].dtype), r)
        new_factors[layer] = FakeLoraLayer(A=A_new, B=B_new, scaling=1.0)
        del tau, deltas
    model.add_adapter(merged_adapter_name, new_factors)
