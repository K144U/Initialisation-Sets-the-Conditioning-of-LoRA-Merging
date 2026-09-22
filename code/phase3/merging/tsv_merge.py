"""TSV-Merge (Gargiulo et al., CVPR 2025), ported from the official code.

Source: github.com/AntoAndGar/task_singular_vectors,
src/utils/TSVM_utils.py::compute_and_sum_svd_mem_reduction. Per 2-D layer:

  1. SVD each task's update, u, s, v = svd(Delta_t), and keep the top
     int(min(out, in) / T) components (sv_reduction = 1/T);
  2. place them side by side in sum_u, sum_s, sum_v (zero-padded to the
     full SVD width, as the official code allocates zeros_like(u));
  3. orthogonalise: u_u, _, v_u = svd(sum_u); u_v, _, v_v = svd(sum_v);
  4. merged = u_u @ v_u @ diag(sum_s) @ u_v @ v_v.

Step 3 is a whitening of the stacked singular vectors. When the tasks' input
subspaces nearly coincide, as under shared LoRA initialisation, the stacked
right singular vectors are nearly linearly dependent and the whitening
promotes their small differences to full orthonormal directions. That is why
this method is in the study; nothing about it is changed here.

The only additions are the ones every method in this pipeline gets: the
update is computed on the LoRA deltas and SVD-truncated back to rank r.
The scaling coefficient is the official default of 1.0.
"""
from __future__ import annotations

import torch

from ._adapter_utils import svd_truncate_to_rank
from .tests._fake_model import FakeLoraLayer


def tsv_merge_matrix(deltas: list[torch.Tensor]) -> torch.Tensor:
    T = len(deltas)
    sv_reduction = 1 / T
    sum_u = sum_s = sum_v = None
    for i, vec in enumerate(deltas):
        u, s, v = torch.linalg.svd(vec.float(), full_matrices=False)
        if i == 0:
            sum_u = torch.zeros_like(u)
            sum_s = torch.zeros_like(s)
            sum_v = torch.zeros_like(v)
        k = int(s.shape[0] * sv_reduction)
        sum_u[:, i * k:(i + 1) * k] = u[:, :k]
        sum_s[i * k:(i + 1) * k] = s[:k]
        sum_v[i * k:(i + 1) * k, :] = v[:k, :]
    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))


def merge_tsv(model, adapter_names, weights, merged_adapter_name,
              alpha: float = 1.0, **_kwargs) -> None:
    new_factors = {}
    for layer in model.layer_names():
        _, _, r = model.layer_topology[layer]
        deltas = [model.get_delta(a, layer).float() for a in adapter_names]
        merged = alpha * tsv_merge_matrix(deltas)
        A, B = svd_truncate_to_rank(merged.to(deltas[0].dtype), r)
        new_factors[layer] = FakeLoraLayer(A=A, B=B, scaling=1.0)
    model.add_adapter(merged_adapter_name, new_factors)
