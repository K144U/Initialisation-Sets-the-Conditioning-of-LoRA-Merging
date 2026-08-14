"""R2: measure the mechanism the paper currently asserts.

Section 7.2 explains the regime null by saying that task arithmetic, TIES, DARE
and TVQ "never go near the ill-conditioned direction" because "each produces a
bounded-norm, roughly mean-like solution, whereas the quantity that blows up is
the norm of the exact interpolator". No measurement of that appears anywhere in
the paper. This is that measurement, and it needs no GPU: every merge in this
set is weight arithmetic on the adapter factors.

Two numbers per (base, cohort, method), in the same projected coordinates as
floor_conditioning.py so they sit in the same units as Table 2's 43-64 against
4.0:

  amplification  ||Delta_merged Q|| / ||mean_t Delta_t Q||
                 the interpolator tau_H is the same ratio and is printed as the
                 reference row.

  tail fraction  the share of ||Delta_merged Q||^2 lying in the bottom quartile
                 of Hbar's eigen-directions. This is "how far into the
                 ill-conditioned direction does this solution actually reach".
                 If the assertion is right, every heuristic sits near the mean's
                 value and only tau_H is large.

Layer sampling matches floor_conditioning.py exactly (8 evenly spaced lora_A
keys), so the two tables are comparable row by row.

Usage:  python code/phase3/scripts/merged_solution_norms.py
"""
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
sys.path.insert(0, str(ROOT / "code/phase3"))

from merging.registry import DEFAULT_KWARGS, REGISTRY          # noqa: E402
from merging.tests._fake_model import FakeLoraLayer, FakePeftModel  # noqa: E402

ART, RES = ROOT / "artifacts/lora", ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
COHORTS = ["seed1", "indep1"]
RANK = 16

# The methods the null is stated over, plus the two solvers.
METHODS = [
    ("task_arithmetic", {}),
    ("ties", {}),
    ("dare", {}),
    ("tvq_b2", {"rate_bits": 2}),
    ("knots", {}),
    ("regmean", {"ridge_lambda": 1e-6}),
    ("regmean_l0p13", {"ridge_lambda": 0.13}),
    ("rd_encoder_l0", {"ridge_lambda": 0.0}),
    ("rd_encoder_l0p05", {"ridge_lambda": 0.05}),
]


def base_method(name):
    for prefix in ("tvq", "regmean", "rd_encoder", "task_arithmetic", "ties",
                   "dare", "knots"):
        if name.startswith(prefix):
            return prefix if prefix != "tvq" else "tvq"
    raise KeyError(name)


class ViewModel(FakePeftModel):
    """FakePeftModel plus the get_layer hook PeftModelView provides.

    RegMean asks for the per-task LoRA A factor and falls back to a coarse
    Delta^T Delta surrogate when it cannot get one. The fallback is a different
    method, so measuring production behaviour means providing the hook.
    """

    def get_layer(self, adapter, layer):
        return self.adapters[adapter][layer]


def merged_delta(method, kwargs, factors, topology, layer):
    model = ViewModel(layer_topology={layer: topology})
    for task, (A, B) in factors.items():
        model.add_adapter(task, {layer: FakeLoraLayer(A=A, B=B, scaling=1.0)})
    name = base_method(method)
    kw = dict(DEFAULT_KWARGS.get(name, {}))
    kw.update(kwargs)
    REGISTRY[name](model, list(factors), [1.0 / len(factors)] * len(factors),
                   "merged", **kw)
    return model.get_delta("merged", layer)


def analyse(base, cohort):
    paths = {t: ART / base / t / cohort / "adapter_model.safetensors"
             for t in TASKS}
    if any(not p.exists() for p in paths.values()):
        return None
    mats = {t: load_file(str(p)) for t, p in paths.items()}
    keys = sorted(k for k in mats[TASKS[0]] if k.endswith("lora_A.weight"))
    step = max(1, len(keys) // 8)
    keys = keys[::step][:8]

    acc = {m: {"amp": [], "tail": []} for m, _ in METHODS}
    acc["mean"] = {"amp": [], "tail": []}
    acc["tau_H"] = {"amp": [], "tail": []}

    for ka in keys:
        kb = ka.replace("lora_A", "lora_B")
        A = {t: mats[t][ka].float() for t in TASKS}
        B = {t: mats[t][kb].float() for t in TASKS}
        D = {t: B[t] @ A[t] for t in TASKS}

        # Projected coordinates, identical construction to floor_conditioning.
        V = {t: torch.linalg.qr(A[t].T)[0][:, :RANK] for t in TASKS}
        M = torch.cat([V[t] for t in TASKS], dim=1)
        U, S, _ = torch.linalg.svd(M, full_matrices=False)
        q = int((S > 1e-10 * S[0]).sum())
        Q = U[:, :q]
        Hbar = sum((Q.T @ V[t]) @ (Q.T @ V[t]).T for t in TASKS) / len(TASKS)
        evals, evecs = torch.linalg.eigh(Hbar)
        cut = max(1, q // 4)                      # bottom quartile of Hbar
        tail_basis = evecs[:, :cut]

        G = sum(D[t] @ Q for t in TASKS) / len(TASKS)
        g_norm = G.norm()
        tauH = G @ torch.linalg.pinv(Hbar, rtol=1e-10)

        def record(key, dq):
            acc[key]["amp"].append(float(dq.norm() / g_norm))
            total = float((dq ** 2).sum())
            inside = float(((dq @ tail_basis) ** 2).sum())
            acc[key]["tail"].append(inside / total if total > 0 else 0.0)

        record("mean", G)
        record("tau_H", tauH)
        out_dim, in_dim = D[TASKS[0]].shape
        for m, kwargs in METHODS:
            dm = merged_delta(m, kwargs, {t: (A[t], B[t]) for t in TASKS},
                              (out_dim, in_dim, RANK), ka)
            record(m, dm.float() @ Q)

    return {k: {kk: sum(vv) / len(vv) for kk, vv in v.items()}
            for k, v in acc.items()}


def main():
    out = {}
    for cohort in COHORTS:
        print("=" * 88)
        print(f"cohort = {cohort}")
        print(f"{'base':<13}{'solution':<20}{'amplification':>15}"
              f"{'tail fraction':>15}")
        print("=" * 88)
        for base in BASES:
            rows = analyse(base, cohort)
            if rows is None:
                print(f"{base:<13} SKIP (missing adapters)")
                continue
            out[f"{cohort}|{base}"] = rows
            for key in ["mean"] + [m for m, _ in METHODS] + ["tau_H"]:
                r = rows[key]
                print(f"{base:<13}{key:<20}{r['amp']:>15.2f}{r['tail']:>15.4f}")
            print()
    (RES / "merged_solution_norms.json").write_text(json.dumps(out, indent=2))
    print("wrote", RES / "merged_solution_norms.json")


if __name__ == "__main__":
    main()
