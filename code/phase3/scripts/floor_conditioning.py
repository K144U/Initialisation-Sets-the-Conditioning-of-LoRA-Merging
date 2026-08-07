"""Follow-up: the exact floor is zero in BOTH cohorts, so why does shared init
look different at all?

Because floor = 0 only says an exact interpolator w exists (the V_t are in
direct sum, q = Tr = 64, in both cohorts). It says nothing about its NORM. If
the V_t are near-collinear, the interpolator exists but is enormous, and no
practical merge with bounded norm, weight decay, or rank truncation can reach
it.

This measures the thing that actually differs: the conditioning of Hbar and
the norm of the exact solution, per layer, averaged.
"""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
ART, RES = ROOT / "artifacts/lora", ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASK_DIRS = ["gsm8k", "alpaca", "magicoder", "flores"]


def layer(A, D, r=16):
    tasks = list(A)
    T = len(tasks)
    V = {t: torch.linalg.qr(A[t].T)[0][:, :r] for t in tasks}
    M = torch.cat([V[t] for t in tasks], dim=1)
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    q = int((S > 1e-10 * S[0]).sum())
    Q = U[:, :q]
    C = {t: Q.T @ V[t] for t in tasks}
    P = {t: C[t] @ C[t].T for t in tasks}
    Dt = {t: D[t] @ Q for t in tasks}
    Hbar = sum(P[t] for t in tasks) / T
    G = sum(Dt[t] for t in tasks) / T
    ev = torch.linalg.eigvalsh(Hbar).clamp(min=0)
    tauH = G @ torch.linalg.pinv(Hbar, rtol=1e-10)
    # norm of the mean delta, as the natural scale to compare against
    return {
        "q": q,
        "sv_min_M": float(S[min(q, len(S)) - 1]),
        "sv_max_M": float(S[0]),
        "cond_M": float(S[0] / S[min(q, len(S)) - 1]),
        "eig_min_Hbar": float(ev[0]),
        "eig_max_Hbar": float(ev[-1]),
        "cond_Hbar": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
        "norm_tauH": float(tauH.norm()),
        "norm_mean_delta": float(G.norm()),
        "amplification": float(tauH.norm() / G.norm()),
    }


out = {}
for cohort in ["seed1", "indep1"]:
    out[cohort] = {}
    print("=" * 92)
    print(f"cohort = {cohort}")
    print(f"{'base':<13}{'q':>4}{'cond(Hbar)':>13}{'min eig':>11}"
          f"{'||tauH||':>11}{'||mean D||':>12}{'amplif.':>10}")
    for base in BASES:
        paths = {t: ART / base / t / cohort / "adapter_model.safetensors"
                 for t in TASK_DIRS}
        if any(not p.exists() for p in paths.values()):
            print(f"{base:<13} SKIP")
            continue
        mats = {t: load_file(str(p)) for t, p in paths.items()}
        keys = sorted(k for k in mats[TASK_DIRS[0]] if k.endswith("lora_A.weight"))
        step = max(1, len(keys) // 8)
        keys = keys[::step][:8]
        per = []
        for ka in keys:
            kb = ka.replace("lora_A", "lora_B")
            A = {t: mats[t][ka].float() for t in TASK_DIRS}
            D = {t: mats[t][kb].float() @ mats[t][ka].float() for t in TASK_DIRS}
            per.append(layer(A, D))
        n = len(per)
        agg = {k: sum(l[k] for l in per) / n for k in per[0]}
        out[cohort][base] = agg
        print(f"{base:<13}{agg['q']:>4.0f}{agg['cond_Hbar']:>13.1f}"
              f"{agg['eig_min_Hbar']:>11.2e}{agg['norm_tauH']:>11.4f}"
              f"{agg['norm_mean_delta']:>12.4f}{agg['amplification']:>10.1f}")
    print()

(RES / "floor_conditioning.json").write_text(json.dumps(out, indent=2))
print("wrote", RES / "floor_conditioning.json")
