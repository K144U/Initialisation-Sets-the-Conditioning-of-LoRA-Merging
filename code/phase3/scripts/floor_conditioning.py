"""Follow-up: the exact floor is zero in BOTH cohorts, so why does shared init
look different at all?

Because floor = 0 only says an exact interpolator w exists (the V_t are in
direct sum, q = Tr, in both cohorts). It says nothing about its NORM. If the
V_t are near-collinear, the interpolator exists but is enormous, and no
practical merge with bounded norm, weight decay, or rank truncation can reach
it.

This measures the thing that actually differs: the conditioning of Hbar and the
norm of the exact solution, per layer, averaged.

It also runs the same measurement with the translation adapter EXCLUDED.
`flores` is degenerate: on all four bases in both arms it fails to beat the
base model at its own task, or is beaten there by another task's adapter (see
adapter_quality.py, commit 059d277). An undertrained adapter's A barely moves
from its init, which is exactly the mechanism that produces collinearity under
shared init, so the degenerate adapter could be manufacturing the conditioning
contrast that E2 is built on. If the contrast survives at T=3 it is real.
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
TASK_SETS = [("T=4 (all)", TASK_DIRS),
             ("T=3 (no translation)", [t for t in TASK_DIRS if t != "flores"])]


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
    return {
        "q": q,
        "Tr": T * r,
        "eig_min_Hbar": float(ev[0]),
        "eig_max_Hbar": float(ev[-1]),
        "cond_Hbar": float(ev[-1] / ev[0]) if ev[0] > 0 else float("inf"),
        "norm_tauH": float(tauH.norm()),
        "norm_mean_delta": float(G.norm()),
        "amplification": float(tauH.norm() / G.norm()),
    }


def run(cohort, tasks_used):
    rows = {}
    for base in BASES:
        paths = {t: ART / base / t / cohort / "adapter_model.safetensors"
                 for t in tasks_used}
        if any(not p.exists() for p in paths.values()):
            print(f"{base:<13} SKIP")
            continue
        mats = {t: load_file(str(p)) for t, p in paths.items()}
        keys = sorted(k for k in mats[tasks_used[0]] if k.endswith("lora_A.weight"))
        step = max(1, len(keys) // 8)
        keys = keys[::step][:8]
        per = []
        for ka in keys:
            kb = ka.replace("lora_A", "lora_B")
            A = {t: mats[t][ka].float() for t in tasks_used}
            D = {t: mats[t][kb].float() @ mats[t][ka].float() for t in tasks_used}
            per.append(layer(A, D))
        n = len(per)
        agg = {k: sum(l[k] for l in per) / n for k in per[0]}
        rows[base] = agg
        print(f"{base:<13}{agg['q']:>4.0f}{agg['Tr']:>5.0f}{agg['cond_Hbar']:>13.1f}"
              f"{agg['eig_min_Hbar']:>11.2e}{agg['amplification']:>10.1f}")
    return rows


out = {}
for label, tasks_used in TASK_SETS:
    for cohort in ["seed1", "indep1"]:
        key = f"{cohort} | {label}"
        print("=" * 78)
        print(f"cohort = {cohort}    tasks = {label}")
        print(f"{'base':<13}{'q':>4}{'Tr':>5}{'cond(Hbar)':>13}"
              f"{'min eig':>11}{'amplif.':>10}")
        out[key] = run(cohort, tasks_used)
        print()

print("=" * 78)
print("DOES THE CONTRAST SURVIVE DROPPING THE DEGENERATE ADAPTER?")
print("=" * 78)
print(f"{'base':<13}{'cond ratio T=4':>16}{'cond ratio T=3':>16}"
      f"{'amp T=4':>11}{'amp T=3':>11}")
for base in BASES:
    try:
        s4 = out["seed1 | T=4 (all)"][base]
        i4 = out["indep1 | T=4 (all)"][base]
        s3 = out["seed1 | T=3 (no translation)"][base]
        i3 = out["indep1 | T=3 (no translation)"][base]
    except KeyError:
        continue
    print(f"{base:<13}{s4['cond_Hbar']/i4['cond_Hbar']:>16.1f}"
          f"{s3['cond_Hbar']/i3['cond_Hbar']:>16.1f}"
          f"{s4['amplification']:>11.1f}{s3['amplification']:>11.1f}")
print()
print("cond ratio = shared / independent. If the T=3 ratio collapses toward 1,")
print("the conditioning contrast was an artifact of the degenerate adapter and")
print("E2 is measuring the wrong thing.")

(RES / "floor_conditioning.json").write_text(json.dumps(out, indent=2))
print("\nwrote", RES / "floor_conditioning.json")
