"""Scoping deliverable: the full geometry table at T=4 and at T=3.

Decision (2026-08-08, user): scope honestly, do NOT retrain. The translation
adapter is degenerate on all four bases in both arms (adapter_quality.py,
059d277), so a nominal T=4 cohort has three tasks that learned. Rather than
retrain and invalidate every cell measured on these cohorts, we disclose the
defect and report every geometry quantity at both T=4 (as run) and T=3
(translation excluded), so a reader can see for themselves that nothing rests
on the degenerate adapter.

Conventions are taken verbatim from measure_subspace_geometry.py:
  V_t = orthonormal basis of rowspace(A_t)
  soft d_eff = participation ratio of the squared singular values of
               M = [V_1 | ... | V_T]
  B^2 = max_t ||Delta_t||_F^2
"""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
ART, RES = ROOT / "artifacts/lora", ROOT / "results/phase3"
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
ALL_TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
SETS = [("T=4", ALL_TASKS), ("T=3", [t for t in ALL_TASKS if t != "flores"])]
COHORTS = [("shared", "seed1"), ("independent", "indep1")]
EPS = [1e-6, 1e-3, 1e-2, 1e-1]


def analyse_layer(A, D, r=16):
    tasks = list(A)
    T = len(tasks)
    Tr = T * r
    V = {t: torch.linalg.qr(A[t].T)[0][:, :r] for t in tasks}
    cos = []
    for i in range(T):
        for j in range(i + 1, T):
            cos.append(float(torch.linalg.svdvals(V[tasks[i]].T @ V[tasks[j]])
                             .clamp(max=1.0).median()))
    M = torch.cat([V[t] for t in tasks], dim=1)
    S = torch.linalg.svdvals(M)
    S2 = S * S
    soft = float((S2.sum() ** 2) / (S2 * S2).sum())
    out = {"median_principal_cosine": sum(cos) / len(cos),
           "sigma_max": float(S[0]), "soft_d_eff": soft, "Tr": Tr,
           "soft_floor_frac": max(0.0, 1.0 - soft / Tr)}
    for e in EPS:
        out[f"hard_{e}"] = float(int((S > e * S[0]).sum()))
    return out


rows = {}
for label, tasks in SETS:
    for arm, cohort in COHORTS:
        key = f"{arm} | {label}"
        rows[key] = {}
        for base in BASES:
            paths = {t: ART / base / t / cohort / "adapter_model.safetensors"
                     for t in tasks}
            if any(not p.exists() for p in paths.values()):
                continue
            mats = {t: load_file(str(p)) for t, p in paths.items()}
            keys = sorted(k for k in mats[tasks[0]] if k.endswith("lora_A.weight"))
            step = max(1, len(keys) // 8)
            keys = keys[::step][:8]
            per = []
            for ka in keys:
                kb = ka.replace("lora_A", "lora_B")
                A = {t: mats[t][ka].float() for t in tasks}
                D = {t: mats[t][kb].float() @ mats[t][ka].float() for t in tasks}
                per.append(analyse_layer(A, D))
            n = len(per)
            rows[key][base] = {k: sum(l[k] for l in per) / n for k in per[0]}

print("=" * 104)
print("GEOMETRY AT T=4 (as run) AND T=3 (degenerate translation adapter excluded)")
print("=" * 104)
print(f"{'base':<13}{'arm':<14}{'T':<5}{'med cos':>9}{'sig_max':>9}"
      f"{'soft d_eff':>12}{'Tr':>5}{'floor/B^2':>11}{'hard@1e-1':>11}")
for base in BASES:
    for arm, _ in COHORTS:
        for label, _ in SETS:
            k = f"{arm} | {label}"
            if base not in rows.get(k, {}):
                continue
            a = rows[k][base]
            print(f"{base:<13}{arm:<14}{label:<5}{a['median_principal_cosine']:>9.4f}"
                  f"{a['sigma_max']:>9.3f}{a['soft_d_eff']:>12.2f}{a['Tr']:>5.0f}"
                  f"{a['soft_floor_frac']:>11.4f}{a[f'hard_{EPS[-1]}']:>11.1f}")
    print()

print("=" * 104)
print("DOES THE CONCLUSION DEPEND ON THE DEGENERATE ADAPTER?")
print("=" * 104)
print(f"{'base':<13}{'cos shared/indep T=4':>24}{'cos shared/indep T=3':>24}")
for base in BASES:
    try:
        s4 = rows["shared | T=4"][base]["median_principal_cosine"]
        i4 = rows["independent | T=4"][base]["median_principal_cosine"]
        s3 = rows["shared | T=3"][base]["median_principal_cosine"]
        i3 = rows["independent | T=3"][base]["median_principal_cosine"]
    except KeyError:
        continue
    print(f"{base:<13}{f'{s4:.4f} / {i4:.4f}':>24}{f'{s3:.4f} / {i3:.4f}':>24}")

(RES / "geometry_T4_vs_T3.json").write_text(json.dumps(rows, indent=2))
print("\nwrote", RES / "geometry_T4_vs_T3.json")
