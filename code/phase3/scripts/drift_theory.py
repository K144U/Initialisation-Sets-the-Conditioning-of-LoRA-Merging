#!/usr/bin/env python3
"""Test the first-order drift theory of the collapse.

Rules: notes/prereg_drift_theory_2026-09-23.md, committed before this file.
Bands and thresholds are hard-coded from that document, not taken as flags.

Exact quantities use the paper's definitions (floor_conditioning.py):
Hbar = mean of row-space projectors, G = mean of D_t = B_t A_t, tauH = G Hbar^+,
kappa = condition number of Hbar on its range, amplification = |tauH|/|G|.

First-order predictions, for a reference subspace W0 (d x r):
  X_t   = P_perp A_t^T (W0^T A_t^T)^{-1}            tilt of V_t off W0
  Y     = [X_t - Xbar]_t                            centred tilts, d x Tr
  kappa_hat = 1 / lambda_min^+(Y Y^T / T)
  M_t   = D_t (W0 + X_t),  b = (M_t - Mbar)_t Y^+,  a = Mbar - b Xbar
  alpha_hat = |a W0^T + b| / |G|

Usage (on the cluster, CPU only):
  PYTHONNOUSERSITE=1 python -u code/phase3/scripts/drift_theory.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path(__file__).resolve().parents[3]))
LORA = ROOT / "artifacts" / "lora"
OUT = ROOT / "results" / "phase3" / "drift_theory.json"

TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
DRIFT_BASE = "llama31_8b"
# 25/50/75/100% of training. Step numbers differ by task (gsm8k 117..468, the
# others 118..470), so checkpoints are taken by position, not by step.
CHECKPOINTS = [0, 1, 2, 3]
DIAL = [0.05, 0.1, 0.2, 0.5, 1.0]
R = 16
N_LAYERS = 8

BAND = (0.75, 1.33)                          # PT1, PT2, PT4
KAPPA_SLOPE = (-2.2, -1.8)                   # PT3
AMP_SLOPE = (-1.2, -0.8)                     # PT3
RIVAL_AMP_SLOPE = (-2.2, -1.8)               # PT3, the review's form
PT5_TOL = 0.05

torch.set_default_dtype(torch.float64)


def norm_key(k: str) -> str:
    return k.replace(".default.weight", ".weight")


def load_ab(path: Path) -> tuple[dict, dict]:
    sd = {norm_key(k): v.double() for k, v in load_file(str(path)).items()}
    A = {k: v for k, v in sd.items() if "lora_A" in k}
    B = {k.replace("lora_B", "lora_A"): v for k, v in sd.items() if "lora_B" in k}
    return A, B


def layer_keys(A: dict) -> list[str]:
    keys = sorted(A)
    step = max(1, len(keys) // N_LAYERS)
    return keys[::step][:N_LAYERS]


def orth_rows(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.qr(A.T)[0][:, :A.shape[0]]


def exact(As: list, Ds: list) -> dict:
    """The paper's kappa(Hbar) and amplification, as floor_conditioning.py."""
    T = len(As)
    V = [orth_rows(A) for A in As]
    M = torch.cat(V, dim=1)
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    q = int((S > 1e-10 * S[0]).sum())
    Q = U[:, :q]
    P = [(Q.T @ v) @ (Q.T @ v).T for v in V]
    Hbar = sum(P) / T
    G = sum(D @ Q for D in Ds) / T
    ev = torch.linalg.eigvalsh(Hbar).clamp(min=0)
    tauH = G @ torch.linalg.pinv(Hbar, rtol=1e-10)
    return {"q": q, "kappa": float(ev[-1] / ev[0]),
            "amp": float(tauH.norm() / G.norm())}


def predicted(As: list, Ds: list, W0: torch.Tensor) -> dict:
    T, r = len(As), As[0].shape[0]
    Pperp = torch.eye(W0.shape[0]) - W0 @ W0.T
    X = [Pperp @ A.T @ torch.linalg.inv(W0.T @ A.T) for A in As]
    Xbar = sum(X) / T
    Y = torch.cat([x - Xbar for x in X], dim=1)
    sv = torch.linalg.svdvals(Y)
    k = (T - 1) * r
    lam_min = float(sv[k - 1] ** 2 / T)
    Ms = [D @ (W0 + x) for D, x in zip(Ds, X)]
    Mbar = sum(Ms) / T
    dM = torch.cat([m - Mbar for m in Ms], dim=1)
    # pinv of Y restricted to its (T-1)r nonzero directions
    Uy, Sy, Vy = torch.linalg.svd(Y, full_matrices=False)
    Ypinv = Vy[:k].T @ torch.diag(1.0 / Sy[:k]) @ Uy[:, :k].T
    b = dM @ Ypinv
    a = Mbar - b @ Xbar
    w = a @ W0.T + b
    G = sum(Ds) / T
    return {"kappa_hat": 1.0 / lam_min, "amp_hat": float(w.norm() / G.norm()),
            "lam_min_sigmaX": lam_min,
            "eps_tilt_median": float(torch.tensor(
                [torch.linalg.matrix_norm(x, ord=2) for x in X]).median())}


def consensus_W0(As: list, r: int) -> torch.Tensor:
    V = torch.cat([orth_rows(A) for A in As], dim=1)
    U, _, _ = torch.linalg.svd(V, full_matrices=False)
    return U[:, :r]


def med(xs):
    return float(torch.tensor(xs).median())


def cohort_layers(paths: list[Path], a0_paths: list[Path] | None = None):
    loaded = [load_ab(p) for p in paths]
    a0 = [load_ab(p)[0] for p in a0_paths] if a0_paths else None
    for key in layer_keys(loaded[0][0]):
        As = [A[key] for A, _ in loaded]
        Bs = [B[key] for _, B in loaded]
        A0s = [x[key] for x in a0] if a0 else None
        yield key, As, Bs, A0s


def in_band(x, band):
    return band[0] <= x <= band[1]


def slope(xs, ys):
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return (sum((a - mx) * (b - my) for a, b in zip(lx, ly))
            / sum((a - mx) ** 2 for a in lx))


def main() -> int:
    report: dict = {"rules": "notes/prereg_drift_theory_2026-09-23.md"}
    task_dirs = [LORA / DRIFT_BASE / t / "drift_shared" for t in TASKS]
    a0_paths = [d / "adapter_A0.safetensors" for d in task_dirs]
    for p in a0_paths:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    # PT1, PT2, PT5: formula against exact, at each real checkpoint
    per_ckpt = {}
    def ckpt_dirs(d: Path) -> list[Path]:
        cks = sorted((d / "trainer_tmp").glob("checkpoint-*"),
                     key=lambda p: int(p.name.split("-")[1]))
        if len(cks) != 4:
            raise SystemExit(f"{d}: {len(cks)} checkpoints, need 4")
        return cks

    for c in CHECKPOINTS:
        paths = [ckpt_dirs(d)[c] / "adapter_model.safetensors" for d in task_dirs]
        rk, ra, ek, ea, eps = [], [], [], [], []
        for key, As, Bs, A0s in cohort_layers(paths, a0_paths):
            Ds = [B @ A for A, B in zip(As, Bs)]
            ex = exact(As, Ds)
            W0 = orth_rows(A0s[0])
            for A0 in A0s[1:]:
                if (A0 - A0s[0]).abs().max() > 0:
                    raise SystemExit("drift_shared A0 differ across tasks")
            pr = predicted(As, Ds, W0)
            rk.append(pr["kappa_hat"] / ex["kappa"])
            ra.append(pr["amp_hat"] / ex["amp"])
            ek.append(ex["kappa"]); ea.append(ex["amp"]); eps.append(pr["eps_tilt_median"])
        per_ckpt[c] = {"kappa_ratio": med(rk), "amp_ratio": med(ra),
                       "kappa_exact": med(ek), "amp_exact": med(ea),
                       "tilt_norm": med(eps)}
        print(f"ckpt {c}: kappa {per_ckpt[c]['kappa_exact']:.3e} ratio "
              f"{per_ckpt[c]['kappa_ratio']:.3f} | amp {per_ckpt[c]['amp_exact']:.2f} "
              f"ratio {per_ckpt[c]['amp_ratio']:.3f} | tilt {per_ckpt[c]['tilt_norm']:.4f}",
              flush=True)
    report["checkpoints"] = per_ckpt
    pt1 = sum(in_band(v["kappa_ratio"], BAND) for v in per_ckpt.values()) >= 3
    pt2 = sum(in_band(v["amp_ratio"], BAND) for v in per_ckpt.values()) >= 3
    ks = [per_ckpt[c]["kappa_exact"] for c in CHECKPOINTS]
    am = [per_ckpt[c]["amp_exact"] for c in CHECKPOINTS]
    pt5 = all(ks[i + 1] <= ks[i] * (1 + PT5_TOL) for i in range(3)) and \
          all(am[i + 1] <= am[i] * (1 + PT5_TOL) for i in range(3))

    # PT3: dial on the final adapters, exact quantities only
    final = [d / "adapter_model.safetensors" for d in task_dirs]
    dial = {}
    for s in DIAL:
        kk, aa = [], []
        for key, As, Bs, A0s in cohort_layers(final, a0_paths):
            As_s = [A0 + s * (A - A0) for A, A0 in zip(As, A0s)]
            Ds = [B @ A for A, B in zip(As_s, Bs)]
            ex = exact(As_s, Ds)
            kk.append(ex["kappa"]); aa.append(ex["amp"])
        dial[s] = {"kappa": med(kk), "amp": med(aa)}
        print(f"dial s={s}: kappa {dial[s]['kappa']:.3e} amp {dial[s]['amp']:.2f}",
              flush=True)
    k_slope = slope(DIAL, [dial[s]["kappa"] for s in DIAL])
    a_slope = slope(DIAL, [dial[s]["amp"] for s in DIAL])
    report["dial"] = {"points": dial, "kappa_slope": k_slope, "amp_slope": a_slope}
    pt3_k = in_band(k_slope, KAPPA_SLOPE)
    pt3_a = in_band(a_slope, AMP_SLOPE)
    rival = in_band(a_slope, RIVAL_AMP_SLOPE)

    # PT4: reference-free, all bases, shared cohorts
    pt4 = {}
    for base in BASES:
        for cohort in ["seed1", "seed2", "seed3"]:
            paths = [LORA / base / t / cohort / "adapter_model.safetensors" for t in TASKS]
            if any(not p.exists() for p in paths):
                pt4[f"{base}/{cohort}"] = None
                print(f"PT4 {base}/{cohort}: MISSING", flush=True)
                continue
            rk, ra = [], []
            for key, As, Bs, _ in cohort_layers(paths):
                Ds = [B @ A for A, B in zip(As, Bs)]
                ex = exact(As, Ds)
                pr = predicted(As, Ds, consensus_W0(As, R))
                rk.append(pr["kappa_hat"] / ex["kappa"])
                ra.append(pr["amp_hat"] / ex["amp"])
            pt4[f"{base}/{cohort}"] = {"kappa_ratio": med(rk), "amp_ratio": med(ra)}
            print(f"PT4 {base}/{cohort}: kappa ratio {med(rk):.3f} amp ratio {med(ra):.3f}",
                  flush=True)
    report["pt4"] = pt4
    scored = [v for v in pt4.values() if v is not None]
    pt4_k = sum(in_band(v["kappa_ratio"], BAND) for v in scored)
    pt4_a = sum(in_band(v["amp_ratio"], BAND) for v in scored)
    pt4_ok = len(scored) == 12 and pt4_k >= 10 and pt4_a >= 10

    if pt1 and pt2 and pt3_k and pt3_a and pt4_ok:
        branch = 1
    elif rival:
        branch = 2
    elif pt3_k and pt3_a:
        branch = 3
    elif not pt3_k and not pt3_a:
        branch = 4
    else:
        branch = "none of the four (one PT3 slope in band, one not): report as written"
    report["verdict"] = {"PT1": pt1, "PT2": pt2, "PT3_kappa": pt3_k, "PT3_amp": pt3_a,
                         "PT3_rival_amp": rival, "PT4_kappa_in_band": pt4_k,
                         "PT4_amp_in_band": pt4_a, "PT4_scored": len(scored),
                         "PT4": pt4_ok, "PT5": pt5, "branch": branch}
    print(json.dumps(report["verdict"], indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
