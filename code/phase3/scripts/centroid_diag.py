"""Diagnose the H-weighted centroid blowup (E1 v3 finding, 2026-06-12).

Hypothesis: with H_t = P_{V_t}, Hbar's small eigenvalues (directions weakly
covered by the 4 task subspaces) are amplified by Hbar^+ in the centroid
tau_H = (mean delta) Hbar^+, inflating ||W*|| far beyond the task-vector
scale and out of the quadratic surrogate's validity region.

Checks per sampled layer (llama, CPU, direct from adapter safetensors):
  - spectrum of Hbar's nonzero eigenvalues (via the 64x64 Gram)
  - ||tau_H||_F vs ||mean delta||_F (the TA merge) -> amplification ratio
Run: /cm/local/apps/python311/bin/python3 centroid_diag.py (login node OK
if torch is importable; else conda env python with PYTHONNOUSERSITE=1)
"""

import json
import math
import os

import torch
from safetensors.torch import load_file

ROOT = os.path.expanduser("~/projects/rdmerge")
TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
ADIR = ROOT + "/artifacts/lora/llama31_8b/%s/v1"
SAMPLE_LAYERS = 16  # None = all; set int N for first N


def main():
    cfg = json.load(open(ADIR % TASKS[0] + "/adapter_config.json"))
    scaling = cfg["lora_alpha"] / cfg["r"]
    r = cfg["r"]
    sds = {t: load_file(ADIR % t + "/adapter_model.safetensors")
           for t in TASKS}
    keys = sorted(k for k in sds[TASKS[0]] if k.endswith("lora_A.weight"))
    if SAMPLE_LAYERS:
        keys = keys[:SAMPLE_LAYERS]
    print("layers to scan:", len(keys))

    ratios, eig_mins, eig_meds = [], [], []
    for ka in keys:
        kb = ka.replace("lora_A", "lora_B")
        deltas, bases = [], []
        for t in TASKS:
            A = sds[t][ka].float()      # (r, in)
            B = sds[t][kb].float()      # (out, r)
            deltas.append(scaling * (B @ A))
            _, _, v = torch.svd_lowrank(deltas[-1], q=r + 8, niter=4)
            bases.append(v[:, :r])      # (in, r)
        Mw = torch.cat([V / math.sqrt(len(TASKS)) for V in bases], dim=1)
        G = Mw.T @ Mw
        S, U = torch.linalg.eigh(G)
        keep = S > 1e-6 * S.max()
        S, U = S[keep], U[:, keep]
        Q = Mw @ U @ torch.diag(S.rsqrt())
        N = sum(deltas) / len(TASKS)
        tau = (N @ Q) @ torch.diag(1.0 / S) @ Q.T
        ratio = float(tau.norm() / N.norm())
        ratios.append(ratio)
        eig_mins.append(float(S.min()))
        eig_meds.append(float(S.median()))
    tens = torch.tensor
    print("Hbar nonzero-eig: min(min)=%.2e med(min)=%.2e med(median)=%.3f" % (
        tens(eig_mins).min(), tens(eig_mins).median(), tens(eig_meds).median()))
    rt = tens(ratios)
    print("||tau_H|| / ||TA delta||: median=%.1f mean=%.1f max=%.1f min=%.1f" % (
        rt.median(), rt.mean(), rt.max(), rt.min()))
    print("layers with ratio > 3: %d/%d" % (int((rt > 3).sum()), len(ratios)))


if __name__ == "__main__":
    main()
