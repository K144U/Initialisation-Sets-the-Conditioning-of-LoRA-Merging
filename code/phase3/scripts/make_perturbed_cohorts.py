#!/usr/bin/env python3
"""Perturbed copies of independent cohorts for prereg_heterogeneity Part G.

Writes artifacts/lora_perturbed/<base>/<task>/<cohort>__<pert>/ with the
original adapter_config.json and tokenizer files and a float32
adapter_model.safetensors. Originals are only read. Perturbations (fixed in
the registration):
  gauge : gsm8k A <- 4A, B <- B/4              (update unchanged exactly)
  scale : gsm8k B <- 4B                         (update x4)
  canon : every adapter A = S^1/2 V^T, B = U S^1/2 per module (update unchanged)
  rank  : canon, then gsm8k and magicoder truncated to rank 4 (zero-padded)
  module: gsm8k k_proj and v_proj A and B zeroed (module absent)
A JSON next to each copy records the max relative change of every update.
"""
from __future__ import annotations

import json
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

from _c1019 import ROOT

LORA = ROOT / "artifacts" / "lora"
OUT = ROOT / "artifacts" / "lora_perturbed"
BASES = ["llama31_8b", "qwen25_7b"]
COHORTS = ["indep1", "indep2", "indep3"]
TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
PERTS = ["gauge", "scale", "canon", "rank", "module"]


def pairs(sd):
    for ka in sorted(k for k in sd if ".lora_A." in k):
        yield ka, ka.replace(".lora_A.", ".lora_B.")


def canon(A, B, keep=None):
    # SVD of B A through the factors: B = Qb Rb, A^T = Qa Ra, so
    # B A = Qb (Rb Ra^T) Qa^T and only an r x r SVD is needed.
    Qb, Rb = torch.linalg.qr(B.double())
    Qa, Ra = torch.linalg.qr(A.double().T)
    u, S, vh = torch.linalg.svd(Rb @ Ra.T)
    U, Vh = Qb @ u, vh @ Qa.T
    if keep is not None:
        S = S.clone()
        S[keep:] = 0
    s = S.sqrt()
    return (s[:, None] * Vh).float(), (U * s[None, :]).float()


def _fro2(A, B):
    """||B A||_F^2 from the factors: tr((A A^T)(B^T B))."""
    return float(torch.trace((A @ A.T) @ (B.T @ B)))


def rel_change(A1, B1, A0, B0):
    """||B1 A1 - B0 A0||_F / ||B0 A0||_F via [B1, -B0] [A1; A0] (small Grams)."""
    A = torch.cat([A1.double(), A0.double()], 0)
    B = torch.cat([B1.double(), -B0.double()], 1)
    den = _fro2(A0.double(), B0.double())
    return (max(_fro2(A, B), 0.0) / den) ** 0.5 if den > 0 else 0.0


def perturb(sd, task, pert):
    out = {k: v.float().clone() for k, v in sd.items()}
    for ka, kb in pairs(sd):
        A, B = out[ka], out[kb]
        if pert == "gauge" and task == "gsm8k":
            out[ka], out[kb] = 4 * A, B / 4
        elif pert == "scale" and task == "gsm8k":
            out[kb] = 4 * B
        elif pert == "canon":
            out[ka], out[kb] = canon(A, B)
        elif pert == "rank":
            out[ka], out[kb] = canon(A, B, keep=4 if task in ("gsm8k", "magicoder") else None)
        elif pert == "module" and task == "gsm8k" and (".k_proj." in ka or ".v_proj." in ka):
            out[ka], out[kb] = torch.zeros_like(A), torch.zeros_like(B)
    return out


def main() -> int:
    n = 0
    for base in BASES:
        for cohort in COHORTS:
            for task in TASKS:
                src = LORA / base / task / cohort
                sd = load_file(str(src / "adapter_model.safetensors"))
                for pert in PERTS:
                    dst = OUT / base / task / f"{cohort}__{pert}"
                    if (dst / "adapter_model.safetensors").exists():
                        continue
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in src.iterdir():
                        if f.is_file() and f.name != "adapter_model.safetensors":
                            shutil.copy(f, dst / f.name)
                    new = perturb(sd, task, pert)
                    change = max(rel_change(new[ka], new[kb], sd[ka], sd[kb])
                                 for ka, kb in pairs(sd))
                    save_file(new, str(dst / "adapter_model.safetensors"))
                    (dst / "perturbation.json").write_text(json.dumps(
                        {"source": str(src), "perturbation": pert,
                         "max_rel_update_change": change}, indent=1))
                    n += 1
    print(f"wrote {n} perturbed adapters under {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
