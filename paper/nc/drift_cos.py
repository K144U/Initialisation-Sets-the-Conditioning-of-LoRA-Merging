#!/usr/bin/env python3
"""Principal cosine and drift along training for the drift cohorts (figure 1a).

For the Llama-3.1-8B drift cohorts (drift_shared, drift_indep), whose initial
A factors were saved before the first optimiser step, record at A_0 and at each
checkpoint: the mean relative displacement ||A - A_0|| / ||A_0|| and the median
principal cosine between the four tasks' row spaces. Same layers and same
checkpoint discovery as code/phase3/scripts/analyze_drift.py.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

TASKS = ["gsm8k", "alpaca", "magicoder", "flores"]
ARMS = ["drift_shared", "drift_indep"]
N_LAYERS = 8


def norm_key(k):
    return k.replace(".default.weight", ".weight")


def a_factors(path):
    return {norm_key(k): v.double() for k, v in load_file(str(path)).items() if "lora_A" in k}


def checkpoints(task_dir):
    out = []
    tmp = task_dir / "trainer_tmp"
    if tmp.exists():
        for d in tmp.glob("checkpoint-*"):
            m = re.search(r"checkpoint-(\d+)", d.name)
            f = d / "adapter_model.safetensors"
            if m and f.exists():
                out.append((int(m.group(1)), f))
    return sorted(out)


def median_cos(As):
    Q = [torch.linalg.qr(A.T)[0] for A in As]
    cos = []
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            cos.extend(torch.linalg.svdvals(Q[i].T @ Q[j]).clamp(max=1).tolist())
    return float(torch.tensor(cos).median())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lora = Path(a.art) / "llama31_8b"
    rep = {}
    for arm in ARMS:
        A0 = {t: a_factors(lora / t / arm / "adapter_A0.safetensors") for t in TASKS}
        names = sorted(A0[TASKS[0]])[:N_LAYERS]
        cks = {t: checkpoints(lora / t / arm) for t in TASKS}
        n = min(len(v) for v in cks.values())
        rows = [{"step": 0,
                 "drift": 0.0,
                 "cos": sum(median_cos([A0[t][k] for t in TASKS]) for k in names) / len(names)}]
        for i in range(n):
            Af = {t: a_factors(cks[t][i][1]) for t in TASKS}
            drift = [float((Af[t][k] - A0[t][k]).norm() / A0[t][k].norm())
                     for t in TASKS for k in names]
            cos = [median_cos([Af[t][k] for t in TASKS]) for k in names]
            rows.append({"step": cks[TASKS[0]][i][0],
                         "drift": sum(drift) / len(drift),
                         "cos": sum(cos) / len(cos)})
            print(arm, rows[-1], flush=True)
        rep[arm] = rows
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
