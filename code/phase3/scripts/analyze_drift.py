#!/usr/bin/env python3
"""Apply the registered drift rule to the checkpoint sweep.

Rules: notes/prereg_drift_2026-08-16.md (9202d3b). Committed before any adapter
of this arm is trained, so the thresholds below cannot have been chosen with
the traces in view.

Registered, and hard-coded here rather than taken as flags:

  drift(f) = median over tasks of ||A_f - A_0||_F / ||A_0||_F
  cos(f)   = median principal cosine between the T task row spaces
  P1  drift(100) < 0.5 on the shared arm
  P2  drift is monotone non-decreasing in f, to within 0.01
  P3  cos(100) > 0.9 on the shared arm

All four branches of the decision rule are printed with the verdict, so the
consequence is read off the registration rather than decided afterwards.

Usage:
  python code/phase3/scripts/analyze_drift.py
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
LORA = ROOT / "artifacts" / "lora"
OUT = ROOT / "results" / "phase3" / "drift_summary.json"

BASE = "llama31_8b"
TASKS = ["alpaca", "gsm8k", "magicoder", "flores"]
ARMS = ["drift_shared", "drift_indep"]

P1_MAX_DRIFT = 0.5
P2_TOL = 0.01
P3_MIN_COS = 0.9
N_LAYERS = 8


def _norm_key(k: str) -> str:
    """Drop the PEFT adapter name so A_0 and the saved adapter agree.

    A_0 comes from model.named_parameters() and carries the adapter name:
        ...lora_A.default.weight
    PEFT's save_pretrained does not:
        ...lora_A.weight
    Without this every lookup misses, every trace is empty, and the analyzer
    reports nothing rather than reporting a drift of zero, which is the
    failure mode worth avoiding.
    """
    return k.replace(".default.weight", ".weight")


def a_factors(path: Path) -> dict[str, torch.Tensor]:
    sd = load_file(str(path))
    out = {_norm_key(k): v.float() for k, v in sd.items() if "lora_A" in k}
    if not out:
        raise SystemExit(f"{path}: no lora_A tensors")
    return out


def checkpoints(task_dir: Path) -> list[tuple[int, Path]]:
    """(step, adapter file) for each intermediate checkpoint, in order."""
    out = []
    tmp = task_dir / "trainer_tmp"
    if tmp.exists():
        for d in tmp.glob("checkpoint-*"):
            m = re.search(r"checkpoint-(\d+)", d.name)
            f = d / "adapter_model.safetensors"
            if m and f.exists():
                out.append((int(m.group(1)), f))
    final = task_dir / "adapter_model.safetensors"
    if final.exists():
        step = max((s for s, _ in out), default=0) + 1
        out.append((step, final))
    return sorted(out)


def principal_cosines(mats: list[torch.Tensor]) -> float:
    """Median principal cosine between the row spaces, as the audit computes."""
    bases = []
    for A in mats:
        q, _ = torch.linalg.qr(A.T)
        bases.append(q)
    cos = []
    for i in range(len(bases)):
        for j in range(i + 1, len(bases)):
            s = torch.linalg.svdvals(bases[i].T @ bases[j])
            cos.append(float(s.median()))
    return float(torch.tensor(cos).median()) if cos else float("nan")


def main() -> int:
    report: dict = {"base": BASE, "arms": {}}
    missing = []

    for arm in ARMS:
        per_task_traces: dict[str, list[tuple[float, float]]] = {}
        layer_names: list[str] = []
        for task in TASKS:
            d = LORA / BASE / task / arm
            a0p = d / "adapter_A0.safetensors"
            if not a0p.exists():
                missing.append(f"{arm}/{task}: no A_0")
                continue
            A0 = a_factors(a0p)
            if not layer_names:
                layer_names = sorted(A0)[:N_LAYERS]
            cks = checkpoints(d)
            if len(cks) < 4:
                missing.append(f"{arm}/{task}: {len(cks)} checkpoints, need 4")
                continue
            trace = []
            for step, f in cks:
                Af = a_factors(f)
                rel = []
                for k in layer_names:
                    if k in Af and k in A0:
                        rel.append(float((Af[k] - A0[k]).norm() / A0[k].norm()))
                if rel:
                    trace.append((step, sum(rel) / len(rel)))
            if not trace:
                raise SystemExit(
                    f"{arm}/{task}: every checkpoint produced an empty trace, "
                    f"which means the tensor names did not match between A_0 "
                    f"and the checkpoints. Refusing to report a drift of zero "
                    f"that is really a lookup failure.")
            per_task_traces[task] = trace

        if not per_task_traces:
            continue

        n = min(len(v) for v in per_task_traces.values())
        drift = [float(torch.tensor(
            [per_task_traces[t][i][1] for t in per_task_traces]).median())
            for i in range(n)]

        # Cosines at the final checkpoint only: that is what P3 gates on.
        finals = []
        for task in per_task_traces:
            cks = checkpoints(LORA / BASE / task / arm)
            finals.append(a_factors(cks[-1][1]))
        cos_final = float(torch.tensor([
            principal_cosines([f[k] for f in finals if k in f])
            for k in layer_names
            if sum(k in f for f in finals) == len(finals)]).median())

        report["arms"][arm] = {
            "drift_trace": drift,
            "drift_final": drift[-1] if drift else None,
            "cos_final": cos_final,
            "per_task_final": {t: v[-1][1] for t, v in per_task_traces.items()},
        }
        print(f"\n{arm}")
        print(f"  drift trace   {['%.4f' % d for d in drift]}")
        print(f"  cos at 100%   {cos_final:.4f}")
        for t, v in per_task_traces.items():
            print(f"    {t:<12} {v[-1][1]:.4f}")

    if missing:
        print(f"\nINCOMPLETE: {len(missing)} runs not ready")
        for m in missing[:10]:
            print("   ", m)
        print("\nBinding constraint 1: nothing is read until all eight runs "
              "have all four checkpoints.")
        return 1

    sh = report["arms"].get("drift_shared", {})
    drift = sh.get("drift_trace") or []
    p1 = bool(drift) and drift[-1] < P1_MAX_DRIFT
    p2 = all(b >= a - P2_TOL for a, b in zip(drift, drift[1:]))
    p3 = sh.get("cos_final", 0) > P3_MIN_COS

    print(f"\nP1 drift(100) < {P1_MAX_DRIFT}:  "
          f"{'HOLDS' if p1 else 'FAILS'}  ({drift[-1]:.4f})" if drift else "")
    print(f"P2 monotone (tol {P2_TOL}):   {'HOLDS' if p2 else 'FAILS'}")
    print(f"P3 cos(100) > {P3_MIN_COS}:      "
          f"{'HOLDS' if p3 else 'FAILS'}  ({sh.get('cos_final', float('nan')):.4f})")

    if not p2:
        branch, action = 4, ("Treated as a fault, not a result. Report it, do "
                             "not use the trace, do not discharge E3.")
    elif not p3:
        branch, action = 2, ("The cohort did not reproduce the collapse. "
                             "Report drift descriptively, discharge nothing, "
                             "keep the scoping caveat.")
    elif not p1:
        branch, action = 3, ("MECHANISM FALSIFIED. A moves a long way and the "
                             "subspaces still collapse. Rewrite the mechanism "
                             "subsection to say the cause is not what we "
                             "claimed, and say so in the abstract.")
    else:
        branch, action = 1, ("E3 discharged for this base and this training "
                             "budget. C.3 becomes a measurement; drop the "
                             "scoping caveat from geometry claims.")
    print(f"\nregistered branch {branch}: {action}")

    report["verdict"] = {"P1": p1, "P2": p2, "P3": p3,
                         "branch": branch, "action": action}
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
