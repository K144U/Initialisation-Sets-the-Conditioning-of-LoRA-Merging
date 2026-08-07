"""W1: the EXACT per-instance floor from Lemma 1, not the P*-averaged surrogate.

The paper reports the floor as B^2 (1 - d_eff/(Tr)), which is Lemma 2's
expectation under the synthetic prior P*. The reviewer is right that this is a
minimax/prior-averaged quantity being applied to specific real cohorts, and
that the exact instance floor is computable from data we already hold:

    floor = (1/T) sum_t (tau_t - tauH)^T H_t (tau_t - tauH)

with Hbar = (1/T) sum_t H_t and tauH = Hbar^+ (1/T) sum_t H_t tau_t.

Conventions are taken verbatim from measure_subspace_geometry.py so the result
is comparable to the published d_eff:
  V_t = orthonormal basis of rowspace(A_t), shape (k, r)
  H_t = right-projection onto V_t, i.e. D_t(W) = ||(W - Delta_t) V_t V_t^T||_F^2
  B^2 = max_t ||Delta_t||_F^2

Everything lives in span([V_1 ... V_T]), of dimension q <= Tr = 64, so we work
in an orthonormal basis Q of that span and all operators are q x q.

Because rowspace(Delta_t) is contained in V_t, we have Delta_t P_t = Delta_t,
so the numerator average is just the plain mean of the deltas. That identity is
asserted numerically rather than assumed.

We also report, in the SAME weight-space metric, the worst-task distortion
actually achieved by the task-arithmetic merge. The ratio floor/D_max is the
number the audit claims to provide and never did: the fraction of achieved
distortion that is irreducible.

Hbar is near-singular under shared initialisation, so the pseudoinverse
tolerance matters; the floor is reported at several tolerances rather than at
one convenient value.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
ART = ROOT / "artifacts/lora"
RES = ROOT / "results/phase3"

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
TASK_DIRS = ["gsm8k", "alpaca", "magicoder", "flores"]
RTOLS = [1e-6, 1e-4, 1e-2, 1e-1]


def analyse_layer(A, D, r):
    tasks = list(A)
    T = len(tasks)
    Tr = T * r

    V = {t: torch.linalg.qr(A[t].T)[0][:, :r] for t in tasks}
    M = torch.cat([V[t] for t in tasks], dim=1)          # (k, Tr)

    # orthonormal basis of the union subspace
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    q = int((S > 1e-10 * S[0]).sum())
    Q = U[:, :q]                                          # (k, q)

    C = {t: Q.T @ V[t] for t in tasks}                    # (q, r)
    P = {t: C[t] @ C[t].T for t in tasks}                 # (q, q)
    Dt = {t: D[t] @ Q for t in tasks}                     # (m, q)

    # rowspace(Delta_t) subset V_t  =>  Delta_t P_t = Delta_t. Assert it.
    for t in tasks:
        lhs, rhs = Dt[t] @ P[t], Dt[t]
        assert torch.allclose(lhs, rhs, atol=1e-3 * float(rhs.abs().max()) + 1e-6), \
            f"Delta_t P_t != Delta_t for {t}"

    Hbar = sum(P[t] for t in tasks) / T                    # (q, q)
    G = sum(Dt[t] for t in tasks) / T                      # (m, q)

    B2 = max(float(D[t].norm()) ** 2 for t in tasks)

    # task-arithmetic merge in the same metric (plain mean of deltas)
    ta_max = max(float(((G - Dt[t]) @ P[t]).norm()) ** 2 for t in tasks)

    out = {"B2": B2, "q": q, "Tr": Tr, "ta_worst": ta_max}
    for rt in RTOLS:
        Hp = torch.linalg.pinv(Hbar, rtol=rt)
        tauH = G @ Hp                                      # (m, q)
        fl = sum(float(((Dt[t] - tauH) @ P[t]).norm()) ** 2 for t in tasks) / T
        cen_max = max(float(((tauH - Dt[t]) @ P[t]).norm()) ** 2 for t in tasks)
        out[f"floor_{rt}"] = fl
        out[f"centroid_worst_{rt}"] = cen_max
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+",
                    default=["seed1", "indep1", "indep2", "indep3"])
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    allout = {}
    for cohort in args.cohorts:
        allout[cohort] = {}
        print("=" * 100)
        print(f"cohort = {cohort}")
        print("=" * 100)
        print(f"{'base':<13}{'B^2':>10}{'exactFloor':>12}{'/B^2':>9}"
              f"{'surrogate':>11}{'TAworst':>10}{'floor/TA':>10}")
        for base in BASES:
            paths = {t: ART / base / t / cohort / "adapter_model.safetensors"
                     for t in TASK_DIRS}
            if any(not p.exists() for p in paths.values()):
                print(f"{base:<13} SKIP (missing adapters)")
                continue
            mats = {t: load_file(str(p)) for t, p in paths.items()}
            keys = sorted(k for k in mats[TASK_DIRS[0]] if k.endswith("lora_A.weight"))
            if args.layers and args.layers < len(keys):
                step = max(1, len(keys) // args.layers)
                keys = keys[::step][:args.layers]

            per = []
            for ka in keys:
                kb = ka.replace("lora_A", "lora_B")
                A = {t: mats[t][ka].float() for t in TASK_DIRS}
                D = {t: mats[t][kb].float() @ mats[t][ka].float() for t in TASK_DIRS}
                per.append(analyse_layer(A, D, args.rank))

            n = len(per)
            agg = {k: sum(l[k] for l in per) / n for k in per[0]}
            # surrogate the paper published, recomputed here for comparison
            surro = 1.0 - agg["q"] / agg["Tr"]
            f6 = agg["floor_1e-06"]
            allout[cohort][base] = dict(agg, surrogate_from_q=surro)
            print(f"{base:<13}{agg['B2']:>10.5f}{f6:>12.6f}"
                  f"{f6/agg['B2']:>9.4f}{surro:>11.4f}"
                  f"{agg['ta_worst']:>10.5f}{f6/agg['ta_worst']:>10.4f}")
        print()
        print("  floor/B^2 at each pinv tolerance:")
        for base in allout[cohort]:
            a = allout[cohort][base]
            s = "  ".join(f"{rt:.0e}:{a[f'floor_{rt}']/a['B2']:.4f}" for rt in RTOLS)
            print(f"    {base:<13}{s}")
        print()

    (RES / "exact_instance_floor.json").write_text(json.dumps(allout, indent=2))
    print("wrote", RES / "exact_instance_floor.json")


if __name__ == "__main__":
    raise SystemExit(main())
