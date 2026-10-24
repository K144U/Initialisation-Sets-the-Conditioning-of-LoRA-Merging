#!/usr/bin/env python3
"""Solve precision check: notes/check_solve_precision_2026-10-24.md (professor B1).

float32: merge_regmean (ridge 1e-6) and merge_rd_encoder (bits 32, ridge 0)
called unchanged on CPU, the pre-truncation update captured where each method
hands it to svd_truncate_to_rank. float64: the same formulas from the same
stored factors, with exact factor-based SVDs. Per module: relative difference
before and after rank-r truncation, amplification ||W|| / ||task arithmetic||
in both precisions, and the encoder's kept dimension.

  python check_solve_precision.py --selftest
  python check_solve_precision.py <out.json> [base ...]
"""
from __future__ import annotations

import contextlib
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code" / "phase3"))
import merging.rd_encoder as rde  # noqa: E402
import merging.regmean as rm  # noqa: E402
from merging._adapter_utils import svd_truncate_to_rank  # noqa: E402
from merging.tests._fake_model import FakePeftModel, make_random_fake_model  # noqa: E402

BASES = ["llama31_8b", "mistral_7b", "qwen25_7b", "yi15_9b"]
ARMS = ["seed1", "indep1"]
OLD = "/home/sanjay.g/projects/rdmerge/"
RM_LAMBDA = 1e-6
ENC_FLOOR = 1e-6            # rd_encoder's eig_rel_floor default


class View(FakePeftModel):
    """FakePeftModel plus the get_layer hook, so regmean takes its A_t^T A_t branch."""

    def get_layer(self, adapter, layer):
        return self.adapters[adapter][layer]


def specs(base, arm):
    cfg = json.loads((ROOT / "results/phase3/eval_ridge_cond" / f"{base}__rd_l0__{arm}.json")
                     .read_text())["meta"]["config"]
    return [(s["name"], ROOT / s["dir"].replace(OLD, "")) for s in cfg["adapter_specs"]]


def load_view(sp):
    from merging.tests._fake_model import FakeLoraLayer
    v = View()
    for name, d in sp:
        acfg = json.loads((d / "adapter_config.json").read_text())
        assert not acfg.get("use_rslora") and not acfg.get("use_dora"), d
        sc = acfg["lora_alpha"] / acfg["r"]
        sd = load_file(str(d / "adapter_model.safetensors"))
        layers = {}
        for k in sorted(sd):
            if k.endswith(".lora_A.weight"):
                L = k[: -len(".lora_A.weight")]
                A, B = sd[k].float(), sd[L + ".lora_B.weight"].float()
                layers[L] = FakeLoraLayer(A=A, B=B, scaling=sc)
                v.layer_topology.setdefault(L, (B.shape[0], A.shape[1], A.shape[0]))
        v.add_adapter(name, layers)
    return v


def sub(v, L):
    """A one-layer view, so a merge's full-size updates never pile up in memory."""
    return View(layer_topology={L: v.layer_topology[L]},
                adapters={n: {L: layers[L]} for n, layers in v.adapters.items()})


def production(v, names, fn, **kw):
    """Run a production merge, returning {layer: pre-truncation W} (float32)."""
    captured = []

    def grab(delta, rank):
        captured.append(delta.detach().clone())
        out_dim, in_dim = delta.shape
        return torch.zeros(rank, in_dim), torch.zeros(out_dim, rank)

    mod = rm if fn is rm.merge_regmean else rde
    orig = mod.svd_truncate_to_rank
    mod.svd_truncate_to_rank = grab
    try:
        with contextlib.redirect_stdout(io.StringIO()):   # rd_encoder prints per call
            fn(v, names, [1.0 / len(names)] * len(names), "_merged", **kw)
    finally:
        mod.svd_truncate_to_rank = orig
        v.delete_adapter("_merged")
    return dict(zip(v.layer_names(), captured))


def factors64(v, n, L):
    f = v.adapters[n][L]
    return f.A.double(), f.scaling * f.B.double()


def regmean64(v, names, lam):
    w = 1.0 / len(names)
    out = {}
    for L in v.layer_names():
        AB = [factors64(v, n, L) for n in names]
        in_dim = AB[0][0].shape[1]
        den = lam * torch.eye(in_dim, dtype=torch.float64)
        num = 0.0
        for A, B in AB:
            den += w * (A.T @ A)
            num = num + w * (B @ (A @ A.T) @ A)       # = w * Delta_t @ (A^T A)
        out[L] = torch.linalg.solve(den, num.T).T
    return out


def right_basis64(A, B):
    """Exact right-singular basis of B @ A (rank r) from the factors."""
    Qa, Ra = torch.linalg.qr(A.T)
    Qb, Rb = torch.linalg.qr(B)
    _, _, Vh = torch.linalg.svd(Rb @ Ra.T)
    return Qa @ Vh.T


def encoder64(v, names, lam=0.0):
    w = 1.0 / len(names)
    out, deff = {}, {}
    for L in v.layer_names():
        AB = [factors64(v, n, L) for n in names]
        M = torch.cat([w ** 0.5 * right_basis64(A, B) for A, B in AB], dim=1)
        S, U = torch.linalg.eigh(M.T @ M)
        keep = S > ENC_FLOOR * S.max()
        S, U = S[keep], U[:, keep]
        Q = M @ U @ torch.diag(S.rsqrt())
        N = sum(w * (B @ A) for A, B in AB)
        out[L] = (N @ Q) @ torch.diag(1.0 / (S + lam)) @ Q.T
        deff[L] = int(S.numel())
    return out, deff


def deff32(v, names):
    """The kept dimension production's float32 encoder used, recomputed."""
    w = 1.0 / len(names)
    d = {}
    for L in v.layer_names():
        M = torch.cat([w ** 0.5 * rde._right_basis(v.get_delta(n, L).float(), v.layer_topology[L][2])
                       for n in names], dim=1)
        S = torch.linalg.eigvalsh(M.T @ M)
        d[L] = int((S > ENC_FLOOR * S.max()).sum())
    return d


def trunc_prod(W, r):
    """Production's rank-r realisation (randomised SVD, q = r, 4 power iterations)."""
    torch.manual_seed(0)
    A, B = svd_truncate_to_rank(W, r)
    return B.double() @ A.double()


def trunc_exact(W, r):
    """Exact rank-r truncation in float64. Oversampled to q = r + 64, which covers the
    whole rank-64 range of a four-adapter merge; checked per case against the full SVD
    ('trunc_check')."""
    torch.manual_seed(0)
    U, S, V = torch.svd_lowrank(W.double(), q=min(r + 64, *W.shape), niter=8)
    return (U[:, :r] * S[:r]) @ V[:, :r].T


def trunc_full(W, r):
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def rel(a, b):
    return float((a - b).norm() / b.norm())


def compare(v, names, W32, W64, check=False):
    """rel_trunc and amp* use exact truncations (the decision rule); prod_gap and
    amp_prod describe production's randomised truncation of the float32 W."""
    ta = {L: sum(factors64(v, n, L)[1] @ factors64(v, n, L)[0] for n in names) / len(names)
          for L in v.layer_names()}
    rows = {}
    for L in v.layer_names():
        r = v.layer_topology[L][2]
        e32, e64, p32 = trunc_exact(W32[L], r), trunc_exact(W64[L], r), trunc_prod(W32[L], r)
        n_ta = float(ta[L].norm())
        rows[L] = {"rel_pre": rel(W32[L].double(), W64[L]), "rel_trunc": rel(e32, e64),
                   "amp32": float(e32.norm()) / n_ta, "amp64": float(e64.norm()) / n_ta,
                   "prod_gap": rel(p32, e32), "amp_prod": float(p32.norm()) / n_ta}
        if check:
            rows[L]["trunc_check"] = max(rel(e32, trunc_full(W32[L], r)),
                                         rel(e64, trunc_full(W64[L], r)))
    return rows


def summarise(rows, extra=None):
    out = {}
    for k in ("rel_pre", "rel_trunc", "amp32", "amp64", "prod_gap", "amp_prod", "trunc_check"):
        xs = [r[k] for r in rows.values() if k in r]
        if xs:
            out[k] = {"median": st.median(xs), "max": max(xs)}
    if extra:
        out.update(extra)
    return out


def selftest() -> int:
    """Production float32 vs this file's float64 on a well-conditioned random cohort."""
    spec = {f"l{i}": (48, 64, 4) for i in range(3)}
    names = ["a", "b", "c", "d"]
    fake = make_random_fake_model(spec, names, scale=0.3)
    v = View(layer_topology=fake.layer_topology, adapters=fake.adapters)
    ok = True
    for label, fn, kw, ref in (
            ("regmean", rm.merge_regmean, {"ridge_lambda": 0.1}, lambda: regmean64(v, names, 0.1)),
            ("encoder", rde.merge_rd_encoder, {"bits": 32, "ridge_lambda": 0.0, "realize": "rank_r"},
             lambda: encoder64(v, names)[0])):
        W32, W64 = production(v, names, fn, **kw), ref()
        err = max(rel(W32[L].double(), W64[L]) for L in v.layer_names())
        print(f"selftest {label}: max relative difference {err:.2e}")
        ok &= err < 1e-4
    print("selftest", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    if sys.argv[1:] == ["--selftest"]:
        return selftest()
    out_path = Path(sys.argv[1])
    bases = sys.argv[2:] or BASES
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    for base in bases:
        for arm in ARMS:
            key = f"{base}/{arm}"
            if key in res:
                continue
            t0 = time.time()
            sp = specs(base, arm)
            names = [n for n, _ in sp]
            v = load_view(sp)
            rows_rm, rows_en, d32, d64 = {}, {}, {}, {}
            for i, L in enumerate(v.layer_names()):
                s, check = sub(v, L), i < 4                # full-SVD check on the first layer
                rows_rm.update(compare(s, names,
                                       production(s, names, rm.merge_regmean, ridge_lambda=RM_LAMBDA),
                                       regmean64(s, names, RM_LAMBDA), check))
                W64, d = encoder64(s, names)
                rows_en.update(compare(s, names,
                                       production(s, names, rde.merge_rd_encoder, bits=32,
                                                  ridge_lambda=0.0, realize="rank_r"), W64, check))
                d64.update(d)
                d32.update(deff32(s, names))
            res[key] = {
                "n_modules": len(v.layer_names()),
                "regmean_1e-6": summarise(rows_rm),
                "encoder_l0": summarise(rows_en,
                                        {"deff32_median": st.median(d32.values()),
                                         "deff64_median": st.median(d64.values()),
                                         "deff_differs": sum(d32[L] != d64[L] for L in d64)}),
            }
            out_path.write_text(json.dumps(res, indent=1))
            s = res[key]
            print(f"{key:18s} {s['n_modules']} modules, {time.time() - t0:.0f} s", flush=True)
            for m in ("regmean_1e-6", "encoder_l0"):
                x = s[m]
                print(f"   {m:13s} rel_trunc median {x['rel_trunc']['median']:.2e} max {x['rel_trunc']['max']:.2e}"
                      f"  rel_pre median {x['rel_pre']['median']:.2e}"
                      f"  amp32 {x['amp32']['median']:.3f} amp64 {x['amp64']['median']:.3f}"
                      f"  prod_gap median {x['prod_gap']['median']:.2e}"
                      f"  trunc_check max {x['trunc_check']['max']:.1e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
