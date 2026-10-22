#!/usr/bin/env python3
"""Re-read the 45-cohort prevalence under the maximum pairwise cosine.

prereg_heterogeneity_2026-10-19, Part M, branch "M-a holds": the audit's
statistic changes from the median to the maximum pairwise cosine (or the count
of collapsed pairs) and the 45-cohort prevalence is re-read under it.

Same cohorts, members, modules, rank floor and lora_A tensors as
audit_public_cohorts.py phase 2 (fetch_lora_A, range reads). Per cohort:
  pooled_median   the original statistic (median over all pair x module
                  values, upper median), recomputed as a reproduction check
                  against prevalence_summary.json;
  max_pair_cos    per pair, the median over shared modules (statistics.median,
                  as analyze_heterogeneity.geometry); then the maximum over
                  pairs;
  n_pairs, n_collapsed_pairs (per-pair median > 0.9).
Statuses use the registered cutoffs (COLLAPSED > 0.9, INTERMEDIATE [0.5, 0.9]).

  python reread_prevalence_maxpair.py <cohorts.json> <prevalence_summary.json> <cache dir> <out.json>
"""
from __future__ import annotations

import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import audit_public_cohorts as apc


def status(x):
    return ("COLLAPSED" if x > apc.COLLAPSED else
            "INTERMEDIATE" if x >= apc.INTERMEDIATE else "SEPARATED")


def main() -> int:
    import torch
    spec, prev, cache, out = (Path(p) for p in sys.argv[1:5])
    scored = {(c["namespace"], c["base_model"]): c
              for c in json.loads(prev.read_text())["cohorts"]
              if c.get("status") in ("COLLAPSED", "INTERMEDIATE", "SEPARATED")}
    cohorts = [c for c in json.loads(spec.read_text())["cohorts"]
               if (c["namespace"], c["base_model"]) in scored]
    repos_all = sorted({r for c in scored.values() for r in c["repos"]})
    print(f"{len(cohorts)} scored cohorts, {len(repos_all)} adapters", flush=True)

    def path(repo):
        return cache / repo.replace("/", "__") / "lora_A.pt"

    # Fetch into the cache and keep only a flag: holding every adapter's tensors
    # at once (about 18 GB) exhausted a 32 GB machine. Each cohort is loaded
    # back from the cache below and dropped when it is done.
    def one(repo):
        try:
            return repo, bool(apc.fetch_lora_A(repo, path(repo)))
        except Exception as e:                # noqa: BLE001
            print(f"  fetch failed {repo}: {e}", flush=True)
            return repo, False

    with ThreadPoolExecutor(max_workers=apc.WORKERS) as ex:
        fetched = dict(ex.map(one, repos_all))
    print(f"fetched {sum(fetched.values())}/{len(fetched)}", flush=True)

    rows = []
    for c in cohorts:
        ref = scored[(c["namespace"], c["base_model"])]
        repos = ref["repos"]                  # the members phase 2 actually scored
        mats = {rp: apc.fetch_lora_A(rp, path(rp)) for rp in repos if fetched.get(rp)}
        if len(mats) != len(repos):
            rows.append({"namespace": c["namespace"], "base_model": c["base_model"],
                         "status": "unreadable now"})
            print(f"{c['namespace']}: unreadable now", flush=True)
            continue
        keys = sorted(set.intersection(*(set(m) for m in mats.values())))
        pooled, pair_vals = [], {}
        for k in keys:
            As = {rp: mats[rp][k].float() for rp in repos}
            r = min(A.shape[0] for A in As.values())
            if r < apc.MIN_RANK:
                continue
            V = {rp: torch.linalg.qr(A.T)[0][:, :r] for rp, A in As.items()}
            for i in range(len(repos)):
                for j in range(i + 1, len(repos)):
                    v = float(torch.linalg.svdvals(V[repos[i]].T @ V[repos[j]]).clamp(max=1.0).median())
                    pooled.append(v)
                    pair_vals.setdefault((i, j), []).append(v)
        pooled.sort()
        med = pooled[len(pooled) // 2]
        pair_med = {p: st.median(v) for p, v in pair_vals.items()}
        mx = max(pair_med.values())
        row = {"namespace": c["namespace"], "base_model": c["base_model"],
               "n_adapters": len(repos), "pooled_median": med,
               "stored_median": ref["median_principal_cosine"],
               "max_pair_cos": mx, "n_pairs": len(pair_med),
               "n_collapsed_pairs": sum(v > apc.COLLAPSED for v in pair_med.values()),
               "status_median": ref["status"], "status_max_pair": status(mx)}
        rows.append(row)
        print(f"{c['namespace']:<26} n={len(repos):2d} median {med:.3f} (stored "
              f"{ref['median_principal_cosine']:.3f}) max-pair {mx:.3f} "
              f"collapsed pairs {row['n_collapsed_pairs']}/{len(pair_med)} "
              f"{ref['status']} -> {row['status_max_pair']}", flush=True)

    ok = [r for r in rows if "max_pair_cos" in r]
    repro = max(abs(r["pooled_median"] - r["stored_median"]) for r in ok)
    n_med = sum(r["status_median"] == "COLLAPSED" for r in ok)
    n_max = sum(r["status_max_pair"] == "COLLAPSED" for r in ok)
    n_any = sum(r["n_collapsed_pairs"] > 0 for r in ok)
    summ = {"rules": "notes/prereg_heterogeneity_2026-10-19.md (Part M branch)",
            "n_scored": len(ok), "max_abs_reproduction_error": repro,
            "collapsed_median": n_med, "collapsed_max_pair": n_max,
            "with_any_collapsed_pair": n_any,
            "fraction_median": n_med / len(ok), "fraction_max_pair": n_max / len(ok),
            "cohorts": rows}
    out.write_text(json.dumps(summ, indent=1))
    print(f"\n{len(ok)} cohorts; reproduction error {repro:.2e}; collapsed: median "
          f"{n_med} ({n_med / len(ok):.1%}), max pair {n_max} ({n_max / len(ok):.1%}), "
          f"any collapsed pair {n_any}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
