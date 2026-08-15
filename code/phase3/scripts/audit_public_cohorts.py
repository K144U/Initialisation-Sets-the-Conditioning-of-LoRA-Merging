#!/usr/bin/env python3
"""Does shared-A initialisation occur in public LoRA cohorts?

Rules: notes/prereg_prevalence_2026-08-15.md (84526b9), committed before any
adapter was downloaded. This script implements the registered sampling frame
and nothing else.

The frame, from the registration, is mechanical on purpose:

  * walk HuggingFace `library_name=peft` sorted by downloads descending;
  * group into cohorts by (base_model, namespace), two or more adapters;
  * take the first 50 cohorts meeting the inclusion criteria, or stop at
    2000 models examined;
  * NO cohort is skipped after inspection.

Two phases, so that every inclusion decision is made on metadata before any
geometry exists:

  phase 1  metadata only. Reads adapter_config.json per candidate. Decides
           inclusion. Writes the cohort list. Downloads no weights.
  phase 2  downloads weights for the cohorts phase 1 selected, measures, and
           reports.

Measured per cohort, over pairs of adapters and shared target modules:
  * median principal cosine between rowspace(A_i) and rowspace(A_j), the same
    quantity as the paper's Table 2;
  * direct similarity ||A_i - A_j||_F / ||A_i||_F.

Thresholds, registered: COLLAPSED > 0.9, INTERMEDIATE [0.5, 0.9], SEPARATED
below 0.5. P1 is that at least 20% of cohorts are COLLAPSED.

Usage:
  python code/phase3/scripts/audit_public_cohorts.py --phase 1
  python code/phase3/scripts/audit_public_cohorts.py --phase 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))
OUT = ROOT / "results/phase3/prevalence"
CACHE = OUT / "adapters"

API = "https://huggingface.co/api/models"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{fname}"

# Registered constants. None of these may change after a cohort is measured.
TARGET_COHORTS = 50
MAX_EXAMINED = 2000
MIN_ADAPTERS = 2
MIN_RANK = 4
COLLAPSED = 0.9
INTERMEDIATE = 0.5
P1_FRACTION = 0.20

UA = {"User-Agent": "rdmerge-prevalence-audit/1.0"}


def get_json(url: str, tries: int = 3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def get_bytes(url: str, dest: Path, tries: int = 3) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=300) as r, \
                    open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return False
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return False


# ---------------------------------------------------------------- phase 1

def phase1() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    examined = 0
    candidates: dict[tuple, list] = defaultdict(list)
    reject = defaultdict(int)
    seen_repos = set()

    print(f"walking {API} filter=peft sort=downloads desc, "
          f"stop at {TARGET_COHORTS} cohorts or {MAX_EXAMINED} models")

    cursor = None
    while examined < MAX_EXAMINED:
        url = (f"{API}?filter=peft&sort=downloads&direction=-1&limit=100"
               + (f"&cursor={cursor}" if cursor else ""))
        page = get_json(url)
        if not page:
            break
        if isinstance(page, dict):
            page = page.get("models", [])
        if not page:
            break

        for m in page:
            repo = m.get("id") or m.get("modelId")
            if not repo or repo in seen_repos:
                continue
            seen_repos.add(repo)
            examined += 1
            if examined % 100 == 0:
                print(f"  examined {examined}, cohorts so far "
                      f"{sum(1 for v in candidates.values() if len(v) >= MIN_ADAPTERS)}")

            cfg = get_json(RESOLVE.format(repo=repo, fname="adapter_config.json"))
            if not cfg:
                reject["no adapter_config.json"] += 1
                continue
            if str(cfg.get("peft_type", "")).upper() != "LORA":
                reject["not LoRA"] += 1
                continue
            base = cfg.get("base_model_name_or_path")
            r = cfg.get("r")
            tm = cfg.get("target_modules")
            if not base:
                reject["no base_model"] += 1
                continue
            if not isinstance(r, int) or r < MIN_RANK:
                reject[f"rank < {MIN_RANK}"] += 1
                continue
            if not tm:
                reject["no target_modules"] += 1
                continue
            ns = repo.split("/")[0]
            candidates[(ns, str(base))].append(
                {"repo": repo, "rank": r,
                 "target_modules": sorted(tm) if isinstance(tm, list) else [str(tm)],
                 "downloads": m.get("downloads", 0)})

        cursor = page[-1].get("_id") if isinstance(page[-1], dict) else None
        if not cursor:
            break

        done = [k for k, v in candidates.items() if len(v) >= MIN_ADAPTERS]
        if len(done) >= TARGET_COHORTS:
            break

    cohorts = []
    for (ns, base), members in candidates.items():
        if len(members) < MIN_ADAPTERS:
            reject["cohort of one"] += 1
            continue
        shared = set(members[0]["target_modules"])
        for m in members[1:]:
            shared &= set(m["target_modules"])
        if not shared:
            reject["no shared target module"] += 1
            continue
        cohorts.append({"namespace": ns, "base_model": base,
                        "members": members, "shared_modules": sorted(shared)})

    cohorts = cohorts[:TARGET_COHORTS]
    payload = {"examined": examined, "n_cohorts": len(cohorts),
               "rejected": dict(reject), "cohorts": cohorts}
    (OUT / "cohorts.json").write_text(json.dumps(payload, indent=2))

    print(f"\nexamined {examined} models")
    print(f"cohorts meeting criteria: {len(cohorts)}")
    print("rejections:")
    for k, v in sorted(reject.items(), key=lambda kv: -kv[1]):
        print(f"  {v:>6}  {k}")
    print(f"\nwrote {OUT / 'cohorts.json'}  (no weights downloaded)")
    if len(cohorts) < 10:
        print("\nNOTE: fewer than 10 cohorts. The registration calls this "
              "inconclusive on sample size and forbids relaxing a criterion.")
    return 0


# ---------------------------------------------------------------- phase 2

def phase2() -> int:
    import torch
    from safetensors.torch import load_file

    spec = json.loads((OUT / "cohorts.json").read_text())
    cohorts = spec["cohorts"]
    print(f"measuring {len(cohorts)} cohorts from the phase-1 list, in order\n")

    rows, identical = [], []
    for ci, c in enumerate(cohorts, 1):
        members = c["members"]
        mats = {}
        ok = True
        for m in members:
            dest = CACHE / m["repo"].replace("/", "__") / "adapter_model.safetensors"
            url = RESOLVE.format(repo=m["repo"], fname="adapter_model.safetensors")
            if not get_bytes(url, dest):
                url = RESOLVE.format(repo=m["repo"], fname="adapter_model.bin")
                dest = dest.with_suffix(".bin")
                if not get_bytes(url, dest):
                    ok = False
                    break
            try:
                sd = (load_file(str(dest)) if dest.suffix == ".safetensors"
                      else torch.load(dest, map_location="cpu"))
            except Exception:
                ok = False
                break
            mats[m["repo"]] = {k: v for k, v in sd.items() if "lora_A" in k}
        if not ok or len(mats) < MIN_ADAPTERS:
            print(f"[{ci}/{len(cohorts)}] {c['namespace']} :: SKIP, weights unreadable")
            rows.append({**{k: c[k] for k in ("namespace", "base_model")},
                         "status": "unreadable"})
            continue

        repos = list(mats)
        keys = set(mats[repos[0]])
        for rp in repos[1:]:
            keys &= set(mats[rp])
        keys = sorted(k for k in keys)
        if not keys:
            print(f"[{ci}/{len(cohorts)}] {c['namespace']} :: SKIP, no common key")
            rows.append({**{k: c[k] for k in ("namespace", "base_model")},
                         "status": "no common key"})
            continue

        cos_all, rel_all, allsame = [], [], True
        for k in keys:
            As = {rp: mats[rp][k].float() for rp in repos}
            r = min(A.shape[0] for A in As.values())
            if r < MIN_RANK:
                continue
            V = {rp: torch.linalg.qr(A.T)[0][:, :r] for rp, A in As.items()}
            for i in range(len(repos)):
                for j in range(i + 1, len(repos)):
                    a, b = repos[i], repos[j]
                    cos_all.append(float(torch.linalg.svdvals(
                        V[a].T @ V[b]).clamp(max=1.0).median()))
                    Aa, Ab = As[a], As[b]
                    if Aa.shape == Ab.shape:
                        rel_all.append(float((Aa - Ab).norm() / Aa.norm()))
                        if not torch.equal(Aa, Ab):
                            allsame = False
                    else:
                        allsame = False
        if not cos_all:
            rows.append({**{k: c[k] for k in ("namespace", "base_model")},
                         "status": "no measurable module"})
            continue

        cos_all.sort()
        med = cos_all[len(cos_all) // 2]
        rel = sum(rel_all) / len(rel_all) if rel_all else float("nan")
        if allsame:
            identical.append(c["namespace"])
            status = "byte-identical"
        else:
            status = ("COLLAPSED" if med > COLLAPSED else
                      "INTERMEDIATE" if med >= INTERMEDIATE else "SEPARATED")
        print(f"[{ci}/{len(cohorts)}] {c['namespace']:<28} n={len(repos)} "
              f"cos={med:.3f}  dA/A={rel:.3f}  {status}")
        rows.append({"namespace": c["namespace"], "base_model": c["base_model"],
                     "n_adapters": len(repos), "repos": repos,
                     "median_principal_cosine": med, "rel_A_diff": rel,
                     "status": status})

    scored = [r for r in rows if r.get("status") in
              ("COLLAPSED", "INTERMEDIATE", "SEPARATED")]
    n = len(scored)
    n_col = sum(1 for r in scored if r["status"] == "COLLAPSED")
    frac = n_col / n if n else 0.0

    print("\n" + "=" * 78)
    print(f"cohorts measured and scored : {n}")
    print(f"  COLLAPSED    (cos > {COLLAPSED})      : {n_col}")
    print(f"  INTERMEDIATE [{INTERMEDIATE}, {COLLAPSED}]  : "
          f"{sum(1 for r in scored if r['status'] == 'INTERMEDIATE')}")
    print(f"  SEPARATED    (cos < {INTERMEDIATE})      : "
          f"{sum(1 for r in scored if r['status'] == 'SEPARATED')}")
    print(f"  byte-identical, excluded from denominator: {len(identical)}")
    print(f"\ncollapsed fraction: {frac:.1%}  (P1 threshold {P1_FRACTION:.0%})")

    if n < 10:
        verdict = "INCONCLUSIVE ON SAMPLE SIZE"
    elif frac >= P1_FRACTION:
        verdict = "P1 HOLDS"
    elif n_col > 0:
        verdict = "SCOPED: some but under threshold"
    else:
        verdict = "NO COHORT COLLAPSED"
    print(f"VERDICT: {verdict}")

    (OUT / "prevalence_summary.json").write_text(json.dumps(
        {"verdict": verdict, "n_scored": n, "n_collapsed": n_col,
         "collapsed_fraction": frac, "byte_identical": identical,
         "thresholds": {"collapsed": COLLAPSED, "intermediate": INTERMEDIATE,
                        "p1_fraction": P1_FRACTION},
         "examined": spec["examined"], "rejected": spec["rejected"],
         "cohorts": rows}, indent=2))
    print(f"wrote {OUT / 'prevalence_summary.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    return phase1() if a.phase == 1 else phase2()


if __name__ == "__main__":
    raise SystemExit(main())
