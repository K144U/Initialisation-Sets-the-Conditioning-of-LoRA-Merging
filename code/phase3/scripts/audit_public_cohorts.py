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
import struct
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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

# Cohorts are independent, and the cost is network round-trips rather than
# compute, so they are fetched concurrently. This changes no measured value
# and no ordering: results are collected back into the phase-1 order before
# anything is scored.
WORKERS = 8

_DTYPES = None


def _dtypes():
    global _DTYPES
    if _DTYPES is None:
        import torch
        _DTYPES = {"F64": torch.float64, "F32": torch.float32,
                   "F16": torch.float16, "BF16": torch.bfloat16}
    return _DTYPES


class _LazyDtypes(dict):
    def get(self, k, default=None):
        return _dtypes().get(k, default)


DTYPES = _LazyDtypes()


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


def get_json_paged(url: str, tries: int = 3):
    """Return (payload, next_url). The models API paginates with an opaque
    cursor handed back in a Link: rel="next" header, not with anything present
    in the body, so the body cannot be used to walk the list."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                body = json.load(r)
                nxt = None
                link = r.headers.get("Link")
                if link:
                    for part in link.split(","):
                        if 'rel="next"' in part:
                            nxt = part.split(">")[0].split("<")[-1].strip()
                            break
                return body, nxt
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None, None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None, None


def _range(url: str, a: int, b: int, tries: int = 3):
    """One HTTP Range read. Returns bytes, or None."""
    for k in range(tries):
        try:
            req = urllib.request.Request(
                url, headers={**UA, "Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 416):
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    return None


def fetch_lora_A(repo: str, cache: Path):
    """Fetch ONLY the lora_A tensors, via HTTP Range on the safetensors file.

    The audit measures row spaces of A and nothing else. A published adapter
    can be gigabytes, because PEFT's modules_to_save writes full copies of
    embed_tokens and lm_head alongside the LoRA factors; on one measured
    cohort lora_A was 16.78 MB of a 4230 MB file. Reading the whole file to
    use 0.4% of it is the entire cost of this audit.

    A safetensors file is 8 bytes of little-endian header length, then that
    many bytes of JSON carrying every tensor's dtype, shape and byte range,
    then the data block. So three requests suffice: length, header, and one
    span per contiguous run of wanted tensors.

    Verified byte-identical against a full download before being adopted; the
    tensors this returns are the same tensors, so no measured value changes.

    Falls back to a whole-file download for .bin adapters, which are zip
    archives and cannot be range-sliced this way.
    """
    import torch
    from safetensors.torch import load_file

    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return torch.load(cache, map_location="cpu")

    # A full file from an earlier run is as good a source as the network.
    for old in (cache.parent / "adapter_model.safetensors",
                cache.parent / "adapter_model.bin"):
        if old.exists() and old.stat().st_size > 0:
            try:
                sd = (load_file(str(old)) if old.suffix == ".safetensors"
                      else torch.load(old, map_location="cpu"))
                A = {k: v for k, v in sd.items() if "lora_A" in k}
                if A:
                    torch.save(A, cache)
                    return A
            except Exception:
                pass

    url = RESOLVE.format(repo=repo, fname="adapter_model.safetensors")
    head = _range(url, 0, 7)
    if head is None or len(head) != 8:
        dest = cache.parent / "adapter_model.bin"
        if get_bytes(RESOLVE.format(repo=repo, fname="adapter_model.bin"), dest):
            try:
                sd = torch.load(dest, map_location="cpu")
                A = {k: v for k, v in sd.items() if "lora_A" in k}
                torch.save(A, cache)
                return A
            except Exception:
                return None
        return None

    n = struct.unpack("<Q", head)[0]
    if n <= 0 or n > (1 << 28):
        return None
    hdr_raw = _range(url, 8, 8 + n - 1)
    if hdr_raw is None:
        return None
    try:
        hdr = json.loads(hdr_raw)
    except Exception:
        return None

    want = {k: v for k, v in hdr.items()
            if "lora_A" in k and isinstance(v, dict) and "data_offsets" in v}
    if not want:
        return None

    base = 8 + n
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1], k)
                   for k, v in want.items())
    # Merge runs separated by less than 4 MB: fewer requests, and the wasted
    # bytes stay far below the whole-file cost either way.
    runs, cur = [], [spans[0][0], spans[0][1]]
    for s, e, _ in spans[1:]:
        if s - cur[1] < (4 << 20):
            cur[1] = max(cur[1], e)
        else:
            runs.append(tuple(cur))
            cur = [s, e]
    runs.append(tuple(cur))

    blobs = []
    for s, e in runs:
        b = _range(url, base + s, base + e - 1)
        if b is None:
            return None
        blobs.append((s, e, b))

    out = {}
    for k, v in want.items():
        s, e = v["data_offsets"]
        dt = DTYPES.get(v["dtype"])
        if dt is None:
            continue
        for rs, re_, blob in blobs:
            if rs <= s and e <= re_:
                raw = blob[s - rs: e - rs]
                out[k] = torch.frombuffer(
                    bytearray(raw), dtype=dt).reshape(v["shape"])
                break
    if not out:
        return None
    torch.save(out, cache)
    return out


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

    url = f"{API}?filter=peft&sort=downloads&direction=-1&limit=100"
    while url and examined < MAX_EXAMINED:
        page, next_url = get_json_paged(url)
        url = next_url
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

    # Prefetch every repo's lora_A tensors concurrently. Ordering of the
    # SCORING loop below is untouched: it still walks the phase-1 list in
    # order, and only reads from this cache.
    repos_all = sorted({m["repo"] for c in cohorts for m in c["members"]})
    print(f"prefetching lora_A for {len(repos_all)} repos with "
          f"{WORKERS} workers (range reads, not whole files)")

    def _one(repo: str):
        cache = CACHE / repo.replace("/", "__") / "lora_A.pt"
        try:
            return repo, fetch_lora_A(repo, cache)
        except Exception:
            return repo, None

    fetched: dict[str, object] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, (repo, A) in enumerate(ex.map(_one, repos_all), 1):
            fetched[repo] = A
            if i % 10 == 0 or i == len(repos_all):
                got = sum(1 for v in fetched.values() if v)
                print(f"  prefetched {i}/{len(repos_all)}, {got} usable")

    rows, identical = [], []
    for ci, c in enumerate(cohorts, 1):
        members = c["members"]
        mats = {}
        for m in members:
            A = fetched.get(m["repo"])
            if A:
                mats[m["repo"]] = A
        if len(mats) < MIN_ADAPTERS:
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
