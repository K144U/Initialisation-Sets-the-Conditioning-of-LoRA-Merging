#!/usr/bin/env python3
"""Prove the range-read fetch returns exactly what a whole-file download does.

The prevalence audit was switched from downloading each adapter in full to
reading only its lora_A tensors over HTTP Range, because a published adapter
can be gigabytes of saved embeddings around a few MB of LoRA factors. That is
a 250x cost reduction and it would be worthless if it changed a measured
value.

This compares, for every repo whose whole file is still on disk from the
earlier run, the tensors obtained each way: same keys, same shapes, same
dtypes, bitwise-equal contents. It is the same argument
check_nll_tau_determinism.py makes for the nll_tau cache, and for the same
reason: an optimisation adopted without this check is an unverified change to
the measurement.

Usage:
  python code/phase3/scripts/verify_range_fetch.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_public_cohorts import CACHE, fetch_lora_A  # noqa: E402

ROOT = Path(os.environ.get("RDMERGE_ROOT", "/home/sanjay.g/projects/rdmerge"))


def main() -> int:
    wholes = sorted(CACHE.glob("*/adapter_model.safetensors"))
    if not wholes:
        print("no whole-file downloads on disk to compare against")
        return 1

    print(f"comparing {len(wholes)} repos fetched both ways\n")
    n_ok = n_bad = 0
    tot_whole = tot_a = 0

    for w in wholes:
        repo = w.parent.name.replace("__", "/", 1)
        try:
            ref = {k: v for k, v in load_file(str(w)).items() if "lora_A" in k}
        except Exception as e:
            print(f"  {repo}: unreadable whole file ({e})")
            continue

        # Force the network path: ignore both caches for this comparison.
        tmp = w.parent / "_verify_lora_A.pt"
        if tmp.exists():
            tmp.unlink()
        saved = w.rename(w.with_suffix(".safetensors.hidden"))
        try:
            got = fetch_lora_A(repo, tmp)
        finally:
            saved.rename(w)

        if got is None:
            print(f"  {repo}: range fetch returned nothing")
            n_bad += 1
            continue

        same_keys = set(got) == set(ref)
        bad = []
        for k in sorted(set(got) & set(ref)):
            a, b = got[k], ref[k]
            if a.shape != b.shape or a.dtype != b.dtype or not torch.equal(a, b):
                bad.append(k)

        whole_mb = w.stat().st_size / 1e6
        a_mb = sum(v.numel() * v.element_size() for v in ref.values()) / 1e6
        tot_whole += whole_mb
        tot_a += a_mb

        if same_keys and not bad:
            n_ok += 1
            print(f"  OK   {repo:<52} {len(ref):>3} tensors  "
                  f"{a_mb:>7.2f} / {whole_mb:>9.2f} MB  "
                  f"{whole_mb / max(a_mb, 1e-9):>5.0f}x")
        else:
            n_bad += 1
            print(f"  FAIL {repo}: keys_match={same_keys} mismatched={bad[:3]}")
        if tmp.exists():
            tmp.unlink()

    print(f"\n{n_ok} identical, {n_bad} mismatched")
    if tot_a > 0:
        print(f"bytes needed: {tot_a:.1f} MB of {tot_whole:.1f} MB "
              f"= {tot_whole / tot_a:.0f}x less")
    if n_bad:
        print("\nDO NOT ADOPT: the fast path changes a measured value.")
        return 2
    print("\nThe two paths agree bitwise on every tensor compared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
