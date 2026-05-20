"""Seed conventions mirroring code/synthetic/day14g_final_lock.py.

Global RNG is date-keyed (e.g. 20260517 for 2026-05-17). Per-variant
sub-RNGs are derived from `hash(label) & 0xffff` to keep bootstrap
resamples reproducible across re-runs without collisions between
sibling variants.
"""

import hashlib
import os
import random

import numpy as np


def global_rng(seed: int = 20260517) -> np.random.Generator:
    return np.random.default_rng(seed)


def subrun_rng(label: str) -> np.random.Generator:
    # stable across Python invocations (Python's built-in hash() is salted)
    h = int(hashlib.blake2b(label.encode("utf-8"), digest_size=8).hexdigest(), 16)
    return np.random.default_rng(h & 0xFFFFFFFF)


def seed_everything(seed: int = 20260517) -> None:
    """Seed numpy, python stdlib random, and torch if importable."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
