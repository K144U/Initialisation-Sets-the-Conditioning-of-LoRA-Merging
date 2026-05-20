"""Startup checks for Phase 3 PBS jobs.

Fail early with clear messages if the environment isn't what we expect:
- HF cache pointed at project BeeGFS path (not user home)
- CUDA_VISIBLE_DEVICES pinned by the wrapper
- torch can see exactly one GPU and bf16 is supported
"""

import os
import sys


def assert_env(min_free_gb: float = 70.0) -> dict[str, str]:
    errors: list[str] = []

    expected_hf = "/home/sanjay.g/projects/rdmerge/cache/hf_home"
    hf_home = os.environ.get("HF_HOME", "")
    if not hf_home.startswith(expected_hf):
        errors.append(f"HF_HOME={hf_home!r}; expected to start with {expected_hf!r}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        errors.append("CUDA_VISIBLE_DEVICES is empty; the PBS wrapper must pin a GPU")

    try:
        import torch
    except ImportError as e:
        errors.append(f"torch import failed: {e}")
        torch = None

    if torch is not None:
        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is False")
        elif torch.cuda.device_count() != 1:
            errors.append(
                f"torch.cuda.device_count()={torch.cuda.device_count()} "
                f"(expected 1; CUDA_VISIBLE_DEVICES must pin exactly one GPU)"
            )
        else:
            cap = torch.cuda.get_device_capability(0)
            if cap < (8, 0):
                errors.append(f"GPU compute capability {cap} < (8, 0) — bf16 unreliable")
            free, total = torch.cuda.mem_get_info(0)
            if free / (1024 ** 3) < min_free_gb:
                errors.append(
                    f"GPU 0 has only {free / 1024**3:.1f} GB free "
                    f"(need >= {min_free_gb} GB)"
                )

    if errors:
        for e in errors:
            print(f"PBS_ENV_ERROR: {e}", file=sys.stderr)
        sys.exit(87)

    return {
        "hf_home": hf_home,
        "cuda_visible_devices": cvd,
        "pbs_jobid": os.environ.get("PBS_JOBID", ""),
        "hostname": os.environ.get("HOSTNAME", ""),
    }
