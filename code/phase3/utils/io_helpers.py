"""JSON IO with reproducibility metadata.

Every Phase 3 result file should pass through `write_result` so that
the JSON carries enough context to reproduce: git SHA, wall-clock time,
hostname, GPU index, config hash.
"""

import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


def git_sha(repo_root: str | Path = "/home/sanjay.g/projects/rdmerge") -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(blob, digest_size=8).hexdigest()


def write_result(
    path: str | Path,
    payload: dict[str, Any],
    config: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Atomic JSON write with metadata stamp.

    Writes to <path>.tmp then renames to <path>.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "git_sha": git_sha(),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pbs_jobid": os.environ.get("PBS_JOBID", ""),
        "wallclock_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wallclock_unix": time.time(),
    }
    if config is not None:
        meta["config_hash"] = config_hash(config)
        meta["config"] = config
    if extra_meta:
        meta.update(extra_meta)
    out = {"meta": meta, **payload}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2, default=str))
    tmp.replace(path)
    return path
