#!/usr/bin/env python3
"""Resubmit campaign jobs that die at walltime before their cells are done.

Runs detached on the login node of the student account. Every 10 minutes, for
each registered job spec: if no job with that name is queued or running and
the shard still has cells whose done-file is missing, qsub it again. The
orchestrator skips finished cells, so a resubmission only does what is left.
Each spec is resubmitted at most MAX_RESUB times, so a cell that fails
deterministically cannot loop forever; it stays parked for inspection.

Specs live in logs/keeper_jobs.json so jobs can be added without a restart:
  [{"name": "rdm_tost0", "manifest": "...", "state": "tost_0.json",
    "shard": "0/2", "sentinel": "_TOST_DONE_0"}, ...]
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(os.environ.get("RDMERGE_ROOT", Path.home() / "projects/rdmerge_v2"))
SPECS = ROOT / "logs" / "keeper_jobs.json"
LOG = ROOT / "logs" / "keeper.log"
MAX_RESUB = 5
PERIOD_S = 600


def log(msg: str) -> None:
    with open(LOG, "a") as f:
        f.write(time.strftime("%F %T ") + msg + "\n")


def active_names() -> set[str]:
    """Job names as `qstat -u` prints them (truncated to 10 characters, so
    every spec name must be 10 characters or fewer)."""
    names = set()
    for line in subprocess.run(["qstat", "-u", os.environ.get("USER", "")],
                               capture_output=True, text=True).stdout.splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[9] in ("Q", "R", "H", "E"):
            names.add(parts[3])
    return names


def remaining(spec: dict) -> int:
    cells = json.loads((ROOT / spec["manifest"]).read_text())
    if spec.get("shard"):
        i, n = (int(x) for x in spec["shard"].split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]
    return sum(not (ROOT / c["done"]).exists() for c in cells)


def submit(spec: dict) -> None:
    v = (f"MANIFEST={spec['manifest']},ORCH_STATE={spec['state']},"
         f"ORCH_SENTINEL={spec['sentinel']}")
    if spec.get("shard"):
        v += f",ORCH_SHARD={spec['shard']}"
    extra = []
    if spec.get("ngpus", 1) > 1:
        v += f",ORCH_WORKERS={spec['ngpus']}"
        extra = ["-l", f"select=1:ncpus=3:ngpus={spec['ngpus']}:mem=60gb"]
    r = subprocess.run(["qsub", "-N", spec["name"], "-v", v, *extra, "-o",
                        str(ROOT / "logs" / "pbs" / f"{spec['name']}.out"),
                        "code/phase3/scripts/pbs_v2.sh"],
                       cwd=ROOT, capture_output=True, text=True)
    log(f"resubmit {spec['name']}: {r.stdout.strip()} {r.stderr.strip()}")


def main() -> None:
    counts: dict[str, int] = {}
    log("keeper start")
    while True:
        try:
            specs = json.loads(SPECS.read_text())
            active = active_names()
            for s in specs:
                left = remaining(s)
                if left == 0 or s["name"][:10] in active:
                    continue
                if counts.get(s["name"], 0) >= MAX_RESUB:
                    continue
                counts[s["name"]] = counts.get(s["name"], 0) + 1
                log(f"{s['name']}: {left} cells left, not queued; attempt {counts[s['name']]}")
                submit(s)
        except Exception as e:  # never die on a transient qstat or JSON error
            log(f"error {type(e).__name__}: {e}")
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    main()
