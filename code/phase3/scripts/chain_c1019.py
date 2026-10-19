#!/usr/bin/env python3
"""Queue a successor for each running 2026-10-19 job, so a 24 h walltime kill
does not end its manifest.

  python3 code/phase3/scripts/chain_c1019.py <jobid> [<jobid> ...]

Each successor is the same pbs_v2.sh submission (manifest, state file, shard,
sentinel, workers, ngpus) held with depend=afterany:<jobid>: it starts when the
predecessor ends for any reason and resumes, because the orchestrator skips
cells whose done-file exists. A successor of a finished manifest exits at once.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "projects" / "rdmerge_v2"
KEEP = ("MANIFEST", "ORCH_STATE", "ORCH_SHARD", "ORCH_SENTINEL", "ORCH_WORKERS")


def job_info(jid):
    out = subprocess.run(["qstat", "-f", jid], capture_output=True, text=True, check=True).stdout
    out = re.sub(r"\n\t", "", out)             # qstat -f wraps long values
    name = re.search(r"Job_Name = (\S+)", out).group(1)
    ngpus = re.search(r"Resource_List.ngpus = (\d+)", out).group(1)
    vl = re.search(r"Variable_List = (.*)", out).group(1)
    env = dict(kv.split("=", 1) for kv in re.split(r",(?=[A-Z_][A-Z0-9_]*=)", vl) if "=" in kv)
    return name, ngpus, {k: env[k] for k in KEEP if k in env}


def main():
    for jid in sys.argv[1:]:
        name, ngpus, env = job_info(jid)
        if "MANIFEST" not in env:
            print(f"{jid}: no MANIFEST in its variables, skipped")
            continue
        v = ",".join(f"{k}={val}" for k, val in env.items())
        cmd = ["qsub", "-N", name[:13] + "c", "-W", f"depend=afterany:{jid}",
               "-l", f"select=1:ncpus=3:ngpus={ngpus}:mem=60gb", "-v", v,
               "-o", str(ROOT / "logs" / "pbs") + "/", str(ROOT / "code/phase3/scripts/pbs_v2.sh")]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        print(f"{jid} ({name}) -> {r.stdout.strip() or r.stderr.strip()}")


if __name__ == "__main__":
    main()
