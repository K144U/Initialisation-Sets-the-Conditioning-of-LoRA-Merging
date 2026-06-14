"""Opportunistic GPU pin expansion.

Each invocation:
  1. SSH to jiit-gpu01, read nvidia-smi free-memory per GPU.
  2. For each non-pinned GPU: increment a stability counter if free >= threshold,
     reset to 0 otherwise.
  3. A GPU is "stable-candidate" once its counter reaches `stable_ticks`
     (default 4 = 2 hours at the 30-min cron cadence).
  4. Write the union (pinned + stable-candidates) to an output file, which
     the next PBS job submission auto-discovers and uses.

This deliberately does NOT touch any currently-running orchestrator —
orchestrator.py reads the GPU set at startup. The dynamic expansion benefits
the NEXT job submission (e.g., the E5 Arm 2 main run after the pilot gate
fires GO).

Run every cron tick:
    python code/phase3/scripts/gpu_opportunity.py \
        --pinned 2,4,6 --threshold-gb 45 --stable-ticks 4 \
        --state-file logs/gpu_opportunity_state.json \
        --output-file _ORCH_GPUS_DYN

Prints one summary line to stdout. Idempotent; safe to invoke from cron.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/home/sanjay.g/projects/rdmerge")
ALL_GPUS = (0, 1, 2, 3, 4, 5, 6, 7)


def query_free_gb() -> dict[int, float]:
    """SSH to gpu01, return {gpu_idx: free_mem_GB}."""
    cmd = [
        "ssh", "jiit-gpu01",
        "nvidia-smi --query-gpu=index,memory.free "
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr}")
    res: dict[int, float] = {}
    for line in out.stdout.strip().splitlines():
        idx_s, mem_s = line.split(",")
        res[int(idx_s.strip())] = float(mem_s.strip()) / 1024.0  # MiB -> GiB
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pinned", default="2,4,6",
                   help="Comma-separated list of always-included GPUs.")
    p.add_argument("--threshold-gb", type=float, default=45.0,
                   help="Min free VRAM (GB) to count as a candidate this tick.")
    p.add_argument("--stable-ticks", type=int, default=4,
                   help="Consecutive ticks above threshold required to "
                        "promote a GPU to the dynamic set.")
    p.add_argument("--state-file", default="logs/gpu_opportunity_state.json")
    p.add_argument("--output-file", default="_ORCH_GPUS_DYN",
                   help="Where to write the comma-separated final set.")
    p.add_argument("--max-extras", type=int, default=5,
                   help="Cap on how many non-pinned GPUs to promote.")
    args = p.parse_args()

    pinned = sorted({int(x) for x in args.pinned.split(",") if x})

    state_path = PROJECT_ROOT / args.state_file
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"counters": {str(g): 0 for g in ALL_GPUS}}

    try:
        free = query_free_gb()
    except Exception as e:
        print(f"gpu_opp: ERR query_free_gb: {e}", file=sys.stderr)
        return 1

    # Update counters for NON-pinned GPUs only; pinned are always in the set.
    candidates: list[int] = []
    summary_parts: list[str] = []
    for g in ALL_GPUS:
        if g in pinned:
            continue
        free_gb = free.get(g, 0.0)
        if free_gb >= args.threshold_gb:
            state["counters"][str(g)] = int(state["counters"].get(str(g), 0)) + 1
        else:
            state["counters"][str(g)] = 0
        c = state["counters"][str(g)]
        summary_parts.append(f"gpu{g}:{free_gb:.0f}gb({c}t)")
        if c >= args.stable_ticks:
            candidates.append(g)

    candidates = candidates[: args.max_extras]
    final = sorted(set(pinned) | set(candidates))
    out_path = PROJECT_ROOT / args.output_file
    new_text = ",".join(str(g) for g in final)
    prev_text = out_path.read_text().strip() if out_path.exists() else ""
    if new_text != prev_text:
        out_path.write_text(new_text + "\n")
        change = "WROTE"
    else:
        change = "no change"

    state_path.write_text(json.dumps(state, indent=1))

    print(f"gpu_opp: pinned={pinned} candidates={candidates} "
          f"final={final} ({change}); " + " ".join(summary_parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
