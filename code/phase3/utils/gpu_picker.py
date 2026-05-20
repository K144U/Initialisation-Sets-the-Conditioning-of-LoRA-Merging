"""Pick the least-loaded GPU on jiit-gpu01.

PBS does not track GPUs as schedulable resources, so the 8 A100s on
jiit-gpu01 are shared informally. This script polls nvidia-smi and
returns the index of the GPU with the most free memory iff that free
memory exceeds --min-free-gb.

Exit code 0 + index on stdout if a suitable GPU is found.
Exit code 87 (no stdout) if no GPU meets the threshold — PBS wrapper
catches this and exits the job cleanly so it can be resubmitted.
"""

import argparse
import shutil
import subprocess
import sys


def query_free_memory_mib() -> list[int]:
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not on PATH — likely on login node", file=sys.stderr)
        sys.exit(87)
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return [int(line.strip()) for line in out.splitlines() if line.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--min-free-gb", type=float, default=75.0,
                   help="Skip GPUs with less than this many free GB")
    args = p.parse_args()

    free_mib = query_free_memory_mib()
    if not free_mib:
        print("No GPUs visible", file=sys.stderr)
        return 87

    best_idx = max(range(len(free_mib)), key=lambda i: free_mib[i])
    best_free_gb = free_mib[best_idx] / 1024.0
    if best_free_gb < args.min_free_gb:
        print(
            f"Best GPU {best_idx} only has {best_free_gb:.1f} GB free "
            f"(need >= {args.min_free_gb}); all: {[f'{m/1024:.1f}' for m in free_mib]}",
            file=sys.stderr,
        )
        return 87

    print(best_idx)
    print(
        f"Picked GPU {best_idx} with {best_free_gb:.1f} GB free "
        f"(all: {[f'{m/1024:.1f}' for m in free_mib]})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
