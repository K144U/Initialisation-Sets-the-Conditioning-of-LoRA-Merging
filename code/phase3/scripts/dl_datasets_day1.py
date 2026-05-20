"""Pre-download Magicoder + FLORES datasets into HF cache."""

import os
import sys
import time

from datasets import load_dataset


CASES = [
    ("ise-uiuc/Magicoder-OSS-Instruct-75K", None),
    ("facebook/flores", "eng_Latn-deu_Latn"),
]


def main() -> int:
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN missing", flush=True)
        return 1
    for repo, cfg in CASES:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] loading {repo} (config={cfg})", flush=True)
        try:
            ds = load_dataset(repo, cfg) if cfg else load_dataset(repo)
            print(f"  -> done in {time.time()-t0:.1f}s, splits={list(ds.keys())}", flush=True)
        except Exception as e:
            print(f"  FAIL {repo}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            return 2
    print("DATASETS_DAY1_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
