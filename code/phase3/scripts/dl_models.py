"""Download Llama-3.1-8B + Qwen-2.5-7B safetensors to HF cache.

Invoked by pbs_dl_models.sh from a compute node — compute nodes have
much higher effective bandwidth to HF CDN than the login node.

Skips the `original/` Meta-pth duplicates and only fetches what we
need: safetensors + config + tokenizer files.
"""

import os
import sys
import time

from huggingface_hub import snapshot_download


REPOS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]
ALLOW = [
    "config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "*.safetensors",
    "model.safetensors.index.json",
]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set", flush=True)
        return 1
    for repo in REPOS:
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] starting {repo}", flush=True)
        path = snapshot_download(
            repo_id=repo,
            token=token,
            allow_patterns=ALLOW,
            max_workers=4,
        )
        elapsed = time.time() - t0
        print(f"[{time.strftime('%H:%M:%S')}] {repo} done in {elapsed:.1f}s -> {path}", flush=True)
    print("DOWNLOAD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
