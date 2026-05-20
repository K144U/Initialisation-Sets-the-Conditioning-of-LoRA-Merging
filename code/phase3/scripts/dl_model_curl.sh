#!/bin/bash
# Download a HF model repo to a flat directory via wget.
# Bypasses huggingface_hub's Xet protocol (which self-throttles on this
# cluster). Resumable via wget --continue.
#
# Usage: dl_model_curl.sh <repo_id> <out_dir>
#   e.g. dl_model_curl.sh meta-llama/Llama-3.1-8B-Instruct \
#        /home/sanjay.g/projects/rdmerge/models/Llama-3.1-8B-Instruct
#
# Requires HF_TOKEN in env. Logs to stdout.

set -uo pipefail

REPO=$1
OUT=$2

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set" >&2
    exit 1
fi

mkdir -p "$OUT"
cd "$OUT"

WGET_OPTS=(
    --continue
    --tries=10
    --timeout=120
    --waitretry=30
    --header="Authorization: Bearer $HF_TOKEN"
    --user-agent="rdmerge-curl/1.0"
)

# Always-present small files
SMALL=(
    config.json
    generation_config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    model.safetensors.index.json
)

echo "[$(date)] downloading small files for $REPO"
for f in "${SMALL[@]}"; do
    if [ -s "$f" ]; then
        echo "  skip (exists): $f"
        continue
    fi
    URL="https://huggingface.co/$REPO/resolve/main/$f"
    echo "  fetch: $f"
    if ! wget "${WGET_OPTS[@]}" -O "$f" "$URL"; then
        echo "  WARN: $f not present in repo (ok if optional)"
        rm -f "$f"
    fi
done

# Optional: tokenizer.model (some repos have it, some don't)
if [ ! -s tokenizer.model ]; then
    URL="https://huggingface.co/$REPO/resolve/main/tokenizer.model"
    echo "  fetch: tokenizer.model (optional)"
    wget "${WGET_OPTS[@]}" -O tokenizer.model "$URL" 2>/dev/null || rm -f tokenizer.model
fi

# Parse safetensors index for shard names
if [ ! -s model.safetensors.index.json ]; then
    echo "ERROR: no model.safetensors.index.json — cannot determine shards" >&2
    exit 2
fi

SHARDS=$(python3 -c "
import json, sys
idx = json.load(open('model.safetensors.index.json'))
print('\n'.join(sorted(set(idx['weight_map'].values()))))
")

if [ -z "$SHARDS" ]; then
    echo "ERROR: no shards parsed from index" >&2
    exit 3
fi

echo "[$(date)] shards to download:"
echo "$SHARDS"
N_SHARDS=$(echo "$SHARDS" | wc -l)
i=0
for shard in $SHARDS; do
    i=$((i + 1))
    echo "[$(date)] shard $i/$N_SHARDS: $shard"
    URL="https://huggingface.co/$REPO/resolve/main/$shard"
    t0=$(date +%s)
    wget "${WGET_OPTS[@]}" --progress=dot:giga -O "$shard" "$URL"
    rc=$?
    t1=$(date +%s)
    size=$(stat -c %s "$shard" 2>/dev/null || echo 0)
    elapsed=$((t1 - t0))
    if [ "$elapsed" -gt 0 ] && [ "$size" -gt 0 ]; then
        rate=$((size / elapsed))
    else
        rate=0
    fi
    echo "[$(date)] shard $i done: size=$(( size / 1024 / 1024 ))MB elapsed=${elapsed}s rate=$(( rate / 1024 ))KB/s rc=$rc"
    [ $rc -ne 0 ] && { echo "ERROR: shard $shard failed"; exit 4; }
done

echo "[$(date)] DOWNLOAD_DONE: $REPO -> $OUT"
