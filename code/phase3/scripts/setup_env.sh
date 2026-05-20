#!/bin/bash
# Sourced by every PBS job in Phase 3. Sets project paths and HF cache.
# Does NOT activate the conda env — the caller does that after this so
# the activation can read PROJECT_ROOT.

export PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
export HF_HOME=$PROJECT_ROOT/cache/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/cache/torch_extensions
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_ROOT/code/phase3:${PYTHONPATH:-}

# HF auth — load HF_TOKEN from project .env (gitignored) if present,
# else fall back to standard huggingface-cli login token file.
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
