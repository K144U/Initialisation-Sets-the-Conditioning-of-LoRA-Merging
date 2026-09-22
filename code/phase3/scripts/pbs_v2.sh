#!/bin/bash
#PBS -N rdm_v2
#PBS -q gpu
#PBS -l select=1:ncpus=3:ngpus=1:mem=60gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -V

# Generic campaign launcher for the student account (STUDENT-ACCOUNT).
#
# The sanjay.g tree is READ-ONLY from here: adapters and base models are read
# from it, everything is written under ~/projects/rdmerge_v2. PROTECT_USERS
# makes the orchestrator refuse any GPU on which a sanjay.g process is
# running, so these cells never share a card with that account's jobs.
#
#   qsub -v MANIFEST=code/phase3/configs/<x>_manifest.json,ORCH_STATE=<x>.json,ORCH_SHARD=0/5 \
#        -o ~/projects/rdmerge_v2/logs/pbs/ code/phase3/scripts/pbs_v2.sh
#
# Queue limits as of 2026-09-23: mem <= 64gb per job, 5 running jobs and 16
# cpus per user, GPUs allocated by PBS (ngpus). Five shards of one manifest,
# ORCH_SHARD=0/5 .. 4/5, use the whole allowance with no cell run twice.

set -uo pipefail
export PROJECT_ROOT=$HOME/projects/rdmerge_v2
export RDMERGE_ROOT=$PROJECT_ROOT
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate /home/sanjay.g/projects/rdmerge/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export HF_HOME=$PROJECT_ROOT/cache/hf_home
export TRANSFORMERS_CACHE=$HF_HOME/hub
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
fi
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/cache/torch_extensions
export UNSLOTH_COMPILE_LOCATION=$PROJECT_ROOT/cache/unsloth_compiled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$PROJECT_ROOT/code/phase3:${PYTHONPATH:-}
# 2026-09-23: sanjay.g jobs held 7 of 8 GPUs, so the user chose to share cards
# behind a memory gate (need + ORCH_HEADROOM_GB free, one of our cells per
# card). Set PROTECT_USERS=sanjay.g to go back to never sharing.
export PROTECT_USERS="${PROTECT_USERS:-}"
export ORCH_HEADROOM_GB="${ORCH_HEADROOM_GB:-10}"

# ngpus=1 is accounting only on this queue (checked 2026-09-23 with a probe
# job: CUDA_VISIBLE_DEVICES empty, all 8 devices visible, exec_vnode names no
# GPU). So the job picks its own card: one worker (ORCH_DYNAMIC=1) choosing,
# per cell, the allowed GPU with the most free memory that carries no
# PROTECT_USERS process (if set) and no claim by another of our jobs. It never
# holds more than the one GPU it was billed.
# 5 and 7 are outside this project's allocation.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPUS="$CUDA_VISIBLE_DEVICES"
else
  GPUS="${GPUS_ALLOWED:-0 1 2 3 4 6}"
  GPUS="${GPUS// /,}"
  export ORCH_DYNAMIC=1
fi
export GPUS
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv || true
export ORCH_STATE="${ORCH_STATE:-orchestrator_state_v2.json}"
export ORCH_SENTINEL="${ORCH_SENTINEL:-_QUEUE_COMPLETE_V2}"

MANIFEST="${MANIFEST:?set MANIFEST}"
[ -f "$MANIFEST" ] || { echo "[v2] ABORT: no manifest $MANIFEST"; exit 2; }

echo "[$(date '+%F %T')] rdm_v2 job=${PBS_JOBID:-none} manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python -c "import torch,peft,transformers; print('[v2] torch',torch.__version__,'peft',peft.__version__,'tf',transformers.__version__)"
python code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
