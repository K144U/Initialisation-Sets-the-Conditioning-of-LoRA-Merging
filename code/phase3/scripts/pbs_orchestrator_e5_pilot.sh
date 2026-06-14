#!/bin/bash
#PBS -N rdm_e5pilot
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=24:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E5 Arm 2 pilot orchestrator: 12 qwen-2.5-7B trainings (4 tasks x 3 alpha).
# Isolated sentinel/state so it cannot clobber other orchestrators.
# Default GPUs 2,4,6 (the pinned safe set).
# Each cell needs ~40GB VRAM (bf16 7B base + LoRA + optimizer state).

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E5_PILOT_COMPLETE
export ORCH_STATE=orchestrator_state_e5_pilot.json

MANIFEST="${MANIFEST:-code/phase3/configs/e5_pilot_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E5" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E5)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e5pilot job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
