#!/bin/bash
#PBS -N rdm_s3tr
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=8:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Phase 1 step 4 — train 16 seed-3 LoRA adapters (4 bases x 4 tasks)
# at training seed 20260520, to support the §6.1 multi-seed expansion
# from 2 seeds to 3 seeds.
#
# Compute: ~16 cells x ~25 min each / 3 lanes = ~140 min wallclock.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_SEED3_TRAIN_COMPLETE
export ORCH_STATE=orchestrator_state_seed3_train.json

MANIFEST="${MANIFEST:-code/phase3/configs/seed3_train_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E6" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E6)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_s3tr job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
