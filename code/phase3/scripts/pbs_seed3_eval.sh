#!/bin/bash
#PBS -N rdm_s3ev
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=4:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Phase 1 step 4 — eval the 20 seed-3 §6.1 matrix cells
# (4 bases x 5 methods at training seed 3 adapters).
#
# PRE-REQUISITE: rdm_s3tr (pbs_seed3_train.sh) must have completed —
# the 16 /seed3/ adapter directories must exist.
#
# Compute: ~20 cells x ~15 min each / 3 lanes = ~100 min wallclock.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_SEED3_EVAL_COMPLETE
export ORCH_STATE=orchestrator_state_seed3_eval.json

MANIFEST="${MANIFEST:-code/phase3/configs/seed3_matrix_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E6" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E6)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_s3ev job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
