#!/bin/bash
#PBS -N rdm_smoke
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=02:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Smoke gate for the review-response campaign. One cell per new path plus one
# control, per hard-won rule #2 (context.md): smoke-first is MANDATORY for any
# new merge/eval path, with an explicit PASS/FAIL bar before any batch.
#
#   python code/phase3/scripts/gen_smoke_review.py
#   qsub code/phase3/scripts/pbs_smoke_review.sh
#   python code/phase3/scripts/check_smoke_review.py    # exit 0 = safe to dispatch

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_smoke_review.json
export GPUS=0,1,2,3,4,6

MANIFEST="${MANIFEST:-code/phase3/configs/smoke_review_manifest.json}"
echo "[$(date '+%F %T')] rdm_smoke job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
echo
echo "=== smoke gate verdict ==="
python $PROJECT_ROOT/code/phase3/scripts/check_smoke_review.py
