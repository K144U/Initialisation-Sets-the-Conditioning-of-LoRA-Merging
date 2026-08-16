#!/bin/bash
#PBS -N rdm_drift
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# The drift cohort: 8 training runs, one base, two initialisation arms.
#
# Rules: notes/prereg_drift_2026-08-16.md (9202d3b). Generator and analyzer
# both committed before this job was first dispatched.
#
#   qsub code/phase3/scripts/pbs_drift_cohort.sh
#
# Each run writes A_0 before the first optimiser step and keeps checkpoints at
# 25/50/75/100%. That is the whole point: the original cohorts kept only the
# final checkpoint, which is why E3 has stood undischarged.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1

export ORCH_STATE="orchestrator_state_drift.json"

if [ -n "${GPUS_FILE:-}" ] && [ -f "$GPUS_FILE" ]; then
  GPUS=$(cat "$GPUS_FILE")
fi
export GPUS="${GPUS:-1,2,3,4,6}"

MANIFEST="${MANIFEST:-code/phase3/configs/drift_cohort_manifest.json}"
if [ ! -f "$MANIFEST" ]; then
  echo "[drift] ABORT: no manifest at $MANIFEST, run gen_drift_cohort.py first"
  exit 2
fi

echo "[$(date '+%F %T')] rdm_drift job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
