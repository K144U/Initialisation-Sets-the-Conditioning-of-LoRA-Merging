#!/bin/bash
#PBS -N rdm_w3a
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E2: ridge sweep against conditioning. 56 cells, 4 bases x 7 lambdas x 2
# cohorts (seed1 shared, indep1 independent).
#
# Rules fixed in notes/prereg_conditioning_2026-08-07.md, committed at f9d230e
# before these cells were generated. Primary prediction: lambda* is large on
# the shared arm and collapses toward zero on the independent one.
#
#   qsub code/phase3/scripts/pbs_ridge_cond.sh
#   qsub -v MANIFEST=code/phase3/configs/ridge_cond_smoke_manifest.json,GPUS=3 \
#        code/phase3/scripts/pbs_ridge_cond.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_w3_asym.json
# Overridable. A single-cell smoke must pin one known-free GPU: the
# orchestrator starts one worker per pinned GPU and the others exit on
# queue.Empty after 30s, so a lone cell that lands on a card below its
# min_free_gb requeues into a queue with no live consumer.
export GPUS="${GPUS:-0,1,2,3,4,6}"

# Both cohorts must be fully trained: 16 adapters each.
for C in seed1 indep1; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[w3a] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[w3a] $C $N/16 adapters present"
done

MANIFEST="${MANIFEST:-code/phase3/configs/w3_asymptote_manifest.json}"
echo "[$(date '+%F %T')] rdm_w3a job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
