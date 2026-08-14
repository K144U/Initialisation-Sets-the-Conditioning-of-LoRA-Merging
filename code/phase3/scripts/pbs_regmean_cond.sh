#!/bin/bash
#PBS -N rdm_rm
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# R1: does cohort conditioning affect a solver we did NOT build?
# 56 cells, 4 bases x 7 lambdas x 2 cohorts (seed1 shared, indep1 independent),
# method = regmean.
#
# Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a), amended by
# notes/prereg_tmlr_amendment_2026-08-14.md (7ce15b1). Both committed before
# the generator existed. Analyzer committed before any cell landed.
#
#   qsub code/phase3/scripts/pbs_regmean_cond.sh
#   qsub -v MANIFEST=code/phase3/configs/regmean_cond_smoke_manifest.json,GPUS=3 \
#        code/phase3/scripts/pbs_regmean_cond.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_regmean_cond.json
# Overridable. A single-cell smoke must pin one known-free GPU: the orchestrator
# starts one worker per pinned GPU and the others exit on queue.Empty after 30s,
# so a lone cell that lands on a card below its min_free_gb requeues into a
# queue with no live consumer.
export GPUS="${GPUS:-0,1,2,3,4,6}"

# Both cohorts must be fully trained: 16 adapters each.
for C in seed1 indep1; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[rm] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[rm] $C $N/16 adapters present"
done

MANIFEST="${MANIFEST:-code/phase3/configs/regmean_cond_manifest.json}"
echo "[$(date '+%F %T')] rdm_rm job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
