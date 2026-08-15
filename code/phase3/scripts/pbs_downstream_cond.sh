#!/bin/bash
#PBS -N rdm_dsc
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Downstream accuracy on the lambda = 0 conditioning collapse. 32 cells.
#
# Rules: notes/prereg_downstream_2026-08-15.md (df832fe), amended by
# notes/prereg_downstream_amendment_2026-08-15.md (4d48987). Generator and
# analyzer both committed before any cell of this arm landed.
#
#   qsub code/phase3/scripts/pbs_downstream_cond.sh
#   qsub -v MANIFEST=code/phase3/configs/downstream_cond_smoke_manifest.json \
#        code/phase3/scripts/pbs_downstream_cond.sh
#
# Do NOT pass a comma-separated GPUS through -v: PBS splits the value on
# commas and reads the tail as further variable names. Use GPUS_FILE.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1

export ORCH_STATE="orchestrator_state_downstream_cond.json"

# GPU 0 is held by another user; 5 and 7 are outside our allocation.
if [ -n "${GPUS_FILE:-}" ] && [ -f "$GPUS_FILE" ]; then
  GPUS=$(cat "$GPUS_FILE")
fi
export GPUS="${GPUS:-1,2,3,4,6}"

MANIFEST="${MANIFEST:-code/phase3/configs/downstream_cond_manifest.json}"
if [ ! -f "$MANIFEST" ]; then
  echo "[dsc] ABORT: no manifest at $MANIFEST, run gen_downstream_conditioning.py first"
  exit 2
fi

# These are the two arms of the published E2 sweep; both must be intact.
for C in seed1 indep1; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[dsc] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[dsc] $C $N/16 adapters present"
done

echo "[$(date '+%F %T')] rdm_dsc job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
