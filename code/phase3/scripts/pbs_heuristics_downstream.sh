#!/bin/bash
#PBS -N rdm_hds
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# The heuristics null, measured in downstream accuracy. 80 cells.
#
# Rules: notes/prereg_heuristics_downstream_2026-08-16.md (8ff7fb4).
# Generator and analyzer both committed before any cell of this run landed.
#
#   qsub code/phase3/scripts/pbs_heuristics_downstream.sh
#   qsub -v MANIFEST=... code/phase3/scripts/pbs_heuristics_downstream.sh
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

export ORCH_STATE="orchestrator_state_heur_downstream.json"

if [ -n "${GPUS_FILE:-}" ] && [ -f "$GPUS_FILE" ]; then
  GPUS=$(cat "$GPUS_FILE")
fi
export GPUS="${GPUS:-1,2,3,4,6}"

MANIFEST="${MANIFEST:-code/phase3/configs/heur_downstream_manifest.json}"
if [ ! -f "$MANIFEST" ]; then
  echo "[hds] ABORT: no manifest at $MANIFEST, run gen_heuristics_downstream.py first"
  exit 2
fi

# Both arms must be intact: the registered comparison is paired within a
# (method, base) and a missing adapter on one side silently changes the merge
# on that side only.
for C in seed1 indep1; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[hds] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[hds] $C $N/16 adapters present"
done

echo "[$(date '+%F %T')] rdm_hds job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
