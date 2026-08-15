#!/bin/bash
#PBS -N rdm_shr
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# The ridge sweep's shared arm at n = 3: seed2 and seed3, 56 cells.
#
# Rules: notes/prereg_shared_arm_2026-08-15.md (603e013). Generator and
# analyzer both committed before any cell of this arm landed.
#
#   qsub code/phase3/scripts/pbs_shared_arm.sh
#   qsub -v GPUS_FILE=_ORCH_GPUS_SHR code/phase3/scripts/pbs_shared_arm.sh
#
# Do NOT pass a comma-separated GPUS through -v: PBS splits the value on
# commas and reads the tail as further variable names, failing with "cannot
# send environment with the job". Use GPUS_FILE, read below, or the default.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1

export ORCH_STATE="orchestrator_state_shared_arm.json"

# GPU 0 sits outside the default: another user's job has held 78.9 of its
# 81.9 GB since 2026-08-14, so a worker there fails the 25 GB gate on every
# cell. GPUs 5 and 7 are outside our allocation.
if [ -n "${GPUS_FILE:-}" ] && [ -f "$GPUS_FILE" ]; then
  GPUS=$(cat "$GPUS_FILE")
fi
export GPUS="${GPUS:-1,2,3,4,6}"

# Overridable so that binding constraint 5 of the registration, one real cell
# inspected before the other fifty-five are dispatched, runs through this same
# path rather than around it. No commas in the value, so -v is safe here.
MANIFEST="${MANIFEST:-code/phase3/configs/shared_arm_manifest.json}"
if [ ! -f "$MANIFEST" ]; then
  echo "[shr] ABORT: no manifest at $MANIFEST, run gen_shared_arm.py first"
  exit 2
fi

# The cohorts this arm adds. The seed1 arm is reused and is not re-run, so it
# is deliberately not checked for here.
for C in seed2 seed3; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[shr] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[shr] $C $N/16 adapters present"
done

echo "[$(date '+%F %T')] rdm_shr job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
