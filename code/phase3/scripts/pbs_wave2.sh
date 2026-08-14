#!/bin/bash
#PBS -N rdm_w2
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# Wave 2 of the TMLR revision campaign. One arm per submission.
#
#   ARM=r7  untruncated ridge sweep, seed1 + indep1, realize=rank_deff  (R7)
#   ARM=r8  ridge sweep on indep2 + indep3 so the 2xSE gate exists       (R8)
#
# Rules: notes/prereg_tmlr_2026-08-14.md (acebd1a). Analyzer committed before
# any cell of either arm landed.
#
#   qsub -v ARM=r7 code/phase3/scripts/pbs_wave2.sh
#   qsub -v ARM=r8,GPUS_FILE=_ORCH_GPUS_W2B code/phase3/scripts/pbs_wave2.sh
#
# Do NOT pass a comma-separated GPUS through -v: PBS splits the value on
# commas and reads the tail as further variable names, failing with "cannot
# send environment with the job". Use GPUS_FILE, read below, or edit the
# default.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1

ARM="${ARM:-r7}"
export ORCH_STATE="orchestrator_state_wave2_${ARM}.json"

# GPU 0 sits outside the default: another user's job has held 78.9 of its
# 81.9 GB since 2026-08-14, so a worker there fails the 25 GB gate on every
# cell. GPUS_FILE lets two arms run concurrently on disjoint cards.
if [ -n "${GPUS_FILE:-}" ] && [ -f "$GPUS_FILE" ]; then
  GPUS=$(cat "$GPUS_FILE")
fi
export GPUS="${GPUS:-1,2,3,4,6}"

case "$ARM" in
  r7) MANIFEST=code/phase3/configs/wave2_r7_manifest.json; COHORTS="seed1 indep1" ;;
  r8) MANIFEST=code/phase3/configs/wave2_r8_manifest.json; COHORTS="indep2 indep3" ;;
  r4) MANIFEST=code/phase3/configs/r4_knots_indep_manifest.json
      COHORTS="indep1 indep2 indep3" ;;
  r6) MANIFEST=code/phase3/configs/r6_t3_manifest.json
      COHORTS="indep1 indep2 indep3" ;;
  *)  echo "[w2] ABORT: unknown ARM=$ARM"; exit 2 ;;
esac

for C in $COHORTS; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[w2] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[w2] $C $N/16 adapters present"
done

echo "[$(date '+%F %T')] rdm_w2 arm=$ARM job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
