#!/bin/bash
#PBS -N rdm_dt
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# DARE-TIES: 36 cells, 4 bases x 3 cohorts (indep1/2/3) x 3 dare densities
# (0.5, 0.2, 0.1). ties_density pinned at 0.2 so the only difference from the
# existing `ties` arm is the DARE mask.
#
# Rules fixed in notes/prereg_dare_ties_2026-08-04.md, committed at efb593f
# before the dare_ties method existed. Primary verdict is dare_density 0.2 only.
#
#   qsub code/phase3/scripts/pbs_dare_ties.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_dare_ties.json
export GPUS=0,1,2,3,4,6

# All three cohorts must be fully trained: 16 adapters each.
for C in indep1 indep2 indep3; do
  N=$(ls artifacts/lora/*/*/$C/adapter_model.safetensors 2>/dev/null | wc -l)
  if [ "$N" -lt 16 ]; then
    echo "[dt] ABORT: $C has $N/16 adapters"
    exit 2
  fi
  echo "[dt] $C $N/16 adapters present"
done

# The comparison is against the EXISTING ties cells; refuse to run without them
# rather than produce an arm with nothing to compare to.
NT=$(ls results/phase3/eval_a1_indep/*__ties__indep[123].json 2>/dev/null | wc -l)
if [ "$NT" -lt 12 ]; then
  echo "[dt] ABORT: only $NT/12 TIES reference cells present"
  exit 2
fi
echo "[dt] $NT/12 TIES reference cells present"

MANIFEST="${MANIFEST:-code/phase3/configs/dare_ties_manifest.json}"
echo "[$(date '+%F %T')] rdm_dt job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
