#!/bin/bash
#PBS -N rdm_a1rm
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# A1 replication stage 2: the T=4 matrix on indep2 and indep3. 56 cells,
# 5 baselines + rd_ridge + rd_rank16 per base per cohort.
#
# Run only after stage 1 has produced all 32 adapters. The guard below refuses
# a half-trained cohort, which is what the equivalent check in
# pbs_a1_indep_matrix.sh is for.
#
#   qsub code/phase3/scripts/pbs_a1_rep_matrix.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_a1_rep_matrix.json
export GPUS=0,1,2,3,4,6

# Refuse to run against a half-trained cohort. 16 adapters per cohort.
N2=$(ls artifacts/lora/*/*/indep2/adapter_model.safetensors 2>/dev/null | wc -l)
N3=$(ls artifacts/lora/*/*/indep3/adapter_model.safetensors 2>/dev/null | wc -l)
if [ "$N2" -lt 16 ] || [ "$N3" -lt 16 ]; then
  echo "[a1rm] ABORT: indep2 $N2/16, indep3 $N3/16 adapters present; run stage 1 first"
  exit 2
fi
echo "[a1rm] indep2 $N2/16, indep3 $N3/16 adapters present"

MANIFEST="${MANIFEST:-code/phase3/configs/a1_rep_matrix_manifest.json}"
echo "[$(date '+%F %T')] rdm_a1rm job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
