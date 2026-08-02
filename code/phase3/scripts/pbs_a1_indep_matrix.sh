#!/bin/bash
#PBS -N rdm_a1ev
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=06:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# A1 stage 3: the T=4 matrix on the independent-init cohort. 28 cells,
# 5 baselines + rd-ridge in both realizations, per base. Run only after
# stage 1 completes (16/16 adapters) and stage 2 has reported the geometry.
#
#   qsub code/phase3/scripts/pbs_a1_indep_matrix.sh

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_a1_indep_matrix.json
export GPUS=0,1,2,3,4,6

# Refuse to run against a half-trained cohort.
N=$(ls artifacts/lora/*/*/indep1/adapter_model.safetensors 2>/dev/null | wc -l)
if [ "$N" -lt 16 ]; then
  echo "[a1ev] ABORT: only $N/16 indep1 adapters present; run stage 1 first"
  exit 2
fi
echo "[a1ev] $N/16 indep1 adapters present"

MANIFEST="${MANIFEST:-code/phase3/configs/a1_indep_matrix_manifest.json}"
echo "[$(date '+%F %T')] rdm_a1ev job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
