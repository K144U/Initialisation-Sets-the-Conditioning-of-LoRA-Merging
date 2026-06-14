#!/bin/bash
#PBS -N rdm_e5arm3
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E5 Arm 3 (geometric forcing): 4 qwen-7B trainings at rank=64.
# Arm 2 pilot fired NO-GO at 0/112 layers <0.8 on alpha=0.9 (2026-06-14),
# so we fall back to geometric forcing per master_plan §E5.
# Rank 64 with T=4 tasks -> Tr = 256, which can approach the layer dimension
# (4096 for qwen) and mechanically force d_eff < Tr by counting.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E5_ARM3_COMPLETE
export ORCH_STATE=orchestrator_state_e5_arm3.json

MANIFEST="${MANIFEST:-code/phase3/configs/e5_arm3_manifest.json}"
# Prefer dynamic GPU set (auto-expanded by gpu_opportunity), then user
# override, then default 2,4,6.
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E5" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E5)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e5arm3 job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
