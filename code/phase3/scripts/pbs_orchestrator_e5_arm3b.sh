#!/bin/bash
#PBS -N rdm_e5arm3b
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=6:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E5 Arm 3b: 4 Llama-3.2-3B trainings at rank=64. Cross-architecture
# confirmation of the Arm 3 null on qwen-7B. Layer dim 3072 vs qwen's
# ~3584 — mechanical forcing still unreached (would need r > 768).
# Cells are fast (~30 min each on 3B model).

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E5_ARM3B_COMPLETE
export ORCH_STATE=orchestrator_state_e5_arm3b.json

MANIFEST="${MANIFEST:-code/phase3/configs/e5_arm3b_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E5" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E5)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e5arm3b job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
