#!/bin/bash
#PBS -N rdm_e6Lev
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=8:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# E6 full sweep stage 2 — 54 merge-eval cells (9 subsets x 6 methods) on
# Llama-3.1-8B-Instruct across T in {2, 4, 7}. Submit only after stage 1
# trainings complete (artifacts/lora/llama31_8b/e6_pilot/
# {codealpaca,dolly,xsum}/v1 all present).

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_SENTINEL=_E6_LLAMA_EVAL_COMPLETE
export ORCH_STATE=orchestrator_state_e6_llama_eval.json

MANIFEST="${MANIFEST:-code/phase3/configs/e6_llama_eval_manifest.json}"
if [ -f "$PROJECT_ROOT/_ORCH_GPUS_E6" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_E6)
elif [ -f "$PROJECT_ROOT/_ORCH_GPUS_DYN" ]; then
    export GPUS=$(cat $PROJECT_ROOT/_ORCH_GPUS_DYN)
else
    export GPUS=2,4,6
fi

echo "[$(date '+%F %T')] rdm_e6Lev job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
