#!/bin/bash
#PBS -N rdm_w5rs
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=120gb
#PBS -l walltime=12:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

# W5 / A4 / A5: re-run the 144 downstream cells with the fixed scorers.
#   GSM8K   last-number-anywhere fallback (old code failed on 60-81% of
#           generations, method-dependently)
#   HumanEval  completion stripper no longer returns "" for markdown-fenced
#           answers (old code discarded up to 79% of some methods' output)
# Cells now store FULL generations, so any future scorer change is a CPU
# re-score instead of another 36 GPU-hours.
#
# 144 cells at ~10-25 min => ~10-12 h over 6 lanes. Walltime 12 h; the
# orchestrator skips done cells on requeue, so a walltime kill is safe.
#
#   python code/phase3/scripts/gen_w5_rescore.py
#   qsub code/phase3/scripts/pbs_w5_rescore.sh
#
# Published results in eval_{e3_gsm8k,b4_humaneval,...}_seed/ are NOT touched;
# the new cells land in eval_downstream_v2/ so the delta stays auditable.

set -uo pipefail
PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
cd $PROJECT_ROOT
mkdir -p logs/pbs logs/orch

source $PROJECT_ROOT/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge
export PYTHONNOUSERSITE=1
export ORCH_STATE=orchestrator_state_w5_rescore.json
export GPUS=0,1,2,3,4,6

# Fail fast if the scorer fixes are not in place on this checkout.
python -c "
import sys; sys.path.insert(0, 'code/phase3')
from eval.downstream_metrics import gsm8k_extract_answer, _strip_humaneval_completion
assert gsm8k_extract_answer('Thus, 12 students are good at math.') == '12', 'A5 fix missing'
assert _strip_humaneval_completion('\ndef f(x):\n    return x') != '', 'A4 fix missing'
print('[w5] scorer fixes verified')
" || { echo "[w5] ABORT: scorer fixes not present"; exit 2; }

MANIFEST="${MANIFEST:-code/phase3/configs/w5_rescore_manifest.json}"
echo "[$(date '+%F %T')] rdm_w5rs job=$PBS_JOBID manifest=$MANIFEST gpus=$GPUS host=$(hostname)"
python $PROJECT_ROOT/code/phase3/scripts/orchestrator.py --manifest "$MANIFEST"
echo "[$(date '+%F %T')] orchestrator exited"
