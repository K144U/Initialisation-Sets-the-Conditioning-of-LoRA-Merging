#!/bin/bash
#PBS -N rdm_eval
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -l walltime=03:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -e /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR
source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge

PICKED_GPU=$(python $PROJECT_ROOT/code/phase3/utils/gpu_picker.py --min-free-gb 25)
[ -z "$PICKED_GPU" ] && { echo "no GPU with >=25GB free; aborting" >&2; exit 87; }
export CUDA_VISIBLE_DEVICES=$PICKED_GPU
echo "[$(date)] eval_cell pinned to GPU $PICKED_GPU; CONFIG=$CONFIG"

python $PROJECT_ROOT/code/phase3/eval/run_eval_cell.py --config $CONFIG
