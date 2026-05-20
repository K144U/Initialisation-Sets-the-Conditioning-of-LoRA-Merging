#!/bin/bash
#PBS -N rdm_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=04:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -e /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR

source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge

# CONFIG is passed via -v at qsub time.
# We need ~40 GB free VRAM for 8B-class bf16 training.
PICKED_GPU=$(python $PROJECT_ROOT/code/phase3/utils/gpu_picker.py --min-free-gb 40)
if [ -z "$PICKED_GPU" ]; then
    echo "ERROR: no GPU with >=40GB free; aborting" >&2
    exit 87
fi
export CUDA_VISIBLE_DEVICES=$PICKED_GPU
echo "[$(date)] training pinned to GPU $PICKED_GPU; CONFIG=$CONFIG"

python $PROJECT_ROOT/code/phase3/training/train_lora.py --config $CONFIG
