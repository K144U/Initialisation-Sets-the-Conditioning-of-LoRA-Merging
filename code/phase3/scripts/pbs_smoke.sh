#!/bin/bash
#PBS -N rdm_smoke
#PBS -q gpu
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -l walltime=01:00:00
#PBS -o /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -e /home/sanjay.g/projects/rdmerge/logs/pbs/
#PBS -j oe
#PBS -V

set -uo pipefail
cd $PBS_O_WORKDIR

source /home/sanjay.g/projects/rdmerge/code/phase3/scripts/setup_env.sh
module load anaconda3/anaconda cuda12.2/toolkit/12.2.2 cudnn/9.14
source activate $PROJECT_ROOT/.conda/envs/rdmerge

# Pick the least-loaded GPU — for smoke we only need ~10 GB free
PICKED_GPU=$(python $PROJECT_ROOT/code/phase3/utils/gpu_picker.py --min-free-gb 12)
if [ -z "$PICKED_GPU" ]; then
    echo "ERROR: no GPU with >=12GB free; smoke aborted" >&2
    exit 87
fi
export CUDA_VISIBLE_DEVICES=$PICKED_GPU
echo "[$(date)] Smoke pinned to GPU $PICKED_GPU"

python $PROJECT_ROOT/code/phase3/smoke/smoke_test.py \
    --config $PROJECT_ROOT/code/phase3/smoke/smoke_config.yaml
