#!/bin/bash
# Submit the remaining 7 LoRA training jobs for Day 2.
# Respects PBS gpu queue's 3-concurrent-job cap (max_run = 3 per user).
# Logs each submission to logs/submit_all_lora.log

set -uo pipefail

PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
LOG=$PROJECT_ROOT/logs/submit_all_lora.log
SCRIPT=$PROJECT_ROOT/code/phase3/scripts/pbs_train.sh
CONFIG_DIR=$PROJECT_ROOT/code/phase3/configs/lora

CONFIGS=(
    llama31_8b_magicoder
    llama31_8b_alpaca
    llama31_8b_flores
    qwen25_7b_gsm8k
    qwen25_7b_magicoder
    qwen25_7b_alpaca
    qwen25_7b_flores
)

echo "[$(date)] launching ${#CONFIGS[@]} LoRA jobs" >> $LOG

for c in "${CONFIGS[@]}"; do
    # Wait until <3 of our rdm_train jobs are running/queued
    while [ "$(qstat -u $USER 2>/dev/null | grep -c 'rdm_train')" -ge 3 ]; do
        sleep 60
    done

    cfg=$CONFIG_DIR/$c.yaml
    if [ ! -f "$cfg" ]; then
        echo "[$(date)] MISSING CONFIG: $cfg" >> $LOG
        continue
    fi

    jobid=$(qsub -v CONFIG=$cfg $SCRIPT)
    echo "[$(date)] submitted $c -> $jobid" >> $LOG
done

echo "[$(date)] all 7 submitted; waiting for completion" >> $LOG

# Block until our gpu-queue jobs drain
while [ "$(qstat -u $USER 2>/dev/null | grep -c 'rdm_train')" -gt 0 ]; do
    sleep 120
done
echo "[$(date)] ALL_TRAINING_DONE" >> $LOG
