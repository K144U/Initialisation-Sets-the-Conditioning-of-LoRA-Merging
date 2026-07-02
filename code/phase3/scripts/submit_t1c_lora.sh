#!/bin/bash
# T1.C — submit 16 LoRA training jobs for the 4 robustness-panel models.
#   Mistral-7B-Instruct-v0.3, Yi-1.5-9B-Chat, gemma-3-12b-it, Qwen2.5-14B-Instruct
# Respects the 3-concurrent gpu-queue cap by counting BOTH rdm_eval AND rdm_train
# jobs — so this launcher yields cleanly to the in-flight v2 eval matrix
# (job-name "rdm_eval") and only fills slots as eval cells drain.
#
# Logs each submission to logs/submit_t1c_lora.log
# Order: by-task batches across models (gsm8k for all 4 first, then alpaca, ...)
# so that any model-specific failure surfaces early across the model dimension
# rather than burning hours of the same model failing on every task.

set -uo pipefail

PROJECT_ROOT=/home/sanjay.g/projects/rdmerge
LOG=$PROJECT_ROOT/logs/submit_t1c_lora.log
SCRIPT=$PROJECT_ROOT/code/phase3/scripts/pbs_train.sh
CONFIG_DIR=$PROJECT_ROOT/code/phase3/configs/lora

CONFIGS=(
    mistral_7b_gsm8k
    yi15_9b_gsm8k
    mistral_7b_alpaca
    yi15_9b_alpaca
    mistral_7b_magicoder
    yi15_9b_magicoder
    mistral_7b_flores
    yi15_9b_flores
)

echo "[$(date)] T1.C launching ${#CONFIGS[@]} LoRA jobs (yields to rdm_eval)" >> $LOG

for c in "${CONFIGS[@]}"; do
    # Wait until <3 of OUR gpu-queue jobs (eval + train combined) are running/queued.
    # This makes the launcher cooperate with the in-flight v2 eval matrix.
    while [ "$(qstat -u $USER 2>/dev/null | grep -cE 'rdm_(eval|train)')" -ge 3 ]; do
        sleep 60
    done

    cfg=$CONFIG_DIR/$c.yaml
    if [ ! -f "$cfg" ]; then
        echo "[$(date)] MISSING CONFIG: $cfg" >> $LOG
        continue
    fi

    jobid=$(qsub -v CONFIG=$cfg $SCRIPT)
    echo "[$(date)] submitted $c -> $jobid" >> $LOG
    # Small pause so qstat reflects the new submission before the next loop iter.
    sleep 5
done

echo "[$(date)] all ${#CONFIGS[@]} T1.C submissions queued; waiting for completion" >> $LOG

# Block until our rdm_train jobs drain (eval jobs not our concern here).
while [ "$(qstat -u $USER 2>/dev/null | grep -c 'rdm_train')" -gt 0 ]; do
    sleep 120
done
echo "[$(date)] T1C_ALL_TRAINING_DONE" >> $LOG
