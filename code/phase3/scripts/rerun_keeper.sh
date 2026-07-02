#!/bin/bash
# Self-healing keeper for the matched-seed re-run (jobs rdm_rls + rdm_dssd).
# Resubmits either PBS job if it dies before its manifest is 100% complete
# (both are done-file resumable). Exits when both are complete or after ~12h.
cd /home/sanjay.g/projects/rdmerge || exit 1
LOG=logs/rerun_keeper.log
SCR=code/phase3/scripts
RLS=results/phase3/eval_ridge_seed
DS1=results/phase3/eval_e3_gsm8k_seed
DS2=results/phase3/eval_b4_humaneval_seed
DS3=results/phase3/eval_e3b_gsm8k_rdridge_seed
DS4=results/phase3/eval_b4b_humaneval_rdridge_seed
echo "$(date) rerun_keeper started pid $$" >> $LOG
i=0
while [ $i -lt 150 ]; do
  i=$((i+1))
  rls=$(ls $RLS/*.json 2>/dev/null | wc -l)
  ds=$(ls $DS1/*.json $DS2/*.json $DS3/*.json $DS4/*.json 2>/dev/null | wc -l)
  if [ $rls -ge 30 ] && [ $ds -ge 144 ]; then
    echo "$(date) COMPLETE rls=$rls/30 ds=$ds/144 -> exiting" >> $LOG
    break
  fi
  q=$(qstat -u sanjay.g 2>/dev/null)
  if [ $rls -lt 30 ] && ! echo "$q" | grep -q rdm_rls; then
    echo "$(date) rdm_rls dead at $rls/30 -> resubmit" >> $LOG
    ( cd $SCR && qsub pbs_ridge_lambda_seed_sweep.sh ) >> $LOG 2>&1
    sleep 20
  fi
  if [ $ds -lt 144 ] && ! echo "$q" | grep -q rdm_dssd; then
    echo "$(date) rdm_dssd dead at $ds/144 -> resubmit" >> $LOG
    ( cd $SCR && qsub pbs_downstream_seed.sh ) >> $LOG 2>&1
    sleep 20
  fi
  sleep 300
done
echo "$(date) rerun_keeper exiting (i=$i rls=$rls ds=$ds)" >> $LOG
