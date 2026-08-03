#!/bin/bash
# Self-healing keeper for the A1 replication run (indep2 + indep3).
#
# Same pattern as campaign_keeper.sh: resubmit any stage whose PBS job died
# before its manifest completed, dispatch waiting stages as slots free.
#
# NOTE the real concurrency limit. The PBS `gpu` queue allows 2 RUNNING jobs
# per user (measured 2026-08-03: "User has reached queue gpu running job
# limit" while two ran on a node with 2 TB free memory). MAXJOBS counts
# queued + running, so 3 keeps one warm in the queue without violating policy.
#
# Stage 2 is gated on all 32 adapters existing; the PBS script also refuses a
# half-trained cohort, so the gate is belt and braces.
#
# Stop:  touch _REPLICATE_KEEPER_STOP
# Log:   logs/replicate_keeper.log
# Start: setsid nohup bash code/phase3/scripts/replicate_keeper.sh >/dev/null 2>&1 </dev/null &
#
# Verify exactly one is alive with:  pgrep -af replicate_keeper.sh
# Never pkill -f it: an ssh command line containing the pattern matches itself.

cd /home/sanjay.g/projects/rdmerge || exit 1
LOG=logs/replicate_keeper.log
Q=/opt/pbs/bin/qstat
QSUB=/opt/pbs/bin/qsub
MAXJOBS=3
mkdir -p logs
echo $$ > logs/replicate_keeper.pid

log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "replicate keeper started pid $$ (cap ${MAXJOBS} queued+running)"

# name | pbs jobname | script | glob of done-files | target count
STAGES=(
  "a1rt|rdm_a1rt|pbs_a1_rep_train.sh|results/phase3/lora_train/*indep[23]*.json|32"
  "a1rm|rdm_a1rm|pbs_a1_rep_matrix.sh|results/phase3/eval_a1_indep/*indep[23]*.json|56"
)

count() { ls $1 2>/dev/null | wc -l; }

for i in $(seq 1 400); do          # 400 x 5 min = ~33 h
  if [ -f _REPLICATE_KEEPER_STOP ]; then
    log "stop sentinel found -> exiting"; rm -f _REPLICATE_KEEPER_STOP; exit 0
  fi

  qout=$($Q -u sanjay.g 2>/dev/null)
  running=$(echo "$qout" | grep -c "rdm_")
  alldone=1
  summary=""

  for s in "${STAGES[@]}"; do
    IFS='|' read -r name job script glob target <<< "$s"
    n=$(count "$glob")
    summary="$summary $name=$n/$target"
    [ "$n" -ge "$target" ] && continue
    alldone=0

    echo "$qout" | grep -q "$job" && continue
    [ "$running" -ge "$MAXJOBS" ] && continue
    # the matrix needs BOTH cohorts fully trained
    if [ "$name" = "a1rm" ]; then
      [ "$(count 'results/phase3/lora_train/*indep[23]*.json')" -ge 32 ] || continue
    fi

    jid=$($QSUB "code/phase3/scripts/$script" 2>>$LOG)
    log "$name at $n/$target and no $job in queue -> submitted $jid"
    running=$((running+1))
    sleep 20
    qout=$($Q -u sanjay.g 2>/dev/null)
  done

  [ $((i % 6)) -eq 1 ] && log "tick $i running=$running |$summary"

  if [ "$alldone" -eq 1 ]; then
    log "ALL STAGES COMPLETE |$summary -> exiting"
    exit 0
  fi
  sleep 300
done
log "replicate keeper hit its iteration limit; exiting"
