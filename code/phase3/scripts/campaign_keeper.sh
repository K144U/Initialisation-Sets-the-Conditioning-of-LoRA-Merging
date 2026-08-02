#!/bin/bash
# Self-healing keeper for the 2026-08-02 review-response campaign.
#
# Does two jobs at once:
#   1. resubmits any stage whose PBS job died before its manifest completed
#      (every stage is done-file resumable, so a walltime kill costs nothing)
#   2. dispatches the stages still waiting, as slots free
#
# Honours the documented 3-concurrent-PBS-job cap. Stages are tried in
# priority order, so the decisive experiment (W1, the review's main objection)
# is always resubmitted before the cheap ones are started.
#
# A1 stage 3 (the indep-init matrix) is gated on A1 stage 1 producing all 16
# adapters; the keeper will not start it early.
#
# Stop:  touch _CAMPAIGN_KEEPER_STOP
# Log:   logs/campaign_keeper.log
# Start: setsid nohup bash code/phase3/scripts/campaign_keeper.sh >/dev/null 2>&1 &

cd /home/sanjay.g/projects/rdmerge || exit 1
LOG=logs/campaign_keeper.log
Q=/opt/pbs/bin/qstat
QSUB=/opt/pbs/bin/qsub
MAXJOBS=3
mkdir -p logs
echo $$ > logs/campaign_keeper.pid

log() { echo "$(date '+%F %T') $*" >> $LOG; }
log "campaign keeper started pid $$ (cap ${MAXJOBS} concurrent)"

# name | pbs jobname | script | glob of done-files | target count
STAGES=(
  "w1|rdm_w1a|pbs_w1_alpha_sweep.sh|results/phase3/eval_w1_alpha/*.json|52"
  "w5|rdm_w5rs|pbs_w5_rescore.sh|results/phase3/eval_downstream_v2/*.json|144"
  "a1tr|rdm_a1tr|pbs_a1_indep_train.sh|results/phase3/lora_train/*indep1*.json|16"
  "a2|rdm_a2kt|pbs_a2_knots_ties.sh|results/phase3/eval_a2_knots_ties/*.json|12"
  "w3|rdm_w3rt|pbs_w3_rate_sweep.sh|results/phase3/eval_w3_rate/*.json|24"
  "a1ev|rdm_a1ev|pbs_a1_indep_matrix.sh|results/phase3/eval_a1_indep/*.json|28"
)

count() { ls $1 2>/dev/null | wc -l; }

for i in $(seq 1 400); do          # 400 x 5 min = ~33 h
  if [ -f _CAMPAIGN_KEEPER_STOP ]; then
    log "stop sentinel found -> exiting"; rm -f _CAMPAIGN_KEEPER_STOP; exit 0
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

    # already in the queue? nothing to do for this stage
    echo "$qout" | grep -q "$job" && continue
    # at the cap? wait for a slot
    [ "$running" -ge "$MAXJOBS" ] && continue
    # a1ev is gated on a1tr finishing all 16 adapters
    if [ "$name" = "a1ev" ]; then
      [ "$(count 'results/phase3/lora_train/*indep1*.json')" -ge 16 ] || continue
    fi

    jid=$($QSUB "code/phase3/scripts/$script" 2>>$LOG)
    log "$name at $n/$target and no $job in queue -> submitted $jid"
    running=$((running+1))
    sleep 20
    qout=$($Q -u sanjay.g 2>/dev/null)
  done

  # log a heartbeat every 6th tick (~30 min) to keep the log readable
  [ $((i % 6)) -eq 1 ] && log "tick $i running=$running |$summary"

  if [ "$alldone" -eq 1 ]; then
    log "ALL STAGES COMPLETE |$summary -> exiting"
    exit 0
  fi
  sleep 300
done
log "keeper hit its iteration limit; exiting"
