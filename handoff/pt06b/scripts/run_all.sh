#!/usr/bin/env bash
# Drive the whole campaign on 6 GPUs: two slots of three cards, each taking the next unfinished run.
#   bash scripts/run_all.sh                 # train everything in the recommended order, then evaluate
#   ORDER="sft seqkd fastopd1024" bash scripts/run_all.sh
#   SLOT_A=0,1,2 SLOT_B=3,4,5 bash scripts/run_all.sh
#
# Runs in the foreground and prints what it is doing; use tmux/screen. Re-running after any interruption
# picks up exactly where it stopped, because every step below is idempotent.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
SLOT_A=${SLOT_A:-0,1,2}
SLOT_B=${SLOT_B:-3,4,5}
# Cheapest first: the table fills in gradually and an early mistake costs little.
ORDER=${ORDER:-"sft fastopd1024 seqkd fastopd2048 trd fastopd4096 grpo fastopd8192 opd relayopd skd"}
QUEUE="$WORK/.queue"; : > "$QUEUE"
for m in $ORDER; do echo "$m" >> "$QUEUE"; done
LOCKS="$WORK/.locks"; mkdir -p "$LOCKS"

worker() {  # worker <slot-name> <gpus>
  local slot=$1 gpus=$2 m
  while true; do
    m=""
    for cand in $(cat "$QUEUE"); do
      if mkdir "$LOCKS/$cand" 2>/dev/null; then m=$cand; break; fi
    done
    [ -z "$m" ] && { log "[$slot] queue empty, worker exits"; return 0; }
    if [ "$(latest_step "$CKPT_DIR/${m}_pt06b")" -ge "$TARGET_STEP" ]; then
      log "[$slot] $m already trained, moving on"
    else
      log "[$slot] training $m on GPUs $gpus"
      bash "$HANDOFF_DIR/scripts/03_train.sh" "$m" "$gpus" || log "[$slot] !!! $m FAILED -- continuing with the next run"
    fi
    if [ "$(latest_step "$CKPT_DIR/${m}_pt06b")" -ge "$TARGET_STEP" ]; then
      log "[$slot] math evaluation of $m on GPUs $gpus"
      bash "$HANDOFF_DIR/scripts/04_eval_math.sh" "$m" "" "$gpus" || log "[$slot] !!! eval of $m failed"
    fi
  done
}

log "slot A = $SLOT_A, slot B = $SLOT_B, order: $ORDER"
worker A "$SLOT_A" & pa=$!
worker B "$SLOT_B" & pb=$!
wait $pa $pb
log "=== all workers finished ==="
bash "$HANDOFF_DIR/scripts/status.sh"
