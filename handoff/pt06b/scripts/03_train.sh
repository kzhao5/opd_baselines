#!/usr/bin/env bash
# Step 3: train ONE baseline to step 139, resuming automatically after any interruption.
#   bash scripts/03_train.sh <method> [GPUS]
#   method: sft seqkd grpo opd trd fastopd1024 fastopd2048 fastopd4096 fastopd8192 skd relayopd
#   GPUS:   three comma-separated indices, default 0,1,2 (2 actor + 1 teacher)
#
# The script relaunches verl until latest_checkpointed_iteration reaches 139. verl itself resumes from
# the last checkpoint (trainer.resume_mode=auto), so a crash costs at most the steps since the last save.
# If three consecutive launches fail to advance the step counter, the script stops and says so rather
# than burning GPU time in a loop.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
METHOD=${1:?usage: 03_train.sh <method> [GPUS]}
GPUS=${2:-0,1,2}
NG=$(tr ',' ' ' <<< "$GPUS" | wc -w)

case "$METHOD" in
  sft)          SCRIPT=opd/scripts/baselines/sft.sh            ; TDATA="$TEACHER_PARQUET" ;;
  seqkd)        SCRIPT=opd/scripts/baselines/seqkd.sh          ; TDATA="$TEACHER_PARQUET" ;;
  trd)          SCRIPT=opd/scripts/baselines/trd.sh            ; TDATA="$TRD_PARQUET"     ;;
  grpo)         SCRIPT=opd/scripts/baselines/grpo.sh           ; TDATA="$DAPO_PARQUET"    ;;
  opd)          SCRIPT=opd/scripts/baselines/opd.sh            ; TDATA="$DAPO_PARQUET"    ;;
  fastopd1024)  SCRIPT=opd/scripts/baselines/fastopd/1024.sh   ; TDATA="$DAPO_PARQUET"    ;;
  fastopd2048)  SCRIPT=opd/scripts/baselines/fastopd/2048.sh   ; TDATA="$DAPO_PARQUET"    ;;
  fastopd4096)  SCRIPT=opd/scripts/baselines/fastopd/4096.sh   ; TDATA="$DAPO_PARQUET"    ;;
  fastopd8192)  SCRIPT=opd/scripts/baselines/fastopd/8192.sh   ; TDATA="$DAPO_PARQUET"    ;;
  skd)          SCRIPT=opd/scripts/baselines/skd.sh            ; TDATA="$DAPO_PARQUET"    ;;
  relayopd)     SCRIPT=opd/scripts/relay_opd/train.sh          ; TDATA="$DAPO_PARQUET"    ;;
  *) die "unknown method '$METHOD' (see RUN_MATRIX.tsv)" ;;
esac

RUN="${METHOD}_pt06b"
# SMOKE=1 trains two steps into a throw-away directory: it proves the stack works without touching the
# real run or its checkpoints.
if [ "${SMOKE:-0}" = 1 ]; then
  RUN="${RUN}_smoke"
  export TOTAL_TRAINING_STEPS=2 SAVE_FREQ=1 TARGET_STEP=1
  log "SMOKE MODE: 2 steps into $CKPT_DIR/$RUN (delete it afterwards)"
fi
export OUTPUT_DIR="$CKPT_DIR/$RUN"
export EXP_ID="$RUN"
export TRAIN_DATA="$TDATA"
export METHOD_SCRIPT="$SCRIPT"
LOG="$LOG_DIR/$RUN.log"
mkdir -p "$OUTPUT_DIR"

[ -s "$TRAIN_DATA" ] || die "$METHOD trains on $TRAIN_DATA which does not exist -- run 02_offline_data.sh first"
[ -f "$STUDENT_MODEL/config.json" ] || die "student model missing: $STUDENT_MODEL (run 01_fetch_models_data.sh)"
[ -f "$TEACHER_MODEL/config.json" ] || die "teacher model missing: $TEACHER_MODEL (run 01_fetch_models_data.sh)"

# SFT and GRPO have no teacher engine, so they use every card in the slot; the distillation methods
# split the slot into 2 actor GPUs + 1 teacher GPU.
case "$METHOD" in
  sft)  export NUM_GPUS=$NG ;;
  grpo) export N_GPUS=$NG   ;;
  *)    export ACTOR_GPUS_PER_NODE=2 TEACHER_GPUS_PER_NODE=$((NG-2)) ;;
esac
# verl's SFT trainer runs with resume_mode=disable: a second launch would restart from step 0 and
# overwrite the run. It therefore gets exactly one attempt and must finish it.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-40}; [ "$METHOD" = sft ] && MAX_ATTEMPTS=1
[ "${SMOKE:-0}" = 1 ] && MAX_ATTEMPTS=1

activate_env relay-opd
export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}"
export VERL_OPD_DIR="$VERL_DIR"

stall=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  before=$(latest_step "$OUTPUT_DIR")
  if [ "${before:-0}" -ge "$TARGET_STEP" ]; then
    log "$RUN: already at step $before >= $TARGET_STEP -- nothing to do"; break
  fi
  require_gpus "$GPUS" "$NG"
  export CUDA_VISIBLE_DEVICES=$GPUS
  log "$RUN attempt $attempt: resuming from step $before on GPUs $GPUS (log: $LOG)"
  ( cd "$VERL_DIR" && bash "$SCRIPT" ) >>"$LOG" 2>&1
  rc=$?
  after=$(latest_step "$OUTPUT_DIR")
  log "$RUN attempt $attempt finished rc=$rc, step $before -> $after"
  # vLLM EngineCore children survive a killed parent and hold the cards; clear them before retrying.
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 20
  if [ "${after:-0}" -ge "$TARGET_STEP" ]; then log "$RUN reached step $after"; break; fi
  if [ "${after:-0}" -le "${before:-0}" ]; then
    stall=$((stall+1))
    [ "$stall" -ge 3 ] && die "$RUN made no progress in 3 consecutive attempts (still at step $after). Read the tail of $LOG and see HANDBOOK.md -> Troubleshooting."
  else
    stall=0
  fi
done

final=$(latest_step "$OUTPUT_DIR")
[ "${final:-0}" -ge "$TARGET_STEP" ] || die "$RUN stopped at step $final of $TARGET_STEP"
log "=== $RUN COMPLETE at step $final ==="
ls -d "$OUTPUT_DIR"/global_step_* 2>/dev/null | sed 's/.*global_step_/  checkpoint step /'
