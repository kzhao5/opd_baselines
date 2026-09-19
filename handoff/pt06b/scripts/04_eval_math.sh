#!/usr/bin/env bash
# Step 4: math evaluation of one finished run (AIME24, AIME25, AMC23, MATH500).
#   bash scripts/04_eval_math.sh <method> [STEPS] [GPUS]
#   STEPS defaults to every saved checkpoint: 20 40 60 80 100 120 139
#
# Protocol (identical to the table we already have): 8 samples, temperature 1.0, top_p 1.0,
# 16384 new tokens, max_model_len 18433, seed 42, stop tokens 151643/151645.
# A (step, bench) pair whose summary already exists is skipped, so re-running costs nothing.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
METHOD=${1:?usage: 04_eval_math.sh <method> [STEPS] [GPUS]}
STEPS=${2:-"20 40 60 80 100 120 139"}
GPUS=${3:-0,1,2}
NG=$(tr ',' ' ' <<< "$GPUS" | wc -w)
RUN="${METHOD}_pt06b"
OUT="$CKPT_DIR/$RUN"
TAG="${RUN}_b16k"

activate_env relay-opd
export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}" VERL_OPD_DIR="$VERL_DIR"
export HF_HUB_OFFLINE=1

for st in $STEPS; do
  MODEL=$(hf_dir "$OUT" "$st") || { log "$RUN@$st: no checkpoint, skipping"; continue; }
  d="$EVAL_DIR/$TAG/step_$st"
  if [ -n "$(ls "$d"/shard_*/math500.summary.json 2>/dev/null)" ]; then
    log "$RUN@$st: already evaluated, skipping"; continue
  fi
  require_gpus "$GPUS" "$NG"
  export CUDA_VISIBLE_DEVICES=$GPUS
  log "evaluating $RUN@$st ($MODEL)"
  ( cd "$VERL_DIR" && \
    RUN_NAME="$TAG" STEP="$st" MODEL="$MODEL" \
    DATA_DIR="$BENCH_DIR" OUT_ROOT="$EVAL_DIR" \
    BENCHES=aime24,aime25,amc23,math500 \
    N_SAMPLES=8 TEMPERATURE=1.0 TOP_P=1.0 MAX_NEW=16384 MAX_MODEL_LEN=18433 \
    DP_SIZE=$NG TP=1 SEED=42 EVAL_STOP_TOKEN_IDS='151643;151645' \
    bash opd/scripts/evaluation/math.sh ) >>"$LOG_DIR/eval_${RUN}.log" 2>&1
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 15
  log "$RUN@$st done -> $d"
done
bash "$HANDOFF_DIR/scripts/status.sh" "$METHOD"
