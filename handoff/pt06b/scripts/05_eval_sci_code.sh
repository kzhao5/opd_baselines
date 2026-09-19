#!/usr/bin/env bash
# Step 5: science / code generation for ONE run at ONE step (normally the peak step).
#   bash scripts/05_eval_sci_code.sh <method> <step> [GPUS]
#
# Four benchmarks, greedy decoding, 1 sample, 16384 new tokens, max_model_len 20480, seed 42:
#   mmlu_pro  gpqa_diamond   -> scored by the eval script itself (summary.json holds pass@1)
#   humanevalplus  lcb       -> generations only; see 06_score_heplus.sh and REPORT_BACK.md
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
METHOD=${1:?usage: 05_eval_sci_code.sh <method> <step> [GPUS]}
STEP=${2:?step is required (use the peak step from status.sh)}
GPUS=${3:-0,1,2}
NG=$(tr ',' ' ' <<< "$GPUS" | wc -w)
RUN="${METHOD}_pt06b"
MODEL=$(hf_dir "$CKPT_DIR/$RUN" "$STEP") || die "no checkpoint for $RUN@$STEP"

activate_env relay-opd
export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}" VERL_OPD_DIR="$VERL_DIR"
export HF_HUB_OFFLINE=1

for bench in mmlu_pro gpqa_diamond humanevalplus lcb; do
  d="$EVAL_DIR/${RUN}_${bench}/step_${STEP}"
  if [ -f "$d/$bench.summary.json" ] || [ -n "$(ls "$d"/shard_*/$bench.summary.json 2>/dev/null)" ]; then
    log "$RUN@$STEP $bench: already generated, skipping"; continue
  fi
  require_gpus "$GPUS" "$NG"
  export CUDA_VISIBLE_DEVICES=$GPUS
  log "generating $bench for $RUN@$STEP"
  ( cd "$VERL_DIR" && \
    RUN_NAME="${RUN}_${bench}" STEP="$STEP" MODEL="$MODEL" \
    DATA_DIR="$BENCH_DIR" OUT_ROOT="$EVAL_DIR" BENCHES="$bench" \
    N_SAMPLES=1 TEMPERATURE=0.0 TOP_P=1.0 MAX_NEW=16384 MAX_MODEL_LEN=20480 \
    DP_SIZE=$NG TP=1 SEED=42 EVAL_STOP_TOKEN_IDS='151643;151645' \
    bash opd/scripts/evaluation/math.sh ) >>"$LOG_DIR/scicode_${RUN}.log" 2>&1
  pkill -9 -u "$(id -u)" -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 15
  log "$bench done -> $d"
done
log "science/code generation finished for $RUN@$STEP"
