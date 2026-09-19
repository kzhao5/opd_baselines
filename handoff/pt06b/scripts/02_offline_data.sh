#!/usr/bin/env bash
# Step 2: generate the offline trajectories three baselines train on.
#   bash scripts/02_offline_data.sh teacher [GPUS]   # -> SFT and SeqKD   (~10 h on 3 L40S)
#   bash scripts/02_offline_data.sh trd     [GPUS]   # -> TRD             (~20 h on 3 L40S)
# Both are resumable only with the SAME number of GPUs, so pass the same GPUS on a retry.
# Skip this step entirely if you only run the online methods (GRPO / OPD / FastOPD / SKD / RelayOPD).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
MODE=${1:?usage: 02_offline_data.sh teacher|trd [GPUS]}
GPUS=${2:-0,1,2}
case "$MODE" in teacher|trd) ;; *) die "MODE must be 'teacher' or 'trd'" ;; esac

case "$MODE" in
  teacher) OUT="$TEACHER_PARQUET" ;;
  trd)     OUT="$TRD_PARQUET" ;;
esac
if [ -s "$OUT" ]; then log "$MODE trajectories already present: $OUT"; exit 0; fi

activate_env relay-opd
require_gpus "$GPUS" "$(tr ',' ' ' <<< "$GPUS" | wc -w)"
export CUDA_VISIBLE_DEVICES=$GPUS
# The generators address cards by position inside CUDA_VISIBLE_DEVICES, one worker group per card.
NG=$(tr ',' ' ' <<< "$GPUS" | wc -w)
GROUPS=$(seq 0 $((NG-1)) | paste -sd';')
mkdir -p "$(dirname "$OUT")"

# Both students of this project declare only <|endoftext|> as EOS; the table's rollouts and evals also
# stop at <|im_end|>, so the generated trajectories must use the same pair.
export GEN_STOP_TOKEN_IDS='151643;151645'
export MAX_NEW=${MAX_NEW:-16384}
export SMOKE_ROWS=${SMOKE_ROWS:-}

SOURCE="$DAPO_PARQUET"
if [ -n "$SMOKE_ROWS" ]; then
  SOURCE="$(dirname "$OUT")/source_smoke_${SMOKE_ROWS}.parquet"
  python -c "import pandas as pd; pd.read_parquet('$DAPO_PARQUET').head($SMOKE_ROWS).to_parquet('$SOURCE')"
  log "SMOKE MODE: $SMOKE_ROWS rows -> $SOURCE"
fi

cd "$VERL_DIR"
log "generating $MODE trajectories: student=$STUDENT_MODEL teacher=$TEACHER_MODEL groups=$GROUPS out=$OUT"
if [ "$MODE" = teacher ]; then
  STUDENT_MODEL=$STUDENT_MODEL TEACHER_MODEL=$TEACHER_MODEL SOURCE_DATA=$SOURCE OUTPUT_PARQUET=$OUT \
    GPU_GROUPS="$GROUPS" bash opd/scripts/data/teacher_trajectories.sh 2>&1 | tee -a "$LOG_DIR/data_$MODE.log"
else
  STUDENT_MODEL=$STUDENT_MODEL TEACHER_MODEL=$TEACHER_MODEL SOURCE_DATA=$SOURCE OUTPUT_PARQUET=$OUT \
    STUDENT_GPU_GROUPS="$GROUPS" REWRITE_GPU_GROUPS="$GROUPS" \
    bash opd/scripts/data/trd_trajectories.sh 2>&1 | tee -a "$LOG_DIR/data_$MODE.log"
fi

[ -s "$OUT" ] || die "$MODE generation produced no parquet -- see $LOG_DIR/data_$MODE.log"
python - <<PY
import pandas as pd
d = pd.read_parquet("$OUT")
print("[check] rows", len(d), "eos_rate", round(d.eos_reached.mean(), 4), "gen_len mean", round(d.gen_len.mean(), 1))
assert len(d) > 0, "empty parquet"
PY
log "$MODE trajectories done: $OUT"
