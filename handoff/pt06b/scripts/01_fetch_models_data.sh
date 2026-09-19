#!/usr/bin/env bash
# Step 1: fetch the two models and the training set. Re-running skips what is already complete.
#   bash scripts/01_fetch_models_data.sh
# Needs internet. If the machine is offline, copy $MODELS_DIR and $DATA_DIR from another host instead.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
activate_env relay-opd
pip install -q "huggingface_hub[cli]" 2>/dev/null || true

fetch_model() {  # fetch_model <hf repo id> <target dir>
  local repo=$1 dst=$2
  if [ -f "$dst/config.json" ]; then log "model present: $dst"; return 0; fi
  log "downloading $repo -> $dst"
  huggingface-cli download "$repo" --local-dir "$dst" || die "download of $repo failed"
  [ -f "$dst/config.json" ] || die "$dst has no config.json after download"
}

# The STUDENT is the post-trained (instruction-tuned, non-thinking) 0.6B model, NOT Qwen3-0.6B-Base.
fetch_model "Qwen/Qwen3-0.6B"                "$STUDENT_MODEL"
fetch_model "Qwen/Qwen3-4B-Instruct-2507"    "$TEACHER_MODEL"

# The training set and the benchmarks ship with this branch (verl format, 9 MB total), so there is
# nothing to build and no dataset version to get wrong.
if [ ! -s "$DAPO_PARQUET" ]; then
  mkdir -p "$(dirname "$DAPO_PARQUET")"
  cp "$HANDOFF_DIR/data/dapo-math-17k.parquet" "$DAPO_PARQUET" || die "cannot copy the shipped training parquet"
  log "training set installed: $DAPO_PARQUET"
else
  log "training set present: $DAPO_PARQUET"
fi

log "=== benchmark parquets shipped with this branch ==="
ls -1 "$BENCH_DIR" || die "bench directory missing: $BENCH_DIR"
python - <<PY
import pandas as pd, glob, os
for p in sorted(glob.glob(os.path.join("$BENCH_DIR", "*.parquet"))):
    print(f"  {os.path.basename(p):24s} rows={len(pd.read_parquet(p))}")
print("train:", len(pd.read_parquet("$DAPO_PARQUET")), "rows (expected 17917 = 139 steps x batch 128)")
PY
log "models and data ready"
