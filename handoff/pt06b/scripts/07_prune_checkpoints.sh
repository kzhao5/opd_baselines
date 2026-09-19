#!/usr/bin/env bash
# Step 7 (disk hygiene): drop the optimizer/model shards of checkpoints the run has already passed.
#   bash scripts/07_prune_checkpoints.sh [method ...]     # default: every finished run
#   DRY=1 bash scripts/07_prune_checkpoints.sh            # list what would be deleted
#
# verl saves [model, optimizer, extra, hf_model] at every save point: 9.5 GB per checkpoint for a 0.6B
# student, 67 GB per run, ~740 GB for all eleven. Only the LATEST checkpoint needs the .pt shards (that is
# what resume reads); evaluation only ever reads global_step_N/actor/huggingface, which is 1.2 GB.
# Pruning brings a finished run down to about 18 GB and cannot affect a run that is still training,
# because the newest checkpoint is always kept intact.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
DRY=${DRY:-0}
METHODS=${*:-"sft seqkd grpo opd trd fastopd1024 fastopd2048 fastopd4096 fastopd8192 skd relayopd"}
freed=0
for m in $METHODS; do
  out="$CKPT_DIR/${m}_pt06b"; [ -d "$out" ] || continue
  latest=$(latest_step "$out")
  for d in "$out"/global_step_*; do
    [ -d "$d" ] || continue
    st=${d##*global_step_}
    [ "$st" = "$latest" ] && { log "keep  ${m}@$st (latest, needed to resume)"; continue; }
    [ -f "$d/actor/huggingface/config.json" ] || [ -f "$d/huggingface/config.json" ] || {
      log "SKIP  ${m}@$st: no huggingface export here, refusing to prune"; continue; }
    sz=$(du -sm "$d" 2>/dev/null | cut -f1)
    n=$(find "$d" -name "*.pt" | wc -l)
    [ "$n" -eq 0 ] && continue
    if [ "$DRY" = 1 ]; then
      log "would prune ${m}@$st: $n shard files (~${sz} MB -> keeps huggingface/)"
    else
      find "$d" -name "*.pt" -delete
      log "pruned ${m}@$st: $n shard files removed"
      freed=$((freed + sz))
    fi
  done
done
[ "$DRY" = 1 ] || log "freed roughly ${freed} MB"
df -h "$CKPT_DIR" | tail -1
