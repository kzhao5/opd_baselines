#!/usr/bin/env bash
# Step 6 (optional, CPU only): score the HumanEval+ generations.
#   bash scripts/06_score_heplus.sh <method> <step>
# LiveCodeBench is scored on our side -- just send the generations back (see REPORT_BACK.md).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
METHOD=${1:?usage: 06_score_heplus.sh <method> <step>}; STEP=${2:?step required}
RUN="${METHOD}_pt06b"
activate_env codeeval
python "$HANDOFF_DIR/scripts/score_heplus.py" "$EVAL_DIR" "$RUN" "$STEP"
