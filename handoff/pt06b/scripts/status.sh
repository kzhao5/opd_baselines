#!/usr/bin/env bash
# Progress of the whole handoff (safe to run any time, reads only files).
#   bash scripts/status.sh [method ...]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
# Reads files only, so it must work even where conda is not set up.
PY=python3
if find_conda 2>/dev/null && conda activate relay-opd 2>/dev/null; then PY=python; fi
command -v "$PY" >/dev/null 2>&1 || die "no python interpreter found"
"$PY" "$HANDOFF_DIR/scripts/status.py" "$CKPT_DIR" "$EVAL_DIR" "$@"
