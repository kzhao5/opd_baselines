#!/usr/bin/env bash
# Shared helpers for the post-trained-0.6B baseline handoff. Source this, do not run it.
#
# Every script in this directory is idempotent: re-running it after a crash, a reboot or an OOM
# continues from what is already on disk and never overwrites a finished artefact.

set -uo pipefail

# ---------------------------------------------------------------- paths
HANDOFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$HANDOFF_DIR/../.." && pwd)"          # the Relay-OPD checkout
VERL_DIR="$REPO_ROOT/relay-opd"                        # the patched verl/opd tree
WORK="${WORK:-$REPO_ROOT/work_pt06b}"                  # everything this handoff writes
export MODELS_DIR="${MODELS_DIR:-$WORK/models}"
export DATA_DIR="${DATA_DIR:-$WORK/data}"
export CKPT_DIR="${CKPT_DIR:-$WORK/checkpoints}"
export EVAL_DIR="${EVAL_DIR:-$WORK/eval_out}"
export LOG_DIR="${LOG_DIR:-$WORK/logs}"
export BENCH_DIR="${BENCH_DIR:-$HANDOFF_DIR/bench}"    # benchmark parquets shipped with this branch
mkdir -p "$WORK" "$MODELS_DIR" "$DATA_DIR" "$CKPT_DIR" "$EVAL_DIR" "$LOG_DIR"

# ---------------------------------------------------------------- models and data
export STUDENT_MODEL="${STUDENT_MODEL:-$MODELS_DIR/Qwen3-0.6B}"                  # POST-TRAINED, not -Base
export TEACHER_MODEL="${TEACHER_MODEL:-$MODELS_DIR/Qwen3-4B-Instruct-2507}"
export DAPO_PARQUET="${DAPO_PARQUET:-$DATA_DIR/dapo-math-17k.parquet}"
export TEACHER_PARQUET="${TEACHER_PARQUET:-$DATA_DIR/teacher_traj_pt06b/teacher_trajectories.parquet}"
export TRD_PARQUET="${TRD_PARQUET:-$DATA_DIR/trd_pt06b/trd_trajectories.parquet}"

# ---------------------------------------------------------------- the frozen protocol
# These are the numbers the base-student table was produced with. Do not change them: a run that
# deviates cannot go in the same table as the runs we already have.
export MAX_PROMPT_LENGTH=2048
export MAX_RESPONSE_LENGTH=16384      # methods with their own budget (FastOPD@N) override this themselves
export VAL_MAX_RESPONSE_LENGTH=16384  # engine context only -- in-training validation is off
export TRAIN_BATCH_SIZE=128
export PPO_MINI_BATCH_SIZE=128
export TOTAL_EPOCHS=1                 # 1 epoch over DAPO-Math-17k == 139 optimizer steps
export TARGET_STEP=139
export SAVE_FREQ=20                   # checkpoints at 20,40,60,80,100,120,139
export TEST_FREQ=-1
export VAL_BEFORE_TRAIN=False
# Terminal tokens: the post-trained student already declares BOTH <|im_end|> (151645) and
# <|endoftext|> (151643) in its generation_config, so rollouts stop correctly with the upstream code and
# nothing has to be overridden here. Evaluation passes them explicitly (EVAL_STOP_TOKEN_IDS) because the
# eval harness builds its own sampling parameters.
export EVAL_STOP_TOKEN_IDS='151643;151645'
export BENCH="$BENCH_DIR"
export MATH_GRADER_PATH="$VERL_DIR/opd/reward/grader"

# ---------------------------------------------------------------- 6x L40S layout
# One training run = 2 actor GPUs + 1 teacher GPU. Six cards therefore host two runs at a time.
# Slot A = GPUs 0,1,2   Slot B = GPUs 3,4,5
export ACTOR_GPUS_PER_NODE="${ACTOR_GPUS_PER_NODE:-2}"
export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-1}"
# 48 GB cards: the upstream defaults (0.85 rollout / 0.45 teacher) were tuned for 80 GB A100s.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
export TEACHER_GPU_MEMORY_UTILIZATION="${TEACHER_GPU_MEMORY_UTILIZATION:-0.80}"
export ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}"

# ---------------------------------------------------------------- runtime environment
# sm89 (L40S) runs the standard wheels. FlashInfer is left off on purpose: its JIT needs a matching
# nvcc, and on a card without it vLLM silently falls back after a long compile.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] FATAL: $*" >&2; exit 1; }

find_conda() {  # put conda on PATH, trying the usual places; returns 1 if there is none
  command -v conda >/dev/null 2>&1 && return 0
  # environment-module clusters
  source /etc/profile.d/lmod.sh 2>/dev/null || true
  module load miniforge3 2>/dev/null || module load anaconda3 2>/dev/null || true
  command -v conda >/dev/null 2>&1 && return 0
  local c
  for c in "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda /usr/local/miniforge3; do
    [ -f "$c/etc/profile.d/conda.sh" ] && { source "$c/etc/profile.d/conda.sh"; return 0; }
  done
  return 1
}

activate_env() {  # activate_env <env-name>
  local name=${1:-relay-opd}
  find_conda || die "conda is not on PATH and was not found in the usual places; see 00_setup_env.sh"
  eval "$(conda shell.bash hook)" 2>/dev/null || true
  conda activate "$name" || die "conda env '$name' missing; run 00_setup_env.sh first"
  # Some clusters ship a FIPS-mode OpenSSL that crashes cv2/vLLM imports.
  export OPENSSL_CONF=/dev/null
  # Slurm/ROCm leftovers make verl refuse to start when both device lists are set.
  unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES 2>/dev/null || true
}

require_gpus() {  # require_gpus <csv> <count> -- every listed card must be idle (<2 GB used)
  local csv=$1 need=$2 free n=0
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2+0<2048{print $1}' | tr '\n' ' ')
  for g in ${csv//,/ }; do [[ " $free " == *" $g "* ]] && n=$((n+1)); done
  [ "$n" -ge "$need" ] || { nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; die "need $need idle GPUs out of ($csv); only $n are idle"; }
}

latest_step() {  # latest_step <output_dir> -- 0 when nothing has been saved yet
  cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0
}

hf_dir() {  # hf_dir <output_dir> <step> -- the HuggingFace export verl writes next to the checkpoint
  local d=$1 s=$2
  for c in "$d/global_step_$s/actor/huggingface" "$d/global_step_$s/huggingface"; do
    [ -f "$c/config.json" ] && { echo "$c"; return 0; }
  done
  return 1
}
