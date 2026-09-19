#!/usr/bin/env bash
# Step 0: build the two conda environments and check the machine.
#   bash scripts/00_setup_env.sh
# Safe to re-run: existing environments are left alone.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

command -v conda >/dev/null 2>&1 || die "conda/miniforge is required and not on PATH"
eval "$(conda shell.bash hook)"

log "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || die "no usable nvidia-smi"
n_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
[ "$n_gpu" -ge 3 ] || die "this handoff assumes at least 3 GPUs (6 to run two slots); found $n_gpu"
log "found $n_gpu GPUs; the run plan assumes 6 x 48 GB (L40S)"

# ---- training / evaluation environment -------------------------------------------------------
if conda env list | grep -qE "^relay-opd\s"; then
  log "conda env 'relay-opd' already exists, skipping"
else
  log "creating conda env 'relay-opd' (python 3.12)"
  conda create -y -n relay-opd python=3.12 || die "env creation failed"
  conda activate relay-opd
  # torch first, then vllm: vllm pins the torch it was built against.
  pip install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" || die "torch install failed"
  pip install "vllm==0.21.0" || die "vllm install failed"
  pip install "transformers==5.15.0" "datasets==5.0.1" "accelerate==1.14.0" "ray==2.57.0" \
              "tensordict==0.10.0" "hydra-core==1.3.5" "omegaconf==2.3.1" "peft==0.20.0" \
              "numpy==1.26.4" "pyarrow==25.0.1" "codetiming==1.4.0" \
              "math-verify==0.9.0" "latex2sympy2_extended==1.11.0" || die "python deps failed"
  pip install -e "$VERL_DIR" || log "WARN: 'pip install -e relay-opd' failed; PYTHONPATH fallback is used by the scripts"
  conda deactivate
fi

# ---- CPU environment used only to score the code benchmarks -----------------------------------
if conda env list | grep -qE "^codeeval\s"; then
  log "conda env 'codeeval' already exists, skipping"
else
  log "creating conda env 'codeeval' (python 3.11, CPU only)"
  conda create -y -n codeeval python=3.11 || die "env creation failed"
  conda activate codeeval
  pip install "evalplus==0.3.1" "datasets==2.21.0" "pyarrow==25.0.1" "Pebble==5.2.2" "multiprocess==0.70.16" \
    || die "codeeval deps failed"
  conda deactivate
fi

log "=== smoke: import torch + vllm on one GPU ==="
activate_env relay-opd
python - <<'PY'
import torch, vllm, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
print("vllm", vllm.__version__, "transformers", transformers.__version__)
cc = torch.cuda.get_device_capability(0)
print("compute capability", cc)
assert cc >= (8, 0), "bf16 training needs Ampere or newer"
PY
log "environment ready"
