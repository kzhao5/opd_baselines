# Post-trained 0.6B baselines — run handbook

**Read this file top to bottom before running anything.** It is written so that an agent with shell
access can execute the whole campaign unattended, decide when something has gone wrong, and know what to
hand back. Every command is idempotent: after a crash, a reboot or an OOM, re-run the same command and it
continues from what is on disk.

---

## 1. What you are producing

Eleven distillation/RL **baselines** for one student model, each trained for exactly one epoch
(139 optimizer steps) and evaluated on four math benchmarks:

| | |
|---|---|
| Student | `Qwen/Qwen3-0.6B` — the **post-trained** (instruction-tuned, non-thinking) model, **not** `Qwen3-0.6B-Base` |
| Teacher | `Qwen/Qwen3-4B-Instruct-2507` |
| Training set | DAPO-Math-17k, 17 917 rows, shipped in `data/` (verl format) |
| Baselines | SFT, SeqKD, GRPO, OPD, TRD, FastOPD@{1024,2048,4096,8192}, SKD, RelayOPD |
| Evaluation | AIME24, AIME25, AMC23, MATH500 at each saved step; MMLU-Pro, GPQA-Diamond, HumanEval+, LiveCodeBench at the best step |

These numbers go into a table next to runs we already produced for other students, so **the protocol in
section 4 is frozen**. A run that deviates from it cannot be used, no matter how good it looks.

## 2. Hardware this assumes

One node, **6 × L40S (48 GB)**, split into two independent slots of three cards:

```
slot A = GPU 0,1,2      slot B = GPU 3,4,5
         │ │ └── teacher engine (Qwen3-4B, vLLM, TP=1)
         └─┴──── actor: FSDP training + vLLM rollout (2 GPUs)
```

SFT and GRPO have no teacher engine and use all three cards of their slot. Two runs train at once; more
than that will OOM or thrash.

**Disk: budget 1 TB.** A 0.6B checkpoint with optimizer state is 9.5 GB, seven of them per run, 67 GB per
run, ~740 GB for all eleven. `scripts/07_prune_checkpoints.sh` brings a finished run down to ~18 GB by
deleting the shard files of checkpoints the run has already passed; run it as you go.

**Time: 12–17 days** for all eleven with both slots busy. See `RUN_MATRIX.tsv` for per-run estimates
measured on the same methods and the same student size. If you only have a few days, run the first six
rows of that file (~4–5 days) and say so.

## 3. Quick start

```bash
cd <this repo>/handoff/pt06b

bash scripts/00_setup_env.sh                 # conda envs + machine check           (~30 min)
bash scripts/01_fetch_models_data.sh         # two models from HF, data is shipped  (~15 min)

# Offline trajectories: needed only by SFT, SeqKD (teacher) and TRD (trd). Skip if you skip those.
bash scripts/02_offline_data.sh teacher 0,1,2   # ~10 h
bash scripts/02_offline_data.sh trd     3,4,5   # ~20 h   (can run at the same time as the line above)

# Smoke test before committing days of GPU time -- see section 5.
SMOKE=1 bash scripts/03_train.sh fastopd1024 0,1,2

# The campaign: two slots, cheapest runs first, training then math eval, restartable.
bash scripts/run_all.sh                      # run it under tmux/screen

bash scripts/status.sh                       # progress table, safe to run any time
```

## 4. The frozen protocol

`scripts/lib.sh` sets all of this. **Do not change any of it**; if a card cannot fit something, change
the memory knobs in section 7, never the protocol.

| Setting | Value | Why it matters |
|---|---|---|
| `TRAIN_BATCH_SIZE` / `PPO_MINI_BATCH_SIZE` | 128 | 17 917 rows / 128 = 139 steps = one epoch |
| `TOTAL_EPOCHS` | 1 | the table is defined at one epoch |
| `TARGET_STEP` | 139 | a run that stops earlier is incomplete |
| `SAVE_FREQ` | 20 | checkpoints at 20, 40, 60, 80, 100, 120, 139 |
| `MAX_PROMPT_LENGTH` | 2048 | |
| `MAX_RESPONSE_LENGTH` | 16384 | FastOPD@N overrides this with its own budget N — that is the method |
| terminal tokens | `151645` + `151643` | the post-trained student declares both in `generation_config.json`, so rollouts stop on either; evaluation passes them explicitly via `EVAL_STOP_TOKEN_IDS` |
| `TEST_FREQ` | -1 | no in-training validation; evaluation is a separate step |
| math eval | 8 samples, temp 1.0, top_p 1.0, 16384 new tokens, seed 42 | the reported metric is **avg@k** (mean accuracy over the 8 samples), pooled over shards — not `pass@1`, which the same summary file also contains |
| sci/code eval | greedy, 1 sample, 16384 new tokens, seed 42 | |

One documented deviation from the A100 runs: `VAL_MAX_RESPONSE_LENGTH` is 16384 instead of 32768. It only
sets the inference engine's context window (validation is off), and 32768 does not fit comfortably in
48 GB next to training. Keep it at 16384 and mention it in your report.

## 5. Smoke test before the real thing

Do not start a 100-hour run on an untested stack.

```bash
SMOKE_ROWS=64 bash scripts/02_offline_data.sh teacher 0,1,2   # tiny trajectory file
SMOKE=1 bash scripts/03_train.sh fastopd1024 0,1,2            # stops after a few steps
```

`SMOKE=1` trains two steps into `work_pt06b/checkpoints/fastopd1024_pt06b_smoke`, a throw-away
directory that the real run never reads. It passes when verl starts, the teacher engine answers, and
`.../fastopd1024_pt06b_smoke/latest_checkpointed_iteration.txt` appears. Delete the `_smoke` directory
afterwards; it is worth ~10 GB of disk.

## 6. What each script does, and how to tell it worked

| Script | Produces | Passed when |
|---|---|---|
| `00_setup_env.sh` | conda envs `relay-opd`, `codeeval` | prints torch/vllm versions and compute capability ≥ 8.0 |
| `01_fetch_models_data.sh` | `work_pt06b/models/*`, `work_pt06b/data/dapo-math-17k.parquet` | prints `train: 17917 rows` |
| `02_offline_data.sh <mode>` | `teacher_traj_pt06b/` or `trd_pt06b/` parquet | prints `[check] rows ... eos_rate ...` with rows > 0 |
| `03_train.sh <method>` | `checkpoints/<method>_pt06b/global_step_*` | prints `=== <run> COMPLETE at step 139 ===` |
| `04_eval_math.sh <method>` | `eval_out/<method>_pt06b_b16k/step_*/shard_*/{aime24,aime25,amc23,math500}.summary.json` | `status.sh` shows all seven steps |
| `05_eval_sci_code.sh <method> <step>` | `eval_out/<method>_pt06b_<bench>/step_<step>/` | four benches present |
| `06_score_heplus.sh <method> <step>` | `heplus_score.json` | prints HumanEval / HumanEval+ pass@1 |
| `07_prune_checkpoints.sh` | frees disk | `DRY=1` first to see what goes |

`status.sh` is the single source of truth for progress. Example:

```
method       train    math evaluated                peak   Avg4    sci/code at peak
sft          DONE     20,40,60,80,100,120,139       139    18.42   mmlu:Y gpqa:Y humanevalplus:. lcb:.
fastopd1024  @60      20,40                         40     15.10   -
```

## 7. Troubleshooting — every item here is a failure we actually hit

**A run stops making progress.** `03_train.sh` aborts after three attempts that do not advance the step
counter, and prints the log path. Read the last 100 lines of `work_pt06b/logs/<run>.log` first.

**GPUs look busy but nothing is running.** vLLM `EngineCore` children survive a killed parent and keep
the memory. The scripts clear them between attempts; to do it by hand:
`pkill -9 -f "VLLM::EngineCore"`, then check `nvidia-smi`.

**verl refuses to start, complaining about device lists.** Some clusters export `ROCR_VISIBLE_DEVICES`
next to `CUDA_VISIBLE_DEVICES`. `lib.sh` unsets it; if you launch verl by hand, unset it yourself.

**Import errors from OpenSSL / cv2 on a hardened host.** Set `OPENSSL_CONF=/dev/null` (lib.sh does).

**CUDA OOM during rollout or the optimizer step.** Lower these, in this order — they change speed, not
results: `ROLLOUT_GPU_MEMORY_UTILIZATION` (0.80 → 0.70), `ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU` (16384 →
8192), `TEACHER_MAX_NUM_BATCHED_TOKENS` (4096 → 2048). SKD is the tightest run: it also uses
`ACTOR_ULYSSES_SEQUENCE_PARALLEL_SIZE=2` and needs both actor cards.

**A long compile then a fallback message about FlashInfer.** Expected and harmless: its JIT wants a
matching `nvcc`, which sm89 boxes usually do not have. It is disabled by default here.

**`Disk quota exceeded` from inside HuggingFace `datasets`.** The dataset cache defaults to `$HOME`.
Point it somewhere with room *before* starting: `export HF_HOME=/big/disk/hf`.

**SFT restarts from step 0.** verl's SFT trainer runs with `resume_mode=disable`, so it gets exactly one
attempt (`MAX_ATTEMPTS=1` in `03_train.sh`). If it dies, delete its output directory and run it again
from scratch — do not let it half-resume.

**Offline generation restarted with a different GPU count.** Trajectory generation is resumable only with
the same number of shards. Use the same GPU list you started with, or delete the partial output.

## 8. Rules

1. Never edit the protocol constants in `lib.sh`.
2. Never delete a `global_step_*/actor/huggingface` directory — that is the deliverable. `07_prune_checkpoints.sh` only removes resume shards.
3. If a run fails three times, stop it and report; do not "fix" it by lowering the step count, shrinking the batch, or switching the student model.
4. Do not upgrade `torch`, `vllm` or `transformers`. The pins in `env/` are what these results were produced with.
5. When in doubt, run `scripts/status.sh` and send that table with your question.

## 9. What is in this branch

The training code is the **upstream** Relay-OPD tree: the baselines under
`relay-opd/opd/scripts/baselines/` are the authors' own scripts, unmodified. Four files carry fixes we
made while reproducing the paper, and they are the only code changes here:

| File | Fix |
|---|---|
| `opd/eval/math_benchmarks.py` | multiple-choice and code benchmarks, `EVAL_STOP_TOKEN_IDS`, per-shard summaries |
| `opd/scripts/evaluation/math.sh` | data-parallel sharding, resume, engine cleanup |
| `opd/data/generate_teacher_trajectories.py` | honour the stop-token list when generating |
| `opd/data/generate_trd_trajectories.py` | same |

## 10. When you are done

Follow `REPORT_BACK.md`. Short version: send the eval summaries, the training logs and the
`huggingface/` export of each run's best step — not the whole checkpoint directory.
