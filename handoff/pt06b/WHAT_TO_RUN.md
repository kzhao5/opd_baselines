# What we are asking you to run

This is the *assignment*: which runs matter, in what order, and what "finished" means.
For the mechanics — commands, environment, troubleshooting — see [HANDBOOK.md](HANDBOOK.md).

## The gap you are filling

We are comparing on-policy distillation methods across four student configurations. Three of them are
done or in progress on our cluster. The fourth — the **post-trained Qwen3-0.6B** student — has nothing:
no baseline has ever been trained for it. That is the column you own, end to end.

Every run uses the same teacher (`Qwen3-4B-Instruct-2507`), the same data (DAPO-Math-17k), and the same
budget (one epoch = 139 optimizer steps, responses capped at 16 384 tokens). The only thing that varies
between runs is the distillation method.

## Priorities

Work top-down. Each tier is useful on its own, so stopping at the end of a tier is a perfectly good
outcome — stopping in the middle of one is much less useful.

### Phase 0 — offline trajectories (~1 day, do first)

SFT, SeqKD and TRD train on pre-generated trajectories, so these two jobs gate three runs. Run them on
the two slots at the same time:

```bash
bash scripts/02_offline_data.sh teacher 0,1,2    # ~10 h   -> SFT, SeqKD
bash scripts/02_offline_data.sh trd     3,4,5    # ~20 h   -> TRD
```

### Phase 1 — the six table rows that are affordable (~5–7 days)

These are rows of the paper table. Cheapest first, so the table fills in steadily:

| order | method | expected on one 3-GPU slot |
|---|---|---|
| 1 | `sft` | 10–14 h |
| 2 | `relayopd` | 40–56 h |
| 3 | `trd` | 44–61 h |
| 4 | `seqkd` | 48–66 h |
| 5 | `fastopd8192` | 52–73 h |
| 6 | `grpo` | 56–79 h |

```bash
ORDER="sft relayopd trd seqkd fastopd8192 grpo" bash scripts/run_all.sh
```

`run_all.sh` trains and then evaluates each run, keeps both slots busy, and resumes wherever it stopped.

### Phase 2 — the two expensive table rows (~6–8 days)

`opd` (125–175 h) and `skd` (143–200 h). Together they cost as much as all of phase 1. Start them only
once phase 1 is complete; they are the two runs most likely to be worth doing on our A100s instead, so
tell us before you commit a week of GPU time to them.

### Phase 3 — the FastOPD budget ablation (~1.5–2 days)

`fastopd1024`, `fastopd2048`, `fastopd4096`. These are appendix numbers, not main-table rows. Lowest
priority, but they are cheap and they resume well, so they are good filler for a slot that is free while
a long run occupies the other.

### Phase 4 — science and code benchmarks (optional, ~1 day per run)

MMLU-Pro, GPQA-Diamond, HumanEval+ and LiveCodeBench, at each run's best step:

```bash
bash scripts/05_eval_sci_code.sh <method> <peak-step> 0,1,2
```

`status.sh` prints the peak step. MMLU-Pro dominates the cost. Do this only for runs you have already
finished, and only if a slot would otherwise be idle — **if you would rather send us the checkpoints, we
will run these ourselves.** Say which you prefer.

## What "finished" means for one run

A run counts as delivered when all four of these hold:

1. `latest_checkpointed_iteration.txt` reads **139** — not 120, not "close enough";
2. the math benchmarks are evaluated at every saved step (20, 40, 60, 80, 100, 120, 139), so we can pick
   the peak the same way we did for the other students;
3. `scripts/status.sh` shows the run as `DONE` with seven evaluated steps and a peak step;
4. you have sent the files listed in [REPORT_BACK.md](REPORT_BACK.md).

Send each run as it finishes. Do not wait for the whole set.

## Keeping us in the loop

Once a week, or whenever a run finishes, send the output of:

```bash
bash scripts/status.sh
```

That table is enough for us to see where you are. If something is broken, send it together with the last
100 lines of the run's log — that pair answers most questions immediately.

## Things that would waste your time

- **Training a different student.** It must be `Qwen/Qwen3-0.6B`, the post-trained model. `Qwen3-0.6B-Base` is a different column of our table and we already have it.
- **Changing the protocol** to make a run fit — batch size, step count, response budget, sampling. A run with different settings cannot go in the table. Memory pressure has its own knobs; see HANDBOOK.md §7.
- **Starting with `opd` or `skd`** because they look most important. They are the most expensive by a wide margin and phase 1 delivers six rows for the same GPU time.
- **Deleting checkpoints to save disk.** Use `scripts/07_prune_checkpoints.sh`; it keeps every `huggingface/` export, which is what we actually need.
- **Silently working around a failure.** A documented deviation is usable. An undocumented one costs us the run, and we only find out when the numbers do not line up with the other three students.
