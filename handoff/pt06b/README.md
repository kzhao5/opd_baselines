# Post-trained 0.6B baseline handoff

Everything needed to train and evaluate the eleven baselines of the **post-trained Qwen3-0.6B** student
on a single node with 6 × L40S.

Two documents, in this order:

1. **[WHAT_TO_RUN.md](WHAT_TO_RUN.md)** — the assignment: which runs matter, in what priority, and what counts as finished.
2. **[HANDBOOK.md](HANDBOOK.md)** — the mechanics, written to be executed end to end, including by an agent.

```
WHAT_TO_RUN.md     the assignment: priorities, phases, definition of done
HANDBOOK.md        how to run it, with an acceptance check per step
RUN_MATRIX.tsv     the eleven runs, their scripts, and measured cost
REPORT_BACK.md     what to send back and in what form
scripts/           00_setup_env … 07_prune_checkpoints, status.sh, run_all.sh
bench/             benchmark parquets (math + science + code)
data/              DAPO-Math-17k in verl format
env/               exact package pins for both conda environments
```

The training code itself is the repository this directory lives in: `relay-opd/` is the patched
verl/OPD tree, and the baselines are its own scripts under `relay-opd/opd/scripts/baselines/`.
