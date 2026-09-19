# Post-trained 0.6B baseline handoff

Everything needed to train and evaluate the eleven baselines of the **post-trained Qwen3-0.6B** student
on a single node with 6 × L40S.

Start with **[HANDBOOK.md](HANDBOOK.md)** — it is written to be executed by an agent, end to end.

```
HANDBOOK.md        what to run, in order, with an acceptance check per step
RUN_MATRIX.tsv     the eleven runs, their scripts, and measured cost
REPORT_BACK.md     what to send back and in what form
scripts/           00_setup_env … 07_prune_checkpoints, status.sh, run_all.sh
bench/             benchmark parquets (math + science + code)
data/              DAPO-Math-17k in verl format
env/               exact package pins for both conda environments
```

The training code itself is the repository this directory lives in: `relay-opd/` is the patched
verl/OPD tree, and the baselines are its own scripts under `relay-opd/opd/scripts/baselines/`.
