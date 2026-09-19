# What to send back

Send results as they finish — do not wait for all eleven runs. A finished run is more useful to us the
day it finishes than a complete set three weeks later.

## 1. After every run (small, send by git)

Push to your branch of this repo, or attach to the issue thread:

```
work_pt06b/eval_out/<method>_pt06b_b16k/step_*/shard_*/*.summary.json      # math, 4 files per step
work_pt06b/eval_out/<method>_pt06b_<bench>/step_*/*.summary.json           # sci/code
work_pt06b/eval_out/<method>_pt06b_<bench>/step_*/heplus_score.json        # if you scored HumanEval+
work_pt06b/logs/<method>_pt06b.log                                         # gzip it
```

Plus the output of `bash scripts/status.sh` as a text file. Collect everything with:

```bash
bash -c 'cd work_pt06b && tar czf ../results_$(date +%F).tgz \
  eval_out/*/step_*/*.summary.json eval_out/*/step_*/shard_*/*.summary.json \
  eval_out/*/step_*/heplus_score.json logs/*.log'
```

That archive is a few MB.

## 2. The code-benchmark generations (medium, send once per run)

LiveCodeBench is scored on our side, so we need the raw generations, not a score:

```
work_pt06b/eval_out/<method>_pt06b_lcb/step_<peak>/**/lcb.jsonl
work_pt06b/eval_out/<method>_pt06b_humanevalplus/step_<peak>/**/humanevalplus.jsonl
```

These are 10–30 MB each — too big for git, fine for a HuggingFace dataset repo or any file drop.

## 3. The models (large, one upload per run)

For each finished run, only the **best step** matters, and only its HuggingFace export:

```
work_pt06b/checkpoints/<method>_pt06b/global_step_<peak>/actor/huggingface/   # ~1.2 GB
```

`scripts/status.sh` prints the peak step (highest math Avg4). If you can, upload to a **private**
HuggingFace repo, one folder per run, and send us the repo name; otherwise tell us and we will arrange
a transfer. Do not ship the `.pt` shard files — we cannot use them.

## 4. Tell us explicitly

- which runs you did **not** do, and why (time, OOM, anything else);
- every deviation from the protocol in HANDBOOK.md section 4, including the ones you were forced into;
- anything in `scripts/` you had to edit to make it work on your cluster — send the diff, not a description.

A run with an honest note about a deviation is usable. A run that silently deviates is not, and we will
find out when the numbers do not line up with the other students.
