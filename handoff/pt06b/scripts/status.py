#!/usr/bin/env python
"""Progress of every run in the handoff. Usage: python status.py <ckpt_dir> <eval_dir> [method ...]"""
import glob, json, os, sys

CKPT, EVAL = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3:]
METHODS = ["sft", "seqkd", "grpo", "opd", "trd", "fastopd1024", "fastopd2048",
           "fastopd4096", "fastopd8192", "skd", "relayopd"]
MATH = ("aime24", "aime25", "amc23", "math500")
SCI = ("mmlu_pro", "gpqa_diamond", "humanevalplus", "lcb")
STEPS = (20, 40, 60, 80, 100, 120, 139)


def bench_done(tag, step, bench):
    d = f"{EVAL}/{tag}/step_{step}"
    shards = glob.glob(f"{d}/shard_*")
    done = [s for s in shards if os.path.exists(f"{s}/{bench}.summary.json")]
    partial = [s for s in shards if s not in done
               and os.path.exists(f"{s}/{bench}.jsonl") and os.path.getsize(f"{s}/{bench}.jsonl") > 0]
    if os.path.exists(f"{d}/{bench}.summary.json"):
        return True
    return bool(done) and not partial


def acc(tag, step, bench):
    """avg@k pooled over shards, weighted by problems -- the metric our tables use.

    A summary also carries pass@1 and pass@k; do not switch to those, the table is avg@k.
    """
    num = den = 0.0
    for p in glob.glob(f"{EVAL}/{tag}/step_{step}/shard_*/{bench}.summary.json") + \
             glob.glob(f"{EVAL}/{tag}/step_{step}/{bench}.summary.json"):
        try:
            d = json.load(open(p))
            v = d.get("avg@k")
            if v is None:
                continue
            n = float(d.get("n_problems") or 1)
            num += float(v) * n
            den += n
        except Exception:
            pass
    return num / den if den else None


print(f"{'method':<13}{'train':<9}{'math evaluated':<30}{'peak':<7}{'Avg4':<8}{'sci/code at peak'}")
print("-" * 92)
for m in METHODS:
    if ONLY and m not in ONLY:
        continue
    run = f"{m}_pt06b"
    out = f"{CKPT}/{run}"
    latest = 0
    f = f"{out}/latest_checkpointed_iteration.txt"
    if os.path.exists(f):
        latest = int(open(f).read().strip() or 0)
    train = f"@{latest}" if latest else "-"
    if latest >= 139:
        train = "DONE"

    done_steps, best, best_avg = [], None, None
    for st in STEPS:
        if all(bench_done(f"{run}_b16k", st, b) for b in MATH):
            done_steps.append(st)
            vals = [acc(f"{run}_b16k", st, b) for b in MATH]
            if all(v is not None for v in vals):
                a = 100 * sum(vals) / 4
                if best_avg is None or a > best_avg:
                    best, best_avg = st, a
    mstr = ",".join(str(s) for s in done_steps) if done_steps else "-"
    peak = str(best) if best else "-"
    avg = f"{best_avg:.2f}" if best_avg is not None else "-"
    sci = "-"
    if best:
        sci = " ".join(f"{b.split('_')[0]}:{'Y' if bench_done(f'{run}_{b}', best, b) else '.'}" for b in SCI)
    print(f"{m:<13}{train:<9}{mstr:<30}{peak:<7}{avg:<8}{sci}")
print("\nlegend: train DONE = reached step 139 | peak = step with the best math Avg4 | sci/code Y = generated")
