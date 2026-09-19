#!/usr/bin/env python
"""HumanEval+ pass@1 from generated jsonl. Usage: python score_heplus.py <eval_dir> <run> <step>"""
import json, os, re, subprocess, sys

eval_dir, run, step = sys.argv[1], sys.argv[2], sys.argv[3]
D = f"{eval_dir}/{run}_humanevalplus/step_{step}"
gen = f"{D}/humanevalplus.jsonl"
if not os.path.exists(gen):                      # data-parallel runs leave one file per shard
    rows = []
    for sh in sorted(p for p in os.listdir(D) if p.startswith("shard_")):
        f = f"{D}/{sh}/humanevalplus.jsonl"
        if os.path.exists(f):
            rows += [json.loads(l) for l in open(f)]
else:
    rows = [json.loads(l) for l in open(gen)]
assert len(rows) >= 164, f"only {len(rows)} generations under {D}; expected 164"

def extract_code(text):
    """Last fenced block wins; fall back to the whole completion."""
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text

samples = f"{D}/heplus_samples.jsonl"
with open(samples, "w") as w:
    for r in rows:
        w.write(json.dumps({"task_id": r["gt"], "solution": extract_code(r["gen_text"])}) + "\n")

out = subprocess.run(["evalplus.evaluate", "--dataset", "humaneval", "--samples", samples],
                     capture_output=True, text=True, timeout=3600)
txt = out.stdout + out.stderr

def grab(label):
    m = re.search(re.escape(label) + r"[^\n]*\n\s*pass@1:\s*([0-9.]+)", txt)
    return round(100 * float(m.group(1)), 1) if m else None

res = {"run": run, "step": step, "n": len(rows),
       "base": grab("humaneval (base tests)"), "plus": grab("humaneval+ (base + extra tests)")}
json.dump(res, open(f"{D}/heplus_score.json", "w"), indent=1)
open(f"{D}/heplus_stdout.txt", "w").write(txt)
print(f"[heplus] {run}@{step}: HumanEval={res['base']}  HumanEval+={res['plus']}")
