import json, sys
from pathlib import Path
from tokenizers import Tokenizer
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B
tk = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
# sanity: special tokens map to single ids
print("im_start ids", tk.encode("<|im_start|>", add_special_tokens=False).ids, "im_end", tk.encode("<|im_end|>", add_special_tokens=False).ids)
sols = [json.loads(l) for l in Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/solutions_copy.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rows = []
for r in sols:
    if not r["verified"]: continue
    goal = frozenset(map(tuple, r["goal"]))
    states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
    plan = [tuple(x) for x in r["plan"]]
    for t in range(len(plan)):
        rows.append((r["n"], len(plan) - t, B.proposer_prompt(states[t], goal), B.target_actions(plan[t:])))
L = [(nb, k, n(p) + n(t) + 1, n(t)) for nb, k, p, t in rows]
print("distill rows (168 done)", len(L), "max total", max(x[2] for x in L), "over 640:", sum(x[2] > 640 for x in L), "over 256:", sum(x[2] > 256 for x in L))
print("target tokens > 255 (cannot be emitted within M=256 incl EOS):", sum(x[3] + 1 > 256 for x in L), " distinct-plan full targets >255:", sum(1 for x in L if x[3] + 1 > 256))
full = [x for x in L]
import collections
print("max total by blocks", sorted({nb: max(x[2] for x in L if x[0] == nb) for nb in set(x[0] for x in L)}.items()))
# v2 rows max len sample
mx = 0; c = 0
for i, l in enumerate(open(ROOT / "runs/BW_data_v2/proposer_actions_train.jsonl", encoding="utf-8")):
    if i % 10: continue
    r = json.loads(l); m = n(r["prompt"]) + n(r["target"]) + 1; mx = max(mx, m); c += 1
print("v2 rows sampled", c, "max total tokens", mx)
# full-plan (t=0) rows: targets that exceed the eval cap
t0 = [x for x in L]
