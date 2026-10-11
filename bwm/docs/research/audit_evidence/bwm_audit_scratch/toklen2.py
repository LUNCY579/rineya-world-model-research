import json, sys
from pathlib import Path
from tokenizers import Tokenizer
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B
tk = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
sols = [json.loads(l) for l in Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/solutions_copy.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
ver = [r for r in sols if r["verified"]]
tot = []; tmax = 0
for r in ver:
    goal = frozenset(map(tuple, r["goal"]))
    states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
    plan = [tuple(x) for x in r["plan"]]
    for t in range(len(plan)):
        p, tg = B.proposer_prompt(states[t], goal), B.target_actions(plan[t:])
        tot.append(n(p) + n(tg) + 1); tmax = max(tmax, n(tg) + 1)
print("distill rows avg total tokens %.1f" % (sum(tot)/len(tot)), "max target+EOS tokens", tmax)
print("verified plans >40 actions", sum(len(r["plan"]) > 40 for r in ver), ">50", sum(len(r["plan"]) > 50 for r in ver), ">60", sum(len(r["plan"]) > 60 for r in ver))
# optimal length comparison (BFS) for small n only
import time
ratios = []
t0 = time.time()
for r in ver:
    if r["n"] > 7: continue
    init = frozenset(map(tuple, r["init"])); goal = frozenset(map(tuple, r["goal"]))
    opt = B.bfs_plan(init, goal, r["objs"])
    if opt is not None:
        ratios.append(len(r["plan"]) / max(1, len(opt)))
print("6-7 block verified plans: n", len(ratios), "mean len/opt %.2f" % (sum(ratios)/len(ratios)), "max %.2f" % max(ratios), "share optimal %.2f" % (sum(x == 1 for x in ratios)/len(ratios)), "bfs secs %.0f" % (time.time()-t0))
