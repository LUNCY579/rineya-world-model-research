import sys, json, statistics
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
sols = [json.loads(l) for l in open(ROOT + "/runs/BW_DISTILL_r1/solutions.jsonl", encoding="utf-8") if l.strip()]
ver = [r for r in sols if r["verified"]]
L = []
for r in ver:
    goal = frozenset(map(tuple, r["goal"]))
    states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
    plan = [tuple(x) for x in r["plan"]]
    for t in range(len(plan)):
        L.append(len(B.proposer_prompt(states[t], goal)) + len(B.target_actions(plan[t:])))
v2 = []
with open(ROOT + "/runs/BW_data_v2/proposer_actions_train.jsonl", encoding="utf-8") as f:
    for i, l in enumerate(f):
        if i >= 5000: break
        r = json.loads(l); v2.append(len(r["prompt"]) + len(r["target"]))
print("verified", len(ver), "/", len(sols), "distill rows so far", len(L), "proj@240", round(len(L) * 240 / len(sols)))
print("distill chars mean", round(statistics.mean(L)), "v2 chars mean", round(statistics.mean(v2)), "ratio", round(statistics.mean(L) / statistics.mean(v2), 2))
