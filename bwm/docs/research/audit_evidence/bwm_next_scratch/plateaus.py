# Plateau lengths of the beam's ranking key (unmet goal facts) along plans the system found (generated sets only).
import sys, json, collections, statistics as st
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, R)
from bwmn import blocks as B
inst = {}
for i, it in enumerate(json.load(open(R + "/runs/BW_val_large/instances_mirror_hard.json", encoding="utf-8"))):
    inst["mirror/%d" % i] = it
print("example inst keys", list(next(iter(inst.values())).keys())[:8])
def to_state(x): return frozenset(tuple(f) for f in x)
def plateaus(init, goal, plan_strs, objs):
    s = init; um = [len(goal - s)]; 
    for a in plan_strs:
        act = B.parse_action(a, objs); s, ok = B.step(s, act)
        if not ok: break
        um.append(len(goal - s))
    # longest run of steps with no strict decrease below the running minimum; also first-progress step
    best = um[0]; run = 0; longest = 0; first = None
    for t in range(1, len(um)):
        if um[t] < best: best = um[t]; run = 0; first = first or t
        else: run += 1; longest = max(longest, run)
    return longest, first
by = collections.defaultdict(list)
rows = [json.loads(l) for l in open(R + "/runs/BW_TEST_B2/mirror_beam/plans.jsonl", encoding="utf-8")]
for r in rows:
    it = inst[r["id"]]
    init, goal, objs = to_state(it["init"]), to_state(it["goal"]), it["objs"]
    L, first = plateaus(init, goal, r["plan"], objs)
    by[(r["success"], "big" if r["n_blocks"] >= 11 else "small")].append((L, first))
for k, v in sorted(by.items()):
    Ls = [x[0] for x in v]; fs = [x[1] for x in v if x[1]]
    print(k, "n", len(v), "longest plateau mean %.1f, share>=12: %.2f" % (st.mean(Ls), sum(l >= 12 for l in Ls)/len(Ls)),
          "first-progress step mean %.1f, share>12: %.2f" % (st.mean(fs) if fs else -1, sum(f > 12 for f in fs)/max(1,len(fs))))
