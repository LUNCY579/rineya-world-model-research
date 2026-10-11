import sys
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from pathlib import Path
from bwmn import blocks as B
HARD = Path(ROOT) / "bench/planbench/blocksworld_hard/generated"
def towers(facts, objs):
    on = {f[1]: f[2] for f in facts if f[0] == "on"}
    below = set(on.values())
    tops = [b for b in objs if b not in below and (b in on or any(f == ("ontable", b) for f in facts))]
    hs = []
    for t in tops:
        h = 1; x = t
        while x in on: x = on[x]; h += 1
        hs.append(h)
    return tuple(sorted(hs))
g = Counter(); gi = Counter(); full = Counter()
for p in HARD.glob("instance-*.pddl"):
    objs, init, goal = B.parse_problem(p)
    gt = (len(objs), towers(goal, objs)); it = (len(objs), towers(init, objs))
    g[gt] += 1; gi[(it, gt)] += 1
    full[len([f for f in goal if f[0]=='on'])] += 1
print("instances", sum(g.values()), "distinct (n, goal tower shape)", len(g), "max multiplicity", g.most_common(5))
print("distinct (init shape, goal shape)", len(gi), gi.most_common(3))
