import sys, types, random
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
from bwmn import logistics as L
rt = lambda s: L.state_from_json(L.state_json(s)) == s
pb = L.planbench_instances()
n = bad = 0
for x in pb:
    s = x["init"]; n += 1; bad += not rt(s)
    for a in (x["gt"] or []):
        s = L.step(s, a)[0]; n += 1; bad += not rt(s)
    # every applicable action from init (incl. self-loops)
    for a in L.applicable_actions(x["init"], self_loops=True):
        n += 1; bad += not rt(L.step(x["init"], a)[0])
print("planbench states", n, "non-round-trip", bad)
rng = random.Random(123)
n = bad = 0
for i in range(400):
    g = L.generate(rng, rng.choice([1, 2, 3, 4, 5, 6, 7]))
    if g is None: continue
    objs, init, goal = g
    s = init; n += 1; bad += not rt(s)
    for _ in range(30):
        acts = L.applicable_actions(s, self_loops=True)
        s = L.step(s, rng.choice(acts))[0]; n += 1; bad += not rt(s)
print("generated states", n, "non-round-trip", bad)
