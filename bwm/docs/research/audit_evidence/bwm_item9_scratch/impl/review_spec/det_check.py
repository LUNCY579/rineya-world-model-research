import sys, random, hashlib, json
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
h = hashlib.sha256()
rng = random.Random(5)
k = 0
while k < 200:
    g = L.generate(rng, rng.choice((1, 2, 3)), rename=0.5)
    if g is None: continue
    objs, init, goal = g
    p = L.astar_plan(init, goal)
    h.update(repr((sorted(init), sorted(goal), p, L.signature(objs, init, goal), L.proposer_prompt(init, goal),
                   L.forward_prompt_delta(init, p[0]), L.target_actions(p), L.write_problem(objs, init, goal))).encode())
    k += 1
for i in (2, 50, 217):
    objs, init, goal = L.parse_problem(L.planbench_path(i))
    h.update(repr(L.astar_plan(init, goal)).encode())
print(h.hexdigest())
