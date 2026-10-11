import sys, random
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
rng = random.Random(3)
tot = sl = 0; k = 0
while k < 500:
    g = L.generate(rng, rng.choice([1, 2, 2, 3, 3, 3]), rename=0.5)
    if g is None: continue
    k += 1
    objs, init, goal = g
    plan = L.astar_plan(init, goal)
    states = [init]
    for a in plan: states.append(L.step(states[-1], a)[0])
    s = rng.choice(states)
    for _ in range(rng.randrange(8)):
        app = [a for a in L.ground_actions(objs, s, self_loops=True) if L.applicable(s, a)]
        s = L.step(s, rng.choice(app))[0]
    app = [a for a in L.ground_actions(objs, s, self_loops=True) if L.applicable(s, a)]
    tot += 1; sl += sum(map(L.is_self_loop, app)) / len(app)
print("mean share of self-loops among applicable typed actions (F start states):", round(sl / tot, 3))
