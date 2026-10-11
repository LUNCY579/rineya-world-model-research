import random, sys, time
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
rng = random.Random(7311)
for n in (1, 2, 2, 3, 3):
    k = 0
    while True:
        g = L.generate(rng, n)
        k += 1
        if g is None: continue
        objs, init, goal = g
        st = {}
        t = time.time()
        p = L.astar_plan(init, goal, max_expansions=20000, stats=st)
        if p is None or len(p) < 4: print(' skip', st); continue
        break
    print(n, len(p), st, round(time.time()-t, 3), len(objs['city']), len(objs['location']), len(objs['airplane']),
          L.state_from_json(L.state_json(init)) == init)
