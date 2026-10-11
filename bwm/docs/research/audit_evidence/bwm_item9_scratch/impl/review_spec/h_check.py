"""Exhaustive admissibility + consistency check of L._heuristic on reachable state spaces."""
import sys, random
from collections import deque, defaultdict
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
rng = random.Random(77)
shapes = [(2, 1, 1, 2), (2, 2, 1, 2), (2, 2, 2, 2), (3, 1, 2, 2), (2, 3, 1, 2), (3, 2, 1, 1), (3, 2, 2, 2), (2, 3, 2, 3), (3, 2, 3, 2)]
tot_states = 0; bad_adm = 0; bad_cons = 0; worst_states = 0
for shape in shapes:
    for rep in range(6):
        nc, cs, npl, npk = shape
        g = None
        while g is None:
            g = L.generate(rng, npk, n_cities=nc, city_size=cs, n_planes=npl, rename=0.5)
        objs, init, goal = g
        C = L._compile(init, goal)
        assert C["shaped"]
        h = L._heuristic(C)
        s0 = C["state"]
        targets = C["gpk"]
        done = lambda x: all(x[i] == gg for i, gg in targets)
        seen = {s0}; q = deque([s0]); adj = {}
        while q:
            s = q.popleft()
            succ = [ns for _, ns in L._successors(C, s)]
            adj[s] = succ
            for ns in succ:
                if ns not in seen:
                    seen.add(ns); q.append(ns)
        # reverse BFS from goal states
        radj = defaultdict(list)
        for s, ss in adj.items():
            for ns in ss: radj[ns].append(s)
        hstar = {}
        q = deque()
        for s in adj:
            if done(s): hstar[s] = 0; q.append(s)
        while q:
            s = q.popleft()
            for p in radj[s]:
                if p not in hstar:
                    hstar[p] = hstar[s] + 1; q.append(p)
        tot_states += len(adj); worst_states = max(worst_states, len(adj))
        for s, ss in adj.items():
            hs = h(s)
            if s in hstar and hs > hstar[s]:
                bad_adm += 1
                if bad_adm < 5: print("INADMISSIBLE", shape, s, hs, hstar[s])
            for ns in ss:
                if hs > 1 + h(ns):
                    bad_cons += 1
                    if bad_cons < 5: print("INCONSISTENT", shape, s, ns, hs, h(ns))
        # optimality of A* from init
        p = L.astar_plan(init, goal)
        assert len(p) == hstar[s0], (shape, len(p), hstar[s0])
    print(shape, "done; states so far", tot_states, flush=True)
print("total states", tot_states, "max space", worst_states, "inadmissible", bad_adm, "inconsistent edges", bad_cons)
