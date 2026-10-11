import time, json, re, heapq, itertools, statistics as st
from pathlib import Path
src = Path("../lg_bfs.py").read_text().split("d = json.loads")[0]
exec(compile(src, "lgbfs", "exec"))
S = json.load(open("../inst_stats.json"))["generated_basic"]
G = json.load(open("../gt_plans.json"))

def make_h(init, goal):
    city = {f[1]: f[2] for f in init if f[0] == "in-city"}
    airport = {f[1] for f in init if f[0] == "airport"}
    trucks = {f[1] for f in init if f[0] == "truck"}
    gl = {f[1]: f[2] for f in goal if f[0] == "at"}
    def h(s):
        at = {}; inn = {}
        for f in s:
            if f[0] == "at": at[f[1]] = f[2]
            elif f[0] == "in": inn[f[1]] = f[2]
        lu = 0; mv = 0
        for p, g in gl.items():
            if at.get(p) == g: continue
            if p in at:
                l = at[p]
                if city[l] == city[g]: lu += 2; mv = max(mv, 1)
                else:
                    lu += 2 + 2 * (l not in airport) + 2 * (g not in airport); mv = max(mv, 1 + (l not in airport) + (g not in airport))
            else:
                v = inn[p]; l = at[v]
                if v in trucks:
                    if city[l] == city[g]: lu += 1; mv = max(mv, int(l != g))
                    else: lu += 3 + 2 * (g not in airport); mv = max(mv, 1 + (l not in airport) + (g not in airport))
                else:
                    lu += 1 + 2 * (g not in airport); mv = max(mv, int(city[l] != city[g]) + (g not in airport))
        return lu + mv
    return h

def astar(init, goal, cap=2000000, tmax=120):
    acts = typed_ground(init); h = make_h(init, goal)
    g0 = {init: 0}; prev = {init: None}; cnt = itertools.count()
    pq = [(h(init), 0, next(cnt), init)]; t0 = time.time(); exp = 0
    while pq and exp < cap and time.time() - t0 < tmax:
        f, g, _, s = heapq.heappop(pq)
        if g > g0[s]: continue
        if goal <= s:
            return g, exp
        exp += 1
        for a, pre, add, dele in acts:
            if pre <= s:
                ns = (s - dele) | add
                if ns not in g0 or g + 1 < g0[ns]:
                    g0[ns] = g + 1; prev[ns] = (s, a)
                    heapq.heappush(pq, (g + 1 + h(ns), g + 1, next(cnt), ns))
    return None, exp

for npk in (3, 4, 5):
    ts, ex, bad, fails = [], [], 0, 0
    ids = [i for i in range(2, 287) if S[str(i)]["packages"] == npk][:26]
    for i in ids:
        objs, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{i}.pddl")
        t0 = time.time(); n, e = astar(init, goal, tmax=60); ts.append(time.time() - t0); ex.append(e)
        if n is None: fails += 1
        elif n != len(G[str(i)]): bad += 1
    print(f"pk {npk}: n {len(ids)} fails {fails} len!=GT {bad} time med {st.median(ts):.2f} max {max(ts):.1f} exp max {max(ex)}")
