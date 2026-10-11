"""Token lengths of GENERATED <=3-package training rows (pool names, 50% renamed, A*-optimal suffixes) in the compact
merged-verb rendering of rep_cost.py. CPU only."""
import random, statistics as st, sys
from pathlib import Path
exec(compile(Path("gen_proto.py").read_text().split("pb_sig = {}")[0], "gp", "exec"))
src = Path("rep_cost.py").read_text()
exec(compile(src.split("print(\"sys tokens")[0], "rc", "exec"))
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
rng = random.Random(20261011)
shapes = [(c, s, a) for c in (2, 3) for s in (1, 2, 3) for a in range(1, c + 1)]
fr, pr, plen, n = [], [], [], 0
while n < 400:
    c, s, a = rng.choice(shapes); p = rng.choice([1, 2, 3])
    g = gen(rng, c, s, p, a, rename=rng.random() < 0.5)
    if g is None: continue
    init, goal = g
    if goal <= init: continue
    L, e = astar(init, goal, tmax=30)
    if L is None: continue
    # recover plan by re-running a plan-returning A* (copy with parent pointers)
    acts = typed_ground(init); h = make_h(init, goal)
    import heapq, itertools
    g0 = {init: 0}; prev = {init: None}; cnt = itertools.count(); pq = [(h(init), 0, next(cnt), init)]; plan = None
    while pq:
        f_, gg, _, s_ = heapq.heappop(pq)
        if gg > g0[s_]: continue
        if goal <= s_:
            plan = []; cur = s_
            while prev[cur] is not None: cur, aa = prev[cur]; plan.append(aa)
            plan.reverse(); break
        for aa, pre, add, dele in acts:
            if pre <= s_:
                ns = (s_ - dele) | add
                if ns not in g0 or gg + 1 < g0[ns]:
                    g0[ns] = gg + 1; prev[ns] = (s_, aa); heapq.heappush(pq, (gg + 1 + h(ns), gg + 1, next(cnt), ns))
    n += 1; plen.append(len(plan))
    sts = [init]
    for aa in plan: sts.append(M.step(dom, sts[-1], aa)[0])
    for t, aa in enumerate(plan):
        fr.append(T(fwd_prompt(sts[t], aa, True)) + T(B.forward_target_delta(sts[t], sts[t + 1], True)) + 1)
        pr.append(T(prop_prompt(sts[t], goal)) + T(tgt_actions(plan[t:], True)) + 1)
q = lambda v: f"n {len(v)} mean {st.mean(v):.0f} p99 {sorted(v)[int(.99*(len(v)-1))]} max {max(v)}"
print("generated <=3 pkg instances", n, "opt len mean %.1f max %d" % (st.mean(plen), max(plen)))
print("F rows (plan states):", q(fr)); print("Pa suffix rows:", q(pr), "| >640:", sum(x > 640 for x in pr), "| >512:", sum(x > 512 for x in pr))
