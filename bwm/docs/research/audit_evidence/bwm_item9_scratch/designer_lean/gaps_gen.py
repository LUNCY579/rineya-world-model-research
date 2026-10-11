"""LEAN: progress gaps (steps between strict decreases of unmet goal count) on A*-optimal plans of GENERATED
instances (no PlanBench test plan is read). Also A* teacher time per package count."""
import random, time, statistics as st, sys
import lgdom as L
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
rng = random.Random(20261099)
for npk, n in ((1, 40), (2, 40), (3, 40), (4, 20), (5, 6)):
    res, times, fails = [], [], 0
    while len(res) + fails < n:
        nc = rng.choice([2, 3]); sz = rng.choice([1, 2, 3]); na = rng.randint(1, nc)
        init, goal = L.generate(rng, nc, sz, na, npk)
        if goal <= init:
            continue
        t0 = time.time()
        plan, ne = L.astar(init, goal, cap=400000 if npk <= 4 else 300000)
        times.append(time.time() - t0)
        if plan is None:
            fails += 1; continue
        s = init; un = [len(goal - s)]
        for a in plan:
            s, ok = L.step(s, a); assert ok; un.append(len(goal - s))
        last, gaps, first = 0, [], None
        for t in range(1, len(un)):
            if un[t] < un[t - 1]:
                gaps.append(t - last); last = t
                first = t if first is None else first
        res.append((first, max(gaps), len(plan)))
    q = lambda v: (min(v), round(st.median(v), 1), max(v)) if v else None
    print(f"pk {npk}: n {len(res)} fails {fails} | len {q([r[2] for r in res])} | first delivery {q([r[0] for r in res])}"
          f" | max gap {q([r[1] for r in res])} | share maxgap>12 {sum(r[1] > 12 for r in res)}/{len(res)}"
          f" >16 {sum(r[1] > 16 for r in res)} >24 {sum(r[1] > 24 for r in res)} | A* s {q([round(x, 2) for x in times])}",
          flush=True)
