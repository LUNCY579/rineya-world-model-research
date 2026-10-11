"""RIGOROUS designer prototype (CPU only): PlanBench-style Logistics generator (logistics.c distribution), renaming-
invariant exclusion vs the 286 PB instances, A* teacher cost at <= 3 packages, and the admissible lower bound h0 as a
difficulty covariate (correlation with FD-optimal GT length on PB). Run from designer/: python -B gen_proto.py"""
import json, random, statistics as st, sys, time
from collections import Counter
from pathlib import Path
src = Path("astar_time.py").read_text().split("for npk in")[0]
exec(compile(src, "astar", "exec"))  # brings M, LG, S, G, typed_ground, make_h, astar
sys.path.insert(0, ".")
from canon import signature
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))

def gen(rng, c, s, p, a, rename=False, pool=12):
    """logistics.c: truck i in city i at a random location; package origin/destination uniform over all locations;
    airplane at a random city's airport (lX-0). rename: break PB naming (random location/vehicle indices; airport
    not necessarily index 0) -- a surface change only."""
    cities = [f"c{i}" for i in range(c)]
    locs = {(i, j): f"l{i}-{j}" for i in range(c) for j in range(s)}
    trucks = [f"t{i}" for i in range(c)]
    planes = [f"a{i}" for i in range(a)]
    pk_names = sorted(rng.sample(range(pool), p))  # package names drawn from a pool so p3..p6 are seen in training
    pkgs = [f"p{k}" for k in pk_names]
    if rename:
        cperm = rng.sample(range(4), c)
        lnames = {}
        for i in range(c):
            idx = rng.sample(range(4), s)
            for j in range(s):
                lnames[(i, j)] = f"l{rng.choice(range(4))}-{idx[j]}"
        if len(set(lnames.values())) < len(lnames):
            return None
        locs = lnames; cities = [f"c{cperm[i]}" for i in range(c)]
        trucks = [f"t{x}" for x in rng.sample(range(4), c)]; planes = [f"a{x}" for x in rng.sample(range(4), a)]
    init = set()
    for x in planes: init.add(("airplane", x))
    for x in cities: init.add(("city", x))
    for x in trucks: init.add(("truck", x))
    for (i, j), l in locs.items():
        init |= {("location", l), ("in-city", l, cities[i])}
    for i in range(c): init.add(("airport", locs[(i, 0)]))
    for x in pkgs: init.add(("obj", x))
    for i, t in enumerate(trucks): init.add(("at", t, locs[(i, rng.randrange(s))]))
    goal = set()
    for x in pkgs: init.add(("at", x, locs[(rng.randrange(c), rng.randrange(s))]))
    for x in pkgs: goal.add(("at", x, locs[(rng.randrange(c), rng.randrange(s))]))
    for x in planes: init.add(("at", x, locs[(rng.randrange(c), 0)]))
    return frozenset(init), frozenset(goal)

pb_sig = {}
for i in range(1, 287):
    o, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{i}.pddl")
    pb_sig.setdefault(signature(init, goal), []).append(i)
# h0 vs optimal GT length on PB
xs, ys = [], []
for i in range(2, 287):
    o, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{i}.pddl")
    xs.append(make_h(init, goal)(init)); ys.append(len(G[str(i)]))
mx, my = st.mean(xs), st.mean(ys)
r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
print(f"PB h0 vs optimal length: pearson r {r:.3f}; mean ratio opt/h0 {st.mean(y / x for x, y in zip(xs, ys) if x):.3f}")

rng = random.Random(20261010)
shapes = [(c, s, a) for c in (2, 3) for s in (1, 2, 3) for a in range(1, c + 1)]
coll, dup, n, ts, lens, seen, bad = 0, 0, 0, [], [], set(), 0
t0 = time.time()
while n < 600:
    c, s, a = rng.choice(shapes); p = rng.choice([1, 2, 3])
    g = gen(rng, c, s, p, a, rename=rng.random() < 0.5)
    if g is None: continue
    init, goal = g
    if goal <= init: continue  # trivial (already solved) instances skipped
    sig = signature(init, goal)
    n += 1
    if sig in pb_sig: coll += 1; continue
    if sig in seen: dup += 1; continue
    seen.add(sig)
    t1 = time.time(); L, e = astar(init, goal, tmax=30); ts.append(time.time() - t1)
    if L is None: bad += 1
    else: lens.append(L)
print(f"generated {n} (<=3 pkgs, 50% renamed): PB-signature collisions {coll}, within-sample duplicates {dup}, "
      f"A* fails {bad}, A* time median {st.median(ts):.3f}s max {max(ts):.2f}s, opt length max {max(lens)} mean {st.mean(lens):.1f}, "
      f"wall {time.time()-t0:.0f}s")
# renaming invariance of the signature under the generator's renamed surface: same structure, PB names vs renamed
rng2 = random.Random(7); ok = 0
for k in range(50):
    st_ = rng2.getstate(); g1 = gen(rng2, 3, 3, 3, 2)
    rng2.setstate(st_); r = random.Random(k)
print("PB-OUT signature classes (>=4 pkgs):", len({signature(*M.parse_problem(LG / 'generated_basic' / f'instance-{i}.pddl')[1:]) for i in range(2, 287) if S[str(i)]['packages'] >= 4}))
