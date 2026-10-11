import sys, random, tracemalloc
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
from bwmn import blocks as B
# SEM-2
objs, init, goal = L.parse_problem(L.planbench_path(186))
orig = L._successors
calls = [0]
def wrapped(C, s):
    calls[0] += 1
    return orig(C, s)
L._successors = wrapped
for cap in (0, 1, 7, 100):
    calls[0] = 0; st = {}
    p = L.astar_plan(init, goal, max_expansions=cap, stats=st)
    print("SEM-2 cap", cap, "real", calls[0], "status", st["status"], "reported", st["expansions"])
calls[0] = 0; st = {}; p = L.astar_plan(init, goal, stats=st); full = st["expansions"]
print("SEM-2 uncapped real", calls[0], "reported", full, st["status"])
for cap in (full - 1, full):
    calls[0] = 0; st = {}
    p = L.astar_plan(init, goal, max_expansions=cap, stats=st)
    print("SEM-2 cap", cap, "real", calls[0], "status", st["status"], "reported", st["expansions"])
L._successors = orig
# LG-SPEC-4
for fn in ("parse_plan", "parse_plan_prefix"):
    for objs_ in (None,):
        try:
            print("LG-SPEC-4", fn, getattr(L, fn)('{"actions":["x"]}', objs_))
        except Exception as e:
            print("LG-SPEC-4", fn, "RAISES", type(e).__name__, e)
try:
    print("blocks parse_plan None ->", B.parse_plan('{"actions":["pick-up a"]}', None))
except Exception as e:
    print("blocks raises", e)
try:
    print("LG-SPEC-4 prefix truncated", L.parse_plan_prefix('{"actions":["load-truck p0 t0 l0-0","lo', None))
except Exception as e:
    print("LG-SPEC-4 prefix truncated RAISES", type(e).__name__, e)
# LG-SPEC-5
names = [f"p{k}" for k in range(12)] + [f"t{k}" for k in range(3)] + [f"a{k}" for k in range(3)] + [f"l{x}-{y}" for x in range(3) for y in range(3)] + [f"c{k}" for k in range(3)]
L.ground.cache_clear()
rng = random.Random(0)
tracemalloc.start()
ops = sorted(L.ARITY)
for _ in range(60000):
    op = rng.choice(ops)
    L.ground((op,) + tuple(rng.choice(names) for _ in range(L.ARITY[op])))
cur, peak = tracemalloc.get_traced_memory()
ci = L.ground.cache_info()
print("LG-SPEC-5", ci, "MB", round(cur / 2**20, 1), "bytes/entry", round(cur / ci.currsize), "-> at maxsize", round(cur / ci.currsize * ci.maxsize / 2**20), "MB")
tracemalloc.stop(); L.ground.cache_clear()
# LG-SPEC-2
rng = random.Random(3)
tot = sl = 0; k = 0
while k < 300:
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
print("LG-SPEC-2 mean self-loop share among applicable typed actions:", round(sl / tot, 3))
