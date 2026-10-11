"""LEAN designer prototype (scratch only): pure-python Logistics domain with VAL semantics, PlanBench-shape generator,
A* teacher (admissible per-package load/unload bound), BW-style JSON rendering, canonical renaming signature.
Never imports torch/transformers/peft."""
import heapq, itertools, json, random, re, sys, hashlib
from collections import defaultdict
from pathlib import Path
ROOT = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
from bwmn import mystery as M
PB = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
LG = PB / "instances" / "logistics"
TYPES = ("obj", "truck", "airplane", "location", "city", "airport")


def types(s):
    t = defaultdict(set)
    for f in s:
        if len(f) == 2 and f[0] in TYPES:
            t[f[0]].add(f[1])
    return t


def applicable(s, a):
    op = a[0]
    if op in ("load-truck", "load-airplane", "unload-truck", "unload-airplane"):
        if len(a) != 4:
            return False
        p, v, l = a[1:]
        vt = "truck" if op.endswith("truck") else "airplane"
        need = ("at", p, l) if op.startswith("load") else ("in", p, v)
        return {("obj", p), (vt, v), ("location", l), ("at", v, l), need} <= s
    if op == "drive-truck":
        if len(a) != 5:
            return False
        t, l1, l2, c = a[1:]
        return {("truck", t), ("location", l1), ("location", l2), ("city", c), ("at", t, l1),
                ("in-city", l1, c), ("in-city", l2, c)} <= s
    if op == "fly-airplane":
        if len(a) != 4:
            return False
        p, l1, l2 = a[1:]
        return {("airplane", p), ("airport", l1), ("airport", l2), ("at", p, l1)} <= s
    return False


def effects(a):
    op = a[0]
    if op in ("load-truck", "load-airplane"):
        p, v, l = a[1:]
        return {("at", p, l)}, {("in", p, v)}
    if op in ("unload-truck", "unload-airplane"):
        p, v, l = a[1:]
        return {("in", p, v)}, {("at", p, l)}
    if op == "drive-truck":
        t, l1, l2, c = a[1:]
        return {("at", t, l1)}, {("at", t, l2)}
    p, l1, l2 = a[1:]
    return {("at", p, l1)}, {("at", p, l2)}


def step(s, a):
    """VAL semantics: delete then add (a self-loop move is a legal no-op). Illegal -> unchanged, False."""
    if not applicable(s, a):
        return s, False
    d, ad = effects(a)
    return frozenset((s - d) | ad), True


def execute(init, goal, plan):
    s = init
    for i, a in enumerate(plan):
        s, ok = step(s, a)
        if not ok:
            return {"valid": False, "success": False, "fail_step": i}
    return {"valid": True, "success": goal <= s, "fail_step": None}


def ground_actions(s, self_loops=True):
    t = types(s)
    city = {f[1]: f[2] for f in s if f[0] == "in-city"}
    acts = []
    for p in sorted(t["obj"]):
        for l in sorted(t["location"]):
            for v in sorted(t["truck"]):
                acts += [("load-truck", p, v, l), ("unload-truck", p, v, l)]
            for v in sorted(t["airplane"]):
                acts += [("load-airplane", p, v, l), ("unload-airplane", p, v, l)]
    for v in sorted(t["truck"]):
        for l1 in sorted(t["location"]):
            for l2 in sorted(t["location"]):
                if city.get(l1) == city.get(l2) and (self_loops or l1 != l2):
                    acts.append(("drive-truck", v, l1, l2, city[l1]))
    for v in sorted(t["airplane"]):
        for l1 in sorted(t["airport"]):
            for l2 in sorted(t["airport"]):
                if self_loops or l1 != l2:
                    acts.append(("fly-airplane", v, l1, l2))
    return acts


def h_pkg(s, goal, city, airport):
    """Admissible and consistent: each load/unload touches one package; sum of per-package minimum load/unload
    counts (moves are free). Trucks never leave their city; airplanes only visit airports."""
    pos = {f[1]: f for f in s if f[0] in ("at", "in")}
    vloc = {f[1]: f[2] for f in s if f[0] == "at"}
    planes = {f[1] for f in s if f[0] == "airplane"}
    h = 0
    for g in goal:
        p, gl = g[1], g[2]
        f = pos[p]
        x = 0 if gl in airport else 2
        if f[0] == "at":
            l = f[2]
            if l == gl:
                continue
            h += 2 if city[l] == city[gl] else 2 + (0 if l in airport else 2) + x
        elif f[2] in planes:
            h += 1 + x
        else:  # in a truck
            h += 1 if city[vloc[f[2]]] == city[gl] else 3 + x
    return h


def astar(init, goal, cap=400000):
    acts = ground_actions(init, self_loops=False)
    city = {f[1]: f[2] for f in init if f[0] == "in-city"}
    airport = {f[1] for f in init if f[0] == "airport"}
    cnt = itertools.count()
    g0 = {init: 0}
    prev = {init: None}
    openq = [(h_pkg(init, goal, city, airport), 0, next(cnt), init)]
    closed = set()
    while openq:
        f, g, _, s = heapq.heappop(openq)
        if s in closed:
            continue
        if goal <= s:
            plan = []
            while prev[s] is not None:
                s, a = prev[s]
                plan.append(a)
            return plan[::-1], len(closed)
        closed.add(s)
        if len(closed) > cap:
            return None, len(closed)
        for a in acts:
            if applicable(s, a):
                d, ad = effects(a)
                ns = frozenset((s - d) | ad)
                ng = g + 1
                if ng < g0.get(ns, 1 << 30):
                    g0[ns] = ng
                    prev[ns] = (s, a)
                    heapq.heappush(openq, (ng + h_pkg(ns, goal, city, airport), ng, next(cnt), ns))
    return None, len(closed)


# ---------------------------------------------------------------- generator (PlanBench logistics.c shape)
def generate(rng, n_c, size, n_a, n_p, scramble=False):
    cities = [f"c{i}" for i in range(n_c)]
    locs = {c: [f"l{i}-{j}" for j in range(size)] for i, c in enumerate(cities)}
    s = set()
    for c in cities:
        s.add(("city", c))
        for l in locs[c]:
            s |= {("location", l), ("in-city", l, c)}
        s.add(("airport", locs[c][0]))
    for i, c in enumerate(cities):
        s |= {("truck", f"t{i}"), ("at", f"t{i}", rng.choice(locs[c]))}
    for k in range(n_a):
        s |= {("airplane", f"a{k}"), ("at", f"a{k}", locs[rng.choice(cities)][0])}
    goal = set()
    for k in range(n_p):
        s |= {("obj", f"p{k}"), ("at", f"p{k}", rng.choice(locs[rng.choice(cities)]))}
        goal.add(("at", f"p{k}", rng.choice(locs[rng.choice(cities)])))
    init, goal = frozenset(s), frozenset(goal)
    if scramble:
        init, goal = scramble_names(rng, init, goal)
    return init, goal


def scramble_names(rng, init, goal):
    """Rename within the PlanBench name grammar but break its conventions (lX-0 airport, lX-Y in cX, tX in cX)."""
    t = types(init)
    m = {}
    for ty, pool in (("city", [f"c{i}" for i in range(5)]), ("truck", [f"t{i}" for i in range(5)]),
                     ("airplane", [f"a{i}" for i in range(5)]), ("obj", [f"p{i}" for i in range(15)]),
                     ("location", [f"l{i}-{j}" for i in range(5) for j in range(5)])):
        for o, n in zip(sorted(t[ty]), rng.sample(pool, len(t[ty]))):
            m[o] = n
    ren = lambda fs: frozenset((f[0],) + tuple(m[x] for x in f[1:]) for f in fs)
    return ren(init), ren(goal)


# ---------------------------------------------------------------- rendering (BW recipe: canonical JSON, chat template)
SYS_FORWARD_DELTA_LONG = (
    "Logistics simulator. load-truck P T L needs P and T at L. unload-truck P T L needs P in T and T at L. "
    "load-airplane P A L needs P and A at L. unload-airplane P A L needs P in A and A at L. drive-truck T L1 L2 C "
    "needs T at L1 and L1, L2 in city C. fly-airplane A L1 L2 needs A at L1 and L1, L2 airports. Names: p package, "
    "t truck, a airplane, l location, c city. An action whose requirements fail is illegal and changes nothing. Given "
    "the state and one action, reply with JSON: whether it is legal and the facts it adds and deletes.")
SYS_FORWARD_DELTA = (
    "Logistics simulator. load-truck/load-airplane P V L needs P and V at L. unload-truck/unload-airplane P V L needs "
    "P in V and V at L. drive-truck T L1 L2 C needs T at L1, both in city C. fly-airplane A L1 L2 needs A at L1, both "
    "airports. Illegal actions change nothing. Reply with JSON: whether it is legal and the facts it adds and deletes.")
SYS_PROPOSER = (
    "Logistics planner. Actions: load-truck P T L, unload-truck P T L, load-airplane P A L, unload-airplane P A L, "
    "drive-truck T L1 L2 C, fly-airplane A L1 L2. Reach every goal [P,L] (package P at L). Reply with a JSON plan.")


def state_json(s):
    cities = defaultdict(list)
    for f in s:
        if f[0] == "in-city":
            cities[f[2]].append(f[1])
    return {"airports": sorted(f[1] for f in s if f[0] == "airport"),
            "at": sorted([f[1], f[2]] for f in s if f[0] == "at"),
            "cities": {c: sorted(v) for c, v in sorted(cities.items())},
            "in": sorted([f[1], f[2]] for f in s if f[0] == "in")}


def goal_json(g):
    return sorted([f[1], f[2]] for f in g if f[0] == "at")


def forward_prompt_delta(s, a):
    return B.render_chat(SYS_FORWARD_DELTA, B.canonical({"action": B.action_str(a), "state": state_json(s)}))


def proposer_prompt(s, g, outcome="success"):
    return B.render_chat(SYS_PROPOSER, B.canonical({"goal": goal_json(g), "outcome": outcome, "state": state_json(s)}))


# ---------------------------------------------------------------- renaming signature (exact, brute force)
def signature(init, goal):
    t = types(init)
    city = {f[1]: f[2] for f in init if f[0] == "in-city"}
    airport = {f[1] for f in init if f[0] == "airport"}
    cities = sorted(t["city"])
    nonair = {c: sorted(l for l in t["location"] if city[l] == c and l not in airport) for c in cities}
    air = {c: [l for l in t["location"] if city[l] == c and l in airport] for c in cities}
    trucks, planes = sorted(t["truck"]), sorted(t["airplane"])
    dyn = sorted(f for f in init if f[0] in ("at", "in"))
    best = None
    lperm_choices = [list(itertools.permutations(nonair[c])) for c in cities]
    for cp in itertools.permutations(range(len(cities))):
        for lp in itertools.product(*lperm_choices):
            m = {}
            for ci, c in enumerate(cities):
                m[c] = f"C{cp[ci]}"
                for j, l in enumerate(air[c]):
                    m[l] = f"L{cp[ci]}a{j}"
                for j, l in enumerate(lp[ci]):
                    m[l] = f"L{cp[ci]}n{j}"
            for tp in itertools.permutations(range(len(trucks))):
                for ap in itertools.permutations(range(len(planes))):
                    mm = dict(m)
                    for i, x in enumerate(trucks):
                        mm[x] = f"T{tp[i]}"
                    for i, x in enumerate(planes):
                        mm[x] = f"A{ap[i]}"
                    pk = {f[1]: [f[0], mm[f[2]]] for f in dyn if f[1] in t["obj"]}
                    gl = {f[1]: mm[f[2]] for f in goal}
                    pkg = sorted(pk[p] + [gl.get(p, "-")] for p in t["obj"])
                    veh = sorted([mm[f[1]], mm[f[2]]] for f in dyn if f[1] not in t["obj"])
                    key = json.dumps([len(cities), sorted(len(nonair[c]) for c in cities), veh, pkg])
                    if best is None or key < best:
                        best = key
    return hashlib.sha256(best.encode()).hexdigest()


def norm(fs):
    return frozenset(tuple(w.lower() if i == 0 else w for i, w in enumerate(f)) for f in fs)


def planbench_t1():
    d = json.loads((PB / "prompts/logistics/task_1_plan_generation.json").read_text())
    out = []
    for x in d["instances"]:
        iid = x["instance_id"]
        objs, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{iid}.pddl")
        init, goal = norm(init), norm(goal)
        plan = [tuple(m.split()) for m in re.findall(r"\(([^()]+)\)", x["ground_truth_plan"])]
        t = types(init)
        out.append({"id": iid, "objs": objs, "init": init, "goal": goal, "gt": plan, "np": len(t["obj"]),
                    "nc": len(t["city"]), "na": len(t["airplane"]), "nl": len(t["location"])})
    return out
