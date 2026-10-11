"""A* teacher review: optimality vs FD GT, cap enforcement, heuristic admissibility/consistency on full state spaces."""
import sys, time, random, json
from collections import Counter, deque
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L

part = sys.argv[1] if len(sys.argv) > 1 else "all"
rng = random.Random(99)

if part in ("all", "gt"):
    t0 = time.time()
    pb = L.planbench_instances()
    res = Counter()
    worst = 0
    mism = []
    for x in pb:
        if x["gt"] is None:
            continue
        st = {}
        p = L.astar_plan(x["init"], x["goal"], max_expansions=100_000, stats=st)
        worst = max(worst, st["expansions"])
        if p is None:
            res["unsolved"] += 1
            continue
        res["solved"] += 1
        if len(p) != len(x["gt"]):
            mism.append((x["id"], len(p), len(x["gt"])))
        if not L.execute(x["init"], x["goal"], p)["success"]:
            mism.append((x["id"], "exec fail"))
        if st["h0"] > len(x["gt"]):
            mism.append((x["id"], "h0>GT", st["h0"], len(x["gt"])))
    print("[gt] ", dict(res), "mismatches", mism, "max exp", worst, "secs", round(time.time() - t0, 1))
    # instance 1 (no GT): solve and compare with BFS? just report
    x = pb[0]
    st = {}
    p = L.astar_plan(x["init"], x["goal"], stats=st)
    print("[gt] instance-1 len", None if p is None else len(p), st)

if part in ("all", "cap"):
    calls = {"n": 0}
    orig = L._successors

    def counting(C, s):
        calls["n"] += 1
        return orig(C, s)

    L._successors = counting
    pb = L.planbench_instances(with_gt=True)
    big = [x for x in pb if x["n_packages"] == 7][:5]
    for x in big:
        st = {}
        calls["n"] = 0
        p = L.astar_plan(x["init"], x["goal"], max_expansions=10**7, stats=st)
        full = calls["n"]
        out = []
        for cap in (0, 1, 7, 100, full - 1, full, full + 1):
            if cap < 0:
                continue
            calls["n"] = 0
            st2 = {}
            q = L.astar_plan(x["init"], x["goal"], max_expansions=cap, stats=st2)
            out.append((cap, calls["n"], st2["status"], st2["expansions"], None if q is None else len(q)))
            assert calls["n"] <= cap, (cap, calls["n"])
        print("[cap] id", x["id"], "uncapped expansions(successor calls)", full, "stats", st["expansions"], "|", out)
    # wall-clock cap
    x = big[0]
    for ms in (0.0, 0.05, 0.2):
        st = {}
        t0 = time.time()
        q = L.astar_plan(x["init"], x["goal"], max_seconds=ms, stats=st)
        print("[cap] max_seconds", ms, "->", st["status"], st["expansions"], "elapsed", round(time.time() - t0, 3))
    L._successors = orig

if part in ("all", "space"):
    # full reachable state space; h* by reverse BFS; check admissible + consistent; check _successors vs fact semantics
    def compact_of(C, s):
        li = {l: i for i, l in enumerate(C["L"])}
        vi = {v: i for i, v in enumerate(C["V"])}
        pos = {f[1]: f for f in s if f[0] in ("at", "in")}
        out = [li[pos[p][2]] if pos[p][0] == "at" else C["nL"] + vi[pos[p][2]] for p in C["P"]]
        return tuple(out + [li[pos[v][2]] for v in C["V"]])

    tot = Counter()
    viol = []
    t0 = time.time()
    shapes = [(2, 1, 1, 1), (2, 1, 2, 3), (2, 2, 2, 2), (2, 2, 1, 3), (2, 3, 2, 2), (3, 1, 3, 3), (3, 2, 2, 2),
              (3, 2, 3, 2), (2, 2, 2, 3), (3, 1, 2, 3), (2, 3, 1, 3), (3, 3, 1, 2)]
    k = 0
    while k < 60:
        nc, cs, npl, npk = shapes[k % len(shapes)]
        g = L.generate(rng, npk, n_cities=nc, city_size=cs, n_planes=npl, rename=0.5)
        if g is None:
            continue
        objs, init, goal = g
        C = L._compile(init, goal)
        if not C["shaped"]:
            viol.append(("not shaped", k))
        h = L._heuristic(C)
        s0 = C["state"]
        # forward reachable graph
        seen = {s0: []}
        q = deque([s0])
        while q:
            s = q.popleft()
            succ = [ns for _, ns in L._successors(C, s)]
            seen[s] = succ
            for ns in succ:
                if ns not in seen:
                    seen[ns] = None
                    q.append(ns)
        if len(seen) > 400_000:
            print("big", len(seen))
        # reverse BFS from goal states
        targets = C["gpk"]
        goals = [s for s in seen if all(s[i] == gg for i, gg in targets)]
        rev = {}
        for s, succ in seen.items():
            for ns in succ:
                rev.setdefault(ns, []).append(s)
        hstar = {s: 0 for s in goals}
        q = deque(goals)
        while q:
            s = q.popleft()
            for ps in rev.get(s, ()):
                if ps not in hstar:
                    hstar[ps] = hstar[s] + 1
                    q.append(ps)
        for s, succ in seen.items():
            hv = h(s)
            tot["states"] += 1
            if s in hstar and hv > hstar[s]:
                viol.append(("inadmissible", k, s, hv, hstar[s]))
            if s in goals and hv != 0:
                viol.append(("h(goal)!=0", k, s, hv))
            for ns in succ:
                tot["edges"] += 1
                if hv > 1 + h(ns):
                    viol.append(("inconsistent", k, s, ns, hv, h(ns)))
        # A* from a sample of reachable states == h*
        sample = rng.sample(sorted(seen), min(25, len(seen)))
        # decode compact -> fact set via random walks is complex; instead rebuild fact sets from compact
        L_, P_, V_, nL = C["L"], C["P"], C["V"], C["nL"]
        static = frozenset(f for f in init if f[0] not in ("at", "in"))
        for s in sample:
            facts = set(static)
            for i, p in enumerate(P_):
                x = s[i]
                facts.add(("at", p, L_[x]) if x < nL else ("in", p, V_[x - nL]))
            for j, v in enumerate(V_):
                facts.add(("at", v, L_[s[len(P_) + j]]))
            facts = frozenset(facts)
            assert compact_of(C, facts) == s
            # successor equivalence with fact semantics
            ref = {compact_of(C, L.step(facts, a)[0]) for a in L.ground_actions(L.objs_of(facts), facts)
                   if L.applicable(facts, a)}
            if ref != set(seen[s]):
                viol.append(("succ mismatch", k, s))
            st = {}
            plan = L.astar_plan(facts, goal, stats=st)
            tot["astar_from_states"] += 1
            if s in hstar:
                if plan is None or len(plan) != hstar[s]:
                    viol.append(("astar!=h*", k, s, None if plan is None else len(plan), hstar[s]))
            elif plan is not None:
                viol.append(("astar found plan where none", k, s))
        tot["instances"] += 1
        tot["max_space"] = max(tot["max_space"], len(seen))
        k += 1
    print("[space]", dict(tot), "violations", len(viol), viol[:10], "secs", round(time.time() - t0, 1))
