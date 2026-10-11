import sys, re
from collections import Counter
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/impl/review_sem")
from bwmn import logistics as L
import sexp
probs = []
pk = Counter()
maxobj = maxinit = 0
for i in range(1, 287):
    p = L.planbench_path(i)
    name, dom, objs, init, goal = sexp.problem(p)
    assert dom == "logistics-strips", (i, dom)
    # duplicates in file?
    if len(set(init)) != len(init): probs.append((i, "dup init facts"))
    if len(set(objs)) != len(objs): probs.append((i, "dup objects"))
    o, I, G = L.parse_problem(p)
    if I != frozenset(init): probs.append((i, "init mismatch", set(I) ^ set(init)))
    if G != frozenset(goal): probs.append((i, "goal mismatch"))
    if set(L.all_objects(o)) != set(objs): probs.append((i, "objs mismatch"))
    # every object exactly one primary type
    prim = Counter()
    for f in init:
        if len(f) == 2:
            assert f[0] in ("airplane", "city", "truck", "location", "airport", "obj"), f
            if f[0] != "airport": prim[f[1]] += 1
    if any(prim[x] != 1 for x in objs): probs.append((i, "typing", [x for x in objs if prim[x] != 1]))
    # airports are locations
    if not set(o["airport"]) <= set(o["location"]): probs.append((i, "airport not location"))
    m = re.match(r"logistics-c(\d+)-s(\d+)-p(\d+)-a(\d+)$", name)
    nc, ns, np_, na = map(int, m.groups())
    if (len(o["city"]), len(o["location"]), len(o["obj"]), len(o["airplane"]), len(o["truck"])) != (nc, nc*ns, np_, na, nc):
        probs.append((i, "header mismatch", name, {k: len(v) for k, v in o.items()}))
    if len(goal) != np_ or not all(f[0] == "at" and f[1] in o["obj"] for f in goal) or sorted(f[1] for f in goal) != o["obj"]:
        probs.append((i, "goal shape"))
    if i >= 2: pk[np_] += 1
    maxobj = max(maxobj, len(objs)); maxinit = max(maxinit, len(init))
    # exactly one position each
    pos = Counter(f[1] for f in init if f[0] in ("at", "in"))
    if any(pos[x] != 1 for x in o["obj"] + o["truck"] + o["airplane"]): probs.append((i, "positions"))
    if not all(len(L.objs_of(I)[t]) for t in ("city",)): probs.append((i, "no city"))
print("problems:", probs[:10], len(probs))
print("pkg strata ids 2-286:", sorted(pk.items()), "max objs", maxobj, "max init", maxinit)
