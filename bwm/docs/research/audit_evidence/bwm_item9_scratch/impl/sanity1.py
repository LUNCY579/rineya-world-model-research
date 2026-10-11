import sys, time, collections
sys.path.insert(0, '.')
from bwmn import logistics as L
from bwmn import mystery as M
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
# schema check vs domain file
dom = M.parse_domain(L.DOMAIN_PDDL)
print(dom.name, sorted(dom.actions))
for op, sc in dom.actions.items():
    args = tuple(f"o{i}" for i in range(len(sc.params)))
    b = dict(zip(sc.params, args))
    sub = lambda atoms: frozenset((a[0],) + tuple(b[x] for x in a[1:]) for a in atoms)
    g = L.ground((op,) + args)
    assert g == (sub(sc.pre), sub(sc.add), sub(sc.dele)), op
print("schemas match")
insts = L.planbench_instances()
ok = sum(L.execute(x["init"], x["goal"], x["gt"])["success"] for x in insts if x["gt"])
print("GT exec ok", ok, "of", sum(1 for x in insts if x["gt"]))
cnt = [len(L.ground_actions(x["objs"], x["init"])) for x in insts]
cnt2 = [len(L.ground_actions(x["objs"], x["init"], self_loops=True)) for x in insts]
print("ground counts", min(cnt), max(cnt), "with self loops", min(cnt2), max(cnt2))
# GT actions in ground set
miss = 0
for x in insts:
    if x["gt"]:
        ga = set(L.ground_actions(x["objs"], x["init"]))
        miss += sum(a not in ga for a in x["gt"])
print("GT actions missing from ground set", miss)
t0 = time.time()
sigs = {x["id"]: L.signature(x["objs"], x["init"], x["goal"]) for x in insts}
print("sig time", round(time.time() - t0, 2), "distinct", len(set(sigs.values())))
cls = collections.defaultdict(list)
for i, s in sigs.items(): cls[s].append(i)
print("dups", [v for v in cls.values() if len(v) > 1])
# A* on <=3 packages
t0 = time.time(); res = []
for x in insts:
    if x["gt"] and x["n_packages"] <= 3:
        st = {}
        p = L.astar_plan(x["init"], x["goal"], stats=st)
        res.append((x["id"], x["n_packages"], len(x["gt"]), None if p is None else len(p), st["expansions"], st["seconds"], st["h0"]))
bad = [r for r in res if r[3] != r[2]]
print("astar <=3:", len(res), "mismatch", bad, "max exp", max(r[4] for r in res), "max s", max(r[5] for r in res), "total", round(time.time()-t0,1))
