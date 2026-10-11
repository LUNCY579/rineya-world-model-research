import os, re, sys, collections, statistics
root = sys.argv[1]
for sub in ["generated_basic", "generated"]:
    d = os.path.join(root, sub)
    files = sorted(os.listdir(d))
    others = [f for f in files if not re.fullmatch(r"instance-\d+\.pddl", f)]
    inst = [f for f in files if re.fullmatch(r"instance-\d+\.pddl", f)]
    ids = sorted(int(re.findall(r"\d+", f)[0]) for f in inst)
    rows = []
    for f in inst:
        t = open(os.path.join(d, f), encoding="utf-8", errors="replace").read().lower()
        objs = re.search(r"\(:objects(.*?)\)", t, re.S)
        init = re.search(r"\(:init(.*?)\(:goal", t, re.S).group(1)
        goal = t[t.index("(:goal"):]
        def cnt(p): return len(set(re.findall(r"\(" + p + r"\s+(\S+)\)", init)))
        r = dict(city=cnt("city"), pkg=cnt("obj") or cnt("package"), truck=cnt("truck"), plane=cnt("airplane"),
                 loc=cnt("location"), airport=cnt("airport"),
                 goals=len(re.findall(r"\(at\s", goal)),
                 nobj=len(objs.group(1).split()) if objs else -1)
        rows.append(r)
    print(sub, "n_instance_files", len(inst), "id_range", ids[0], ids[-1], "other_files", others)
    for k in ["city","pkg","truck","plane","loc","airport","goals","nobj"]:
        v = [r[k] for r in rows]
        print(f"  {k:8s} min {min(v)} max {max(v)} mean {statistics.mean(v):.2f} dist {dict(sorted(collections.Counter(v).items()))}")
