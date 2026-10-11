import sys, time, collections, json
sys.path.insert(0, '.')
from bwmn import logistics as L
insts = L.planbench_instances()
by = collections.defaultdict(list)
t0 = time.time()
for x in insts:
    if not x["gt"]:
        continue
    st = {}
    p = L.astar_plan(x["init"], x["goal"], max_expansions=300000, max_seconds=60, stats=st)
    by[x["n_packages"]].append((x["id"], len(x["gt"]), None if p is None else len(p), st["expansions"], st["seconds"], st["status"]))
for k in sorted(by):
    r = by[k]
    solved = [z for z in r if z[2] is not None]
    print(k, "n", len(r), "solved", len(solved), "len!=gt", [z for z in solved if z[2] != z[1]], "max exp", max(z[3] for z in r), "max s", max(z[4] for z in r), "statuses", collections.Counter(z[5] for z in r))
print("total", round(time.time() - t0, 1))
json.dump({k: v for k, v in by.items()}, open('C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/impl/astar_all.json', 'w'))
