"""Typed BFS feasibility on PlanBench Logistics Task-1 instances (analysis only: size of the search, not training)."""
import itertools, json, re, sys, time
from collections import deque, Counter, defaultdict
from pathlib import Path
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import mystery as M
PB = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
LG = PB / "instances" / "logistics"
dom = M.parse_domain(LG / "generated_domain.pddl")

def typed_ground(init):
    ty = defaultdict(list)
    for f in init:
        if len(f) == 2:
            ty[f[0]].append(f[1])
    city = {f[1]: f[2] for f in init if f[0] == "in-city"}
    acts = []
    for p in ty["obj"]:
        for l in ty["location"]:
            acts += [("load-truck", p, t, l) for t in ty["truck"]] + [("unload-truck", p, t, l) for t in ty["truck"]]
            acts += [("load-airplane", p, a, l) for a in ty["airplane"]] + [("unload-airplane", p, a, l) for a in ty["airplane"]]
    for t in ty["truck"]:
        for l1 in ty["location"]:
            for l2 in ty["location"]:
                if l1 != l2 and city[l1] == city[l2]:
                    acts.append(("drive-truck", t, l1, l2, city[l1]))
    for a in ty["airplane"]:
        for l1 in ty["airport"]:
            for l2 in ty["airport"]:
                if l1 != l2:
                    acts.append(("fly-airplane", a, l1, l2))
    return [(a,) + M.ground(dom, a) for a in acts]

def bfs(init, goal, cap=200000, tmax=30):
    acts = typed_ground(init)
    prev = {init: None}; q = deque([init]); t0 = time.time()
    while q and len(prev) < cap and time.time() - t0 < tmax:
        s = q.popleft()
        for a, pre, add, dele in acts:
            if pre <= s:
                ns = (s - dele) | add
                if ns in prev: continue
                prev[ns] = (s, a)
                if goal <= ns:
                    n = 0; cur = ns
                    while prev[cur] is not None:
                        cur = prev[cur][0]; n += 1
                    return n, len(prev), len(acts)
                q.append(ns)
    return None, len(prev), len(acts)

d = json.loads((PB / "prompts/logistics/task_1_plan_generation.json").read_text())
gt = {x["instance_id"]: len(re.findall(r"\(([^()]+)\)", x["ground_truth_plan"])) for x in d["instances"]}
by = defaultdict(list)
for iid in sorted(gt):
    objs, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{iid}.pddl")
    by[sum(1 for f in init if f[0] == "obj")].append((iid, objs, init, goal))
untyped = len(M.ground_actions(dom, by[7][0][1])) if by[7] else None
print("untyped M.ground_actions for a 7-package instance:", untyped, "objects", len(by[7][0][1]) if by[7] else None)
for np_ in sorted(by):
    sample = by[np_][:4]
    out = []
    for iid, objs, init, goal in sample:
        n, ns, na = bfs(init, goal)
        out.append((iid, n, gt[iid], ns, na))
    print(f"packages {np_}: {len(by[np_])} instances; sample (id, bfs_len, gt_len, states, typed_actions): {out}")
