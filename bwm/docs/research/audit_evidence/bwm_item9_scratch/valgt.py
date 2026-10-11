"""bwmn.blocks.val_check on Logistics GT plans (+ negative controls) and BFS optimality check on small instances."""
import json, os, sys, tempfile, time
from collections import deque
from pathlib import Path

SCR = os.path.dirname(os.path.abspath(__file__))
os.environ["TMP"] = os.environ["TEMP"] = os.environ["TMPDIR"] = SCR
tempfile.tempdir = SCR
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import blocks as B
from bwmn import mystery as M

L = "C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench/instances/logistics"
DOM = Path(L + "/generated_domain.pddl")
gt = json.load(open(os.path.join(SCR, "gt_plans.json")))
gt = {int(k): [tuple(a.split()) for a in v] for k, v in gt.items()}

t0 = time.time()
ok = neg_trunc = neg_swap = 0
upper_ok = 0
for i, p in sorted(gt.items()):
    prob = Path(L + f"/generated_basic/instance-{i}.pddl")
    ok += B.val_check(prob, p, domain_path=DOM)
    neg_trunc += B.val_check(prob, p[:-1], domain_path=DOM)
    if len(p) >= 2:
        q = list(p); q[0], q[-1] = q[-1], q[0]
        neg_swap += B.val_check(prob, q, domain_path=DOM)
    if i <= 20:
        upper_ok += B.val_check(prob, [tuple(x.upper() for x in a) for a in p], domain_path=DOM)
print(dict(n=len(gt), val_ok=ok, truncated_valid=neg_trunc, swapped_first_last_valid=neg_swap,
           uppercase_ok_first19=upper_ok, secs=round(time.time() - t0, 1)))


# typed BFS on small instances to check optimality of GT lengths
def typed_ground(dom, objs, init):
    typ = {}
    for f in init:
        if len(f) == 2:
            typ.setdefault(f[1], set()).add(f[0])
    incity = {f[1]: f[2] for f in init if f[0] == "in-city"}
    T = lambda t: sorted(o for o in objs if t in typ.get(o, ()))
    pk, tr, ap, lo, apt, ci = T("obj"), T("truck"), T("airplane"), T("location"), T("airport"), T("city")
    acts = []
    for o in pk:
        for v in tr:
            for l in lo:
                acts += [("load-truck", o, v, l), ("unload-truck", o, v, l)]
        for v in ap:
            for l in lo:
                acts += [("load-airplane", o, v, l), ("unload-airplane", o, v, l)]
    for v in tr:
        for a in lo:
            for b in lo:
                if a != b and incity[a] == incity[b]:
                    acts.append(("drive-truck", v, a, b, incity[a]))
    for v in ap:
        for a in apt:
            for b in apt:
                if a != b:
                    acts.append(("fly-airplane", v, a, b))
    out = []
    for a in acts:
        g = M.ground(dom, a)
        if g:
            out.append((a,) + g)
    return out


def bfs_len(dom, objs, init, goal, cap):
    acts = typed_ground(dom, objs, init)
    if goal <= init:
        return 0
    seen = {init}
    frontier = [init]
    depth = 0
    while frontier and len(seen) < cap:
        depth += 1
        nxt = []
        for s in frontier:
            for a, pre, add, dele in acts:
                if pre <= s:
                    ns = (s - dele) | add
                    if ns not in seen:
                        if goal <= ns:
                            return depth
                        seen.add(ns)
                        nxt.append(ns)
        frontier = nxt
    return None


dom = M.parse_domain(DOM)
checked = match = 0
bad = []
skipped = 0
t0 = time.time()
for i, p in sorted(gt.items()):
    objs, init, goal = M.parse_problem(L + f"/generated_basic/instance-{i}.pddl")
    npk = sum(1 for f in init if f[0] == "obj")
    if npk > 3 or len(p) > 14:
        continue
    r = bfs_len(dom, objs, init, goal, cap=400000)
    if r is None:
        skipped += 1
        continue
    checked += 1
    if r == len(p):
        match += 1
    else:
        bad.append((i, len(p), r))
print(dict(bfs_checked=checked, gt_len_equals_bfs=match, mismatches=bad[:10], bfs_capped=skipped, secs=round(time.time() - t0, 1)))
