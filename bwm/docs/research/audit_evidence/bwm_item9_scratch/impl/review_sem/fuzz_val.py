"""Reviewer fuzz: executor vs local VAL (-v trace) on random sequences. Sequential VAL calls."""
import sys, re, random, json, subprocess, time
from pathlib import Path
from collections import Counter
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/impl/review_sem")
from bwmn import logistics as L
import sexp

SCR = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/impl/review_sem/work")
SCR.mkdir(exist_ok=True)
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 777
rng = random.Random(SEED)
N_SEQ = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def val_trace(prob, plan_text):
    pp = SCR / "plan.txt"
    pp.write_text(plan_text)
    r = subprocess.run([str(L.VAL_EXE), "-v", str(L.DOMAIN_PDDL), str(prob), str(pp)], capture_output=True,
                       text=True, timeout=60)
    out = r.stdout or ""
    steps = []
    cur = None
    fail_time = None
    for line in out.splitlines():
        m = re.match(r"Checking next happening \(time (\d+)\)", line)
        if m:
            cur = {"t": int(m.group(1)), "del": [], "add": []}
            steps.append(cur)
            continue
        m = re.match(r"(Deleting|Adding) \((.*)\)$", line.strip())
        if m and cur is not None:
            cur["del" if m.group(1) == "Deleting" else "add"].append(tuple(m.group(2).lower().split()))
        if line.startswith("Plan failed because of unsatisfied precondition"):
            fail_time = cur["t"] if cur else None
    executed = "Plan executed successfully" in out
    valid = "Plan valid" in out
    return {"rc": r.returncode, "executed": executed, "valid": valid, "steps": steps, "fail_time": fail_time,
            "out": out}


def own_init(prob):
    _, _, objs, init, goal = sexp.problem(prob)
    return objs, frozenset(init), frozenset(goal)


def legal(s):
    return [a for a in L.ground_actions(L.objs_of(s), s, self_loops=True) if L.applicable(s, a)]


def illegal_typed(s):
    return [a for a in L.ground_actions(L.objs_of(s), s, self_loops=True) if not L.applicable(s, a)]


def wild(objs_all, s):
    """right arity, random args, often repeated / type-wrong"""
    op = rng.choice(sorted(L.ARITY))
    k = L.ARITY[op]
    mode = rng.random()
    if mode < 0.3:  # same object in every slot
        x = rng.choice(objs_all)
        return (op,) + (x,) * k
    if mode < 0.6:  # mutate a legal action: one slot -> random object
        b = list(rng.choice(legal(s)))
        j = rng.randrange(1, len(b))
        b[j] = rng.choice(objs_all)
        return tuple(b)
    if mode < 0.8:  # mutate a legal action: copy one argument into another slot (repeated object)
        b = list(rng.choice(legal(s)))
        i, j = rng.sample(range(1, len(b)), 2)
        b[j] = b[i]
        return tuple(b)
    return (op,) + tuple(rng.choice(objs_all) for _ in range(k))


def make_sequences(objs, init, goal, gt):
    seqs = []
    names = L.all_objects(objs)
    if gt:
        for _ in range(2):  # reference plan with self-loops sprinkled in (should still reach the goal)
            s = init
            out = []
            for a in gt:
                while rng.random() < 0.25:
                    sl = [b for b in legal(s) if L.is_self_loop(b)]
                    if not sl:
                        break
                    out.append(rng.choice(sl))
                out.append(a)
                s = L.step(s, a)[0]
            if rng.random() < 0.5:
                sl = [b for b in legal(s) if L.is_self_loop(b)]
                if sl:
                    out.append(rng.choice(sl))
            seqs.append(("gt+selfloops", out))
        for _ in range(2):  # one action replaced by noise
            out = list(gt)
            j = rng.randrange(len(out))
            s = L.execute(init, goal, out[:j])["final"]
            r = rng.random()
            out[j] = rng.choice(legal(s)) if r < 0.4 else (rng.choice(illegal_typed(s)) if r < 0.7 else wild(names, s))
            seqs.append(("gt+noise", out))
        s = L.execute(init, goal, gt)["final"]  # legal actions after the goal is reached
        out = list(gt)
        for _ in range(rng.randrange(1, 4)):
            a = rng.choice(legal(s))
            out.append(a)
            s = L.step(s, a)[0]
        seqs.append(("gt+tail", out))
    for _ in range(N_SEQ):  # random walks ending in a candidate action (legal / typed-illegal / wild)
        s = init
        out = []
        for _ in range(rng.randrange(0, 15)):
            a = rng.choice(legal(s))
            out.append(a)
            s = L.step(s, a)[0]
        r = rng.random()
        if r < 0.25:
            a = rng.choice(legal(s))
        elif r < 0.55:
            a = rng.choice(illegal_typed(s))
        else:
            a = wild(names, s)
        out.append(a)
        s2, ok = L.step(s, a)
        if ok:
            for _ in range(rng.randrange(0, 4)):
                b = rng.choice(legal(s2))
                out.append(b)
                s2 = L.step(s2, b)[0]
        seqs.append(("walk", out))
    return seqs


def compare(prob, init, goal, plan, kind, stats, bad):
    ex = L.execute(init, goal, plan)
    v = val_trace(prob, L.plan_to_pddl(plan) if plan else "")
    stats["val_calls"] += 1
    e_valid, e_succ = ex["valid"], ex["success"]
    if e_valid != v["executed"] or e_succ != v["valid"]:
        bad.append({"kind": kind, "prob": str(prob), "plan": plan, "exec": (e_valid, e_succ, ex["fail_step"]),
                    "val": (v["executed"], v["valid"], v["fail_time"], v["rc"]), "out": v["out"][-800:]})
        return
    if not e_valid:
        if v["fail_time"] is not None and v["fail_time"] != ex["fail_step"] + 1:
            bad.append({"kind": kind, "why": "fail step", "prob": str(prob), "plan": plan, "exec": ex["fail_step"],
                        "val": v["fail_time"]})
            return
        if v["fail_time"] is None:
            stats["val_no_failtime(rc=%s)" % v["rc"]] += 1
    s_e = init
    s_v = init
    n_exec = len(plan) if e_valid else ex["fail_step"]
    applied = [st for st in v["steps"] if not (not e_valid and st["t"] == ex["fail_step"] + 1)]
    if v["fail_time"] is None and not e_valid:
        applied = applied[:n_exec]
    if len(applied) != n_exec:
        bad.append({"kind": kind, "why": "n steps", "prob": str(prob), "plan": plan, "n_exec": n_exec,
                    "n_val": len(applied)})
        return
    for i in range(n_exec):
        s_e = L.step(s_e, plan[i])[0]
        st = applied[i]
        s_v = (s_v - frozenset(st["del"])) | frozenset(st["add"])
        if s_e != s_v:
            bad.append({"kind": kind, "why": "state", "prob": str(prob), "i": i, "plan": plan,
                        "diff": sorted(map(str, s_e ^ s_v))})
            return
    stats["ok"] += 1
    stats["kind:" + kind] += 1
    stats["executed_self_loops"] += sum(L.is_self_loop(a) for a in plan[:n_exec])
    stats["goal_true"] += e_succ
    stats["illegal_end"] += (not e_valid)
    if not e_valid:
        a = plan[ex["fail_step"]]
        stats["fail_repeated_obj"] += len(set(a[1:])) < len(a) - 1
        stats["fail_typewrong"] += any((t, x) not in init for t, x in zip(L.PARAM_TYPES[a[0]], a[1:]))
        stats["fail_selfloop_shape"] += L.is_self_loop(a)


t0 = time.time()
stats = Counter()
bad = []
gt = L.planbench_gt()
ids = sorted(rng.sample(range(2, 287), 36))
insts = []
for i in ids:
    p = L.planbench_path(i)
    objs, init, goal = L.parse_problem(p)
    _, init2, goal2 = own_init(p)
    assert init2 == init and goal2 == goal
    insts.append(("pb%d" % i, p, objs, init, goal, gt[i]))
k = 0
while k < 36:  # module generator, rename 0.5; A* plan as reference
    g = L.generate(rng, rng.choice((1, 2, 2, 3, 3, 3)), rename=0.5)
    if g is None:
        continue
    objs, init, goal = g
    p = SCR / f"gen{SEED}_{k}.pddl"
    p.write_text(L.write_problem(objs, init, goal, name=f"gen{k}"))
    _, init2, goal2 = own_init(p)
    assert init2 == init and goal2 == goal, "write_problem round trip"
    plan = L.astar_plan(init, goal, max_expansions=200000)
    insts.append((f"gen{k}", p, objs, init, goal, plan))
    k += 1
k = 0
for name, p, objs, init, goal, plan in list(insts[::3]):  # intermediate states (packages inside vehicles)
    if not plan or len(plan) < 2:
        continue
    j = rng.randrange(1, len(plan))
    s = L.execute(init, goal, plan[:j])["final"]
    q = SCR / f"mid{SEED}_{k}.pddl"
    q.write_text(L.write_problem(L.objs_of(s), s, goal, name=f"mid{k}"))
    _, s2, g2 = own_init(q)
    assert s2 == s and g2 == goal
    insts.append((f"mid{k}", q, L.objs_of(s), s, goal, plan[j:]))
    k += 1
for name, p, objs, init, goal, plan in insts:
    for kind, seq in make_sequences(objs, init, goal, plan):
        compare(p, init, goal, seq, kind, stats, bad)
for name, p, objs, init, goal, plan in insts[:5]:  # empty plans
    compare(p, init, goal, [], "empty", stats, bad)
    if plan:
        s = L.execute(init, goal, plan)["final"]
        q = SCR / "goalheld.pddl"
        q.write_text(L.write_problem(L.objs_of(s), s, goal, name="gh"))
        compare(q, s, goal, [], "empty_goalheld", stats, bad)
print(json.dumps({"seed": SEED, "instances": len(insts), "stats": stats, "n_bad": len(bad),
                  "secs": round(time.time() - t0, 1)}, default=str))
for b in bad[:10]:
    print(json.dumps(b, default=str)[:1500])
