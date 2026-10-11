"""Read-only inventory of PlanBench Logistics assets (pure python; no torch)."""
import json, re, sys, glob, os
from collections import Counter
from statistics import mean, median

sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import mystery as M  # pure python generic STRIPS parser/executor

L = "C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning"
PB = L + "/plan-bench"
LA = L + "/llm_planning_analysis"
DOM = PB + "/instances/logistics/generated_domain.pddl"
dom = M.parse_domain(DOM)
print("domain", dom.name, "preds", dom.preds)
for n, s in sorted(dom.actions.items()):
    print(" ", n, s.params, "pre", s.pre, "add", s.add, "del", s.dele)


def stats(path):
    objs, init, goal = M.parse_problem(path)
    by = Counter()
    for f in init:
        if len(f) == 2:
            by[f[0]] += 1
    cities = [f[1] for f in init if f[0] == "city"]
    locs = [f[1] for f in init if f[0] == "location"]
    airports = [f[1] for f in init if f[0] == "airport"]
    incity = {f[1]: f[2] for f in init if f[0] == "in-city"}
    locs_per_city = Counter(incity.values())
    pkgs = [f[1] for f in init if f[0] == "obj"]
    at = {f[1]: f[2] for f in init if f[0] == "at"}
    # goals
    gp = [g for g in goal if g[0] == "at"]
    other_goals = [g for g in goal if g[0] != "at"]
    inter = sum(1 for g in gp if incity.get(at.get(g[1])) != incity.get(g[2]))
    already = sum(1 for g in gp if at.get(g[1]) == g[2])
    name = re.search(r"\(problem\s+([^\s)]+)", open(path).read()).group(1)
    return dict(name=name, n_objs=len(objs), packages=len(pkgs), trucks=by["truck"], airplanes=by["airplane"],
                cities=len(cities), locations=len(locs), airports=len(airports),
                max_locs_per_city=max(locs_per_city.values()) if locs_per_city else 0,
                min_locs_per_city=min(locs_per_city.values()) if locs_per_city else 0,
                goal=len(goal), goal_at=len(gp), goal_other=len(other_goals), goal_intercity=inter,
                goal_already=already, n_init=len(init), objs=objs, init=init, goal_set=goal)


def summarize(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows]
        out[k] = dict(min=min(v), max=max(v), mean=round(mean(v), 2), median=median(v), hist=dict(sorted(Counter(v).items())))
    return out


KEYS = ["n_objs", "packages", "trucks", "airplanes", "cities", "locations", "airports", "max_locs_per_city",
        "goal", "goal_intercity", "goal_already", "n_init"]
res = {}
for sub in ["generated_basic", "generated"]:
    d = PB + "/instances/logistics/" + sub
    files = sorted(glob.glob(d + "/instance-*.pddl"), key=lambda p: int(re.search(r"instance-(\d+)", p).group(1)))
    rows = {}
    for p in files:
        i = int(re.search(r"instance-(\d+)", p).group(1))
        rows[i] = stats(p)
    res[sub] = rows
    print("\n==", sub, "n files", len(files), "ids", min(rows), max(rows))
    print(json.dumps(summarize(list(rows.values()), KEYS), indent=None))
    # duplicates by (objs,init,goal)
    sig = Counter((tuple(r["objs"]), r["init"], r["goal_set"]) for r in rows.values())
    print("exact duplicate instances:", sum(c - 1 for c in sig.values() if c > 1))
    names = Counter(r["name"] for r in rows.values())
    print("distinct problem names:", len(names), "most common:", names.most_common(5))
    # combos
    combo = Counter((r["cities"], r["max_locs_per_city"], r["packages"], r["airplanes"]) for r in rows.values())
    print("distinct (cities,city_size,packages,airplanes) combos:", len(combo))
    # ids subset 2..201 (results range) summary
    if sub == "generated_basic":
        for lo, hi in [(2, 201), (202, 286)]:
            sel = [rows[i] for i in range(lo, hi + 1) if i in rows]
            print(f"  ids {lo}-{hi}: n={len(sel)}", json.dumps({k: (min(r[k] for r in sel), max(r[k] for r in sel), round(mean(r[k] for r in sel), 2)) for k in KEYS}))

json.dump({sub: {i: {k: v for k, v in r.items() if k not in ("objs", "init", "goal_set")} for i, r in rows.items()}
           for sub, rows in res.items()},
          open(os.path.join(os.path.dirname(__file__), "inst_stats.json"), "w"), indent=0)

# ---------------------------------------------------------------- ground-truth plans (task_1 prompts)


def parse_plan_text(t):
    acts = []
    for line in t.splitlines():
        line = line.strip().lower()
        if not line or line.startswith(";"):
            continue
        line = line.strip("()")
        acts.append(tuple(line.split()))
    return acts


gb = res["generated_basic"]
t1 = json.load(open(PB + "/prompts/logistics/task_1_plan_generation.json", encoding="utf-8"))
gt = {}
for ins in t1["instances"]:
    gt[ins["instance_id"]] = parse_plan_text(ins["ground_truth_plan"])
# instance 1 GT from other task files
for task in ["task_2_plan_optimality", "task_7_plan_execution", "task_3_plan_verification", "task_4_plan_reuse",
             "task_6_replanning", "task_8_1_goal_shuffling"]:
    dd = json.load(open(PB + f"/prompts/logistics/{task}.json", encoding="utf-8"))
    ids = [x["instance_id"] for x in dd["instances"]]
    print(task, "n", len(ids), "ids", min(ids), max(ids), "prompt_type", dd.get("prompt_type"), "keys", sorted(dd["instances"][0].keys()))
lens = {i: len(p) for i, p in gt.items()}
print("\nGT plans (task_1 prompts): n", len(lens), "len min/max/mean/median", min(lens.values()), max(lens.values()),
      round(mean(lens.values()), 2), median(lens.values()))
print("GT len hist", dict(sorted(Counter(lens.values()).items())))
for lo, hi in [(2, 201), (202, 286)]:
    v = [lens[i] for i in range(lo, hi + 1)]
    print(f"  ids {lo}-{hi}: n={len(v)} min {min(v)} max {max(v)} mean {round(mean(v), 2)} median {median(v)}")
opc = Counter(a[0] for p in gt.values() for a in p)
print("GT action-type counts", dict(opc))
# simulate GT with generic STRIPS executor
bad = []
for i, p in gt.items():
    r = gb[i]
    ex = M.execute(dom, r["init"], r["goal_set"], p)
    if not ex["success"]:
        bad.append((i, ex["fail_step"]))
print("GT plans failing generic executor:", len(bad), bad[:10])
# self-loop actions in GT (repeated objects, rejected by M.ground)
sl = [(i, a) for i, p in gt.items() for a in p if len(set(a[1:])) != len(a) - 1]
print("GT actions with repeated objects:", len(sl), sl[:5])
json.dump({str(i): [" ".join(a) for a in p] for i, p in gt.items()},
          open(os.path.join(os.path.dirname(__file__), "gt_plans.json"), "w"), indent=0)
