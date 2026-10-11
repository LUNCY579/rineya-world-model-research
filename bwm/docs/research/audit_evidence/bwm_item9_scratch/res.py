"""Success rates of published PlanBench Logistics results + independent re-check (own executor, local VAL)."""
import json, re, sys, glob, os, subprocess
from collections import Counter

ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import mystery as M
from bwmn import blocks as B

SCR = os.path.dirname(os.path.abspath(__file__))
L = "C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning"
LA = L + "/llm_planning_analysis"
PB = L + "/plan-bench"
DOMS = {"logistics": LA + "/instances/logistics/generated_domain.pddl",
        "obfuscated_randomized_logistics": LA + "/instances/obfuscated_randomized_logistics/generated_domain.pddl"}
INST = {"logistics": LA + "/instances/logistics/generated_basic/instance-{}.pddl",
        "obfuscated_randomized_logistics": LA + "/instances/obfuscated_randomized_logistics/generated_basic/instance-{}.pddl"}
DO_VAL = "--val" in sys.argv


def ground_rep(dom, a):  # like M.ground but allows repeated objects (PDDL/VAL semantics)
    sc = dom.actions.get(a[0]) if a else None
    if sc is None or len(a) - 1 != len(sc.params):
        return None
    b = dict(zip(sc.params, a[1:]))
    return M._subst(sc.pre, b), M._subst(sc.add, b), M._subst(sc.dele, b)


def execute(dom, init, goal, plan):
    s = init
    for i, a in enumerate(plan):
        g = ground_rep(dom, a)
        if g is None or not g[0] <= s:
            return False, i
        s = (s - g[2]) | g[1]
    return goal <= s, None


OBF = False


def parse_plan(p):
    if isinstance(p, list):
        lines = p
    else:
        lines = (p or "").splitlines()
    out = []
    for ln in lines:
        ln = ln.replace("\\", " ").strip().lower()
        if not ln or ln.startswith(";"):
            continue
        toks = ln.replace("(", " ").replace(")", " ").split()
        toks = [("object_" + t[1:]) if (OBF and re.fullmatch(r"o[0-9]+", t)) else t for t in toks]
        out.append(tuple(toks))
    return out


def val(domain, inst, plan):
    d = os.path.join(SCR, "valtmp")
    os.makedirs(d, exist_ok=True)
    pp = os.path.join(d, "plan.txt")
    with open(pp, "w") as fh:
        fh.write("".join("(" + " ".join(a) + ")\n" for a in plan))
    out = subprocess.run([str(B.VAL_EXE), domain, inst, pp], capture_output=True, text=True, timeout=60).stdout
    return "Plan valid" in out


summary = []
for f in sorted(glob.glob(LA + "/results/*logistics/*/*.json")):
    rel = os.path.relpath(f, LA).replace("\\", "/")
    dname = rel.split("/")[1]
    d = json.load(open(f, encoding="utf-8"))
    dom = M.parse_domain(DOMS[dname])
    globals()["OBF"] = dname.startswith("obfuscated")
    ins = d["instances"]
    ids = [x["instance_id"] for x in ins]
    field = "llm_correct" if "llm_correct" in ins[0] else "correct"
    flags = [bool(x.get(field)) for x in ins]
    missing_field = sum(1 for x in ins if field not in x)
    n_ok = sum(flags)
    # independent re-check
    agree = dis = 0
    ours_ok = 0
    val_ok = val_dis = 0
    dis_list = []
    lens_ok = []
    for x, fl in zip(ins, flags):
        plan = parse_plan(x.get("extracted_llm_plan"))
        objs, init, goal = M.parse_problem(INST[dname].format(x["instance_id"]))
        ok, _ = execute(dom, init, goal, plan)
        ours_ok += ok
        if ok == fl:
            agree += 1
        else:
            dis += 1
            dis_list.append(x["instance_id"])
        if DO_VAL:
            v = val(DOMS[dname], INST[dname].format(x["instance_id"]), plan) if plan else False
            val_ok += v
            val_dis += (v != fl)
    rec = dict(file=rel, engine=d.get("engine"), task=d.get("task"), prompt_type=d.get("prompt_type"),
               domain=d.get("domain"), n=len(ins), ids=f"{min(ids)}-{max(ids)}", unique_ids=len(set(ids)),
               field=field, missing_field=missing_field, n_correct=n_ok, rate=round(n_ok / len(ins), 4),
               ours_exec_correct=ours_ok, ours_disagree=dis, ours_disagree_ids=dis_list[:10])
    if DO_VAL:
        rec.update(val_correct=val_ok, val_disagree_with_flag=val_dis)
    if "relaxed_checks" in ins[0]:
        rec["relaxed_no_relax_eq1"] = sum(1 for x in ins if x["relaxed_checks"].get("no_relax") == 1)
    if "steps" in ins[0]:
        rec["steps_hist"] = dict(sorted(Counter(x["steps"] for x in ins).items()))
        rec["context_window_hit"] = sum(bool(x.get("context_window_hit")) for x in ins)
        rec["could_not_extract"] = sum(bool(x.get("could_not_extract")) for x in ins)
    summary.append(rec)
    print(json.dumps(rec))

json.dump(summary, open(os.path.join(SCR, "results_summary.json"), "w"), indent=1)
