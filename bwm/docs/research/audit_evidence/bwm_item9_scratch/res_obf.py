import json, os, re, sys, subprocess
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import mystery as M
from bwmn import blocks as B
SCR = os.path.dirname(os.path.abspath(__file__))
L = "C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/llm_planning_analysis"
d = json.load(open(L + "/results/obfuscated_randomized_logistics/o1-preview_chat/task_1_plan_generation_zero_shot_pddl.json", encoding="utf-8"))
tmp = os.path.join(SCR, "obf_tmp")
os.makedirs(tmp, exist_ok=True)
ok_flag = ok_ex = ok_val = dis = dis_val = 0
gt_ok = 0
NL = chr(10)


def parse(t):
    out = []
    for ln in (t or "").replace(chr(92), " ").lower().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith(";"):
            continue
        out.append(tuple(ln.replace("(", " ").replace(")", " ").split()))
    return out


for x in d["instances"]:
    q = x["query"]
    dom_txt = q.split("[DOMAIN]" + NL)[1].split("[QUERY PROBLEM]" + NL)[0]
    prob_txt = q.split("[QUERY PROBLEM]" + NL)[1].split("[PLAN]")[0]
    dp, pp, plp = [os.path.join(tmp, n) for n in ("d.pddl", "p.pddl", "plan.txt")]
    open(dp, "w").write(dom_txt)
    open(pp, "w").write(prob_txt)
    dom = M.parse_domain(dp)
    objs, init, goal = M.parse_problem(pp)

    def ex(plan):
        s = init
        for a in plan:
            sc = dom.actions.get(a[0])
            if sc is None or len(a) - 1 != len(sc.params):
                return False
            b = dict(zip(sc.params, a[1:]))
            pre, add, dele = M._subst(sc.pre, b), M._subst(sc.add, b), M._subst(sc.dele, b)
            if not pre <= s:
                return False
            s = (s - dele) | add
        return goal <= s

    plan = parse(x["extracted_llm_plan"])
    e = ex(plan)
    open(plp, "w").write("".join("(" + " ".join(a) + ")" + NL for a in plan))
    out = subprocess.run([str(B.VAL_EXE), dp, pp, plp], capture_output=True, text=True, timeout=60).stdout
    v = ("Plan valid" in out) if plan else False
    f = bool(x["llm_correct"])
    ok_flag += f; ok_ex += e; ok_val += v; dis += (e != f); dis_val += (v != f)
    gt_ok += ex(parse(x["ground_truth_plan"]))
print(dict(n=len(d["instances"]), flag=ok_flag, ours=ok_ex, val=ok_val, dis_ours=dis, dis_val=dis_val, gt_valid_ours=gt_ok))
