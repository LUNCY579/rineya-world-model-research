import json, os, sys
from collections import Counter, defaultdict
from statistics import mean
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import mystery as M
SCR = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(SCR, "valgt.py")).read().split("def bfs_len")[0].split("# typed BFS")[1].replace("on small instances to check optimality of GT lengths", ""))
L = "C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning"
LA = L + "/llm_planning_analysis/results/logistics/"
PB = L + "/plan-bench"
d = json.load(open(LA + "gpt-4_chat/task_1_zero_shot_plan_generation.json", encoding="utf-8"))
x1 = [x for x in d["instances"] if x["instance_id"] == 1][0]
print("instance 1 GT:", x1["ground_truth_plan"])
gt = {int(k): v for k, v in json.load(open(os.path.join(SCR, "gt_plans.json"))).items()}
dom = M.parse_domain(PB + "/instances/logistics/generated_domain.pddl")
byp = defaultdict(list); app0 = []; nground = []
for i in range(1, 287):
    objs, init, goal = M.parse_problem(PB + f"/instances/logistics/generated_basic/instance-{i}.pddl")
    acts = typed_ground(dom, objs, init)
    nground.append(len(acts))
    app0.append(sum(1 for a, pre, add, dele in acts if pre <= init))
    npk = sum(1 for f in init if f[0] == "obj")
    if i in gt:
        byp[npk].append(len(gt[i]))
print("typed ground actions per instance: min/max/mean", min(nground), max(nground), round(mean(nground), 1))
print("applicable actions in initial state: min/max/mean", min(app0), max(app0), round(mean(app0), 1))
print("GT length by #packages:", {k: (len(v), min(v), max(v), round(mean(v), 1)) for k, v in sorted(byp.items())})
t5 = json.load(open(PB + "/prompts/logistics/task_5_plan_generalization.json", encoding="utf-8"))
ids = [x["instance_id"] for x in t5["instances"]]
print("task5 n", len(ids), min(ids), max(ids), t5.get("prompt_type"), sorted(t5["instances"][0].keys()))
for f in ["task_8_2_full_to_partial", "task_8_3_partial_to_full"]:
    t = json.load(open(PB + f"/prompts/logistics/{f}.json", encoding="utf-8"))
    print(f, len(t["instances"]))
