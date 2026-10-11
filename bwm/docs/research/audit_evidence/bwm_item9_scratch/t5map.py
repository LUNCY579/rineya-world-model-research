import json, re, sys
from pathlib import Path
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import mystery as M
PB = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
LG = PB / "instances" / "logistics"
dom = M.parse_domain(LG / "generated_domain.pddl")
d = json.loads((PB / "prompts/logistics/task_5_plan_generalization.json").read_text())
insts = {int(p.stem.split("-")[1]): M.parse_problem(p) for p in (LG / "generated").glob("instance-*.pddl")}
for x in d["instances"]:
    plan = [tuple(m.split()) for m in re.findall(r"\(([^()]+)\)", x["ground_truth_plan"].lower())]
    hits = [k for k, (o, i, g) in insts.items() if M.execute(dom, i, g, plan)["success"]]
    o, i, g = insts[x["instance_id"]]
    ex = M.execute(dom, i, g, plan)
    print(x["instance_id"], len(plan), "valid", ex["valid"], "fail_step", ex["fail_step"], "succeeds on", hits[:5], "| first action", plan[0], "| query has", re.findall(r"\[STATEMENT\]", x["query"])[:1])
