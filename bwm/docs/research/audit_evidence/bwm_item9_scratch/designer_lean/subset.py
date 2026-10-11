"""LEAN design: pre-registered out-of-range subset of PlanBench Logistics Task 1, and published per-instance flags."""
import json, sys
from collections import Counter
from pathlib import Path
import lgdom as L
from bwmn import mystery as M

CL = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning")
R = CL / "llm_planning_analysis" / "results" / "logistics"
T = L.planbench_t1()
by = {x["id"]: x for x in T}
oor = sorted((x for x in T if x["np"] >= 4), key=lambda x: (x["np"], x["id"]))
inr = [x for x in T if x["np"] <= 3]
N = 60
k = len(oor) / N
sub = [oor[int(i * k)] for i in range(N)]
print("in-range (1-3 packages):", len(inr), "| out-of-range (4-7):", len(oor))
print("subset n", len(sub), "packages", sorted(Counter(x["np"] for x in sub).items()),
      "| OOR population", sorted(Counter(x["np"] for x in oor).items()))
print("subset ids <=200:", sum(x["id"] <= 200 for x in sub), "| 2..201:", sum(x["id"] <= 201 for x in sub),
      "| >=202:", sum(x["id"] >= 202 for x in sub))
print("subset GT length", sorted(Counter((len(x["gt"]) // 10) * 10 for x in sub).items()),
      "| mean", round(sum(len(x["gt"]) for x in sub) / len(sub), 1),
      "| OOR mean", round(sum(len(x["gt"]) for x in oor) / len(oor), 1))
print("subset (cities, airplanes)", sorted(Counter((x["nc"], x["na"]) for x in sub).items()))
print("OOR ids 2..201:", sum(x["id"] <= 201 for x in oor), "| in-range ids 2..201:", sum(x["id"] <= 201 for x in inr))
sub_ids = [x["id"] for x in sub]
json.dump({"oor_subset_ids": sub_ids, "rule": "out-of-range = packages >= 4; sort by (packages, id); take index int(i*212/60), i=0..59"},
          open("oor_subset.json", "w"))
print("subset ids", sub_ids)


def flags(path, field):
    d = json.loads(path.read_text())
    ins = d["instances"]
    field = "llm_correct" if "llm_correct" in ins[0] else "correct"
    tv = Counter(type(x.get(field)).__name__ + ":" + str(x.get(field)) for x in ins)
    print("   value types", dict(tv))
    return {int(x["instance_id"]): (x.get(field) is True or x.get(field) == "True") for x in ins if field in x}


files = {"o1-preview 0-shot PDDL": (R / "o1-preview_chat/task_1_plan_generation_zero_shot_pddl.json", "correct"),
         "GPT-4 1-shot NL": (R / "gpt-4_chat/task_1_plan_generation.json", "correct"),
         "GPT-4 0-shot NL": (R / "gpt-4_chat/task_1_zero_shot_plan_generation.json", "correct"),
         "LLaMA-3.1-405B 0-shot PDDL": (R / "llama-3.1-405b_aws/task_1_plan_generation_zero_shot_pddl.json", "correct"),
         "GPT-3.5-instruct 1-shot NL": (R / "gpt-3.5-turbo-instruct/task_1_plan_generation.json", "llm_correct")}
for name, (p, f) in files.items():
    fl = flags(p, f)
    ov_sub = [i for i in sub_ids if i in fl]
    ov_oor = [x["id"] for x in oor if x["id"] in fl]
    ov_in = [x["id"] for x in inr if x["id"] in fl]
    print(f"{name}: ids {min(fl)}-{max(fl)} n {len(fl)} correct {sum(fl.values())} | on OOR subset overlap "
          f"{sum(fl[i] for i in ov_sub)}/{len(ov_sub)} | all OOR overlap {sum(fl[i] for i in ov_oor)}/{len(ov_oor)} "
          f"| in-range overlap {sum(fl[i] for i in ov_in)}/{len(ov_in)}")
