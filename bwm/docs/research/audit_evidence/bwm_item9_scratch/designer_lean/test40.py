"""LEAN test list: PlanBench Logistics t1, packages 4-7, ids 2..200 (fully paired with the published per-instance
files), 10 per package count; within a stratum sort by id and take index int(i*n_k/10), i=0..9. Sizes only decide
membership; published flags are read afterwards for documentation."""
import json, sys, hashlib, statistics as st
from collections import Counter
from pathlib import Path
import lgdom as L
from bwmn import stats as S
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
T = L.planbench_t1()
pool = [x for x in T if x["np"] >= 4 and 2 <= x["id"] <= 200]
ids = []
for k in (4, 5, 6, 7):
    s = sorted((x for x in pool if x["np"] == k), key=lambda x: x["id"])
    ids += [s[int(i * len(s) / 10)]["id"] for i in range(10)]
by = {x["id"]: x for x in T}
print("ids", ids)
print("sha256(json ids)", hashlib.sha256(json.dumps(ids).encode()).hexdigest())
gl = [len(by[i]["gt"]) for i in ids]
print("GT len min/mean/max", min(gl), round(st.mean(gl), 1), max(gl), "| by pk",
      {k: (min(len(by[i]["gt"]) for i in ids if by[i]["np"] == k), max(len(by[i]["gt"]) for i in ids if by[i]["np"] == k)) for k in (4, 5, 6, 7)})
print("GT bins", sorted(Counter((min(l, 39) // 10) * 10 for l in gl).items()))
print("(cities, planes)", sorted(Counter((by[i]["nc"], by[i]["na"]) for i in ids).items()))
CL = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/llm_planning_analysis/results/logistics")
files = {"o1-preview 0-shot PDDL": "o1-preview_chat/task_1_plan_generation_zero_shot_pddl.json",
         "GPT-4 1-shot NL": "gpt-4_chat/task_1_plan_generation.json",
         "GPT-4 1-shot PDDL": "gpt-4_chat/task_1_plan_generation_pddl.json",
         "GPT-4 0-shot NL": "gpt-4_chat/task_1_zero_shot_plan_generation.json",
         "GPT-4 0-shot PDDL": "gpt-4_chat/task_1_zero_shot_plan_generation_pddl.json",
         "LLaMA-3.1-405B 0-shot PDDL": "llama-3.1-405b_aws/task_1_plan_generation_zero_shot_pddl.json",
         "GPT-3.5-instruct 1-shot NL": "gpt-3.5-turbo-instruct/task_1_plan_generation.json"}
for name, f in files.items():
    p = CL / f
    if not p.exists():
        print(name, "MISSING", f); continue
    ins = json.loads(p.read_text())["instances"]
    fld = "llm_correct" if "llm_correct" in ins[0] else "correct"
    fl = {int(x["instance_id"]): bool(x.get(fld)) for x in ins}
    cov = [i for i in ids if i in fl]
    per = {k: sum(fl[i] for i in ids if by[i]["np"] == k and i in fl) for k in (4, 5, 6, 7)}
    print(f"{name}: covers {len(cov)}/40, correct {sum(fl[i] for i in cov)}/{len(cov)} by pk {per}")
print("McNemar sanity: b=6,c=0 ->", S.mcnemar_exact(6, 0), "| b=10,c=1 ->", S.mcnemar_exact(10, 1), "| b=5,c=0 ->", S.mcnemar_exact(5, 0))
