import json, subprocess, sys
from pathlib import Path
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
CL = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
VAL = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/bench/planbench/VAL/bin/validate/validate.exe"
D = CL / "instances/logistics/generated_domain.pddl"
ok = bad = 0
for x in json.loads((CL / "prompts/logistics/task_1_plan_generation.json").read_text())["instances"]:
    i = int(x["instance_id"]); p = Path("valgt") / f"p{i}.txt"; p.write_text(x["ground_truth_plan"])
    out = subprocess.run([VAL, str(D), str(CL / f"instances/logistics/generated_basic/instance-{i}.pddl"), str(p)], capture_output=True, text=True, timeout=60).stdout
    if "Plan valid" in out: ok += 1
    else: bad += 1
print("GT plans VAL-valid", ok, "invalid", bad)
