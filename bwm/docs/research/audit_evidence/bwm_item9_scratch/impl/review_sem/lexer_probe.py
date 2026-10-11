import sys, json
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L

gt = L.planbench_gt()
p = L.planbench_path(2)
objs, init, goal = L.parse_problem(p)
G = L.plan_to_pddl(gt[2])
for ch in [",", ".", ":", "\\", "[", "]", "{", "}", "|", "'", '"', "\t", "!", "?", "-", "_", "#", "="]:
    txt = G.replace("p0 a1", "p0" + ch + " a1", 1)
    acts = L.plan_from_pddl(txt)
    ex = L.execute(init, goal, acts)
    v = L.val_run(p, txt)
    flag = "DISAGREE" if ex["success"] != v["valid"] else ""
    print(repr(ch), "exec", ex["success"], "VAL", v["valid"], v["returncode"], flag)
d = json.loads((L.RESULTS_DIR / "gpt-4_chat" / "task_1_zero_shot_plan_generation.json").read_text(encoding="utf-8"))
x = [x for x in d["instances"] if int(x["instance_id"]) == 76][0]
raw = x["extracted_llm_plan"]
raw = "\n".join(raw) if isinstance(raw, list) else raw
print([l for l in raw.splitlines() if "," in l])
o, i, g = L.parse_problem(L.planbench_path(76))
acts = L.plan_from_pddl(raw)
ex = L.execute(i, g, acts)
print("76 exec fail step", ex["fail_step"], "comma action idx", [k for k, a in enumerate(acts) if any("," in t for t in a)])
# id 29 with commas replaced: does the outcome change?
d = json.loads((L.RESULTS_DIR / "gpt-4_chat" / "task_1_plan_generation.json").read_text(encoding="utf-8"))
x = [x for x in d["instances"] if int(x["instance_id"]) == 29][0]
raw = x["extracted_llm_plan"]
raw = "\n".join(raw) if isinstance(raw, list) else raw
o, i, g = L.parse_problem(L.planbench_path(29))
for t in (raw, raw.replace(",", " ")):
    ex = L.execute(i, g, L.plan_from_pddl(t))
    print("29 exec", ex["success"], "fail_step", ex["fail_step"])
