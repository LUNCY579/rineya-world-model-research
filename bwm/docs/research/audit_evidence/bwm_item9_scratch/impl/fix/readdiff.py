import sys, json
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
d = json.loads((L.RESULTS_DIR / "gpt-4_chat" / "task_1_zero_shot_plan_generation.json").read_text(encoding="utf-8"))
x = next(y for y in d["instances"] if int(y["instance_id"]) == 50)
raw = x["extracted_llm_plan"]; raw = "\n".join(raw) if isinstance(raw, list) else raw
print(ascii(raw))
out = L.val_run(L.planbench_path(50), raw, verbose=True)
print(out["stdout"][:1500])
print("LISTING", L.val_listing(out["stdout"]))
print("OURS   ", sorted(L.read_val_plan(raw)))
