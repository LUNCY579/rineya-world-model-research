import json, sys, random, importlib.util
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
spec = importlib.util.spec_from_file_location("bw_b3_eval_copy", ROOT / "scripts/bw_b3_eval.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print("torch imported at module import?", "torch" in sys.modules)
OUT = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/b3eval_out")
m.OUT = OUT
rng = random.Random(0)
fresh = json.loads((ROOT / "runs/BW_val_large/instances_b3_fresh.json").read_text())
mirror = json.loads((ROOT / "runs/BW_val_large/instances_mirror_hard.json").read_text())
p_succ = {"v3": 0.35, "v2c": 0.2, "v2": 0.05}
for arm in ("v3", "v2c", "v2"):
    for sname, inst in (("b3fresh", fresh), ("mirror", mirror)):
        with open(OUT / f"plans_{arm}_{sname}.jsonl", "w", encoding="utf-8") as fh:
            for i, r in enumerate(inst):
                fh.write(json.dumps({"id": f"{sname}/{i}", "n_blocks": len(r["objs"]), "success": rng.random() < p_succ[arm],
                                     "valid": True, "len": 3, "verified": False, "rounds": 1, "gen_tokens": 100,
                                     "fail": None, "plan": []}) + "\n")
class A: pass
m.analyze(A())
rep = json.loads((OUT / "B3_REPORT.json").read_text())
print(sorted(rep["cells"]), sorted(rep["contrasts"]), rep.get("primary_reading"))
