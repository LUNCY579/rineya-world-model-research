import json, sys, random, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("bw_b3_eval", r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/scripts/bw_b3_eval.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert 'torch' not in sys.modules
m.OUT = Path("evalout")
rng = random.Random(3)
fresh = json.load(open(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_val_large/instances_b3_fresh.json"))
p = {"v3": 0.30, "v2c": 0.15, "v2": 0.05}
for arm in ("v3","v2c","v2"):
    for s in ("b3fresh","mirror"):
        with open(m.OUT / f"plans_{arm}_{s}.jsonl", "w") as fh:
            for i, r in enumerate(fresh):
                fh.write(json.dumps({"id": f"{s}/{i}", "n_blocks": r["n"], "success": rng.random() < p[arm]}) + "\n")
m.analyze(None)
