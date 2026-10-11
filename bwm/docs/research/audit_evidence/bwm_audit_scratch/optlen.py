import json
from pathlib import Path
from collections import defaultdict
R = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
info = json.loads((R / "bench/planbench/blocksworld_hard/info.json").read_text())
ln = {"hard/" + k.replace(".pddl", ""): v["length"] for k, v in info.items()}
ok = lambda r: bool(r.get("val", r["success"]))
b = {json.loads(l)["id"]: ok(json.loads(l)) for l in open(R / "runs/BW_TEST_B2/hard_beam/plans.jsonl")}
d = {json.loads(l)["id"]: ok(json.loads(l)) for l in open(R / "runs/BW_TEST_B2/hard_direct/plans.jsonl")}
assert set(ln) == set(b)
by = defaultdict(lambda: [0, 0, 0])
for i, L in ln.items():
    k = "20-24" if L < 25 else "25-29" if L < 30 else "30-34" if L < 35 else "35-40"
    by[k][0] += b[i]; by[k][1] += d[i]; by[k][2] += 1
print("hard by optimal length (beam/direct/n):", dict(sorted(by.items())))
