import json
from pathlib import Path
from scipy.stats import binomtest, beta
T = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_TEST_B2")
def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def ok(r): return bool(r.get("val", r["success"]))
def cp(k,n): return (0 if k==0 else beta.ppf(.025,k,n-k+1), 1 if k==n else beta.ppf(.975,k+1,n-k))
for s, shards in (("hard", range(3)), ("mirror", range(5)), ("hard", range(5))):
    ids = []
    rows = {}
    for i in shards:
        for r in load(T / f"{s}_beam/shard_{i}of8/plans.jsonl"):
            ids.append(r["id"]); rows[r["id"]] = ok(r)
    d = {r["id"]: ok(r) for r in load(T / f"{s}_direct/plans.jsonl")}
    kb = sum(rows.values()); kd = sum(d[i] for i in ids)
    plus = sum(rows[i] and not d[i] for i in ids); minus = sum(d[i] and not rows[i] for i in ids)
    lo, hi = cp(kb, len(ids))
    print(s, list(shards), f"beam {kb}/{len(ids)}={kb/len(ids):.3f} CI {lo:.3f}-{hi:.3f} direct {kd}/{len(ids)} +{plus}/-{minus} p={binomtest(plus,plus+minus,.5).pvalue:.2g}")
# verified precision per beam set
for s in ("hard", "mirror", "planbench"):
    rows = load(T / f"{s}_beam/plans.jsonl")
    v = [r for r in rows if r["verified"]]
    print(s, "verified", len(v), "verified&valid", sum(ok(r) for r in v), "unverified&valid", sum(ok(r) for r in rows if not r["verified"]))
# by-block P-08
from collections import defaultdict
b = {r["id"]: r for r in load(T / "hard_beam/plans.jsonl")}
d = {r["id"]: r for r in load(T / "hard_direct/plans.jsonl")}
by = defaultdict(lambda: [0, 0, 0])
for i, r in b.items():
    x = by[r["n_blocks"]]; x[0] += ok(r); x[1] += ok(d[i]); x[2] += 1
print("hard by blocks beam/direct/n", dict(sorted(by.items())))
