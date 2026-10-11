import json, sys
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
L = lambda p: [json.loads(l) for l in open(f"{ROOT}/{p}")]
b2 = {r["id"]: r for r in L("runs/BW_TEST_B2/hard_beam/plans.jsonl")}
h2 = {r["id"]: r for r in L("runs/BW_B3_HARD/h2_beam/plans.jsonl")}
for nm, d in (("b2", b2), ("h2", h2)):
    fails = [r for r in d.values() if not r["success"]]
    for f in ("dead_end", "budget"):
        g = [r["gen_tokens"] for r in fails if r["fail"] == f]
        rd = [r["rounds"] for r in fails if r["fail"] == f]
        if g: print(nm, f, len(g), "mean tokens %.0f" % (sum(g)/len(g)), "rounds", Counter(rd))
    s = [r["gen_tokens"] for r in d.values() if r["success"]]
    print(nm, "solved mean tokens %.0f" % (sum(s)/len(s)), "max", max(s), "all-instances mean %.0f" % (sum(r["gen_tokens"] for r in d.values())/110))
print("anytime success by proposer-token budget T (solved with gen_tokens <= T)")
for T in (2000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, 30000):
    a = sum(1 for r in b2.values() if r["success"] and r["gen_tokens"] <= T)
    b = sum(1 for r in h2.values() if r["success"] and r["gen_tokens"] <= T)
    print(T, a, b)
print("anytime by rounds")
for R in range(1, 13):
    a = sum(1 for r in b2.values() if r["success"] and r["rounds"] <= R)
    b = sum(1 for r in h2.values() if r["success"] and r["rounds"] <= R)
    print(R, a, b)
