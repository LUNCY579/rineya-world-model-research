import json, glob, os, statistics as st
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs"
def load(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8").read().splitlines() if False else [l for l in open(p, encoding="utf-8").read().splitlines() if l.strip()])}
def disc(a, b, lo, hi):
    ids = [i for i in a if lo <= a[i]["n_blocks"] <= hi]
    x = sum(a[i]["success"] and not b[i]["success"] for i in ids)
    y = sum(b[i]["success"] and not a[i]["success"] for i in ids)
    return len(ids), sum(a[i]["success"] for i in ids), sum(b[i]["success"] for i in ids), x, y
E = R + "/BW_B3_EVAL"
for s in ("b3fresh", "mirror"):
    v3, v2c, v2 = (load(f"{E}/plans_{a}_{s}.jsonl") for a in ("v3", "v2c", "v2"))
    for lo, hi in ((6, 10), (11, 15), (13, 15)):
        print(s, f"{lo}-{hi}", "v3 vs v2c (n, k3, kc, +, -)", disc(v3, v2c, lo, hi), "v3 vs v2", disc(v3, v2, lo, hi))
    # gen tokens
    for a, d in (("v3", v3), ("v2c", v2c)):
        for lo, hi in ((6, 10), (11, 15)):
            g = [d[i]["gen_tokens"] for i in d if lo <= d[i]["n_blocks"] <= hi]
            print("  gen_tokens", a, s, f"{lo}-{hi}", "mean", round(st.mean(g), 1), "max", max(g), ">=256", sum(x >= 256 for x in g))
h2 = load(R + "/BW_B3_HARD/h2_beam/plans.jsonl"); b2 = load(R + "/BW_TEST_B2/hard_beam/plans.jsonl")
print("hard H2 vs B2 11-15", disc(h2, b2, 11, 15), "6-10", disc(h2, b2, 6, 10))
mb = load(R + "/BW_TEST_B2/mirror_beam/plans.jsonl")
print("mirror B2 beam 11-15 succ", sum(r["success"] for r in mb.values() if r["n_blocks"] >= 11))
S = {k: load(f"{R}/BW_SEEDS/{k}_beam/plans.jsonl") for k in ("R0", "R1", "R2")}
for a, b in (("R1", "R0"), ("R2", "R0"), ("R1", "R2")):
    print("seeds beam", a, b, "11-15", disc(S[a], S[b], 11, 15), "all", disc(S[a], S[b], 6, 15))
# H2 per-instance beam cost fields
k = next(iter(h2.values())); print("h2 fields", list(k.keys()))
# wall per shard and composition
for sh in sorted(glob.glob(R + "/BW_B3_HARD/h2_beam/shard_*/summary.json")):
    sm = json.load(open(sh))["summary"]; pl = load(os.path.dirname(sh) + "/plans.jsonl")
    print(os.path.basename(os.path.dirname(sh)), sm["wall_s"], sm["n"], "n>=11:", sum(r["n_blocks"] >= 11 for r in pl.values()), "succ", round(sm["success"], 3))
for k2 in ("R0", "R1", "R2"):
    for sh in sorted(glob.glob(f"{R}/BW_SEEDS/{k2}_beam/shard_*/summary.json")):
        sm = json.load(open(sh))["summary"]; pl = load(os.path.dirname(sh) + "/plans.jsonl")
        print(k2, os.path.basename(os.path.dirname(sh)), sm["wall_s"], sm["n"], "n>=11:", sum(r["n_blocks"] >= 11 for r in pl.values()))
