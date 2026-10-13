import json, statistics as st, glob, os, sys
from collections import defaultdict, Counter
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/"
def load(p):
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
def bysize(d, lo=11, hi=15):
    out = {}
    for n in range(6, 16):
        rs = [r for r in d.values() if r["n_blocks"] == n]
        if rs:
            out[n] = (sum(r["success"] for r in rs), len(rs))
    return out
print("== beam runs on generated sets, by size (succ/n), fails, solved len, gen_tokens mean")
for name, p in [("mirror_B2beam", R+"BW_TEST_B2/mirror_beam/plans.jsonl"), ("R0", R+"BW_SEEDS/R0_beam/plans.jsonl"),
                ("R1", R+"BW_SEEDS/R1_beam/plans.jsonl"), ("R2", R+"BW_SEEDS/R2_beam/plans.jsonl"),
                ("hard_B2beam", R+"BW_TEST_B2/hard_beam/plans.jsonl")]:
    d = load(p)
    bs = bysize(d)
    big = [r for r in d.values() if r["n_blocks"] >= 11]
    sl = [r["len"] for r in big if r["success"]]
    fails = Counter(r["fail"] for r in big if not r["success"])
    gt = [r["gen_tokens"] for r in big]
    rounds = [r["rounds"] for r in big]
    print(name, {k: f"{v[0]}/{v[1]}" for k, v in bs.items()}, "11-15:", sum(r["success"] for r in big), "/", len(big),
          "fails", dict(fails), "solvedlen mean/max", round(st.mean(sl), 1) if sl else None, max(sl) if sl else None,
          "gen_tok mean", round(st.mean(gt)), "rounds mean", round(st.mean(rounds), 2))
    sl_by = defaultdict(list)
    for r in big:
        if r["success"]:
            sl_by[r["n_blocks"]].append(r["len"])
    print("   solved len by n:", {k: (round(st.mean(v), 1), max(v)) for k, v in sorted(sl_by.items())})
print("== S-direct on b3fresh / mirror by size")
for s in ("b3fresh", "mirror"):
    ds = {a: load(R+f"BW_B3_EVAL/plans_{a}_{s}.jsonl") for a in ("v3", "v2c", "v2")}
    for a, d in ds.items():
        bs = bysize(d)
        tr = Counter(r["n_blocks"] for r in d.values() if r["gen_tokens"] >= 256)
        print(s, a, {k: f"{v[0]}/{v[1]}" for k, v in bs.items()}, "trunc by n", dict(sorted(tr.items())))
    for lo, hi in ((6, 10), (11, 15)):
        b = sum(ds["v3"][i]["success"] and not ds["v2c"][i]["success"] for i in ds["v3"] if lo <= ds["v3"][i]["n_blocks"] <= hi)
        c = sum(ds["v2c"][i]["success"] and not ds["v3"][i]["success"] for i in ds["v3"] if lo <= ds["v3"][i]["n_blocks"] <= hi)
        n = sum(lo <= ds["v3"][i]["n_blocks"] <= hi for i in ds["v3"])
        print(f"  {s} v3 vs v2c {lo}-{hi}: +{b}/-{c} of {n}")
    # solved direct plan lengths at 11-15
    sl = [r["len"] for r in ds["v3"].values() if r["success"] and r["n_blocks"] >= 11]
    print("  v3 solved len 11-15", sorted(sl))
    trl = [r["len"] for r in ds["v3"].values() if r["gen_tokens"] >= 256]
    print("  v3 truncated parsed len", sorted(trl))
# timings
print("== shard walls")
for name, pat in [("mirror", R+"BW_TEST_B2/mirror_beam/shard_*/summary.json"), ("hard_B2", R+"BW_TEST_B2/hard_beam/shard_*/summary.json"),
                  ("H2", R+"BW_B3_HARD/h2_beam/shard_*/summary.json"), ("R0", R+"BW_SEEDS/R0_beam/shard_*/summary.json"),
                  ("R1", R+"BW_SEEDS/R1_beam/shard_*/summary.json"), ("R2", R+"BW_SEEDS/R2_beam/shard_*/summary.json")]:
    w = []
    for f in sorted(glob.glob(pat)):
        s = json.load(open(f))["summary"]
        w.append((s["n"], round(s["wall_s"]), s.get("proposer_tokens_per_instance")))
    print(name, w, "total", sum(x[1] for x in w))
