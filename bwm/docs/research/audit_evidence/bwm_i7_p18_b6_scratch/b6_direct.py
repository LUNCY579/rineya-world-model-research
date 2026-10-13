import json
from collections import defaultdict
R = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs"
for arm in ("v3", "v2c", "v2"):
    for s in ("b3fresh", "mirror"):
        d = defaultdict(lambda: [0, 0, 0, []])
        for r in map(json.loads, open(f"{R}/BW_B3_EVAL/plans_{arm}_{s}.jsonl")):
            n = r["n_blocks"]
            if n < 11: continue
            d[n][0] += r["success"]; d[n][1] += 1; d[n][2] += r["gen_tokens"] >= 256
            if r["success"]: d[n][3].append(r["len"])
        print(arm, s, {k: f"{a}/{b} trunc {t} len {sum(L)/len(L) if L else 0:.0f}" for k, (a, b, t, L) in sorted(d.items())},
              "11-15", sum(v[0] for v in d.values()), "trunc", sum(v[2] for v in d.values()))
# paired v3 vs v2c at 11-15 b3fresh
a = {r["id"]: r for r in map(json.loads, open(f"{R}/BW_B3_EVAL/plans_v3_b3fresh.jsonl"))}
b = {r["id"]: r for r in map(json.loads, open(f"{R}/BW_B3_EVAL/plans_v2c_b3fresh.jsonl"))}
for lo, hi in ((6, 10), (11, 15)):
    x = sum(a[i]["success"] and not b[i]["success"] for i in a if lo <= a[i]["n_blocks"] <= hi)
    y = sum(b[i]["success"] and not a[i]["success"] for i in a if lo <= a[i]["n_blocks"] <= hi)
    print("v3 vs v2c b3fresh", lo, hi, f"+{x}/-{y}")
print(json.load(open(f"{R}/BW_DISTILL_r1/DISTILL_META.json")), json.load(open(f"{R}/BW_DISTILL_r1/MIX_META.json")))
for p in ("BW_Pa_v3_s1", "BW_Pa_v2c_s1"):
    m = json.load(open(f"{R}/{p}/metrics.json"))
    t = m["train"]
    print(p, {k: t[k] for k in t if k not in ("hist", "loss_hist", "grad_norms", "losses")} if isinstance(t, dict) else str(t)[:300])
