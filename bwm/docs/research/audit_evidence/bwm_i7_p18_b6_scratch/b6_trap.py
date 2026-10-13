import json, sys
from collections import Counter
ROOT = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
exec(open("b6_fail.py").read().split("def parse_act")[0].split("from bwmn import blocks as B")[1])
for label, path, setname in [("H2 hard", "runs/BW_B3_HARD/h2_beam/plans.jsonl", "hard"), ("B2 mirror", "runs/BW_TEST_B2/mirror_beam/plans.jsonl", "mirror"), ("R0 seeds", "runs/BW_SEEDS/R0_beam/plans.jsonl", "seeds_fresh")]:
    S = load_set(setname); c = Counter(); lbs = []
    for p in map(json.loads, open(f"{ROOT}/{path}")):
        if p["n_blocks"] < 11 or p["success"]: continue
        o, i, g = S[p["id"]]; st = i
        for a in p["plan"]:
            r = B.step(st, tuple(a.split()))
            if not r[1]: break
            st = r[0]
        miss = sorted(g - st)
        if len(miss) != 1: c["u>1"] += 1; continue
        _, x, y = miss[0]
        above = {f[2]: f[1] for f in st if f[0] == "on"}
        gon = {f[1]: f[2] for f in g if f[0] == "on"}
        chain = []; z = x
        while z in above:
            z = above[z]; chain.append(z)
        k = len(chain); r_ = sum(1 for z in chain if z in gon)  # blocks above x that have a goal support (must come back)
        lb = 2 * (k + 1) + 2 * r_
        lbs.append(lb); c["lb<=12" if lb <= 12 else "lb>12"] += 1
    print(label, dict(c), "lb list", sorted(lbs))
