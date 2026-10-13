import json, sys
from collections import Counter, defaultdict
ROOT = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from bwmn import blocks as B
def load_set(name):
    if name == "hard":
        out = {}
        for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl"):
            o, i, g = B.parse_problem(p); out[f"hard/{p.stem}"] = (o, i, g)
        return out
    fn = {"mirror": "instances_mirror_hard.json", "seeds_fresh": "instances_seeds_fresh.json", "b3fresh": "instances_b3_fresh.json"}[name]
    rs = json.load(open(f"{ROOT}/runs/BW_val_large/{fn}"))
    return {f"{name}/{k}": (r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))) for k, r in enumerate(rs)}
def parse_act(s):
    return tuple(s.split())
for label, path, setname in [("H2 hard", "runs/BW_B3_HARD/h2_beam/plans.jsonl", "hard"), ("B2 hard", "runs/BW_TEST_B2/hard_beam/plans.jsonl", "hard"),
                             ("B2 mirror", "runs/BW_TEST_B2/mirror_beam/plans.jsonl", "mirror"), ("R0 seeds", "runs/BW_SEEDS/R0_beam/plans.jsonl", "seeds_fresh")]:
    S = load_set(setname)
    pl = [json.loads(l) for l in open(f"{ROOT}/{path}")]
    ids = list(S)
    by = defaultdict(lambda: [0, 0]); fails = Counter(); um = Counter(); toks = defaultdict(list)
    for p in pl:
        n = p["n_blocks"]
        if n < 11: continue
        by[n][0] += p["success"]; by[n][1] += 1
        toks[p["success"]].append(p["gen_tokens"])
        if not p["success"]:
            fails[p["fail"]] += 1
            key = p["id"]
            if key not in S:
                # try mapping
                continue
            o, i, g = S[key]
            st = i
            for a in p["plan"]:
                r = B.step(st, parse_act(a)) if hasattr(B, "step") else None
                if r is None or not r[1]: break
                st = r[0]
            um[len(g - st)] += 1
    tot = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    print(label, dict(sorted((k, f"{a}/{b}") for k, (a, b) in by.items())), "11-15", f"{tot[0]}/{tot[1]}", "fails", dict(fails), "best-node unmet", dict(sorted(um.items())),
          "tok solved", round(sum(toks[True])/max(1,len(toks[True]))), "tok failed", round(sum(toks[False])/max(1,len(toks[False]))))
