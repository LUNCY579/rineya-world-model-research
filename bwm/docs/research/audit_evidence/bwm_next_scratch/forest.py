import json, sys, glob
from collections import Counter, defaultdict
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from pathlib import Path
from bwmn import blocks as B
HARD = Path(ROOT) / "bench/planbench/blocksworld_hard/generated"
def load(d):
    keys = {}
    for f in glob.glob(f"{ROOT}/{d}/shard_*/fcache.jsonl"):
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            s = frozenset(tuple(x) for x in r["s"]); a = tuple(r["a"])
            keys[(s, a)] = r["text"]
    by_state = defaultdict(list)
    for (s, a), t in keys.items(): by_state[s].append((a, t))
    return keys, by_state
def forest(init, by_state):
    seen = {init}; stack = [init]; bad = []; nkeys = 0
    while stack:
        s = stack.pop()
        for a, t in by_state.get(s, []):
            nkeys += 1
            ns, legal = B.step(s, a); o = B.parse_forward_delta(t, s)
            if o is None: continue
            ps, pl = o
            if not pl:
                if legal: bad.append((s, a))
                continue
            if ps not in seen: seen.add(ps); stack.append(ps)
    return seen, bad, nkeys
res = {}
for name, d, pf in [("b2", "runs/BW_TEST_B2/hard_beam", "runs/BW_TEST_B2/hard_beam/plans.jsonl"),
                    ("h2", "runs/BW_B3_HARD/h2_beam", "runs/BW_B3_HARD/h2_beam/plans.jsonl")]:
    keys, by_state = load(d)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{ROOT}/{pf}")}
    out = {}
    for iid, r in rows.items():
        num = iid.split("-")[-1]
        objs, init, goal = B.parse_problem(HARD / f"instance-{num}.pddl")
        st, bad, nk = forest(init, by_state)
        # best exact progress among imagined states (all imagined states are exact here except pruned)
        best_unmet = min(len(goal - s) for s in st)
        out[iid] = dict(success=r["success"], fail=r["fail"], n=r["n_blocks"], states=len(st), keys=nk,
                        false_illegal=len(bad), best_unmet=best_unmet, gen=r["gen_tokens"], rounds=r["rounds"],
                        goal_size=len(goal))
    res[name] = out
    print(name, "instances touching false-illegal keys:", {k: (v["false_illegal"], v["success"], v["fail"]) for k, v in out.items() if v["false_illegal"]})
b2, h2 = res["b2"], res["h2"]
ct = Counter((b2[i]["fail"] or "solved", h2[i]["fail"] or "solved") for i in b2)
print("B2 outcome x H2 outcome:", dict(ct))
for nm in ("b2", "h2"):
    fails = [v for v in res[nm].values() if not v["success"]]
    print(nm, "failed: best unmet among imagined states (min over forest)", Counter(v["best_unmet"] for v in fails))
json.dump(res, open(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_next_scratch/forest_out.json", "w"), indent=0)
