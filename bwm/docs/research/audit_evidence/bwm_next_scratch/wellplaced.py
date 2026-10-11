import json, sys, glob
from collections import Counter, defaultdict
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from pathlib import Path
from bwmn import blocks as B
HARD = Path(ROOT) / "bench/planbench/blocksworld_hard/generated"
def misplaced(s, goal):
    on = {f[1]: f[2] for f in s if f[0] == "on"}; table = {f[1] for f in s if f[0] == "ontable"}
    hold = {f[1] for f in s if f[0] == "holding"}
    gon = {f[1]: f[2] for f in goal if f[0] == "on"}; gtab = {f[1] for f in goal if f[0] == "ontable"}
    memo = {}
    def ok(b):
        if b in memo: return memo[b]
        memo[b] = False
        if b in hold: r = False
        elif b in gon: r = on.get(b) == gon[b] and ok(gon[b])
        elif b in gtab: r = b in table
        else:
            r = (b in table) or (on.get(b) is not None and ok(on[b]))  # unconstrained: fine if below is fine
        memo[b] = r; return r
    blocks = set(on) | table | hold
    return sum(1 for b in blocks if (b in gon or b in gtab) and not ok(b))
def load(d):
    by_state = defaultdict(list)
    for f in glob.glob(f"{ROOT}/{d}/shard_*/fcache.jsonl"):
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            by_state[frozenset(tuple(x) for x in r["s"])].append((tuple(r["a"]), r["text"]))
    return by_state
for name, d, pf in [("b2", "runs/BW_TEST_B2/hard_beam", "runs/BW_TEST_B2/hard_beam/plans.jsonl"),
                    ("h2", "runs/BW_B3_HARD/h2_beam", "runs/BW_B3_HARD/h2_beam/plans.jsonl")]:
    bs = load(d); out = []
    for l in open(f"{ROOT}/{pf}"):
        r = json.loads(l)
        if r["success"]: continue
        objs, init, goal = B.parse_problem(HARD / f"instance-{r['id'].split('-')[-1]}.pddl")
        seen = {init}; st = [init]
        while st:
            s = st.pop()
            for a, t in bs.get(s, []):
                o = B.parse_forward_delta(t, s)
                if o and o[1] and o[0] not in seen: seen.add(o[0]); st.append(o[0])
        u_best = min(len(goal - s) for s in seen)
        mp_at_ubest = min(misplaced(s, goal) for s in seen if len(goal - s) == u_best)
        mp_best = min(misplaced(s, goal) for s in seen)
        out.append((r["n_blocks"], r["fail"], misplaced(init, goal), u_best, mp_at_ubest, mp_best, len(seen)))
    print(name, "(n, fail, misplaced@init, best_unmet, misplaced@best-unmet-state, min misplaced over forest, |forest|)")
    for o in sorted(out): print("  ", o)
