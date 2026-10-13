import json, sys
from collections import Counter, defaultdict
ROOT = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
exec(open("b6_fail.py").read().split("def parse_act")[0].split("from bwmn import blocks as B")[1])
def above(st, x):
    on = {f[2]: f[1] for f in st if f[0] == "on"}  # below -> above
    k = 0
    while x in on:
        x = on[x]; k += 1
    return k
for label, path, setname in [("H2 hard", "runs/BW_B3_HARD/h2_beam/plans.jsonl", "hard"), ("B2 mirror", "runs/BW_TEST_B2/mirror_beam/plans.jsonl", "mirror")]:
    S = load_set(setname)
    esc = Counter()
    for p in map(json.loads, open(f"{ROOT}/{path}")):
        if p["n_blocks"] < 11 or p["success"]: continue
        o, i, g = S[p["id"]]; st = i
        for a in p["plan"]:
            r = B.step(st, tuple(a.split()))
            if not r[1]: break
            st = r[0]
        miss = sorted(g - st)
        if len(miss) != 1: continue
        _, x, y = miss[0]
        # blocks that must be moved: those above x, those above y (excluding x chain), and x itself
        lb = 2 * (above(st, x) + above(st, y)) + 2
        bfs = None
        esc[(lb <= 12)] += 1
        print(label, p["id"], p["n_blocks"], "missing", miss[0], "above x", above(st, x), "above y", above(st, y), "LB", lb, "plan len", len(p["plan"]))
    print(label, "escape LB<=12:", dict(esc))
# r1 detour rows
rows = 0; det = 0; bysz = Counter(); detsz = Counter()
for r in map(json.loads, open(f"{ROOT}/runs/BW_DISTILL_r1/solutions.jsonl")):
    if not r["verified"]: continue
    goal = frozenset(map(tuple, r["goal"]))
    states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
    L = len(r["plan"])
    for t in range(L):
        u = len(goal - states[t]); rem = L - t
        rows += 1; bysz[r["n"]] += 1
        if u <= 2 and rem >= 2 * u + 4:
            det += 1; detsz[r["n"]] += 1
    # loop removal potential
print("r1 rows", rows, "detour rows (u<=2, L>=2u+4)", det, dict(sorted(detsz.items())))
cut = 0; tot = 0
for r in map(json.loads, open(f"{ROOT}/runs/BW_DISTILL_r1/solutions.jsonl")):
    if not r["verified"]: continue
    states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
    # greedy loop removal: walk; jump to last occurrence of the current state
    last = {}
    for k, s in enumerate(states): last[s] = k
    k = 0; kept = 0
    while k < len(states) - 1:
        k = last[states[k]]
        if k >= len(states) - 1: break
        kept += 1; k += 1
    tot += len(r["plan"]); cut += len(r["plan"]) - kept
print("r1 loop removal: actions", tot, "removed", cut)
