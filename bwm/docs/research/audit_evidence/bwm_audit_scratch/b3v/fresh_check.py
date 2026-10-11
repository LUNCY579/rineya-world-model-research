import json, sys, random, hashlib
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from pathlib import Path
from collections import Counter
from bwmn import blocks as B
import blocks_val_large as VL
exec(open("probs.py").read().split("P = problems(240)")[0])  # defines problems()
assert 'torch' not in sys.modules
ROOTP = Path(ROOT)
P = problems(240)
# verbatim copy of bw_b3_prep.fresh minus the write
hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
inv = {VL.invariant(*h) for h in hard}
for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
    for r in json.loads((ROOTP / "runs/BW_val_large" / fn).read_text()):
        inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
for it in P:
    inv.add(VL.invariant(it["objs"], it["init"], it["goal"]))
sizes = sorted(len(h[0]) for h in hard)
out = VL.make(random.Random(20261011), sizes, B.hard_signatures(), inv)
s = json.dumps(out)
print("regenerated b3fresh sha", hashlib.sha256(s.encode()).hexdigest()[:16], "file sha", hashlib.sha256((ROOTP/"runs/BW_val_large/instances_b3_fresh.json").read_bytes()).hexdigest()[:16])
# exact (state, goal) overlap between distill rows and test instances
sols = [json.loads(l) for l in open("solutions_copy.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
keys = set()
for r in sols:
    if not r["verified"]: continue
    g = frozenset(map(tuple, r["goal"]))
    sts = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, x)) for x in r["f_states"]]
    for t in range(len(r["plan"])):
        keys.add((sts[t], g))
for fn in ("instances_b3_fresh.json", "instances_mirror_hard.json", "instances_val_large.json"):
    tests = json.loads((ROOTP / "runs/BW_val_large" / fn).read_text())
    tk = {(frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))) for r in tests}
    tg = {frozenset(map(tuple, r["goal"])) for r in tests}
    print(fn, "exact (state,goal) overlap with distill rows:", len(keys & tk), "| goal overlap:", len({k[1] for k in keys} & tg))
hk = {(h[1], h[2]) for h in hard}
print("hard exact overlap", len(keys & hk), "goal overlap", len({k[1] for k in keys} & {h[2] for h in hard}))
# optimality for 6-7 blocks
rat = []
for r in sols:
    if r["verified"] and r["n"] <= 7:
        opt = B.bfs_plan(frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"])), r["objs"])
        if opt is not None: rat.append(len(r["plan"]) / len(opt))
print("6-7 block verified: n", len(rat), "mean len/opt", round(sum(rat)/len(rat), 3), "max", round(max(rat),3))
