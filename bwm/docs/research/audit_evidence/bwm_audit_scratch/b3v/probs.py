import json, sys, random, hashlib
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts")
from pathlib import Path
from bwmn import blocks as B
import blocks_val_large as VL
assert 'torch' not in sys.modules
ROOTP = Path(ROOT)
SEED = 20261010
def problems(n):  # verbatim copy of bw_distill.problems
    hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
    sigs = B.hard_signatures()
    inv = {VL.invariant(*h) for h in hard}
    for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
        for r in json.loads((ROOTP / "runs/BW_val_large" / fn).read_text()):
            inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    rng = random.Random(SEED)
    sizes = [6, 7, 8, 8, 9, 9, 10, 10]
    out = VL.make(rng, [sizes[i % len(sizes)] for i in range(n)], sigs, inv)
    return [{"id": f"distill/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"]))} for i, r in enumerate(out)]
P = problems(240)
sols = [json.loads(l) for l in open("solutions_copy.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
mm = 0
for r in sols:
    it = P[int(r["id"].split("/")[1])]
    if it["objs"] != r["objs"] or sorted(map(list, it["init"])) != r["init"] or sorted(map(list, it["goal"])) != r["goal"]:
        mm += 1
done = {r["id"] for r in sols}
todo = [it["id"] for it in P if it["id"] not in done]
h = hashlib.sha256(json.dumps([[it["objs"], sorted(map(list, it["init"])), sorted(map(list, it["goal"]))] for it in P]).encode()).hexdigest()
print("mismatches", mm, "todo", len(todo), todo[0], todo[-1], "hash", h[:16])
from collections import Counter
print("todo block sizes", sorted(Counter(len(P[int(i.split('/')[1])]["objs"]) for i in todo).items()))
# b3fresh disjointness
fresh = json.loads((ROOTP / "runs/BW_val_large/instances_b3_fresh.json").read_text())
finv = {VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))) for r in fresh}
dinv = {VL.invariant(it["objs"], it["init"], it["goal"]) for it in P}
print("b3fresh n", len(fresh), "shares invariants with distill:", len(finv & dinv))
hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
print("fresh sizes == hard sizes", sorted(r["n"] for r in fresh) == sorted(len(h[0]) for h in hard))
