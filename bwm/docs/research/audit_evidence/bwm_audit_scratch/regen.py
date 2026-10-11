import json, sys, random, hashlib, os
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B
import blocks_val_large as VL
SEED = 20261010
def problems(n):  # verbatim copy of bw_distill.problems (bw_distill itself imports torch via blocks_planner)
    hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
    sigs = B.hard_signatures()
    inv = {VL.invariant(*h) for h in hard}
    for fn in ("instances_val_large.json", "instances_mirror_hard.json"):
        for r in json.loads((ROOT / "runs/BW_val_large" / fn).read_text()):
            inv.add(VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))))
    rng = random.Random(SEED)
    sizes = [6, 7, 8, 8, 9, 9, 10, 10]
    out = VL.make(rng, [sizes[i % len(sizes)] for i in range(n)], sigs, inv)
    return [{"id": f"distill/{i}", "objs": r["objs"], "init": frozenset(map(tuple, r["init"])),
             "goal": frozenset(map(tuple, r["goal"]))} for i, r in enumerate(out)], hard, inv
inst, hard, inv = problems(240)
sols = [json.loads(l) for l in Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/solutions_copy.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
by = {it["id"]: it for it in inst}
mism = 0
for r in sols:
    it = by[r["id"]]
    if (it["objs"] != r["objs"] or sorted(map(list, it["init"])) != r["init"] or sorted(map(list, it["goal"])) != r["goal"]):
        mism += 1
print("PYTHONHASHSEED", os.environ.get("PYTHONHASHSEED"), "stored-vs-regenerated mismatches:", mism, "of", len(sols))
h = hashlib.sha256(json.dumps([[it["id"], it["objs"], sorted(map(list, it["init"])), sorted(map(list, it["goal"]))] for it in inst]).encode()).hexdigest()
print("hash of all 240 regenerated problems", h[:16])
print("remaining ids", [it["id"] for it in inst if it["id"] not in {r['id'] for r in sols}][:3], "...", len([1 for it in inst if it["id"] not in {r['id'] for r in sols}]))
# exclusion checks
dinv = [VL.invariant(it["objs"], it["init"], it["goal"]) for it in inst]
print("distill problems colliding with hard/val_large/mirror invariants:", sum(x in inv for x in dinv))
fresh = json.loads((ROOT / "runs/BW_val_large/instances_b3_fresh.json").read_text())
finv = [VL.invariant(r["objs"], frozenset(map(tuple, r["init"])), frozenset(map(tuple, r["goal"]))) for r in fresh]
print("b3fresh n", len(fresh), "collide with distill:", sum(x in set(dinv) for x in finv), "collide with hard/vl/mirror:", sum(x in inv for x in finv))
import collections
hs = sorted(len(h[0]) for h in hard)
print("b3fresh block-count dist == hard:", sorted(collections.Counter(r["n"] for r in fresh).items()) == sorted(collections.Counter(hs).items()))
