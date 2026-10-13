import json, sys, random, time
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/scripts"); sys.path.insert(0, ROOT + "/research/audit_evidence/bwm_h2_scratch/leakhunt")
from bwmn import blocks as B
import blocks_val_large as VL
from canon import canon
R = ROOT + "/runs/BW_val_large/"
fs = lambda x: frozenset(map(tuple, x))
tests = []
for p in B.BENCH.joinpath("blocksworld_hard", "generated").glob("*.pddl"):
    o, i, g = B.parse_problem(p); tests.append(("hard", o, i, g))
for nm in ("val_large", "mirror_hard", "b3_fresh", "seeds_fresh"):
    for r in json.load(open(R + f"instances_{nm}.json")):
        tests.append((nm, r["objs"], fs(r["init"]), fs(r["goal"])))
for nm, p in (("e11a", ROOT + "/research/b4_exclusions/rcx_e11a_instances.json"), ("e11b", ROOT + "/research/b4_exclusions/rcx_e11b_instances.json"),
              ("e07", ROOT + "/runs/BW_B4_SCREEN/rcx_snapshot/runs/RCX-E07/main/data/instances.json")):
    for r in json.load(open(p)):
        tests.append((nm, r["objs"], fs(r["init"]), fs(r["goal"])))
big = [t for t in tests if len(t[1]) >= 11]
print("excl instances at 11-15:", Counter(t[0] for t in big))
inv = {VL.invariant(o, i, g) for _, o, i, g in big if _ in ("hard", "val_large", "mirror_hard", "b3_fresh", "seeds_fresh")}
t0 = time.time()
can = {canon(i, g, o) for _, o, i, g in big}
print("distinct invariants", len(inv), "distinct exact classes", len(can), "canon time", round(time.time() - t0, 2))
rng = random.Random(999)  # scratch only, not a registered seed
rej_inv = rej_can = tot = 0
by = Counter(); byrej = Counter()
for k in range(3000):
    n = 11 + k % 5
    g = B.generate(rng, n, full_goal_frac=1.0)
    if g is None: continue
    o, i, gl = g; tot += 1; by[n] += 1
    if VL.invariant(o, i, gl) in inv: rej_inv += 1; byrej[n] += 1
    if canon(i, gl, o) in can: rej_can += 1
print("candidates", tot, "rejected by coarse invariant", rej_inv, f"({100*rej_inv/tot:.2f}%)", dict(byrej), "exact matches", rej_can)
# goal tower heights distribution of candidates (for disclosure)
