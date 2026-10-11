"""Spec reviewer: independent token checks (standalone tokenizers only)."""
import sys, random, json
from collections import Counter
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from tokenizers import Tokenizer
from bwmn import logistics as L
TK = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(TK.encode(s, add_special_tokens=False).ids)
pb = L.planbench_instances()
gt = [x for x in pb if x["gt"]]
lt = max(gt, key=lambda x: len(x["gt"]))
print("longest GT plan", lt["id"], len(lt["gt"]), "target tokens", n(L.target_actions(lt["gt"])), "+EOS", n(L.target_actions(lt["gt"])) + 1)
print("max target tokens any GT", max(n(L.target_actions(x["gt"])) + 1 for x in gt))
# delta target worst case over the generator name grammar
names_p = [f"p{k}" for k in range(12)]; names_v = [f"t{k}" for k in range(3)] + [f"a{k}" for k in range(3)]
locs = [f"l{x}-{y}" for x in range(3) for y in range(3)]
mx = 0; arg = None
for p in names_p:
    for v in names_v:
        for l in locs:
            for s_, ns_ in (({("at", p, l)}, {("in", p, v)}), ({("in", p, v)}, {("at", p, l)})):
                t = n(L.forward_target_delta(frozenset(s_), frozenset(ns_), True)) + 1
                if t > mx: mx, arg = t, (p, v, l)
for v in names_v:
    for l1 in locs:
        for l2 in locs:
            t = n(L.forward_target_delta(frozenset({("at", v, l1)}), frozenset({("at", v, l2)}), True)) + 1
            if t > mx: mx, arg = t, (v, l1, l2)
print("worst legal delta target over generator name grammar (+EOS):", mx, arg)
# generated training rows 1-3 packages (A* plans), with 50% renaming
rng = random.Random(20261401)
mix = [1, 2, 2, 3, 3, 3]
mxpa = Counter(); mxf = 0; over = 0; nrows = 0; lens = Counter(); maxlen = 0; worst = None
k = 0
while k < 1500:
    npk = rng.choice(mix)
    g = L.generate(rng, npk, rename=0.5)
    if g is None: continue
    objs, init, goal = g
    plan = L.astar_plan(init, goal, max_expansions=200000)
    if plan is None: print("A* fail"); continue
    k += 1
    maxlen = max(maxlen, len(plan))
    s = init
    for i in range(len(plan)):
        row = n(L.proposer_prompt(s, goal)) + n(L.target_actions(plan[i:])) + 1
        nrows += 1
        if row > mxpa[npk]: mxpa[npk] = row
        if row > 640: over += 1
        if i == 0 and (worst is None or row > worst[0]): worst = (row, len(plan), len(objs["location"]), len(objs["airplane"]))
        s = L.step(s, plan[i])[0]
print("generated 1500 instances: max opt len", maxlen, "Pa row max by pk", dict(mxpa), "rows>640", over, "of", nrows, "worst full row", worst)
