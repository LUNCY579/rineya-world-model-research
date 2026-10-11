import sys, random
from collections import Counter
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from tokenizers import Tokenizer
from bwmn import logistics as L
TK = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(TK.encode(s, add_special_tokens=False).ids)
for seed in (20261401, 1, 2):
    rng = random.Random(seed)
    mix = [1, 2, 2, 3, 3, 3]
    k = 0; over = []; lens = Counter(); full_over = 0; fmax = 0
    while k < 3000:
        npk = rng.choice(mix)
        g = L.generate(rng, npk, rename=0.5)
        if g is None: continue
        objs, init, goal = g
        plan = L.astar_plan(init, goal)
        k += 1
        lens[len(plan)] += 1
        ren = any(f[0] == "airport" and not f[1].endswith("-0") for f in init) or "p1" in objs["obj"] and npk == 1
        row = n(L.proposer_prompt(init, goal)) + n(L.target_actions(plan)) + 1
        if row > 640:
            full_over += 1
            over.append((row, n(L.proposer_prompt(init, goal)), len(plan), objs["obj"], len(objs["location"]), len(objs["airplane"]), ren))
        # F rows along the plan (legal actions)
        s = init
        for a in plan:
            fmax = max(fmax, n(L.forward_prompt_delta(s, a)) + n(L.forward_target_delta(s, L.step(s, a)[0], True)) + 1)
            s = L.step(s, a)[0]
    print("seed", seed, "3000 inst: full rows > 640:", full_over, "max len", max(lens), "len>27:", sum(v for l, v in lens.items() if l > 27), "F row max", fmax)
    for o in sorted(over, reverse=True)[:5]: print("  ", o)
