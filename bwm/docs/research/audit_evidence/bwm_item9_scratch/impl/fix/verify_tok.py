import sys, random, time
from collections import Counter
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from tokenizers import Tokenizer
from bwmn import logistics as L
TK = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(TK.encode(s, add_special_tokens=False).ids)
t0 = time.time()
for label, kw, N in (("3pk 3x3", dict(n_cities=3, city_size=3), 1000), ("prereg mix", {}, 1500)):
    rng = random.Random(11)
    lens = Counter(); mxrow = 0; over = 0; k = 0; fmax = 0
    while k < N:
        npk = 3 if kw else rng.choice((1, 2, 2, 3, 3, 3))
        g = L.generate(rng, npk, rename=0.5, **kw)
        if g is None: continue
        k += 1
        objs, init, goal = g
        p = L.astar_plan(init, goal)
        lens[len(p)] += 1
        row = n(L.proposer_prompt(init, goal)) + n(L.target_actions(p)) + 1
        mxrow = max(mxrow, row); over += row > 640
        s = init
        for a in p:
            ns = L.step(s, a)[0]
            fmax = max(fmax, n(L.forward_prompt_delta(s, a)) + n(L.forward_target_delta(s, ns, True)) + 1)
            s = ns
    print(label, N, "inst: max plan len", max(lens), "max full Pa row", mxrow, "rows > 640:", over, "F row max", fmax, round(time.time() - t0, 1), "s")
