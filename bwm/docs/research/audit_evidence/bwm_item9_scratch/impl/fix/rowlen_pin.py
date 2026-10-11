import sys, random, time
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from tokenizers import Tokenizer
from bwmn import logistics as L
tk = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda t: len(tk.encode(t, add_special_tokens=False).ids)
t0 = time.time()
def measure(seed, k, mix, **shape):
    rng = random.Random(seed)
    res = {"len": 0, "pa": 0, "over640": 0, "f": 0, "n": 0}
    while res["n"] < k:
        g = L.generate(rng, rng.choice(mix), rename=0.5, **shape)
        if g is None:
            continue
        objs, init, goal = g
        p = L.astar_plan(init, goal)
        pp = n(L.proposer_prompt(init, goal))
        row = pp + n(L.target_actions(p)) + 1
        res["n"] += 1
        res["len"] = max(res["len"], len(p)); res["pa"] = max(res["pa"], row); res["over640"] += row > 640
        s = init
        for a in p:
            ns = L.step(s, a)[0]
            res["f"] = max(res["f"], n(L.forward_prompt_delta(s, a)) + n(L.forward_target_delta(s, ns, True)) + 1)
            s = ns
    return res
for seed in (4242, 1, 2, 20261401):
    print("seed", seed, "3pk 3x3:", measure(seed, 300, (3,), n_cities=3, city_size=3), "mix:", measure(seed, 300, (1, 2, 2, 3, 3, 3)), round(time.time() - t0, 1))
