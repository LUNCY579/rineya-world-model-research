import json, sys, random
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
from tokenizers import Tokenizer
assert 'torch' not in sys.modules
tk = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
def L(r): return len(tk.encode(r["prompt"], add_special_tokens=False).ids) + len(tk.encode(r["target"], add_special_tokens=False).ids) + 1
v2 = [json.loads(l) for l in open(ROOT + "/runs/BW_data_v2/proposer_actions_train.jsonl", encoding="utf-8")]
lv2 = [L(r) for r in v2]
print("v2 max", max(lv2), "mean", round(sum(lv2)/len(lv2),1))
def order_for(n, steps, batch, seed):
    rng = random.Random(10007 * seed + 17); order = []
    while len(order) < steps * batch:
        idx = list(range(n)); rng.shuffle(idx); order.extend(idx)
    return order
def cost(lens, steps, batch=16, seed=1, micro=4):
    o = order_for(len(lens), steps, batch, seed)
    pad = real = 0
    for s in range(steps):
        b = [lens[o[s*batch+j]] for j in range(batch)]
        for m in range(0, batch, micro):
            mb = b[m:m+micro]; pad += max(mb)*len(mb); real += sum(mb)
    return pad, real
pv2, rv2 = cost(lv2, 1000)
print("Pa_v2 1000 steps: padded", pv2, "real", rv2, "(RUN_META tokens_processed 2781649)")
sols = [json.loads(l) for l in open("solutions_copy.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
d = []
for r in sols:
    if not r["verified"]: continue
    g = frozenset(map(tuple, r["goal"])); sts = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, x)) for x in r["f_states"]]
    pl = [tuple(x) for x in r["plan"]]
    for t in range(len(pl)):
        d.append({"prompt": B.proposer_prompt(sts[t], g), "target": B.target_actions(pl[t:])})
ld = [L(r) for r in d]
# scale D to projected 4329 by resampling distill lengths (approximation)
rng = random.Random(5); Dn = 4329
ldD = ld + [rng.choice(ld) for _ in range(Dn - len(ld))]
pick = random.Random(0).sample(range(len(v2)), 2 * Dn)
v3 = ldD + [lv2[i] for i in pick[:Dn]]
v2c = [lv2[i] for i in pick]
spt = 14805 / pv2
for name, lens in (("Pa_v3", v3), ("Pa_v2c", v2c)):
    p, r = cost(lens, 400)
    print(name, "padded", p, "est wall s (padded-token scaling from Pa_v2)", round(p * spt), "| real-token scaling", round(r * 14805 / rv2))
