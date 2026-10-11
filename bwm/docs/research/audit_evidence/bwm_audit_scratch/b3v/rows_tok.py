import json, sys, random
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
from tokenizers import Tokenizer
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
tk = Tokenizer.from_file(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda x: len(tk.encode(x, add_special_tokens=False).ids)
sols = [json.loads(l) for l in open("solutions_copy.jsonl", encoding="utf-8").read().splitlines() if l.strip()]
ver = [r for r in sols if r["verified"]]
new = []
for r in ver:
    goal = frozenset(map(tuple, r["goal"]))
    states = [frozenset(map(tuple, r["init"]))] + [frozenset(map(tuple, s)) for s in r["f_states"]]
    plan = [tuple(x) for x in r["plan"]]
    for t in range(len(plan)):
        new.append({"prompt": B.proposer_prompt(states[t], goal), "target": B.target_actions(plan[t:]), "n": r["n"]})
print("rows", len(new))
tot = [n(x["prompt"]) + n(x["target"]) + 1 for x in new]
tg = [n(x["target"]) + 1 for x in new]
print("total tokens mean/max", round(sum(tot)/len(tot),1), max(tot), "over 640", sum(t > 640 for t in tot))
print("target+eos max", max(tg), "over 256", sum(t > 256 for t in tg))
# compare to v2 rows format
v2 = [json.loads(l) for l in open(ROOT + "/runs/BW_data_v2/proposer_actions_train.jsonl", encoding="utf-8")]
print("v2 rows", len(v2), "keys", sorted(v2[0].keys()), Counter(r.get("outcome") for r in v2))
print("v2 sample prompt tail:", repr(v2[0]["prompt"][-250:]))
print("v2 sample target:", repr(v2[0]["target"][:120]))
print("distill sample prompt tail:", repr(new[0]["prompt"][-250:]))
print("distill sample target:", repr(new[0]["target"][:120]))
# check v2 prompts are reconstructible by B.proposer_prompt structure (same system text)
print("same system prefix", v2[0]["prompt"].split("<|im_start|>user")[0] == new[0]["prompt"].split("<|im_start|>user")[0])
rng = random.Random(1)
samp = rng.sample(range(len(v2)), 3000)
v2t = [n(v2[i]["prompt"]) + n(v2[i]["target"]) + 1 for i in samp]
print("v2 sample tokens mean/max", round(sum(v2t)/len(v2t),1), max(v2t))
# projection
D_est = len(new) * 240 / 168
print("projected |D|", round(D_est), "2|D|", round(2*D_est), "<= v2", len(v2))
