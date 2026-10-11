import sys, statistics as st
sys.path.insert(0, '.')
from tokenizers import Tokenizer
from bwmn import logistics as L
from bwmn import blocks as B
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
TK = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(TK.encode(s, add_special_tokens=False).ids)
print("sys F", n(L.SYS_FORWARD_DELTA), "P", n(L.SYS_PROPOSER))
pa, full, frow, dmax, per_act = {}, {}, {}, 0, []
for x in L.planbench_instances():
    if not x["gt"]: continue
    k = x["n_packages"]
    pp = n(L.proposer_prompt(x["init"], x["goal"]))
    tg = n(L.target_actions(x["gt"]))
    pa.setdefault(k, []).append(pp); full.setdefault(k, []).append(pp + tg + 1)
    per_act.append((tg - n('{"actions":[]}')) / len(x["gt"]))
    s = x["init"]; mx = 0
    for a in x["gt"]:
        ns, ok = L.step(s, a)
        t = n(L.forward_target_delta(s, ns, ok)) + 1
        dmax = max(dmax, t)
        mx = max(mx, n(L.forward_prompt_delta(s, a)) + t)
        s = ns
    frow.setdefault(k, []).append(mx)
print("delta max +EOS", dmax, "illegal", n(L.forward_target_delta(s, s, False)) + 1)
for k in sorted(pa): print(k, "Pa prompt max", max(pa[k]), "full row max", max(full[k]), "F row max", max(frow[k]))
print("Pa prompt max all", max(max(v) for v in pa.values()), "F row max all", max(max(v) for v in frow.values()))
print("full 1-3 max", max(max(full[k]) for k in (1,2,3)), "4-7 max", max(max(full[k]) for k in (4,5,6,7)))
print("tokens/action mean", round(st.mean(per_act), 2), "max", round(max(per_act), 2))
gt = L.planbench_gt()
print("longest GT", max(len(p) for p in gt.values()), "tokens", max(n(L.target_actions(p)) for p in gt.values()))
