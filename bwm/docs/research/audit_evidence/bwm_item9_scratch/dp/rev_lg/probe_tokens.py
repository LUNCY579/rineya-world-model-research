"""Token budgets with the Titan tokenizer (tokenizers lib only, no weights)."""
import json, random, sys
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
import tokenizers
from bwmn import logistics as L
TOK = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
           "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze") / "tokenizer.json"
tk = tokenizers.Tokenizer.from_file(str(TOK))
n = lambda t: len(tk.encode(t, add_special_tokens=False).ids)
assert "torch" not in sys.modules

def need_tokens(plan):
    """tokens needed to emit the closing quote of each action i (prefix parse accepts i+1 actions), and full+EOS"""
    text = L.target_actions(plan)
    enc = tk.encode(text, add_special_tokens=False)
    offs = enc.offsets
    pos, out = len('{"actions":['), []
    for a in plan:
        q = pos + len(json.dumps(L.action_str(a))) - 1
        t = next(i for i, (s, e) in enumerate(offs) if e > q)
        out.append(t + 1)
        pos = q + 2
    return out, len(enc.ids) + 1

plans = []  # (tag, init, plan)
for x in L.planbench_instances():
    if x["gt"]:
        plans.append((f"pb{x['id']}", x["init"], x["gt"]))
rng = random.Random(55117)
gen = 0
while gen < 150:
    k = rng.choice((4, 5, 6, 7))
    g = L.generate(rng, k, rename=1.0)
    if g is None:
        continue
    p = L.astar_plan(g[1], g[2], max_expansions=150_000)
    if p:
        plans.append((f"gen{gen}", g[1], p)); gen += 1
print("plans", len(plans), "max len", max(len(p) for _, _, p in plans), flush=True)

# F delta budget (cap 40 incl EOS)
dmax, dwhere = 0, None
for tag, s0, plan in plans:
    s = s0
    for a in plan:
        for b in L.applicable_actions(s, self_loops=True):
            ns, ok = L.step(s, b)
            t = n(L.forward_target_delta(s, ns, ok)) + 1
            if t > dmax:
                dmax, dwhere = t, (tag, L.forward_target_delta(s, ns, ok))
        s = L.step(s, a)[0]
print("F delta max tokens incl EOS", dmax, dwhere, "cap 40", flush=True)

# proposer budgets: every suffix of every plan
worst = {}
for R in (12, 24, 48):
    M = 14 * R + 16
    w = (0, None)
    short = 0
    tot = 0
    for tag, s0, plan in plans:
        for k in range(len(plan)):
            suf = plan[k:]
            need, full = need_tokens(suf)
            if len(suf) >= R:
                tot += 1
                if need[R - 1] > w[0]:
                    w = (need[R - 1], tag, k)
                short += need[R - 1] > M
            else:
                tot += 1
                short += full > M  # whole short plan must fit to parse (with prefix it would still parse a prefix)
    worst[R] = (M, w, short, tot)
    print(f"R={R} M={M}: worst tokens for R complete actions {w}; suffixes not fitting {short}/{tot}", flush=True)
fullmax = max((need_tokens(p)[1], tag, len(p)) for tag, _, p in plans)
print("S-direct: max full plan tokens incl EOS", fullmax, "M=704", flush=True)
per = [(need_tokens(p)[0][-1] - n('{"actions":[')) / len(p) for _, _, p in plans]
print("tokens/action mean %.2f max %.2f" % (sum(per) / len(per), max(per)))
# prompts
pmax = max(n(L.proposer_prompt(s0, g)) for (tag, s0, _), g in [((t, s, p), None) for t, s, p in []] ) if False else None
