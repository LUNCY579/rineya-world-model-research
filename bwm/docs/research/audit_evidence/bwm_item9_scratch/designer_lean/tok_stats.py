"""Token statistics for the LEAN design's renderings (standalone `tokenizers`; no torch/transformers/peft)."""
import json, random, sys, statistics as st
from collections import defaultdict
from tokenizers import Tokenizer
import lgdom as L
from bwmn import blocks as B

TOKJ = ("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
        "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
tok = Tokenizer.from_file(TOKJ)
n = lambda t: len(tok.encode(t, add_special_tokens=False).ids)
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))


def q(v):
    v = sorted(v)
    return {"n": len(v), "min": v[0], "mean": round(st.mean(v), 1), "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1]}


T = L.planbench_t1()
rng = random.Random(0)
print("system prompt tokens: F", n(L.SYS_FORWARD_DELTA), "Pa", n(L.SYS_PROPOSER), "| BW F", n(B.SYS_FORWARD_DELTA),
      "BW Pa", n(B.SYS_PROPOSER))
by = defaultdict(lambda: defaultdict(list))
for x in T:
    k = x["np"]
    s = x["init"]
    states = [s]
    for a in x["gt"]:
        s, ok = L.step(s, a)
        assert ok
        states.append(s)
    ga = L.ground_actions(x["init"])
    # F rows: along the GT trajectory, one legal (GT action) and one random illegal per step
    for t, a in enumerate(x["gt"]):
        st0 = states[t]
        r = n(L.forward_prompt_delta(st0, a)) + n(B.forward_target_delta(st0, states[t + 1], True)) + 1
        by[k]["F_row"].append(r)
        by[k]["F_target"].append(n(B.forward_target_delta(st0, states[t + 1], True)))
        ill = [b for b in rng.sample(ga, min(40, len(ga))) if not L.applicable(st0, b)]
        if ill:
            by[k]["F_row"].append(n(L.forward_prompt_delta(st0, ill[0])) + n(B.forward_target_delta(st0, st0, False)) + 1)
    # proposer suffix rows (every suffix of the optimal plan; GT is optimal)
    for t in range(len(x["gt"])):
        pr = n(L.proposer_prompt(states[t], x["goal"]))
        tg = n(B.target_actions(x["gt"][t:]))
        by[k]["Pa_row"].append(pr + tg + 1)
    by[k]["Pa_prompt_init"].append(n(L.proposer_prompt(x["init"], x["goal"])))
    by[k]["full_plan_target"].append(n(B.target_actions(x["gt"])))
    by[k]["tok_per_action"].append(n(B.target_actions(x["gt"])) / max(1, len(x["gt"])))
    by[k]["gt_len"].append(len(x["gt"]))
out = {}
for k in sorted(by):
    out[k] = {m: q(v) for m, v in by[k].items()}
    print("packages", k, json.dumps(out[k]))


def pooled(ks, m):
    v = [r for k in ks for r in by[k][m]]
    return q(v)


for ks in ((1, 2, 3), (1, 2, 3, 4), (4, 5, 6, 7), (5, 6, 7)):
    print("pooled", ks, "F_row", pooled(ks, "F_row"), "Pa_row", pooled(ks, "Pa_row"))
    for cap in (512, 640, 768, 896, 1024):
        v = [r for k in ks for r in by[k]["Pa_row"]]
        print("   Pa rows over", cap, sum(r > cap for r in v), "of", len(v))
json.dump(out, open("tok_stats_planbench.json", "w"), indent=1)
