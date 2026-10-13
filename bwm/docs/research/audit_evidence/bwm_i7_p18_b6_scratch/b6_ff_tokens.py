import json, sys, statistics as st
from collections import defaultdict
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
from bwmn.lm import MODELS
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODELS["titan-sparse"])
nt = lambda s: len(tok(s, add_special_tokens=False).input_ids)
R = ROOT + "/runs/"
sets = {"mirror": json.load(open(R + "BW_val_large/instances_mirror_hard.json")),
        "seeds_fresh": json.load(open(R + "BW_val_large/instances_seeds_fresh.json"))}
srcs = [("mirror", R + "BW_TEST_B2/mirror_beam/plans.jsonl")] + [("seeds_fresh", R + f"BW_SEEDS/{r}_beam/plans.jsonl") for r in ("R0", "R1", "R2")]
rows = defaultdict(list); t0 = defaultdict(list); tpa = []; prompts = defaultdict(list)
for sname, p in srcs:
    for r in map(json.loads, open(p, encoding="utf-8")):
        if not r["success"] or r["n_blocks"] < 11:
            continue
        inst = sets[sname][int(r["id"].split("/")[1])]
        objs = inst["objs"]; s = frozenset(map(tuple, inst["init"])); g = frozenset(map(tuple, inst["goal"]))
        plan = [B.parse_action(a, objs) for a in r["plan"]]
        states = [s]
        for a in plan:
            s, ok = B.step(s, a); assert ok
            states.append(s)
        n = r["n_blocks"]
        for t in range(len(plan)):
            L = nt(B.proposer_prompt(states[t], g)) + nt(B.target_actions(plan[t:])) + 1
            rows[n].append(L)
            if t == 0:
                t0[n].append(L)
                tg = nt(B.target_actions(plan)); tg1 = nt(B.target_actions(plan[:1]))
                tpa.append((tg - tg1) / max(1, len(plan) - 1))
                prompts[n].append(nt(B.proposer_prompt(states[0], g)))
allr = [x for v in rows.values() for x in v]
print("suffix rows from generated-set beam solves (11-15):", len(allr), "mean", round(st.mean(allr), 1), "max", max(allr))
for n in sorted(rows):
    print(n, "rows", len(rows[n]), "mean", round(st.mean(rows[n]), 1), "t0 mean", round(st.mean(t0[n]), 1), "t0 max", max(t0[n]),
          "prompt max", max(prompts[n]))
print("tokens/action mean", round(st.mean(tpa), 3), "max", round(max(tpa), 3))
# worst case: 15-block prompt max + 92 actions at max tokens/action
pm = max(max(v) for v in prompts.values())
print("worst-case 92-action row approx", pm, "+", round(92 * max(tpa) + 6), "=", round(pm + 92 * max(tpa) + 6 + 1))
# D1 rows
d1 = [json.loads(l) for l in open(R + "BW_DISTILL_r1/distill_rows.jsonl", encoding="utf-8")]
d1t = [nt(x["prompt"]) + nt(x["target"]) + 1 for x in d1]
print("D1 rows", len(d1t), "mean", round(st.mean(d1t), 1), "max", max(d1t))
m3 = [json.loads(l) for l in open(R + "BW_DISTILL_r1/mix_v3.jsonl", encoding="utf-8")]
m3t = [nt(x["prompt"]) + nt(x["target"]) + 1 for x in m3]
print("mix_v3 rows", len(m3t), "mean", round(st.mean(m3t), 1), "v2-half mean", round(st.mean(m3t[4254:]), 1))
