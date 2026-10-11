import json, re, collections, sys
sys.path.insert(0, "C:/Users/clclc/AppData/Local/Temp/claude/bwm_h2_scratch")
from bwsim import step  # own exact simulator (stdlib only)
B = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/"
S = "C:/Users/clclc/AppData/Local/Temp/claude/bwm_h2_scratch/dl_upstream_hard/"
def parse(p):
    t = open(p).read().lower()
    init = re.search(r"\(:init(.*?)\)\s*\(:goal", t, re.S).group(1)
    goal = re.search(r"\(:goal(.*)", t, re.S).group(1)
    f = lambda s: [tuple(m.split()) for m in re.findall(r"\(([a-z\-]+(?: [a-z0-9]+)*)\)", s) if m.split()[0] in ("on","ontable","clear","handempty","holding")]
    return frozenset(f(init)), set(f(goal))
def valid(pddl, plan_text):
    S0, G = parse(pddl)
    acts = [tuple(a.split()) for a in re.findall(r"\(([^()]*)\)", plan_text.lower())]
    acts = [a for a in acts if a]
    S_ = S0
    for a in acts:
        S_ = step(S_, a)
        if S_ is None: return False
    return G <= S_
res = {}
for m in ("deepseek-r1-api", "o1-preview_chat", "o1-mini_chat"):
    d = json.load(open(S + m + ".json", encoding="utf-8"))
    s0 = {}; anyv = {}
    for i in d["instances"]:
        pd = B + f"bench/planbench/blocksworld_hard/generated/instance-{i['instance_id']}.pddl"
        sw = i["step_wise_dict"]; ks = sorted(sw, key=int)
        s0[str(i["instance_id"])] = valid(pd, sw["0"]["llm_plan"] or "")
        anyv[str(i["instance_id"])] = any(valid(pd, sw[k]["llm_plan"] or "") for k in ks)
    res[m] = s0
    print(f"{m}: step-0 (direct, no feedback) VAL-equivalent valid {sum(s0.values())}/110 = {100*sum(s0.values())/110:.2f}% ; any-step (verifier loop) valid {sum(anyv.values())}/110 = {100*sum(anyv.values())/110:.2f}% ; ids {len(s0)}")
json.dump(res, open("C:/Users/clclc/AppData/Local/Temp/claude/bwm_h2_scratch/upstream_step0.json", "w"))
# paired vs ours
R = B + "runs/"
def load(p): return {json.loads(l)["id"].split("-")[-1]: bool(json.loads(l)["val"]) for l in open(R + p) if l.strip()}
from scipy.stats import binomtest
ours = {"H2 Pa_v3 S-beam": load("BW_B3_HARD/h2_beam/plans.jsonl"), "B2 S-beam": load("BW_TEST_B2/hard_beam/plans.jsonl"),
        "H1 Pa_v3 S-direct": load("BW_B3_HARD/h1_direct/plans.jsonl"), "B2 S-direct": load("BW_TEST_B2/hard_direct/plans.jsonl")}
for on, O in ours.items():
    assert set(O) == set(res["deepseek-r1-api"]), ("id mismatch", sorted(set(O) ^ set(res["deepseek-r1-api"]))[:10])
    for m, P in res.items():
        b = sum(O[k] and not P[k] for k in O); c = sum(P[k] and not O[k] for k in O)
        p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
        print(f"  {on} {sum(O.values())} vs {m} step-0 {sum(P.values())}: ours-only {b}, theirs-only {c}, exact McNemar p = {p:.3g}")
