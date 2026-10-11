"""RIGOROUS designer check (CPU, pure python, no torch): token cost of a compact Logistics rendering, and an
oracle-proposer simulation of plan_beam's progress rule (default unmet = len(goal - state)) on PlanBench GT plans.
Run from designer/: python -B rep_cost.py"""
import json, re, statistics as st, sys, random
from collections import defaultdict, Counter
from pathlib import Path
from tokenizers import Tokenizer
ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B
from bwmn import mystery as M
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
PB = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/"
          "bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
LG = PB / "instances" / "logistics"
tok = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                          "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
T = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
dom = M.parse_domain(LG / "generated_domain.pddl")
G = json.load(open("../gt_plans.json"))
S = json.load(open("../inst_stats.json"))["generated_basic"]
pk = lambda i: S[str(i)]["packages"]

SYS_F = ("Logistics simulator. load P V L: package P and vehicle V at L. unload P V L: P in V, V at L. "
         "drive T L1 L2 C: truck T at L1, L1 and L2 in city C. fly A L1 L2: airplane A at L1, L1 and L2 airports. "
         "An action whose requirements fail is illegal and changes nothing. Given the map, the state and one action, "
         "reply with JSON: whether it is legal and the facts it adds and deletes.")
SYS_P = ("Logistics planner. load P V L (package P and vehicle V at L), unload P V L (P in V, V at L), drive T L1 L2 C "
         "(truck T at L1, both in city C), fly A L1 L2 (airplane A at L1, both airports). Move every package to its "
         "goal location. Reply with a JSON plan.")
VERB = {"load-truck": "load", "load-airplane": "load", "unload-truck": "unload", "unload-airplane": "unload",
        "drive-truck": "drive", "fly-airplane": "fly"}

def view(s):
    cities = defaultdict(list); pos = {}; inn = {}
    for f in s:
        if f[0] == "in-city": cities[f[2]].append(f[1])
        elif f[0] == "at": pos[f[1]] = f[2]
        elif f[0] == "in": inn[f[1]] = f[2]
    ty = {f[1]: f[0] for f in s if len(f) == 2}
    d = {"map": {"airports": sorted(f[1] for f in s if f[0] == "airport"),
                 "cities": {c: sorted(v) for c, v in cities.items()}},
         "packages": {p: pos.get(p, inn.get(p)) for p, t in ty.items() if t == "obj"},
         "planes": {a: pos[a] for a, t in ty.items() if t == "airplane"},
         "trucks": {x: pos[x] for x, t in ty.items() if t == "truck"}}
    return d

def act_s(a, merged): return " ".join(((VERB[a[0]],) if merged else (a[0],)) + tuple(a[1:]))
def fwd_prompt(s, a, merged): return B.render_chat(SYS_F, B.canonical(dict(view(s), action=act_s(a, merged))))
def prop_prompt(s, goal): return B.render_chat(SYS_P, B.canonical(dict(view(s), goal={f[1]: f[2] for f in goal}, outcome="success")))
def tgt_actions(plan, merged): return B.canonical({"actions": [act_s(a, merged) for a in plan]})

print("sys tokens F", T(SYS_F), "P", T(SYS_P), "| BW F", T(B.SYS_FORWARD_DELTA), "P", T(B.SYS_PROPOSER))
inst = {}
for i in range(2, 287):
    o, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{i}.pddl")
    plan = [tuple(x.split()) for x in G[str(i)]] if isinstance(G[str(i)][0], str) else [tuple(x) for x in G[str(i)]]
    inst[i] = (init, goal, plan)
# tokens per action
for merged in (False, True):
    per = []
    for i, (init, goal, plan) in inst.items():
        per.append((T(tgt_actions(plan, merged)) - T(tgt_actions([], merged))) / len(plan))
    print("tokens/action merged=%s mean %.2f" % (merged, st.mean(per)))
rng = random.Random(0)
def legal_acts(s):
    out = []
    for name in dom.actions:
        pass
    return out
for merged in (False, True):
    print("== merged verbs" if merged else "== PDDL op names")
    for lo, hi, lab in ((1, 3, "IN 1-3"), (4, 7, "OUT 4-7"), (7, 7, "pk7")):
        fr, pr, pp, wins = [], [], [], defaultdict(list)
        for i, (init, goal, plan) in inst.items():
            if not lo <= pk(i) <= hi: continue
            sts = [init]
            for a in plan: sts.append(M.step(dom, sts[-1], a)[0])
            for t, a in enumerate(plan):
                fr.append(T(fwd_prompt(sts[t], a, merged)) + T(B.forward_target_delta(sts[t], sts[t + 1], True)) + 1)
                p = T(prop_prompt(sts[t], goal)); pp.append(p)
                pr.append(p + T(tgt_actions(plan[t:], merged)) + 1)
                for R in (12, 16, 24):
                    wins[R].append(T(tgt_actions(plan[t:t + R], merged)))
        q = lambda v: f"mean {st.mean(v):.0f} p90 {sorted(v)[int(.9*(len(v)-1))]} max {max(v)}"
        print(f" {lab}: F rows {q(fr)} | Pa prompt {q(pp)} | Pa full-suffix rows {q(pr)} >640 {sum(x>640 for x in pr)}/{len(pr)}"
              + " | R-window target tokens " + "; ".join(f"R{R} max {max(v)}" for R, v in wins.items()))

# ---- oracle-proposer simulation of plan_beam's child rule (one oracle candidate per node = GT suffix)
def oracle_rounds(init, goal, plan, R):
    sts = [init]
    for a in plan: sts.append(M.step(dom, sts[-1], a)[0])
    um = [len(goal - s) for s in sts]
    pos, rounds = 0, 0
    while True:
        rounds += 1
        win = list(range(pos + 1, min(pos + R, len(plan)) + 1))  # states after each simulated action
        if any(um[k] == 0 for k in win): return rounds
        tb = min(win, key=lambda k: (um[k], k))
        pos = tb
        if rounds > 200: return None
print("== oracle rounds (sequential expansions needed if the proposer always returns the optimal suffix)")
for R in (8, 12, 16, 24, 32):
    for lo, hi, lab in ((1, 3, "IN"), (4, 6, "pk4-6"), (7, 7, "pk7"), (4, 7, "OUT")):
        v = [oracle_rounds(*inst[i], R) for i in inst if lo <= pk(i) <= hi]
        print(f" R{R} {lab}: n {len(v)} rounds mean {st.mean(v):.1f} max {max(v)} | >6 {sum(x>6 for x in v)} >12 {sum(x>12 for x in v)}")
