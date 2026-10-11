import json, sys, collections
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from bwmn import blocks as B
S = Path(r"C:/Users/clclc/AppData/Local/Temp/claude/bwm_audit_scratch/solutions_copy.jsonl")
lines = S.read_text(encoding="utf-8").splitlines()
sols = []
for i, l in enumerate(lines):
    if not l.strip():
        print("blank line", i); continue
    sols.append(json.loads(l))
print("lines", len(lines), "parsed", len(sols))
ids = [r["id"] for r in sols]
print("unique ids", len(set(ids)), "expected ids 0..167:", ids == [f"distill/{i}" for i in range(168)])
ver = [r for r in sols if r["verified"]]
print("verified", len(ver), "exact_success among verified", sum(r["exact_valid_for_analysis_only"] for r in ver))
print("exact success among unverified", sum(r["exact_valid_for_analysis_only"] for r in sols if not r["verified"]))
short = [(r["id"], len(r["plan"]), len(r["f_states"])) for r in ver if len(r["f_states"]) < len(r["plan"])]
print("verified with f_states shorter than plan:", len(short), short[:10])
crash = [(r["id"], len(r["plan"]), len(r["f_states"])) for r in ver if len(r["f_states"]) < len(r["plan"]) - 1]
print("verified with f_states < plan-1 (rows IndexError):", len(crash))
unv_short = collections.Counter((len(r["f_states"]) == len(r["plan"])) for r in sols if not r["verified"])
print("unverified: f_states==plan len", unv_short)
pl = [len(r["plan"]) for r in ver]
print("verified plan len min/mean/max", min(pl), sum(pl)/len(pl), max(pl), "sum (=rows)", sum(pl))
print("by n verified", sorted(collections.Counter(r["n"] for r in ver).items()), "by n all", sorted(collections.Counter(r["n"] for r in sols).items()))
# malformed facts and goal check under F, and F vs exact divergence
kinds = {"clear": 2, "ontable": 2, "handempty": 1, "holding": 2, "on": 3}
bad_fact = 0; goal_not_met_F = 0; diverge = 0; diverge_rows = 0; tot_rows = 0
invalid_exact = 0; valid_not_goal = 0
for r in ver:
    goal = frozenset(map(tuple, r["goal"])); init = frozenset(map(tuple, r["init"]))
    fst = [frozenset(map(tuple, s)) for s in r["f_states"]]
    for s in fst:
        for f in s:
            if f[0] not in kinds or len(f) != kinds[f[0]]:
                bad_fact += 1
    if not fst or not B.goal_met(fst[-1], goal):
        goal_not_met_F += 1
    plan = [tuple(a) for a in r["plan"]]
    ex = B.execute(init, goal, plan)
    if not ex["valid"]: invalid_exact += 1
    elif not ex["success"]: valid_not_goal += 1
    s = init; dv = False
    states = [init] + fst
    for t, a in enumerate(plan):
        tot_rows += 1
        if states[t] != s:
            diverge_rows += 1; dv = True
        s, ok = B.step(s, a)
    diverge += dv
    # render prompts like rows()
    for t in range(len(plan)):
        B.proposer_prompt(states[t], goal); B.target_actions(plan[t:])
print("malformed facts in verified f_states", bad_fact)
print("verified whose last F-state does not meet goal", goal_not_met_F)
print("verified exactly invalid (illegal action)", invalid_exact, "valid but goal not met", valid_not_goal)
print("verified plans whose F trajectory diverges from exact somewhere", diverge, "rows with F-state != exact state", diverge_rows, "of", tot_rows)
