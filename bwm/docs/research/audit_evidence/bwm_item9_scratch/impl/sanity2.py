import sys, time, random
from pathlib import Path
sys.path.insert(0, '.')
from bwmn import logistics as L
D = Path('C:/Users/clclc/AppData/Local/Temp/claude/bwm_item9_scratch/impl')
insts = {x["id"]: x for x in L.planbench_instances()}
x = insts[2]
print(L.val_check(x["path"], x["gt"]))
pp = D / "w2.pddl"; pp.write_text(L.write_problem(x["objs"], x["init"], x["goal"], "w2"))
print("written problem VAL", L.val_check(pp, x["gt"]), L.parse_problem(pp) == (x["objs"], x["init"], x["goal"]))
sl = [("fly-airplane", "a0", "l1-0", "l1-0"), ("drive-truck", "t1", "l1-0", "l1-0", "c1")] + x["gt"]
print("self-loop exec", L.execute(x["init"], x["goal"], sl)["success"], "VAL", L.val_check(x["path"], sl))
tw = [("load-truck", "p0", "a1", "l1-0")] + x["gt"]
print("type-wrong exec", L.execute(x["init"], x["goal"], tw)["valid"], "VAL", L.val_check(x["path"], tw))
print(L.val_run(x["path"], "(load-airplane p0 a1 l1-0)\n(fly-airplane a1 l1-0)\n"))
print(L.val_run(x["path"], "(LOAD-AIRPLANE p0 a1 l1-0)\n(FLY-AIRPLANE a1 l1-0 l0-0)\n(UNLOAD-AIRPLANE p0 a1 l0-0)\n")["valid"])
# generator sample + VAL
rng = random.Random(1)
n = 0
for k in range(20):
    g = L.generate(rng, rng.choice([1, 2, 3]), rename=0.5)
    if g is None: continue
    objs, init, goal = g
    p = L.astar_plan(init, goal)
    pp = D / f"g{k}.pddl"; pp.write_text(L.write_problem(objs, init, goal, f"g{k}"))
    v = L.val_check(pp, p)
    n += v
    if k < 3: print(objs, sorted(goal), p)
print("generated VAL ok", n)
print(L.proposer_prompt(x["init"], x["goal"]))
print(L.forward_prompt_delta(x["init"], x["gt"][0]))
s1, _ = L.step(x["init"], x["gt"][0])
print(L.forward_target_delta(x["init"], s1, True))
