"""LEAN design prototype: generated training data (<= 3 packages) -> teacher cost, exclusion collisions, row tokens."""
import json, random, sys, time, statistics as st
from collections import Counter, defaultdict
from tokenizers import Tokenizer
import lgdom as L
from bwmn import blocks as B
from bwmn import mystery as M

TOKJ = ("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
        "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
tok = Tokenizer.from_file(TOKJ)
n = lambda t: len(tok.encode(t, add_special_tokens=False).ids)
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
N_INST = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def q(v):
    v = sorted(v)
    return {"n": len(v), "min": v[0], "mean": round(st.mean(v), 1), "p90": v[int(0.9 * (len(v) - 1))], "max": v[-1]}


print("sys tokens F", n(L.SYS_FORWARD_DELTA), "(long", n(L.SYS_FORWARD_DELTA_LONG), ") Pa", n(L.SYS_PROPOSER))
t0 = time.time()
pb_sigs = {}
for i in range(1, 287):
    objs, init, goal = M.parse_problem(L.LG / "generated_basic" / f"instance-{i}.pddl")
    init, goal = L.norm(init), L.norm(goal)
    pb_sigs.setdefault(L.signature(init, goal), []).append(i)
print("PlanBench generated_basic signatures", len(pb_sigs), "of 286 files;", round(time.time() - t0, 1), "s")
dups = {k: v for k, v in pb_sigs.items() if len(v) > 1}
print("PlanBench instances sharing a renaming signature:", sum(len(v) for v in dups.values()), "in", len(dups), "groups")

rng = random.Random(20261011)
inst, coll, fails, ttimes = [], 0, 0, []
pk_mix = [1, 2, 2, 3, 3, 3]
while len(inst) < N_INST:
    nc = rng.choice([2, 3]); sz = rng.choice([1, 2, 3]); na = rng.randint(1, nc); npk = rng.choice(pk_mix)
    init, goal = L.generate(rng, nc, sz, na, npk, scramble=rng.random() < 0.25)
    if goal <= init:
        continue
    if L.signature(init, goal) in pb_sigs:
        coll += 1
        continue
    t1 = time.time()
    plan, ne = L.astar(init, goal, cap=300000)
    ttimes.append(time.time() - t1)
    if plan is None:
        fails += 1
        continue
    inst.append((init, goal, plan, (nc, sz, na, npk)))
print("generated", len(inst), "| dropped as PlanBench renamings", coll, "| teacher failures", fails,
      "| teacher s/instance", q([round(x, 2) for x in ttimes]), "| total", round(time.time() - t0, 1), "s")
print("plan length by packages", {k: q([len(p) for *_, p, sh in inst if sh[3] == k]) for k in (1, 2, 3)})

pa_rows, f_rows, leak = [], [], 0
for init, goal, plan, sh in inst:
    states = [init]
    s = init
    for a in plan:
        s, ok = L.step(s, a)
        assert ok
        states.append(s)
    for t in range(len(plan)):
        if t > 0 and all(f[0] != "in" for f in states[t]) and L.signature(states[t], goal) in pb_sigs:
            leak += 1
            continue
        pa_rows.append(n(L.proposer_prompt(states[t], goal)) + n(B.target_actions(plan[t:])) + 1)
    ga = L.ground_actions(init)
    noisy = list(plan)
    noisy[rng.randrange(len(noisy))] = rng.choice(ga)
    pa_rows.append(n(L.proposer_prompt(init, goal, "failure")) + n(B.target_actions(noisy)) + 1)
    for _ in range(3):  # F rows: a state on the teacher plan, then a 0-7 step random walk, then 50% legal action
        s = states[rng.randrange(len(states))]
        for _ in range(rng.randrange(0, 8)):
            legal = [x for x in ga if L.applicable(s, x)]
            s, _ = L.step(s, rng.choice(legal))
        legal = [x for x in ga if L.applicable(s, x)]
        illegal = [x for x in ga if not L.applicable(s, x)]
        a = rng.choice(legal) if rng.random() < 0.5 else rng.choice(illegal)
        ns, ok = L.step(s, a)
        f_rows.append(n(L.forward_prompt_delta(s, a)) + n(B.forward_target_delta(s, ns, ok)) + 1)
print("suffix rows dropped by the PlanBench-signature guard:", leak)
print("Pa rows", q(pa_rows), "over 512:", sum(r > 512 for r in pa_rows), "over 640:", sum(r > 640 for r in pa_rows))
print("Pa rows per instance", round(len(pa_rows) / len(inst), 2))
print("F rows", q(f_rows), "over 512:", sum(r > 512 for r in f_rows))
mpa, mf = st.mean(pa_rows), st.mean(f_rows)
for name, steps, mean in (("F", 600, mf), ("Pa", 1000, mpa), ("Pa600", 600, mpa)):
    toks = steps * 16 * mean
    print(f"{name}: {steps} steps x 16 x {mean:.0f} = {toks/1e6:.2f}M tokens -> "
          f"{toks/308/3600:.1f}-{toks/188/3600:.1f} h at 308-188 tok/s (BW_F_s0 .. BW_Pa_v2_s1 measured rates)")
json.dump({"pa_rows": q(pa_rows), "f_rows": q(f_rows), "n_inst": len(inst), "collisions": coll, "fails": fails,
           "leak": leak}, open("gen_stats.json", "w"), indent=1)
