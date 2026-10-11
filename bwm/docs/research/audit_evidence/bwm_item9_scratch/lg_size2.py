"""Variants: compact rendering, shared-prefix split, prefix-target proposer rows (Task 1 instances only)."""
import json, re, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import lg_size as L  # re-uses definitions (prints its report again; ignore)
B, M, T, dom = L.B, L.M, L.T, L.dom

def compact_map(static):
    cities = {}
    for f in static:
        if f[0] == "in-city":
            cities.setdefault(f[2], []).append(f[1])
    return {"airports": " ".join(sorted(f[1] for f in static if f[0] == "airport")),
            "cities": {c: " ".join(sorted(v)) for c, v in sorted(cities.items())}}

def compact_state(dyn):
    return {f[1]: f[2] for f in sorted(dyn)}

gts = L.gt_plans(L.PB / "prompts/logistics/task_1_plan_generation.json")
res = {k: [] for k in ("c_fwd_row", "c_prop_prompt", "shared_named", "own_named", "map_named", "state_named",
                       "pref8", "pref12", "pref16", "c_pref12", "c_full")}
for iid, plan in sorted(gts.items()):
    objs, init, goal = M.parse_problem(L.LG / "generated_basic" / f"instance-{iid}.pddl")
    states = [init]
    for a in plan:
        states.append(M.step(dom, states[-1], a)[0])
    static, _ = L.split(init)
    fr, pp, own = [], [], []
    for s0, a in zip(states, plan):
        _, dyn = L.split(s0)
        fr.append(T(B.render_chat(L.SYS_FWD, B.canonical({"action": " ".join(a), "map": compact_map(static),
                                                           "state": compact_state(dyn)}))) + 25 + 1)
        own.append(T(B.canonical({"action": " ".join(a), "state": L.state_json(dyn)})))
    res["c_fwd_row"].append(max(fr))
    res["own_named"].append(max(own))
    res["map_named"].append(T(B.canonical(L.map_json(static, False))))
    res["state_named"].append(max(T(B.canonical(L.state_json(L.split(s)[1]))) for s in states))
    res["shared_named"].append(T(B.render_chat(L.SYS_FWD, "")))
    pps = [T(L.prop_prompt(s0, goal, False)) for s0 in states[:-1]]
    cpp = [T(B.render_chat(L.SYS_PROP, B.canonical({"goal": " ".join(f"{g[1]} {g[2]}" for g in sorted(goal)),
                                                     "map": compact_map(static), "outcome": "success",
                                                     "state": compact_state(L.split(s0)[1])}))) for s0 in states[:-1]]
    res["c_prop_prompt"].append(max(cpp))
    for k in (8, 12, 16):
        res[f"pref{k}"].append(max(p + T(B.canonical({"actions": [" ".join(x) for x in plan[t:t + k]]})) + 1
                                   for t, p in enumerate(pps)))
    res["c_pref12"].append(max(p + T(B.canonical({"actions": [" ".join(x) for x in plan[t:t + 12]]})) + 1
                               for t, p in enumerate(cpp)))
    res["c_full"].append(max(p + T(B.canonical({"actions": [" ".join(x) for x in plan[t:]]})) + 1
                             for t, p in enumerate(cpp)))
print("\n##### variants (Task 1, 285 instances)")
for k, v in res.items():
    v = sorted(v)
    print(f"  {k}: min {v[0]} median {st.median(v)} p90 {v[int(0.9*len(v))]} max {v[-1]}  >256: {sum(x>256 for x in v)}  >640: {sum(x>640 for x in v)}")
