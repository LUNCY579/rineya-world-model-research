"""Read-only sizing of a Logistics BWM rendering at PlanBench Logistics instance sizes (exact token counts with the
standalone `tokenizers` library; no torch / transformers). Reads PDDL text from the scratch clone of LLMs-Planning.
Imports only the pure-python bwmn.blocks / bwmn.mystery modules (python -B: no pycache written)."""
import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
from bwmn import blocks as B  # noqa: E402
from bwmn import mystery as M  # noqa: E402

assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
PB = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/"
          "bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
LG = PB / "instances" / "logistics"
tok = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                          "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
T = lambda s: len(tok.encode(s, add_special_tokens=False).ids)

dom = M.parse_domain(LG / "generated_domain.pddl")
print("domain", dom.name, dom.preds, sorted(dom.actions))
STATIC = {"obj", "truck", "location", "airplane", "city", "airport", "in-city"}
TYPES = {"obj": "packages", "truck": "trucks", "airplane": "airplanes"}

SYS_FWD = (
    "Logistics simulator. load-truck P T L needs package P at L and truck T at L. load-airplane P A L needs package P "
    "at L and airplane A at L. unload-truck P T L needs P in truck T and T at L. unload-airplane P A L needs P in "
    "airplane A and A at L. drive-truck T L1 L2 C needs truck T at L1 and L1, L2 both in city C. fly-airplane A L1 L2 "
    "needs airplane A at L1 and L1, L2 both airports. An action whose requirements fail is illegal and changes "
    "nothing. Given the map, the state and one action, reply with JSON: whether it is legal and the facts it adds and "
    "deletes.")
SYS_PROP = (
    "Logistics planner. Actions: load-truck P T L (package P and truck T at L), load-airplane P A L (P and airplane A "
    "at L), unload-truck P T L (P in T, T at L), unload-airplane P A L (P in A, A at L), drive-truck T L1 L2 C (T at "
    "L1, L1 and L2 in city C), fly-airplane A L1 L2 (A at L1, both airports). Reach every goal pair [P,L] meaning P at "
    "L. Reply with a JSON plan.")
print("BW sys tokens: fwd_delta", T(B.SYS_FORWARD_DELTA), "proposer", T(B.SYS_PROPOSER),
      "| LG draft sys tokens: fwd", T(SYS_FWD), "proposer", T(SYS_PROP))


def split(s):
    return frozenset(f for f in s if f[0] in STATIC), frozenset(f for f in s if f[0] not in STATIC)


def map_json(static, typed):
    cities = {}
    for f in static:
        if f[0] == "in-city":
            cities.setdefault(f[2], []).append(f[1])
    d = {"airports": sorted(f[1] for f in static if f[0] == "airport"),
         "cities": {c: sorted(v) for c, v in sorted(cities.items())}}
    if typed:
        for p, k in TYPES.items():
            d[k] = sorted(f[1] for f in static if f[0] == p)
    return d


def state_json(dyn):
    return {"at": sorted([f[1], f[2]] for f in dyn if f[0] == "at"),
            "in": sorted([f[1], f[2]] for f in dyn if f[0] == "in")}


def goal_json(goal):
    return sorted([f[1], f[2]] for f in goal if f[0] == "at")


def fwd_prompt(s, a, typed):
    static, dyn = split(s)
    return B.render_chat(SYS_FWD, B.canonical({"action": " ".join(a), "map": map_json(static, typed),
                                                "state": state_json(dyn)}))


def prop_prompt(s, goal, typed):
    static, dyn = split(s)
    return B.render_chat(SYS_PROP, B.canonical({"goal": goal_json(goal), "map": map_json(static, typed),
                                                 "outcome": "success", "state": state_json(dyn)}))


def gt_plans(task_file):
    d = json.loads(task_file.read_text())
    return {x["instance_id"]: [tuple(m.split()) for m in re.findall(r"\(([^()]+)\)", x["ground_truth_plan"].lower())]
            for x in d["instances"]}


def summarize(name, vals):
    v = sorted(vals)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return f"{name}: min {v[0]} median {st.median(v)} p90 {q(0.9)} max {v[-1]}"


for sub, task in (("generated_basic", "task_1_plan_generation.json"), ("generated", "task_5_plan_generalization.json")):
    gts = gt_plans(PB / "prompts" / "logistics" / task)
    files = sorted((LG / sub).glob("instance-*.pddl"), key=lambda p: int(p.stem.split("-")[1]))
    stats = Counter()
    rows = []
    for p in files:
        iid = int(p.stem.split("-")[1])
        objs, init, goal = M.parse_problem(p)
        try:
            bo, bi, bg = B.parse_problem(p)
            stats["blocks_parser_same"] += (bo == objs and bi == init and bg == goal)
        except Exception:
            stats["blocks_parser_fail"] += 1
        static, dyn = split(init)
        cnt = Counter(f[0] for f in static if len(f) == 2)
        n_loc_per_city = Counter(f[2] for f in static if f[0] == "in-city")
        rec = {"id": iid, "objs": len(objs), "cities": cnt["city"], "locs": cnt["location"], "trucks": cnt["truck"],
               "planes": cnt["airplane"], "pkgs": cnt["obj"], "static": len(static), "dyn": len(dyn),
               "goal": len(goal), "loc_per_city": max(n_loc_per_city.values()),
               "goal_preds": sorted({g[0] for g in goal})}
        plan = gts.get(iid)
        if plan is not None:
            ex = M.execute(dom, init, goal, plan)
            stats["gt_valid_success"] += ex["success"]
            stats["gt_checked"] += 1
            rec["plan_len"] = len(plan)
            states = [init]
            s = init
            for a in plan:
                s, ok = M.step(dom, s, a)
                states.append(s)
            dts = [T(B.canonical({"add": sorted(" ".join(f) for f in ns - s0), "del": sorted(" ".join(f) for f in s0 - ns),
                                  "legal": True})) for s0, ns in zip(states, states[1:])]
            rec["delta_tok_max"] = max(dts)
            for typed in (False, True):
                tag = "typed" if typed else "named"
                fp = max(T(fwd_prompt(s0, a, typed)) for s0, a in zip(states, plan))
                pp = [T(prop_prompt(s0, goal, typed)) for s0 in states[:-1]]
                pt = [T(B.canonical({"actions": [" ".join(x) for x in plan[t:]]})) for t in range(len(plan))]
                rec[f"fwd_row_{tag}"] = fp + max(dts) + 1
                rec[f"fwd_prompt_{tag}"] = fp
                rec[f"prop_prompt_{tag}"] = max(pp)
                rec[f"prop_row_{tag}"] = max(a + b + 1 for a, b in zip(pp, pt))
            rec["prop_target_full"] = T(B.canonical({"actions": [" ".join(x) for x in plan]}))
            rec["tok_per_action"] = round((rec["prop_target_full"] - 4) / len(plan), 2)
            # mystery-style generic rendering (rules + objects + full fact list), init state only
            rec["mystery_fwd_prompt"] = T(M.forward_prompt(dom, objs, init, plan[0]))
            # chars/3.5 estimate vs exact, for the typed forward prompt at init
            fpi = fwd_prompt(init, plan[0], True)
            rec["fwd_prompt_typed_chars35"] = round(len(fpi) / 3.5)
            rec["fwd_prompt_typed_exact_init"] = T(fpi)
        rows.append(rec)
    print(f"\n=== {sub}: {len(files)} instances; {dict(stats)}")
    print("goal predicates:", Counter(tuple(r["goal_preds"]) for r in rows))
    for k in ("objs", "cities", "locs", "loc_per_city", "trucks", "planes", "pkgs", "static", "dyn", "goal"):
        print(" ", summarize(k, [r[k] for r in rows]))
    pr = [r for r in rows if "plan_len" in r]
    print(f"  -- with ground-truth plans: {len(pr)} (ids {min(r['id'] for r in pr)}..{max(r['id'] for r in pr)})")
    for k in ("plan_len", "tok_per_action", "delta_tok_max", "fwd_prompt_named", "fwd_row_named", "fwd_prompt_typed",
              "fwd_row_typed", "prop_prompt_named", "prop_target_full", "prop_row_named", "prop_prompt_typed",
              "prop_row_typed", "mystery_fwd_prompt", "fwd_prompt_typed_chars35", "fwd_prompt_typed_exact_init"):
        print(" ", summarize(k, [r[k] for r in pr]))
    for cap in (256, 640):
        print(f"  rows over {cap}: fwd named {sum(r['fwd_row_named'] > cap for r in pr)} typed "
              f"{sum(r['fwd_row_typed'] > cap for r in pr)}; proposer named {sum(r['prop_row_named'] > cap for r in pr)}"
              f" typed {sum(r['prop_row_typed'] > cap for r in pr)}; mystery-style fwd prompt "
              f"{sum(r['mystery_fwd_prompt'] > cap for r in pr)}")
    print("  plan_len > 60 (blocks.parse_plan cap):", sum(r["plan_len"] > 60 for r in pr),
          "; full plan target > 80 / 120 / 256 tokens:", sum(r["prop_target_full"] > 80 for r in pr),
          sum(r["prop_target_full"] > 120 for r in pr), sum(r["prop_target_full"] > 256 for r in pr))
    print("  size mix (cities, locs/city, planes, pkgs):",
          sorted(Counter((r["cities"], r["loc_per_city"], r["planes"], r["pkgs"]) for r in rows).items())[:40])
    json.dump(rows, open(Path(__file__).with_name(f"lg_rows_{sub}.json"), "w"))

# example renderings (instance with the median forward row)
objs, init, goal = M.parse_problem(LG / "generated_basic" / "instance-2.pddl")
plan = gt_plans(PB / "prompts" / "logistics" / "task_1_plan_generation.json")[2]
s1, _ = M.step(dom, init, plan[0])
print("\nexample fwd prompt user json (typed):", fwd_prompt(init, plan[0], True).split("user\n")[1].split("<|im_end|>")[0])
print("example delta target:", B.canonical({"add": sorted(" ".join(f) for f in s1 - init),
                                            "del": sorted(" ".join(f) for f in init - s1), "legal": True}))
print("example proposer target:", B.canonical({"actions": [" ".join(x) for x in plan]}))
# token cost of single names
for w in ("a", '"a"', "l0-0", '"l0-0"', "p12", '"p12"', "load-airplane p12 a1 l1-0", "unstack a b",
          '"drive-truck t3 l3-2 l3-0 c3"'):
    print(f"  tokens({w!r}) = {T(w)}")
