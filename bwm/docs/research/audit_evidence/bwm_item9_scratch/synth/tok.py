"""Exact token counts (standalone tokenizers; no torch) for the synthesis rendering on PlanBench Logistics t1.
Draft rendering: grouped JSON (map / packages / planes / trucks), PDDL operator names, B2 delta target."""
import json, re, sys, statistics as st
from pathlib import Path
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from tokenizers import Tokenizer
from bwmn import blocks as B
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
TK = Tokenizer.from_file("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/tokenizer.json")
n = lambda s: len(TK.encode(s, add_special_tokens=False).ids)
CL = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench")
GT = {int(x["instance_id"]): [tuple(l.strip()[1:-1].split()) for l in x["ground_truth_plan"].strip().splitlines() if l.strip()]
      for x in json.loads((CL / "prompts/logistics/task_1_plan_generation.json").read_text())["instances"]}
SYS_F = ("Logistics simulator. Types come from the groups: packages, trucks, planes, locations (map.cities), airports. "
         "load-truck P T L needs truck T and package P at L; P goes into T. unload-truck P T L needs T at L and P in T; "
         "P goes to L. load-airplane P A L and unload-airplane P A L are the same for plane A. drive-truck T L1 L2 C needs "
         "T at L1 and L1, L2 both in city C. fly-airplane A L1 L2 needs A at L1 and L1, L2 airports. An action whose "
         "requirements fail is illegal and changes nothing. Given the state and one action, reply with JSON: whether it "
         "is legal and the facts it adds and deletes.")
SYS_P = ("Logistics planner. Actions: load-truck P T L, unload-truck P T L, load-airplane P A L, unload-airplane P A L "
         "(vehicle and package at L / package in vehicle), drive-truck T L1 L2 C (within city C), fly-airplane A L1 L2 "
         "(between airports). Bring every package to its goal location. Reply with a JSON plan.")


def parse(p):
    t = p.read_text().lower()
    init = [tuple(f.split()) for f in re.findall(r"\(([^()]+)\)", t.split("(:init")[1].split("(:goal")[0])]
    goal = [tuple(f.split()) for f in re.findall(r"\(([^()]+)\)", t.split("(:goal")[1]) if not f.strip().startswith("and")]
    return frozenset(init), frozenset(goal)


def state_json(s):
    ty = lambda k: sorted(f[1] for f in s if f[0] == k and len(f) == 2)
    pos = {f[1]: f[2] for f in s if f[0] in ("at", "in")}
    cities = {c: sorted(f[1] for f in s if f[0] == "in-city" and f[2] == c) for c in ty("city")}
    return {"map": {"airports": ty("airport"), "cities": cities}, "packages": {p: pos[p] for p in ty("obj")},
            "planes": {a: pos[a] for a in ty("airplane")}, "trucks": {t: pos[t] for t in ty("truck")}}


def step(s, a):  # VAL semantics: preconditions (incl. types), delete-then-add
    op, *x = a
    S = set(s)
    def h(*f): return tuple(f) in s
    if op == "load-truck": pre, d, ad = h("obj", x[0]) and h("truck", x[1]) and h("location", x[2]) and h("at", x[1], x[2]) and h("at", x[0], x[2]), [("at", x[0], x[2])], [("in", x[0], x[1])]
    elif op == "load-airplane": pre, d, ad = h("obj", x[0]) and h("airplane", x[1]) and h("location", x[2]) and h("at", x[0], x[2]) and h("at", x[1], x[2]), [("at", x[0], x[2])], [("in", x[0], x[1])]
    elif op == "unload-truck": pre, d, ad = h("obj", x[0]) and h("truck", x[1]) and h("location", x[2]) and h("at", x[1], x[2]) and h("in", x[0], x[1]), [("in", x[0], x[1])], [("at", x[0], x[2])]
    elif op == "unload-airplane": pre, d, ad = h("obj", x[0]) and h("airplane", x[1]) and h("location", x[2]) and h("in", x[0], x[1]) and h("at", x[1], x[2]), [("in", x[0], x[1])], [("at", x[0], x[2])]
    elif op == "drive-truck": pre, d, ad = h("truck", x[0]) and h("location", x[1]) and h("location", x[2]) and h("city", x[3]) and h("at", x[0], x[1]) and h("in-city", x[1], x[3]) and h("in-city", x[2], x[3]), [("at", x[0], x[1])], [("at", x[0], x[2])]
    elif op == "fly-airplane": pre, d, ad = h("airplane", x[0]) and h("airport", x[1]) and h("airport", x[2]) and h("at", x[0], x[1]), [("at", x[0], x[1])], [("at", x[0], x[2])]
    else: return s, False
    if not pre: return s, False
    S -= set(d); S |= set(ad)
    return frozenset(S), True


MERGE = {"load-truck": "load", "load-airplane": "load", "unload-truck": "unload", "unload-airplane": "unload",
         "drive-truck": "drive", "fly-airplane": "fly"}
res = {"pa_prompt": {}, "f_row_max": {}, "tgt_per_action_pddl": [], "tgt_per_action_merged": [], "full_row": {}}
ok_all = True
for i, plan in sorted(GT.items()):
    s0, g = parse(CL / f"instances/logistics/generated_basic/instance-{i}.pddl")
    npk = sum(1 for f in s0 if f[0] == "obj")
    gj = {f[1]: f[2] for f in g}
    pp = B.render_chat(SYS_P, B.canonical({"goal": gj, "outcome": "success", "state": state_json(s0)}))
    tgt = B.target_actions(plan)
    tm = B.canonical({"actions": [" ".join((MERGE[a[0]],) + a[1:]) for a in plan]})
    res["pa_prompt"].setdefault(npk, []).append(n(pp))
    res["full_row"].setdefault(npk, []).append(n(pp) + n(tgt) + 1)
    res["tgt_per_action_pddl"].append((n(tgt) - n('{"actions":[]}')) / len(plan))
    res["tgt_per_action_merged"].append((n(tm) - n('{"actions":[]}')) / len(plan))
    s = s0; mx = 0
    for a in plan:
        ns, ok = step(s, a)
        ok_all &= ok
        row = n(B.render_chat(SYS_F, B.canonical({"action": B.action_str(a), "state": state_json(s)}))) + n(B.forward_target_delta(s, ns, ok)) + 1
        mx = max(mx, row); s = ns
    ok_all &= all(f in s for f in g)
    res["f_row_max"].setdefault(npk, []).append(mx)
print("sys tokens F", n(SYS_F), "P", n(SYS_P), "| all GT plans legal+goal under step():", ok_all)
for k in sorted(res["pa_prompt"]):
    print(f"pk {k}: Pa prompt max {max(res['pa_prompt'][k])} | full GT row max {max(res['full_row'][k])} "
          f"(>640: {sum(x > 640 for x in res['full_row'][k])}/{len(res['full_row'][k])}) | F row max {max(res['f_row_max'][k])}")
print("tokens/action PDDL names mean", round(st.mean(res["tgt_per_action_pddl"]), 2), "max", round(max(res["tgt_per_action_pddl"]), 2),
      "| merged verbs mean", round(st.mean(res["tgt_per_action_merged"]), 2))
print("max GT target tokens (PDDL names):", max(n(B.target_actions(p)) for p in GT.values()), "for len", max(len(p) for p in GT.values()))
