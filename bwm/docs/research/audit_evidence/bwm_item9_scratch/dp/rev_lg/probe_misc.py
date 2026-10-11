import json, sys, types, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import probe_beam as P
from probe_beam import BP, DP, L, astar, AStarProposer
stub = sys.modules["bwmn.lm"]
# 1. hooks: explicit L.goal_met / L.unmet == defaults
inst = P.gen_instances() + P.pb_instances()
for W, E, R in ((4, 24, 12), (2, 6, 48)):
    e1 = DP.plan_beam(inst, AStarProposer(rate=0.5), DP.ExactD(L), width=W, max_expansions=E, max_depth=100, max_rollout=R)
    e2 = DP.plan_beam(inst, AStarProposer(rate=0.5), DP.ExactD(L), width=W, max_expansions=E, max_depth=100, max_rollout=R,
                      goal_met=L.goal_met, unmet=L.unmet)
    print("hooks explicit == default", (W, E, R), e1 == e2)
# 2. instances without objs: LMProposerD gets objs None -> every candidate silently unparseable; score_d still works
class M:
    adapter = None
    def set_adapter(self, a): self.adapter = a
def sample(model, tok, prompts, seeds, **kw):
    out = []
    for p, row in zip(prompts, seeds):
        d = json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
        s = L.state_from_json(d["state"]); g = frozenset(("at", k, v) for k, v in d["goal"].items())
        out.append([{"text": L.target_actions(astar(s, g)), "n_tokens": 9} for _ in row])
    return out
stub.sample_crn_shared = sample
noobjs = [{k: v for k, v in it.items() if k != "objs"} for it in inst[:6]]
prop = DP.LMProposerD(M(), None, "pa", K=4, max_new=184, prefix_ok=True, dom=L)
eps = DP.plan_beam(noobjs, prop, DP.ExactD(L), width=4, max_expansions=24, max_depth=100, max_rollout=12)
summ, rows = DP.score_d(noobjs, eps, L)
print("no objs: fails", [e["fail"] for e in eps], "success", summ["success"], "n_blocks", [r["n_blocks"] for r in rows],
      "| with objs:", DP.score_d(inst[:6], DP.plan_beam(inst[:6], prop, DP.ExactD(L), width=4, max_expansions=24,
                                                         max_depth=100, max_rollout=12), L)[0]["success"])
# 3. score_d vs VAL on PlanBench GT plans and perturbed variants
rng = random.Random(77)
pb = [x for x in L.planbench_instances() if x["gt"]]
pb = rng.sample(pb, 60)
dis, kinds = [], {}
for x in pb:
    gt = list(x["gt"])
    loops = [a for a in L.ground_actions(x["objs"], x["init"], self_loops=True) if L.is_self_loop(a)]
    var = {"gt": gt, "trunc": gt[:-1], "swap": gt[1:2] + gt[:1] + gt[2:],
           "wrongtype": gt[:2] + [("load-truck", x["objs"]["obj"][0], x["objs"]["airplane"][0], x["objs"]["location"][0])] + gt[2:],
           "selfloop_mid": gt[:3] + [rng.choice(loops)] + gt[3:] if loops else gt}
    for tag, p in var.items():
        e = [{"id": x["id"], "plan": p, "verified": False, "rounds": 1, "gen_tokens": 0, "fail": None}]
        _, rows = DP.score_d([x], e, L)
        v = L.val_check(x["path"], [tuple(a.split()) for a in rows[0]["plan"]])
        kinds.setdefault(tag, [0, 0]); kinds[tag][int(v)] += 1
        if v != rows[0]["success"]:
            dis.append((x["id"], tag))
        assert rows[0]["n_blocks"] == rows[0]["n_packages"] == x["n_packages"]
print("score_d vs VAL on", len(pb), "PB instances x 5 variants: disagreements", dis, "VAL [invalid, valid] by kind", kinds)
