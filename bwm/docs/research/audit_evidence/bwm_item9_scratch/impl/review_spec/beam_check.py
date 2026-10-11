"""plan_beam (pinned, unmodified) with the Logistics hooks: exact F, oracle A* proposer passed through the
Logistics text renderer/parser (target_actions -> parse_plan_prefix). bwmn.lm is stubbed (no torch)."""
import sys, types, random
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.modules["bwmn.lm"] = types.ModuleType("bwmn.lm")
from bwmn import logistics as L
from bwmn import blocks_planner as BP
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))

class ExactLG:
    def __init__(self): self.calls = 0
    def predict(self, keys):
        self.calls += len(keys)
        out = []
        for s, a in keys:
            ns, ok = L.step(s, a)
            txt = L.forward_target_delta(s, ns, ok)
            out.append(L.parse_forward_delta(txt, s))
        return out

class OracleText:
    def __init__(self, K=4, cut=None, rng=None): self.K, self.cut, self.rng = K, cut, rng
    def propose(self, items, eval_seed):
        out = []
        for it in items:
            p = L.astar_plan(it["state"], it["goal"]) or []
            txt = L.target_actions(p) if p else '{"actions":[]}'
            if self.cut: txt = txt[: self.cut]
            parsed = L.parse_plan_prefix(txt, it["objs"])
            c = {"plan": parsed[0] if parsed else None, "claimed": None, "n_tokens": len(txt)}
            out.append([c] * self.K)
        return out

pb = {x["id"]: x for x in L.planbench_instances()}
ids = [16, 34, 38, 186, 217, 2, 50]
inst = [{"id": i, "objs": pb[i]["objs"], "init": pb[i]["init"], "goal": pb[i]["goal"]} for i in ids]
for cut in (None, 200):
    fwd = ExactLG()
    eps = BP.plan_beam(inst, OracleText(K=4, cut=cut), fwd, width=4, max_expansions=24, max_depth=100, max_rollout=12,
                       goal_met=L.goal_met, unmet=L.unmet)
    for e in eps:
        ex = L.execute(pb[e["id"]]["init"], pb[e["id"]]["goal"], e["plan"] or [])
        vv = L.val_check(pb[e["id"]]["path"], e["plan"] or [])
        print("cut", cut, "id", e["id"], "len", len(e["plan"] or []), "gt", len(pb[e["id"]]["gt"]), "verified", e["verified"],
              "exec", ex["success"], "VAL", vv, "exp", e["expansions"], "fail", e["fail"])
    print("F calls", fwd.calls)
