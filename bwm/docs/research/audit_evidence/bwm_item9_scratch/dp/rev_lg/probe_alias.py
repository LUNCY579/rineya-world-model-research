import json, sys, types, random
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
stub = types.ModuleType("bwmn.lm"); sys.modules["bwmn.lm"] = stub
from bwmn import blocks_planner as BP, domain_planner as DP, logistics as L
class Model:
    def __init__(self, h): self.h, self.adapter, self.prompts = h, None, []
    def set_adapter(self, a): self.adapter = a
def greedy(model, tok, prompts, *, max_new_tokens=64, max_batch_tokens=2000, min_prefix=16):
    model.prompts += prompts
    return [{"text": model.h(p), "n_tokens": 5} for p in prompts]
stub.greedy_shared_prefix = greedy
user = lambda p: json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
objs, init, goal = L.build_instance(2, 2, [0, 0], [0], [(0, 1)], [(0, 0)])  # p0 at l0-1 -> l0-0, truck t0 at l0-0
plan = L.astar_plan(init, goal); print("plan", plan)
s1 = L.step(L.step(init, plan[0])[0], plan[1])[0]   # drive, load: p0 in t0 at l0-1
s2 = L.step(s1, plan[2])[0]                          # drive back to l0-0
unload = plan[3]
correct = L.forward_target_delta(s2, *L.step(s2, unload))
alias = correct.replace('"add":["at ', '"add":["in ')  # F writes 'in p0 l0-0' for 'at p0 l0-0'
print("correct", correct, "| aliased F output", alias)
def h(p):
    d = user(p); s = L.state_from_json(d["state"]); a = tuple(d["action"].split())
    return alias if a == unload else L.forward_target_delta(s, *L.step(s, a))
m = Model(h)
fwd = DP.LMForwardD(m, None, "fwd", dom=L)
ns_alias = fwd.predict([(s2, unload)])[0][0]
ns_true = L.step(s2, unload)[0]
print("alias state != true state:", ns_alias != ns_true,
      "| F prompts identical:", L.forward_prompt_delta(ns_alias, plan[0]) == L.forward_prompt_delta(ns_true, plan[0]),
      "| Pa prompts identical:", L.proposer_prompt(ns_alias, goal) == L.proposer_prompt(ns_true, goal))
print("goal as rendered to the models:", user(L.proposer_prompt(ns_alias, goal))["state"]["packages"], "goal", dict(L.goal_json(goal)),
      "| goal_met(alias)", L.goal_met(ns_alias, goal), "goal_met(true)", L.goal_met(ns_true, goal))
# the same prompt is sent twice (two keys), and F's (correct) answer for the next action from the aliased state is unsupported
a_next = ("drive-truck", "t0", "l0-0", "l0-1", "c0")
q0 = fwd.model_queries
out = fwd.predict([(ns_alias, a_next), (ns_true, a_next)])
print("model queries for 2 keys with one byte-identical prompt:", fwd.model_queries - q0,
      "| distinct prompts sent", len(set(m.prompts[-2:])))
lt = ("load-truck", "p0", "t0", "l0-0")
print("F on (alias, load-truck p0 t0 l0-0):", fwd.predict([(ns_alias, lt)]), "vs true:", fwd.predict([(ns_true, lt)])[0][1])
