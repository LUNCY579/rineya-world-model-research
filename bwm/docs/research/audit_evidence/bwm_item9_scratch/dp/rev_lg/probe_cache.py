import json, os, sys, types, random
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
stub = types.ModuleType("bwmn.lm"); sys.modules["bwmn.lm"] = stub
from bwmn import blocks_planner as BP, domain_planner as DP, logistics as L

class Tok: pass
class Model:
    def __init__(self, h): self.h, self.adapter, self.prompts = h, None, []
    def set_adapter(self, a): self.adapter = a
def greedy(model, tok, prompts, *, max_new_tokens=64, max_batch_tokens=2000, min_prefix=16):
    model.prompts += prompts
    return [{"text": model.h[model.adapter](p, None), "n_tokens": 5} for p in prompts]
def sample(model, tok, prompts, seeds, *, max_new_tokens=64, temperature=1.0, top_p=0.95, max_batch_tokens=2000):
    return [[{"text": model.h[model.adapter](p, s), "n_tokens": 5} for s in row] for p, row in zip(prompts, seeds)]
stub.greedy_shared_prefix, stub.sample_crn_shared = greedy, sample
user = lambda p: json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
_A = {}
def astar(s, g):
    if (s, g) not in _A: _A[(s, g)] = L.astar_plan(s, g, max_expansions=300_000)
    return _A[(s, g)]
def fwd_h(p, sd):
    d = user(p); s = L.state_from_json(d["state"]); a = tuple(d["action"].split())
    return L.forward_target_delta(s, *L.step(s, a))
def pa_h(p, sd):
    d = user(p); s = L.state_from_json(d["state"]); g = frozenset(("at", k, v) for k, v in d["goal"].items())
    plan = list(astar(s, g) or [])
    if sd is not None and plan:
        r = random.Random(sd)
        if r.random() < 0.3:
            plan[r.randrange(len(plan))] = r.choice(L.ground_actions(L.objs_of(s), s, self_loops=True))
    return L.target_actions(plan)
rng = random.Random(4401)
inst = []
for k in (2, 3, 4, 5, 6, 7):
    while True:
        g = L.generate(rng, k, rename=0.5)
        if g and astar(g[1], g[2]): break
    inst.append({"id": f"c{k}", "objs": g[0], "init": g[1], "goal": g[2]})
cache = Path(__file__).parent / "pc_cache.jsonl"
if sys.argv[1] == "write":
    cache.unlink(missing_ok=True)
m = Model({"fwd": fwd_h, "pa": pa_h})
fwd = DP.LMForwardD(m, Tok(), "fwd", cache=cache, dom=L)
n0 = len(fwd.memo)
eps = DP.plan_beam(inst, DP.LMProposerD(m, Tok(), "pa", K=4, max_new=184, prefix_ok=True, dom=L), fwd,
                   width=4, max_expansions=24, max_depth=100, max_rollout=12)
summ, rows = DP.score_d(inst, eps, L)
print(sys.argv[1], "HASHSEED", os.environ.get("PYTHONHASHSEED"), "memo preloaded", n0, "model_queries", fwd.model_queries,
      "success", summ["success"], "verified", summ["verified_rate"])
keys = list(fwd.memo)
prompts = [L.forward_prompt_delta(*k) for k in keys]
print("distinct keys", len(keys), "distinct prompts", len(set(prompts)), "duplicate prompt sends", len(m.prompts) - len(set(m.prompts)))
print("roundtrip ok on all F-key states", all(L.state_from_json(L.state_json(k[0])) == k[0] for k in keys))
lines = cache.read_text().splitlines()
print("cache lines", len(lines), "unique keys in cache", len({(json.dumps(json.loads(x)["s"]), json.dumps(json.loads(x)["a"])) for x in lines}),
      "s sorted", all(json.loads(x)["s"] == sorted(json.loads(x)["s"]) for x in lines))
out = Path(__file__).parent / f"pc_eps_{sys.argv[1]}.json"
out.write_text(json.dumps(eps, default=str))
