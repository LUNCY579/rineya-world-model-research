import json, sys, types, random, traceback
from pathlib import Path
ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
stub = types.ModuleType("bwmn.lm"); sys.modules["bwmn.lm"] = stub
from bwmn import blocks as B, blocks_planner as BP, domain_planner as DP, logistics as L
sys.path.insert(0, str(ROOT / "tests"))

class Tok:
    def n(self, t): return -(-len(t) // 4)
class Model:
    def __init__(self, h): self.h, self.adapter, self.calls = h, None, 0
    def set_adapter(self, a): self.adapter = a
def greedy(model, tok, prompts, *, max_new_tokens=64, max_batch_tokens=2000, min_prefix=16):
    model.calls += len(prompts)
    return [{"text": model.h[model.adapter](p, None), "n_tokens": 5} for p in prompts]
def sample(model, tok, prompts, seeds, *, max_new_tokens=64, temperature=1.0, top_p=0.95, max_batch_tokens=2000):
    return [[{"text": model.h[model.adapter](p, s), "n_tokens": 5} for s in row] for p, row in zip(prompts, seeds)]
stub.greedy_shared_prefix, stub.sample_crn_shared = greedy, sample

rng = random.Random(3)
objs, init, goal = next(g for g in iter(lambda: L.generate(rng, 2), 0) if g)
a = L.applicable_actions(init, self_loops=False)[0]
print("action", a)
for text in ('{"add":[""],"del":[],"legal":true}', '{"add":"at p0 l0-1","del":[],"legal":true}',
             '{"add":["   "],"del":[],"legal":false}'):
    r = L.parse_forward_delta(text, init)
    print(repr(text), "-> parsed, empty fact in state:", r is not None and () in r[0])
    for fn in (lambda: L.forward_prompt_delta(r[0], a), lambda: L.proposer_prompt(r[0], goal)):
        try:
            fn(); print("   render ok")
        except Exception as e:
            print("   render raises", type(e).__name__, e)
# Blocksworld shares it?
bs = B.parse_forward_delta('{"add":[""],"del":[],"legal":true}', frozenset({("handempty",), ("clear", "a"), ("ontable", "a")}))
try:
    B.forward_prompt_delta(bs[0], ("pick-up", "a")); print("blocks render ok")
except Exception as e:
    print("blocks render raises", type(e).__name__)

# end to end: one degenerate F answer aborts the whole plan_beam call (all instances), and the cache replays it
plan = L.astar_plan(init, goal)
first = L.action_str(plan[0])
def fwd_h(p, sd):
    d = json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    s = L.state_from_json(d["state"]); act = tuple(d["action"].split())
    if d["action"] == first:
        return '{"add":["",' + json.dumps(L.fact_str(("at", act[1], act[3]))) + '],"del":[],"legal":true}' if False else \
               L.forward_target_delta(s, *L.step(s, act)).replace('"add":[', '"add":["",', 1)
    return L.forward_target_delta(s, *L.step(s, act))
def pa_h(p, sd):
    d = json.loads(p.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])
    s = L.state_from_json(d["state"]); g = frozenset(("at", k, v) for k, v in d["goal"].items())
    return L.target_actions(L.astar_plan(s, g) or [])
inst = [{"id": "x0", "objs": objs, "init": init, "goal": goal}]
rng2 = random.Random(9)
while len(inst) < 4:
    g = L.generate(rng2, 2)
    if g: inst.append({"id": f"x{len(inst)}", "objs": g[0], "init": g[1], "goal": g[2]})
cache = Path(__file__).parent / "fp_cache.jsonl"
cache.unlink(missing_ok=True)
m = Model({"fwd": fwd_h, "pa": pa_h})
for attempt in (1, 2):
    fwd = DP.LMForwardD(m, Tok(), "fwd", cache=cache, dom=L)
    prop = DP.LMProposerD(m, Tok(), "pa", K=4, max_new=184, prefix_ok=True, dom=L)
    try:
        eps = DP.plan_beam(inst, prop, fwd, width=4, max_expansions=24, max_depth=100, max_rollout=12)
        print("attempt", attempt, "completed", [e["fail"] for e in eps])
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        print("attempt", attempt, "plan_beam raised", type(e).__name__, repr(e), "at", tb[-1].name, tb[-1].lineno,
              "; memo states with ():", sum(1 for k, v in fwd.memo.items() if v and () in v[0]),
              "; cache lines", len(cache.read_text().splitlines()))
