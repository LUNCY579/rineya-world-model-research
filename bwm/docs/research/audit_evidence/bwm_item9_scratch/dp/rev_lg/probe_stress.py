import json, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import probe_beam as P
from probe_beam import BP, DP, L, astar

class Noisy:
    def __init__(self, K=4, rate=0.5, loops=True):
        self.K, self.rate, self.loops = K, rate, loops
    def propose(self, items, eval_seed):
        out = []
        for it in items:
            base = astar(it["state"], it["goal"]) or []
            cands = []
            for j in range(self.K):
                p = list(base)
                r = random.Random(BP._seed(eval_seed, it["id"], it["round"], j))
                if p and r.random() < self.rate:
                    pool = L.ground_actions(L.objs_of(it["state"]), it["state"], self_loops=self.loops)
                    if r.random() < 0.5:
                        p[r.randrange(len(p))] = r.choice(pool)
                    else:
                        p.insert(r.randrange(len(p) + 1), r.choice([a for a in pool if L.is_self_loop(a)] or pool))
                cands.append({"plan": p or None, "claimed": None, "n_tokens": 13 * len(p)})
            out.append(cands)
        return out

sets = {"gen20": P.gen_instances(), "pb10": P.pb_instances()}
for name, inst in sets.items():
    for cfg, (W, E, R) in {"C1": (4, 24, 12), "C2": (4, 12, 24), "C3": (2, 6, 48)}.items():
        for es in (0, 1):
            eps = DP.plan_beam(inst, Noisy(), DP.ExactD(L), eval_seed=es, width=W, max_expansions=E, max_depth=100,
                               max_rollout=R)
            summ, rows, dis = P.check(inst, eps)
            loops = sum(any(L.is_self_loop(tuple(a.split())) for a in r["plan"]) for r in rows)
            fails = [(r["id"], r["n_packages"], r["fail"]) for r in rows if not r["success"]]
            print(name, cfg, es, json.dumps(summ), "fails", fails, "VALdis", dis, "plans_with_selfloops", loops, flush=True)
