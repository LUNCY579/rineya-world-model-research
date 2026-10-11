"""Reviewer probe (logistics integration): ExactD + A*-based scripted proposer through the unmodified plan_beam at the
prereg grid C1-C3, on 20 generated (test-only seed) and 10 PlanBench Logistics instances; score_d vs execute vs VAL.
Pure python: a stand-in bwmn.lm is registered for the import (no torch)."""
import json
import random
import sys
import time
import types
from pathlib import Path

ROOT = Path(r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
sys.path.insert(0, str(ROOT))
stub = types.ModuleType("bwmn.lm")
sys.modules["bwmn.lm"] = stub
from bwmn import blocks_planner as BP  # noqa: E402
from bwmn import domain_planner as DP  # noqa: E402
from bwmn import logistics as L  # noqa: E402

assert "torch" not in sys.modules
SEED = 990173  # reviewer-only
OUT = Path(__file__).with_suffix(".json")

_A = {}


def astar(s, g):
    k = (s, g)
    if k not in _A:
        _A[k] = L.astar_plan(s, g, max_expansions=400_000)
    return _A[k]


class AStarProposer:
    """candidate 0 = A* from the node's state; candidates 1..K-1 = A* with (rate) one action replaced by a random
    typed ground action (CRN seed as LMProposerD)."""

    def __init__(self, K=4, rate=0.2):
        self.K, self.rate = K, rate
        self.calls = 0

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            self.calls += 1
            base = astar(it["state"], it["goal"])
            cands = []
            for j in range(self.K):
                p = list(base or [])
                if j and p:
                    r = random.Random(BP._seed(eval_seed, it["id"], it["round"], j))
                    if r.random() < self.rate:
                        p[r.randrange(len(p))] = r.choice(L.ground_actions(L.objs_of(it["state"]), it["state"]))
                cands.append({"plan": p or None, "claimed": None, "n_tokens": 13 * len(p)})
            out.append(cands)
        return out


def gen_instances():
    rng = random.Random(SEED)
    d = Path(__file__).parent / "problems"
    d.mkdir(exist_ok=True)
    out = []
    for n in [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7]:
        while True:
            g = L.generate(rng, n)
            if g is None:
                continue
            p = astar(g[1], g[2])
            if p is not None:
                break
        objs, init, goal = g
        i = len(out)
        path = d / f"gen{i}.pddl"
        path.write_text(L.write_problem(objs, init, goal, name=f"gen{i}"))
        out.append({"id": f"gen{i}", "objs": objs, "init": init, "goal": goal, "path": path, "n_packages": n,
                    "opt": len(p)})
    return out


def pb_instances():
    ins = L.planbench_instances()
    pick = []
    for k in (1, 2, 3, 4, 5):
        pick += [i for i in ins if i["n_packages"] == k and i["gt"]][:2]
    out = []
    for i in pick:
        out.append({"id": f"pb{i['id']}", "objs": i["objs"], "init": i["init"], "goal": i["goal"], "path": i["path"],
                    "n_packages": i["n_packages"], "opt": len(i["gt"])})
    return out


def check(inst, eps):
    summ, rows = DP.score_d(inst, eps, L)
    by = {it["id"]: it for it in inst}
    dis = []
    for e, r in zip(eps, rows):
        it = by[e["id"]]
        ex = L.execute(it["init"], it["goal"], e["plan"] or [])
        val = L.val_check(it["path"], e["plan"] or [])
        if (r["success"], r["valid"]) != (ex["success"], ex["valid"]) or val != r["success"]:
            dis.append((e["id"], r["success"], ex["success"], val))
        assert r["n_blocks"] == r["n_packages"] == it["n_packages"]
    return summ, rows, dis


def main():
    t0 = time.time()
    sets = {"gen20": gen_instances(), "pb10": pb_instances()}
    print("instances built", round(time.time() - t0, 1), "s", flush=True)
    res = {}
    for name, inst in sets.items():
        print(name, [(it["id"], it["n_packages"], it["opt"]) for it in inst], flush=True)
        for cfg, (W, E, R) in {"C1": (4, 24, 12), "C2": (4, 12, 24), "C3": (2, 6, 48)}.items():
            t1 = time.time()
            fwd = DP.ExactD(L)
            eps = DP.plan_beam(inst, AStarProposer(K=4, rate=0.2), fwd, eval_seed=0, width=W, max_expansions=E,
                               max_depth=100, max_rollout=R)
            summ, rows, dis = check(inst, eps)
            fails = [(r["id"], r["n_packages"], r["fail"], r["len"]) for r in rows if not r["success"]]
            res[f"{name}/{cfg}"] = {"summary": summ, "fails": fails, "val_disagreements": dis, "f_calls": fwd.calls,
                                    "len_vs_opt": [(r["id"], r["len"], it["opt"]) for r, it in zip(rows, inst)]}
            print(name, cfg, json.dumps(summ), "fails", fails, "VALdis", dis, "Fcalls", fwd.calls,
                  round(time.time() - t1, 1), "s", flush=True)
    OUT.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
