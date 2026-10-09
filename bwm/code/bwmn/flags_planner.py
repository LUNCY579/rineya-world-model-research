"""FlagWorld propose -> forward-check -> select -> execute one -> replan (same rules as bwmn.planner for RW2).

F keys are (state, action-definition text): goal-blind and world-agnostic. Exact-executor diagnostics are recorded
after selection, for analysis only."""
from __future__ import annotations

import hashlib
import random
from collections import Counter

from . import flags as FW
from . import lm


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


class LMForwardFlags:
    """Goal-blind LM forward model; memoised (frozen function). None = unsupported/malformed."""

    def __init__(self, model, tok, adapter):
        self.model, self.tok, self.adapter = model, tok, adapter
        self.memo = {}
        self.model_queries = 0

    def predict(self, keys):
        need = sorted({k for k in keys if k not in self.memo})
        if need:
            self.model_queries += len(need)
            self.model.set_adapter(self.adapter)
            gens = lm.sample_crn(self.model, self.tok, [FW.forward_prompt(s, a) for s, a in need], [None] * len(need),
                                 max_new_tokens=32)
            for k, g in zip(need, gens):
                self.memo[k] = FW.parse_forward(g["text"], len(k[0]))
        return [self.memo[k] for k in keys]


class ExactForwardFlags:
    """ORACLE diagnostic. Keys carry the action definition, so the law is world-agnostic."""

    def __init__(self):
        self.memo = {}

    def predict(self, keys):
        out = []
        for s, a in keys:
            if a == "finish":
                out.append((s, True, True))
                continue
            toks = a.split()
            pre_on = [int(t[1:]) for t in toks if t[0] == "?"]
            pre_off = [int(t[1:]) for t in toks if t[0] == "!"]
            dele = [int(t[1:]) for t in toks if t[0] == "-"]
            add = [int(t[1:]) for t in toks if t[0] == "+"]
            if any(s[b] != "1" for b in pre_on) or any(s[b] != "0" for b in pre_off):
                out.append((s, False, False))
                continue
            x = list(s)
            for b in dele:
                x[b] = "0"
            for b in add:
                x[b] = "1"
            out.append(("".join(x), True, False))
        return out


def fkey(w, state, name):
    a = w.act(name)
    return (state, FW.render_action(a) if a else "finish")


def rollout(fwd, w, state, plans, ep_keys):
    """Batched rollouts of several plans of ONE world from `state`."""
    n = len(plans)
    xs = [state] * n
    res = [{"pred_states": [], "pred_legal": [], "terminated_at": None, "supported": True} for _ in range(n)]
    alive = [True] * n
    d = 0
    while any(alive):
        keys, idx = [], []
        for i in range(n):
            if alive[i] and d < len(plans[i]):
                keys.append(fkey(w, xs[i], plans[i][d]))
                idx.append(i)
            elif alive[i]:
                alive[i] = False
        if not keys:
            break
        for i, k, o in zip(idx, keys, fwd.predict(keys)):
            ep_keys.add(k)
            if o is None:
                res[i]["supported"] = False
                alive[i] = False
                continue
            ns, legal, term = o
            res[i]["pred_states"].append(ns)
            res[i]["pred_legal"].append(legal)
            xs[i] = ns
            if term:
                res[i]["terminated_at"] = d
                alive[i] = False
        d += 1
    for i in range(n):
        res[i]["final"] = xs[i]
    return res


def predicted_utility(w, plan, roll, remaining):
    if not roll["supported"]:
        return 0, -1e9
    if roll["terminated_at"] is None:
        return 0, -0.01 * len(plan) - 0.1 * roll["pred_legal"].count(False)
    used = roll["terminated_at"] + 1
    succ = int(FW.is_goal(roll["final"], w) and used <= remaining)
    return succ, succ - 0.01 * used - 0.1 * roll["pred_legal"][:used].count(False)


class LMProposerFlags:
    def __init__(self, model, tok, adapter, mode, K=4, temperature=1.0, top_p=0.95, outcome="success"):
        self.model, self.tok, self.adapter, self.mode = model, tok, adapter, mode
        self.K, self.temperature, self.top_p, self.outcome = K, temperature, top_p, outcome
        self.max_new = 120 if mode == "actions" else 320

    def propose(self, items, eval_seed):
        """items: dicts {world, state, remaining, family, t}."""
        prompts, seeds, owner = [], [], []
        for i, it in enumerate(items):
            p = FW.proposer_prompt(it["world"], it["state"], it["remaining"], self.outcome)
            for j in range(self.K):
                prompts.append(p)
                seeds.append(None if j == 0 else _seed(eval_seed, it["family"], it["t"], j))
                owner.append(i)
        ctx = None
        if self.adapter is None:
            ctx = self.model.disable_adapter()
            ctx.__enter__()
        else:
            self.model.set_adapter(self.adapter)
        try:
            gens = lm.sample_crn(self.model, self.tok, prompts, seeds, max_new_tokens=self.max_new,
                                 temperature=self.temperature, top_p=self.top_p)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        out = [{"cands": [], "prompt_tokens": 0} for _ in items]
        for p, i, g in zip(prompts, owner, gens):
            parsed = FW.parse_plan(g["text"], self.mode, items[i]["world"])
            out[i]["cands"].append({"plan": parsed[0] if parsed else None, "claimed": parsed[1] if parsed else None,
                                    "n_tokens": g["n_tokens"], "malformed": parsed is None})
            out[i]["prompt_tokens"] += len(p) // 3  # coarse proxy; exact counts are not needed for selection
        return out


def run_episodes(worlds, proposer, fwd, *, eval_seed=0, rule="rerank", consistency_reject=False):
    """worlds: World objects with .budget set. Same selection rules as bwmn.planner.run_episodes."""
    eps = []
    for w in worlds:
        o = FW.opt_moves(w)
        eps.append({"world": w, "family": w.family, "feasible": o is not None and o + 1 <= w.budget, "opt": o,
                    "state": w.initial, "t": 0, "done": False, "actions": [], "success": False, "outcome": None,
                    "gen_tokens": 0, "rounds": 0, "malformed": 0, "candidates": 0, "f_keys": set(),
                    "coverage_first": None, "open_loop_success": None, "first_action_optimal": None,
                    "would_abstain_rounds": 0, "claim_mismatch": 0, "claim_checked": 0})
    while True:
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        items = [{"world": e["world"], "state": e["state"], "remaining": e["world"].budget - e["t"],
                  "family": e["family"], "t": e["t"]} for e in active]
        props = proposer.propose(items, eval_seed)
        for e, p, it in zip(active, props, items):
            w = e["world"]
            e["rounds"] += 1
            cands = p["cands"]
            e["candidates"] += len(cands)
            e["gen_tokens"] += sum(c["n_tokens"] for c in cands)
            e["malformed"] += sum(c["malformed"] for c in cands)
            sub = FW.World(w.n, w.actions, e["state"], w.goal, it["remaining"])
            if e["t"] == 0:
                e["coverage_first"] = any(c["plan"] is not None and FW.execute(sub, c["plan"], it["remaining"])["success"] for c in cands)
            first_valid = next((c for c in cands if c["plan"] is not None), None)
            if fwd is None:
                choice = first_valid
            else:
                valid = [c for c in cands if c["plan"] is not None]
                rolls = rollout(fwd, w, e["state"], [c["plan"] for c in valid], e["f_keys"]) if valid else []
                best, bu = None, None
                for c, r in zip(valid, rolls):
                    if c["claimed"] is not None:
                        m = min(len(c["claimed"]), len(r["pred_states"]))
                        mism = c["claimed"][:m] != r["pred_states"][:m]
                        e["claim_checked"] += 1
                        e["claim_mismatch"] += int(mism)
                        if consistency_reject and mism:
                            continue
                    succ, u = predicted_utility(w, c["plan"], r, it["remaining"])
                    if succ and (bu is None or u > bu):
                        best, bu = c, u
                if best is None:
                    e["would_abstain_rounds"] += 1
                    if rule == "gate":
                        e["done"], e["outcome"] = True, "abstain"
                        choice = None
                    else:
                        choice = first_valid
                else:
                    choice = best
            if e["t"] == 0 and choice is not None:
                e["open_loop_success"] = FW.execute(sub, choice["plan"], it["remaining"])["success"]
                e["first_action_optimal"] = choice["plan"][0] in FW.optimal_next(sub, e["state"])
            if choice is None:
                if not e["done"]:
                    e["done"], e["outcome"] = True, "no_valid_candidate"
                continue
            a = choice["plan"][0]
            ns, legal, term = FW.step(e["state"], a, w)
            e["actions"].append(a)
            e["state"] = ns
            e["t"] += 1
            if term:
                e["done"] = True
                e["success"] = FW.is_goal(ns, w)
                e["outcome"] = "success" if e["success"] else "finished_wrong"
            elif e["t"] >= w.budget:
                e["done"], e["outcome"] = True, "budget_exhausted"
    out = []
    for e in eps:
        w = e.pop("world")
        ex = FW.execute(w, e["actions"], w.budget)
        assert ex["success"] == e["success"]
        e["return"] = ex["return"]
        e["illegal_steps"] = sum(not s["legal"] for s in ex["steps"])
        e["steps_used"] = len(ex["steps"])
        e["f_calls_episode"] = len(e.pop("f_keys"))
        e["n_flags"] = w.n
        e["budget"] = w.budget
        out.append(e)
    return out


def summarize(eps):
    n = len(eps)

    def frac(k):
        v = [e[k] for e in eps if e.get(k) is not None]
        return sum(bool(x) for x in v) / len(v) if v else None
    return {"n": n, "success": sum(e["success"] for e in eps) / n, "outcomes": dict(Counter(e["outcome"] for e in eps)),
            "gen_tokens_per_task": sum(e["gen_tokens"] for e in eps) / n,
            "f_calls_per_task": sum(e["f_calls_episode"] for e in eps) / n,
            "coverage_first": frac("coverage_first"), "open_loop_success": frac("open_loop_success"),
            "first_action_optimal": frac("first_action_optimal"),
            "malformed_rate": sum(e["malformed"] for e in eps) / max(1, sum(e["candidates"] for e in eps)),
            "would_abstain_any": sum(e["would_abstain_rounds"] > 0 for e in eps) / n,
            "claim_mismatch_rate": (sum(e["claim_mismatch"] for e in eps) / max(1, sum(e["claim_checked"] for e in eps)))}
