"""Propose -> forward-check -> select -> execute one action -> replan, batched across episodes.

Selection never touches the exact simulator. Diagnostics computed with the exact executor (coverage, open-loop
success, first-action optimality) are recorded AFTER selection, for analysis only.

F rules (amendment S1/SC-02):
- "rerank" (primary): among candidates F predicts to succeed take the best predicted utility; if none, take the
  first schema-valid candidate in generation order (exactly the no-F rule). A01-A00 then measures pure reranking.
- "gate": as rerank, but abstain when no candidate is predicted to succeed (a separately reported operating point).
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter, deque

from . import lm, rw2


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


# ----------------------------------------------------------------------------- forward models

class ForwardModel:
    """Frozen F. `memo` caches predictions globally (F is a fixed function); per-episode accounting counts what an
    empty per-episode memo would have queried."""
    name = "abstract"

    def __init__(self):
        self.memo: dict = {}
        self.model_queries = 0

    def _query(self, keys):  # -> list of (nx, legal, term) or None
        raise NotImplementedError

    def predict(self, keys):
        need = sorted({k for k in keys if k not in self.memo})
        if need:
            self.model_queries += len(need)
            for k, v in zip(need, self._query(need)):
                self.memo[k] = v
        return [self.memo[k] for k in keys]


class ExactForward(ForwardModel):
    """ORACLE diagnostic: the exact transition law. Never the headline system."""
    name = "exact"

    def _query(self, keys):
        return [rw2.exact_step(x, a, L) for x, a, L in keys]


class TableForward(ForwardModel):
    """A frozen learned F materialised as a table (e.g. F_LM predictions over all (x, action, limit)).
    Keys outside the table are queried from `fallback` (an LMForward) if given, else unsupported."""
    name = "table"

    def __init__(self, table: dict, fallback=None):
        super().__init__()
        self.memo.update(table)
        self.fallback = fallback

    def _query(self, keys):
        if self.fallback is None:
            return [None] * len(keys)
        return self.fallback._query(keys)


class LMForward(ForwardModel):
    """Goal-blind LM forward model: a LoRA adapter on the frozen core. Malformed output = unsupported (None)."""
    name = "lm"

    def __init__(self, model, tok, adapter: str | None):
        super().__init__()
        self.model, self.tok, self.adapter = model, tok, adapter

    def _query(self, keys):
        prompts = [rw2.forward_prompt(*k) for k in keys]
        if self.adapter is None:
            with self.model.disable_adapter():
                gens = lm.sample_crn(self.model, self.tok, prompts, [None] * len(prompts), max_new_tokens=24)
        else:
            self.model.set_adapter(self.adapter)
            gens = lm.sample_crn(self.model, self.tok, prompts, [None] * len(prompts), max_new_tokens=24)
        return [rw2.parse_forward(g["text"]) for g in gens]


def rollout_batch(fwd: ForwardModel, starts, plans, ep_keys=None):
    """Roll out each plan under F from (x, limit); stop at the first finish (executor semantics). A plan with no
    finish is predicted not to terminate. ep_keys[i] (a set) collects the F keys episode i would query."""
    n = len(plans)
    xs = [s[0] for s in starts]
    res = [{"pred_states": [], "pred_legal": [], "terminated_at": None, "supported": True} for _ in range(n)]
    alive = [True] * n
    depth = 0
    while any(alive):
        keys, idx = [], []
        for i in range(n):
            if alive[i] and depth < len(plans[i]):
                keys.append((xs[i], plans[i][depth], starts[i][1]))
                idx.append(i)
            elif alive[i]:
                alive[i] = False
        if not keys:
            break
        outs = fwd.predict(keys)
        for i, k, o in zip(idx, keys, outs):
            if ep_keys is not None:
                ep_keys[i].add(k)
            if o is None:
                res[i]["supported"] = False
                alive[i] = False
                continue
            nx, legal, term = o
            res[i]["pred_states"].append(nx)
            res[i]["pred_legal"].append(legal)
            xs[i] = nx
            if term:
                res[i]["terminated_at"] = depth
                alive[i] = False
        depth += 1
    for i in range(n):
        res[i]["final_x"] = xs[i]
    return res


def predicted_utility(plan, roll, goal, remaining):
    """(predicted_success, utility). Success = F says a finish terminates at x == goal within the remaining budget."""
    if not roll["supported"]:
        return 0, -1e9
    if roll["terminated_at"] is None:
        return 0, -0.01 * len(plan) - 0.1 * roll["pred_legal"].count(False)
    used = roll["terminated_at"] + 1
    illegal = roll["pred_legal"][:used].count(False)
    succ = int(roll["final_x"] == goal and used <= remaining)
    return succ, succ - 0.01 * used - 0.1 * illegal


# ----------------------------------------------------------------------------- proposers

class LMProposer:
    """Candidate 0 is greedy; candidates 1..K-1 are sampled with per-candidate seeds (common random numbers)."""

    def __init__(self, model, tok, adapter: str | None, mode: str, K: int = 4, temperature: float = 1.0,
                 top_p: float = 0.95, max_new_tokens: int | None = None, outcome: str = "success",
                 share_prefix: bool = False):
        self.model, self.tok, self.adapter, self.mode = model, tok, adapter, mode
        self.share_prefix = share_prefix  # lm.sample_crn_shared: identical outputs (verified 24/24), ~38% less compute
        self.K, self.temperature, self.top_p = K, temperature, top_p
        self.max_new_tokens = max_new_tokens or (80 if mode == "actions" else 180)  # longest training target ~103 tokens
        self.outcome = outcome
        self.name = f"lm:{adapter}:{mode}:K{K}"

    def propose(self, states: list[dict], eval_seed: int):
        prompts, seeds, owner = [], [], []
        for si, s in enumerate(states):
            p = rw2.proposer_prompt(s["x"], s["goal"], s["limit"], s["remaining"], self.outcome)
            for j in range(self.K):
                prompts.append(p)
                seeds.append(None if j == 0 else _seed(eval_seed, s["family"], s["t"], j))
                owner.append(si)
        ctx = None
        if self.adapter is None:
            ctx = self.model.disable_adapter()
            ctx.__enter__()
        else:
            self.model.set_adapter(self.adapter)
        try:
            if self.share_prefix:
                K = self.K
                rows = lm.sample_crn_shared(self.model, self.tok, prompts[::K],
                                            [seeds[i:i + K] for i in range(0, len(seeds), K)],
                                            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
                                            top_p=self.top_p)
                gens = [g for row in rows for g in row]
            else:
                gens = lm.sample_crn(self.model, self.tok, prompts, seeds, max_new_tokens=self.max_new_tokens,
                                     temperature=self.temperature, top_p=self.top_p)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
        out = [{"cands": [], "prompt_tokens": 0} for _ in states]
        ptok = {}
        for p, si, g in zip(prompts, owner, gens):
            if p not in ptok:
                ptok[p] = len(self.tok(p, add_special_tokens=False).input_ids)
            parsed = rw2.parse_plan(g["text"], self.mode)
            out[si]["cands"].append({"plan": parsed[0] if parsed else None, "claimed": parsed[1] if parsed else None,
                                     "n_tokens": g["n_tokens"], "malformed": parsed is None, "text": g["text"]})
            out[si]["prompt_tokens"] += ptok[p]
        return out


class RandomProposer:
    """Model-free control: K random plans of random length <= remaining, ending in finish."""

    def __init__(self, K: int = 4):
        self.K, self.name = K, f"random:K{K}"

    def propose(self, states, eval_seed):
        out = []
        for s in states:
            rng = random.Random(_seed(eval_seed, s["family"], s["t"], "rand"))
            row = []
            for _ in range(self.K):
                k = rng.randrange(0, s["remaining"])
                plan = [rng.choice(rw2.MOVES) for _ in range(k)] + ["finish"]
                row.append({"plan": plan, "claimed": None, "n_tokens": 0, "malformed": False, "text": ""})
            out.append({"cands": row, "prompt_tokens": 0})
        return out


# ----------------------------------------------------------------------------- episode runner

def run_episodes(tasks: list[rw2.Task], proposer, fwd: ForwardModel | None, *, eval_seed: int = 0,
                 rule: str = "rerank", consistency_reject: bool = False, log=None) -> list[dict]:
    assert rule in ("rerank", "gate")
    eps = []
    for t in tasks:
        eps.append({"task": t.to_json(), "family": t.family, "feasible": t.feasible, "opt": t.opt, "x": t.start,
                    "t": 0, "done": False, "actions": [], "success": False, "outcome": None, "gen_tokens": 0,
                    "prompt_tokens": 0, "rounds": 0, "malformed": 0, "candidates": 0, "f_keys": set(),
                    "coverage_first": None, "open_loop_success": None, "first_action_optimal": None,
                    "would_abstain_rounds": 0, "pred_success_any_first": None, "claim_mismatch": 0,
                    "claim_checked": 0, "unique_plans_first": None, "no_finish_cands": 0})
    while True:
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        states = [{"x": e["x"], "goal": e["task"]["goal"], "limit": e["task"]["limit"], "family": e["family"],
                   "t": e["t"], "remaining": e["task"]["budget"] - e["t"]} for e in active]
        props = proposer.propose(states, eval_seed)
        rolls = {}
        if fwd is not None:
            starts, plans, keys, ekeys = [], [], [], []
            for ei, (e, p) in enumerate(zip(active, props)):
                for ci, c in enumerate(p["cands"]):
                    if c["plan"] is not None:
                        starts.append((e["x"], e["task"]["limit"]))
                        plans.append(c["plan"])
                        keys.append((ei, ci))
                        ekeys.append(e["f_keys"])
            if plans:
                for k, r in zip(keys, rollout_batch(fwd, starts, plans, ekeys)):
                    rolls[k] = r
        for ei, (e, p, s) in enumerate(zip(active, props, states)):
            e["rounds"] += 1
            e["prompt_tokens"] += p["prompt_tokens"]
            cands = p["cands"]
            e["candidates"] += len(cands)
            e["gen_tokens"] += sum(c["n_tokens"] for c in cands)
            e["malformed"] += sum(c["malformed"] for c in cands)
            e["no_finish_cands"] += sum(c["plan"] is not None and "finish" not in c["plan"] for c in cands)
            task = rw2.Task(**e["task"])
            sub = rw2.Task(e["x"], task.goal, task.limit, s["remaining"])
            if e["t"] == 0:
                e["coverage_first"] = any(c["plan"] is not None and rw2.execute(sub, c["plan"])["success"] for c in cands)
                e["unique_plans_first"] = len({tuple(c["plan"]) for c in cands if c["plan"] is not None})
            first_valid = next((c for c in cands if c["plan"] is not None), None)
            choice = None
            if fwd is None:
                choice = first_valid
            else:
                best, best_u = None, None
                for ci, c in enumerate(cands):
                    if c["plan"] is None:
                        continue
                    r = rolls[(ei, ci)]
                    if c["claimed"] is not None:
                        m = min(len(c["claimed"]), len(r["pred_states"]))
                        mism = c["claimed"][:m] != r["pred_states"][:m]
                        e["claim_checked"] += 1
                        e["claim_mismatch"] += int(mism)
                        if consistency_reject and mism:
                            continue
                    succ, u = predicted_utility(c["plan"], r, s["goal"], s["remaining"])
                    if succ and (best_u is None or u > best_u):
                        best, best_u = c, u
                if e["t"] == 0:
                    e["pred_success_any_first"] = best is not None
                if best is None:
                    e["would_abstain_rounds"] += 1
                    choice = None if rule == "gate" else first_valid
                    if rule == "gate":
                        e["done"], e["outcome"] = True, "abstain"
                else:
                    choice = best
            if e["t"] == 0 and choice is not None:
                e["open_loop_success"] = rw2.execute(sub, choice["plan"])["success"]
                e["first_action_optimal"] = choice["plan"][0] in rw2.optimal_next_actions(e["x"], task.goal, task.limit)
            if choice is None:
                if not e["done"]:
                    e["done"], e["outcome"] = True, "no_valid_candidate"
                continue
            a = choice["plan"][0]
            nx, legal, term = rw2.exact_step(e["x"], a, task.limit)
            e["actions"].append(a)
            e["x"] = nx
            e["t"] += 1
            if term:
                e["done"] = True
                e["success"] = nx == task.goal
                e["outcome"] = "success" if e["success"] else "finished_wrong"
            elif e["t"] >= task.budget:
                e["done"], e["outcome"] = True, "budget_exhausted"
        if log:
            log(f"active {len(active)} -> {sum(not e['done'] for e in eps)}")
    return _finalize(eps)


def _finalize(eps):
    for e in eps:
        ex = rw2.execute(rw2.Task(**e["task"]), e["actions"])
        assert ex["success"] == e["success"], "executor replay mismatch"
        e["return"] = ex["return"]
        e["illegal_steps"] = sum(not st["legal"] for st in ex["steps"])
        e["steps_used"] = len(ex["steps"])
        e["f_calls_episode"] = len(e.pop("f_keys", set()))
    return eps


# ----------------------------------------------------------------------------- model-free baselines under F

def greedy_under_f(tasks, fwd: ForwardModel, lookahead: int = 1):
    """Greedy distance policy whose dynamics come from F (heuristic proposer + the same learned world model)."""
    eps = []
    for t in tasks:
        x, acts, keys = t.start, [], set()
        for _ in range(t.budget):
            if x == t.goal:
                acts.append("finish")
                break
            best, bd = None, None
            for a in rw2.MOVES:
                keys.add((x, a, t.limit))
                o = fwd.predict([(x, a, t.limit)])[0]
                if o is None or not o[1]:
                    continue
                d = abs(o[0] - t.goal)
                if lookahead >= 2:
                    for b in rw2.MOVES:
                        keys.add((o[0], b, t.limit))
                        o2 = fwd.predict([(o[0], b, t.limit)])[0]
                        if o2 is not None and o2[1]:
                            d = min(d, abs(o2[0] - t.goal))
                if bd is None or d < bd:
                    best, bd = a, d
            if best is None:
                break
            acts.append(best)
            x = rw2.exact_step(x, best, t.limit)[0]  # the real environment moves
        eps.append(_episode_record(t, acts, len(keys)))
    return eps


def bfs_under_f(tasks, fwd: ForwardModel, call_cap: int):
    """MPC with breadth-first search over F's predicted transitions, at most `call_cap` distinct F queries per
    decision. Execute the first action of the shortest predicted plan to the goal; if none is found within the cap,
    take the greedy-under-F action. Replan after every real step."""
    eps = []
    for t in tasks:
        x, acts, ep_keys = t.start, [], set()
        for step in range(t.budget):
            rem = t.budget - step
            if x == t.goal:
                acts.append("finish")
                break
            q = deque([(x, [])])
            seen = {x}
            calls = set()
            found = None
            while q and found is None:
                y, path = q.popleft()
                if len(path) + 1 >= rem:
                    continue
                for a in rw2.MOVES:
                    if len(calls) >= call_cap:
                        break
                    calls.add((y, a, t.limit))
                    o = fwd.predict([(y, a, t.limit)])[0]
                    if o is None or not o[1] or o[0] in seen:
                        continue
                    if o[0] == t.goal and len(path) + 2 <= rem:
                        found = path + [a]
                        break
                    seen.add(o[0])
                    q.append((o[0], path + [a]))
                if len(calls) >= call_cap:
                    break
            ep_keys |= calls
            if found is None:
                best, bd = None, None
                for a in rw2.MOVES:
                    ep_keys.add((x, a, t.limit))
                    o = fwd.predict([(x, a, t.limit)])[0]
                    if o is not None and o[1] and (bd is None or abs(o[0] - t.goal) < bd):
                        best, bd = a, abs(o[0] - t.goal)
                if best is None:
                    break
                a0 = best
            else:
                a0 = found[0]
            acts.append(a0)
            x = rw2.exact_step(x, a0, t.limit)[0]
        eps.append(_episode_record(t, acts, len(ep_keys)))
    return eps


def _episode_record(t, acts, f_calls):
    ex = rw2.execute(t, acts)
    return {"task": t.to_json(), "family": t.family, "feasible": t.feasible, "opt": t.opt, "actions": acts,
            "success": ex["success"], "outcome": "success" if ex["success"] else "fail", "return": ex["return"],
            "illegal_steps": sum(not s["legal"] for s in ex["steps"]), "steps_used": len(ex["steps"]),
            "f_calls_episode": f_calls, "gen_tokens": 0, "prompt_tokens": 0, "candidates": 0, "malformed": 0,
            "coverage_first": None, "open_loop_success": None, "first_action_optimal": None}


def summarize(eps: list[dict]) -> dict:
    n = len(eps)
    def frac(key):
        v = [e[key] for e in eps if e.get(key) is not None]
        return (sum(bool(x) for x in v) / len(v)) if v else None
    sol = [e for e in eps if e["success"]]
    out = {
        "n": n, "success": sum(e["success"] for e in eps) / n,
        "outcomes": dict(Counter(e["outcome"] for e in eps)),
        "mean_return": sum(e["return"] for e in eps) / n,
        "gen_tokens_per_task": sum(e["gen_tokens"] for e in eps) / n,
        "prompt_tokens_per_task": sum(e["prompt_tokens"] for e in eps) / n,
        "f_calls_per_task": sum(e.get("f_calls_episode", 0) for e in eps) / n,
        "candidates_per_task": sum(e["candidates"] for e in eps) / n,
        "malformed_rate": sum(e["malformed"] for e in eps) / max(1, sum(e["candidates"] for e in eps)),
        "coverage_first": frac("coverage_first"), "open_loop_success": frac("open_loop_success"),
        "first_action_optimal": frac("first_action_optimal"),
        "illegal_steps_per_task": sum(e["illegal_steps"] for e in eps) / n,
        "optimal_rate_given_success": (sum(e["steps_used"] == e["opt"] + 1 for e in sol) / len(sol)) if sol else None,
    }
    if any("would_abstain_rounds" in e for e in eps):
        out["would_abstain_any"] = sum(e.get("would_abstain_rounds", 0) > 0 for e in eps) / n
    cc = sum(e.get("claim_checked", 0) for e in eps)
    if cc:
        out["claim_mismatch_rate"] = sum(e["claim_mismatch"] for e in eps) / cc
    return out
