"""Blocksworld planning WITHOUT execution feedback (PlanBench single-plan setting): the whole plan is built in
imagination under a forward model F, then submitted once and scored by the exact executor / VAL.

Forward models: ExactF (oracle ceiling), LMForwardBW (learned, goal-blind). Proposers: LMProposerBW, OracleProposer
(BFS, for harness tests only).
Planner "imagine": from the imagined state, propose K complete plans, roll each out under F; if some candidate is
predicted to reach the goal with every step legal, submit prefix + that candidate (shortest wins). Otherwise take the
first action of the candidate whose rollout satisfies the most goal facts (ties -> generation order), advance the
imagined state under F, and repeat (step cap). The exact simulator is never consulted while planning.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from . import blocks as B
from . import lm


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


class ExactF:
    name = "exact"

    def __init__(self):
        self.calls = 0

    def predict(self, keys):
        self.calls += len(keys)
        return [B.step(s, a) for s, a in keys]


class LMForwardBW:
    """Goal-blind learned F; memoised (frozen function). Unparseable output -> None (unsupported).

    cache: optional JSONL of raw model outputs keyed by (state, action), appended (fsynced) after every batch and
    reloaded on construction, so F work carries over across planner configurations, runs and crashes (F is frozen:
    reuse changes no result, only cost; `model_queries` counts only new model calls). max_batch_tokens: GPU decode
    budget; the delta format's outputs are <= 25 tokens, so far wider batches fit than for the state format."""
    name = "lm"

    def __init__(self, model, tok, adapter="fwd", fmt="state", cache=None, max_batch_tokens=2000):
        self.model, self.tok, self.adapter, self.fmt = model, tok, adapter, fmt
        self.memo = {}
        self.model_queries = 0
        self.model_tokens = 0
        self.max_batch_tokens = max_batch_tokens
        self.cache = None if cache is None else Path(cache)
        if self.cache is not None and self.cache.exists():
            for line in self.cache.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except Exception:
                    continue  # torn last line after a crash
                k = (frozenset(tuple(f) for f in r["s"]), tuple(r["a"]))
                self.memo[k] = self._parse(r["text"], k[0])

    def _parse(self, text, s):
        return B.parse_forward_delta(text, s) if self.fmt == "delta" else B.parse_forward(text)

    def predict(self, keys):
        need = list(dict.fromkeys(k for k in keys if k not in self.memo))
        if need:
            self.model_queries += len(need)
            self.model.set_adapter(self.adapter)
            if self.fmt == "delta":
                prompts = [B.forward_prompt_delta(s, a) for s, a in need]
                max_new = 40
            else:
                prompts = [B.forward_prompt(s, a) for s, a in need]
                max_new = max(80, 24 + 6 * max(len(B.blocks_of(s)) for s, _ in need))
            gens = lm.greedy_shared_prefix(self.model, self.tok, prompts, max_new_tokens=max_new,
                                           max_batch_tokens=self.max_batch_tokens)
            for k, g in zip(need, gens):
                self.model_tokens += g["n_tokens"]
                self.memo[k] = self._parse(g["text"], k[0])
            if self.cache is not None:
                with open(self.cache, "a", encoding="utf-8") as fh:
                    for k, g in zip(need, gens):
                        fh.write(json.dumps({"s": sorted(map(list, k[0])), "a": list(k[1]), "text": g["text"]}) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
        return [self.memo[k] for k in keys]


class OracleProposer:
    """BFS plans (harness test only: perfect proposals)."""

    def __init__(self, K=1):
        self.K = K

    def propose(self, items, eval_seed):
        out = []
        for it in items:
            p = B.bfs_plan(it["state"], it["goal"], it["objs"]) or []
            out.append([{"plan": p, "claimed": None, "n_tokens": 0}] * self.K)
        return out


class LMProposerBW:
    def __init__(self, model, tok, adapter, mode="actions", K=8, temperature=1.0, top_p=0.95, max_new=None,
                 prefix_ok=False, max_batch_tokens=2000):
        """max_new: generation cap (default 120 action-mode tokens, ~27 actions). prefix_ok: accept the leading
        complete actions of a truncated output (for the beam planner's rollout cap). max_batch_tokens: GPU budget."""
        self.model, self.tok, self.adapter, self.mode, self.K = model, tok, adapter, mode, K
        self.temperature, self.top_p = temperature, top_p
        self.max_new = max_new or (120 if mode == "actions" else 700)
        self.prefix_ok, self.max_batch_tokens = prefix_ok, max_batch_tokens

    def propose(self, items, eval_seed):
        prompts = [B.proposer_prompt(it["state"], it["goal"]) for it in items]
        seeds = [[None if j == 0 else _seed(eval_seed, it["id"], it["round"], j) for j in range(self.K)] for it in items]
        self.model.set_adapter(self.adapter)
        # Blocksworld evaluations are fresh runs: use the shared-prefix sampler (identical outputs, ~38% faster)
        rows = lm.sample_crn_shared(self.model, self.tok, prompts, seeds, max_new_tokens=self.max_new,
                                    temperature=self.temperature, top_p=self.top_p,
                                    max_batch_tokens=self.max_batch_tokens)
        out = [[] for _ in items]
        owner = [i for i in range(len(items)) for _ in range(self.K)]
        gens = [g for row in rows for g in row]
        for i, g in zip(owner, gens):
            parsed = (B.parse_plan_prefix(g["text"], items[i]["objs"]) if self.prefix_ok and self.mode == "actions"
                      else B.parse_plan(g["text"], items[i]["objs"], self.mode))
            out[i].append({"plan": parsed[0] if parsed else None, "claimed": parsed[1] if parsed else None,
                           "n_tokens": g["n_tokens"]})
        return out


def rollout(fwd, s0, plans):
    """Batched rollout of several plans from s0 under F. Returns per plan: (final_state, all_legal, n_steps, ok)."""
    n = len(plans)
    states = [s0] * n
    legal = [True] * n
    alive = [p is not None and len(p) > 0 for p in plans]
    sup = [True] * n
    d = 0
    while any(alive):
        keys, idx = [], []
        for i in range(n):
            if alive[i] and d < len(plans[i]):
                keys.append((states[i], plans[i][d]))
                idx.append(i)
            elif alive[i]:
                alive[i] = False
        if not keys:
            break
        for i, o in zip(idx, fwd.predict(keys)):
            if o is None:
                sup[i] = False
                alive[i] = False
                continue
            ns, ok = o
            if not ok:
                legal[i] = False
                alive[i] = False  # an illegal step invalidates the plan under PlanBench semantics
                continue
            states[i] = ns
        d += 1
    return [{"final": states[i], "legal": legal[i] and sup[i] and plans[i] is not None} for i in range(n)]


def plan_instances(instances, proposer, fwd, *, eval_seed=0, step_cap=24, log=None):
    """instances: dicts {id, objs, init, goal}. Returns per instance {plan, rounds, gen_tokens, verified}."""
    eps = []
    for it in instances:
        eps.append({"id": it["id"], "objs": it["objs"], "goal": it["goal"], "state": it["init"], "prefix": [],
                    "done": False, "rounds": 0, "gen_tokens": 0, "verified": False, "plan": None, "fail": None})
    rnd = 0
    while True:
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        items = [{"id": e["id"], "objs": e["objs"], "goal": e["goal"], "state": e["state"], "round": e["rounds"]}
                 for e in active]
        props = proposer.propose(items, eval_seed)
        for e, cands in zip(active, props):
            e["rounds"] += 1
            e["gen_tokens"] += sum(c["n_tokens"] for c in cands)
            plans = [c["plan"] for c in cands]
            rolls = rollout(fwd, e["state"], plans)
            best = None
            for p, r in zip(plans, rolls):
                if p is not None and r["legal"] and B.goal_met(r["final"], e["goal"]):
                    if best is None or len(p) < len(best):
                        best = p
            if best is not None:
                e["plan"] = e["prefix"] + list(best)
                e["verified"] = True
                e["done"] = True
                continue
            # no verified candidate: advance one imagined step along the most promising candidate
            score = []
            for i, (p, r) in enumerate(zip(plans, rolls)):
                if p is None:
                    continue
                score.append((len(e["goal"] & r["final"]), -i, p))
            if not score or len(e["prefix"]) >= step_cap:
                e["plan"] = e["prefix"] + (score and max(score)[2] or [])
                e["done"] = True
                e["fail"] = "step_cap" if score else "no_valid_candidate"
                continue
            a = max(score)[2][0]
            o = fwd.predict([(e["state"], a)])[0]
            if o is None or not o[1]:
                e["plan"] = e["prefix"] + max(score)[2]
                e["done"] = True
                e["fail"] = "first_action_illegal_under_F"
                continue
            e["state"] = o[0]
            e["prefix"].append(a)
        rnd += 1
        if log:
            log(f"round {rnd}: active {len(active)} -> {sum(not e['done'] for e in eps)}")
    return eps


def plan_direct(instances, proposer, *, eval_seed=0):
    """No-F arm (plain SFT plan generator, Plansformer-style): submit the proposer's candidate 0 (greedy) as is."""
    items = [{"id": it["id"], "objs": it["objs"], "goal": it["goal"], "state": it["init"], "round": 0}
             for it in instances]
    props = proposer.propose(items, eval_seed)
    return [{"id": it["id"], "plan": cands[0]["plan"] or [], "rounds": 1, "gen_tokens": cands[0]["n_tokens"],
             "verified": False, "fail": None if cands[0]["plan"] else "unparseable", "expansions": 0}
            for it, cands in zip(instances, props)]


def rollout_many(fwd, starts, plans):
    """Batched rollout of plans from per-plan start states under F. Per plan: the imagined states after each legal step
    (stops at the first step F calls illegal or cannot parse) and whether every step was legal and supported."""
    n = len(plans)
    states = list(starts)
    traj = [[] for _ in range(n)]
    ok = [p is not None and len(p) > 0 for p in plans]
    alive = list(ok)
    d = 0
    while True:
        keys, idx = [], []
        for i in range(n):
            if alive[i] and d < len(plans[i]):
                keys.append((states[i], plans[i][d]))
                idx.append(i)
            elif alive[i]:
                alive[i] = False
        if not keys:
            break
        for i, o in zip(idx, fwd.predict(keys)):
            if o is None or not o[1]:
                ok[i] = False
                alive[i] = False
                continue
            states[i] = o[0]
            traj[i].append(o[0])
        d += 1
    return [{"traj": traj[i], "all_legal": ok[i]} for i in range(n)]


def plan_beam(instances, proposer, fwd, *, eval_seed=0, width=4, max_expansions=48, max_depth=80, max_rollout=0,
              log=None, goal_met=None, unmet=None):
    """Proposer-guided beam search in imagination (PlanBench single-plan setting: F only, no execution feedback).

    A node is (imagined state, action prefix). Expanding a node asks the proposer for K complete plans from its state
    and rolls each out under F. If some candidate is predicted to reach the goal with every step legal, the instance is
    solved by the shortest such prefix + candidate found in that round. Otherwise each candidate yields one child: the
    imagined state at the candidate's point of best goal progress (fewest unmet goal facts, earliest; at least one
    step), with its action prefix. Children reaching an already-seen state are dropped (no cycles); the next beam is
    the best `width` children by (unmet goal facts, prefix length, candidate rank). Goal facts are read from the goal
    given in the problem; the domain rules are known only to the learned F. Nodes of all instances are expanded in one
    batched proposer call and one batched F rollout per round. Unsolved instances submit the prefix of the best node.
    max_rollout > 0 simulates only the first max_rollout actions of each candidate (F cost per expansion bounded; a
    solution must then be reached within that prefix, and longer plans are built across expansions).
    Domain hooks (joint roadmap item 3; defaults = Blocksworld): goal_met(state, goal) -> bool and
    unmet(state, goal) -> number (lower is closer). States and actions must be hashable; instances need
    "id", "init", "goal" (and whatever the proposer reads, e.g. "objs")."""
    goal_met = goal_met or B.goal_met
    unmet = unmet or (lambda st, g: len(g - st))
    eps = []
    for it in instances:
        eps.append({"id": it["id"], "objs": it.get("objs"), "goal": it["goal"], "beam": [(it["init"], [])],
                    "seen": {it["init"]}, "done": False, "rounds": 0, "gen_tokens": 0, "verified": False,
                    "plan": None, "fail": None, "expansions": 0,
                    "best": (unmet(it["init"], it["goal"]), 0, it["init"], [])})
    rnd = 0
    while True:
        active = [e for e in eps if not e["done"]]
        if not active:
            break
        items, owner = [], []
        for e in active:
            for j, (s, path) in enumerate(e["beam"]):
                items.append({"id": e["id"], "objs": e["objs"], "goal": e["goal"], "state": s, "round": f"{rnd}.{j}"})
                owner.append((e, s, path))
        props = proposer.propose(items, eval_seed)
        starts, plans, meta = [], [], []
        for (e, s, path), cands in zip(owner, props):
            e["expansions"] += 1
            e["gen_tokens"] += sum(c["n_tokens"] for c in cands)
            for rank, c in enumerate(cands):
                if c["plan"]:
                    starts.append(s)
                    plans.append(c["plan"][:max_rollout] if max_rollout else c["plan"])
                    meta.append((e, path, rank))
        rolls = rollout_many(fwd, starts, plans)
        sols, kids = {}, {}
        for (e, path, rank), plan, r in zip(meta, plans, rolls):
            traj = r["traj"]
            for t, st in enumerate(traj):
                if goal_met(st, e["goal"]):
                    full = path + list(plan[:t + 1])
                    if e["id"] not in sols or len(full) < len(sols[e["id"]]):
                        sols[e["id"]] = full
                    break
            if not traj:
                continue
            um = [unmet(st, e["goal"]) for st in traj]
            t_best = min(range(len(traj)), key=lambda t: (um[t], t))
            kids.setdefault(e["id"], []).append((um[t_best], len(path) + t_best + 1, rank, traj[t_best],
                                                 path + list(plan[:t_best + 1])))
        for e in active:
            e["rounds"] += 1
            if e["id"] in sols:
                e.update(plan=sols[e["id"]], verified=True, done=True)
                continue
            fresh = {}
            for u, ln, rank, st, p in sorted(kids.get(e["id"], []), key=lambda x: x[:3]):
                if st not in e["seen"] and st not in fresh and ln <= max_depth:
                    fresh[st] = (u, ln, rank, st, p)
            nxt = sorted(fresh.values(), key=lambda x: x[:3])[:width]
            for u, ln, rank, st, p in nxt:
                e["seen"].add(st)
                if (u, ln) < e["best"][:2]:
                    e["best"] = (u, ln, st, p)
            e["beam"] = [(st, p) for u, ln, rank, st, p in nxt]
            if not e["beam"] or e["expansions"] >= max_expansions:
                e.update(plan=e["best"][3], done=True, fail="dead_end" if not e["beam"] else "budget")
        rnd += 1
        if log:
            log(f"round {rnd}: active {len(active)} -> {sum(not e['done'] for e in eps)}; "
                f"F queries {getattr(fwd, 'model_queries', getattr(fwd, 'calls', 0))}")
    for e in eps:
        for k in ("beam", "seen", "best", "goal", "objs"):
            e.pop(k, None)
    return eps


def score(instances, eps) -> dict:
    by = {it["id"]: it for it in instances}
    res = Counter()
    rows = []
    for e in eps:
        it = by[e["id"]]
        ex = B.execute(it["init"], it["goal"], e["plan"] or [])
        res["n"] += 1
        res["success"] += ex["success"]
        res["verified"] += e["verified"]
        res["verified_true"] += e["verified"] and ex["success"]
        rows.append({"id": e["id"], "n_blocks": len(it["objs"]), "success": ex["success"], "valid": ex["valid"],
                     "len": len(e["plan"] or []),
                     "verified": e["verified"], "rounds": e["rounds"], "gen_tokens": e["gen_tokens"], "fail": e["fail"],
                     "plan": [B.action_str(a) for a in (e["plan"] or [])]})
    return {"success": res["success"] / max(1, res["n"]), "n": res["n"],
            "verified_rate": res["verified"] / max(1, res["n"]),
            "verification_precision": res["verified_true"] / max(1, res["verified"])}, rows
