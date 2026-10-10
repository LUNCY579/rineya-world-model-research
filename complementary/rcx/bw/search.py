"""Heuristic search with EXACT dynamics for RCX-E08: greedy best-first search and beam search over B.step.

Successors: every a in B.ground_actions(objs) with B.applicable(s, a), next state B.step(s, a). Heuristics: any
h(state, goal) -> float, or an object with .batch(states, goal) -> list[float] (e.g. models.LearnedH); successors of one
expansion are scored in one batch call. h = inf prunes a state (dead end). Duplicate detection is global (a state is
generated once). The goal test is applied at generation time. Budget = node expansions.

Result dict: success (B.execute VAL-semantics check of the returned plan: every action applicable and goal met; the
VAL binary is never called), found (search reached a goal state), plan (list of BWM action tuples), plan_len,
expansions, generated, h_calls (states scored), reason ('goal' | 'budget' | 'exhausted'), seconds.
Ties: GBFS pops the lowest (h, insertion order); beam keeps the `width` lowest (h, generation order) children.
"""
from __future__ import annotations

import heapq
import math
import time

from . import bwm_import as BI


def as_batch(h):
    if hasattr(h, "batch"):
        return h.batch
    return lambda states, goal: [h(s, goal) for s in states]


def _plan(parent, s):
    plan = []
    while parent[s] is not None:
        s, a = parent[s]
        plan.append(a)
    return plan[::-1]


def _expand(B, s, acts):
    out = []
    for a in acts:
        if B.applicable(s, a):
            out.append((a, B.step(s, a)[0]))
    return out


def _result(B, init, goal, plan, found, reason, exp, gen, calls, t0):
    plan = plan or []
    ex = B.execute(init, goal, plan) if found else {"success": False}
    return {"success": bool(ex["success"]), "found": found, "plan": plan, "plan_len": len(plan) if found else None,
            "expansions": exp, "generated": gen, "h_calls": calls, "reason": reason,
            "seconds": round(time.perf_counter() - t0, 4)}


def gbfs(objs, init, goal, h, max_expansions: int = 10_000) -> dict:
    B, _ = BI.load()
    t0 = time.perf_counter()
    hb = as_batch(h)
    if B.goal_met(init, goal):
        return _result(B, init, goal, [], True, "goal", 0, 1, 0, t0)
    acts = B.ground_actions(list(objs))
    parent = {init: None}
    calls, tie, exp, gen = 1, 0, 0, 1
    h0 = hb([init], goal)[0]
    heap = [(h0, 0, init)] if h0 < math.inf else []
    while heap:
        if exp >= max_expansions:
            return _result(B, init, goal, None, False, "budget", exp, gen, calls, t0)
        _, _, s = heapq.heappop(heap)
        exp += 1
        kids = []
        for a, ns in _expand(B, s, acts):
            if ns in parent:
                continue
            parent[ns] = (s, a)
            gen += 1
            if B.goal_met(ns, goal):
                return _result(B, init, goal, _plan(parent, ns), True, "goal", exp, gen, calls, t0)
            kids.append(ns)
        if kids:
            hs = hb(kids, goal)
            calls += len(kids)
            for ns, v in zip(kids, hs):
                if v < math.inf:
                    tie += 1
                    heapq.heappush(heap, (v, tie, ns))
    return _result(B, init, goal, None, False, "exhausted", exp, gen, calls, t0)


def beam_search(objs, init, goal, h, width: int = 8, max_expansions: int = 10_000) -> dict:
    B, _ = BI.load()
    t0 = time.perf_counter()
    hb = as_batch(h)
    if B.goal_met(init, goal):
        return _result(B, init, goal, [], True, "goal", 0, 1, 0, t0)
    acts = B.ground_actions(list(objs))
    parent = {init: None}
    beam, exp, gen, calls = [init], 0, 1, 0
    while beam:
        kids = []
        for s in beam:
            if exp >= max_expansions:
                return _result(B, init, goal, None, False, "budget", exp, gen, calls, t0)
            exp += 1
            for a, ns in _expand(B, s, acts):
                if ns in parent:
                    continue
                parent[ns] = (s, a)
                gen += 1
                if B.goal_met(ns, goal):
                    return _result(B, init, goal, _plan(parent, ns), True, "goal", exp, gen, calls, t0)
                kids.append(ns)
        if not kids:
            break
        hs = hb(kids, goal)
        calls += len(kids)
        ranked = sorted((v, i) for i, v in enumerate(hs) if v < math.inf)
        beam = [kids[i] for _, i in ranked[:width]]
    return _result(B, init, goal, None, False, "exhausted", exp, gen, calls, t0)
