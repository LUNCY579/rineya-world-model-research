"""Baseline heuristics h(state, goal) for RCX-E08 (all take BWM fact sets; objects are read off the state).

  unmet_goal_facts  BWM's beam ranking (blocks_planner.plan_beam default `unmet`): len(goal - state)
  misplaced_blocks  # blocks not in their final position (definition in space.misplaced_compact):
                    done(x) <=> x not held, x's support agrees with the goal (or the goal leaves it free), and if x is
                    on block y then done(y), the goal puts no other block on y and does not ask y to be clear.
  misplaced_lb      admissible lower bound in actions: 2 * (# not-done blocks on the table/blocks) + 1 if a block is
                    held and the goal is unmet (checked against exact h* in tests)
  h_add             classical additive STRIPS heuristic over B.ground_actions (unit costs), as in HSP (Bonet & Geffner)
  h0                0 (blind / breadth-first-like GBFS)
  OracleH           exact h* from the full state space (n <= 8) or A* (any n, budgeted)

`objs` for a state = B.blocks_of(state) (sorted names), which is the order space.from_facts expects.
"""
from __future__ import annotations

import math
from functools import lru_cache

from . import bwm_import as BI
from . import space as SP


def _objs(s):
    B, _ = BI.load()
    return B.blocks_of(s)


def unmet_goal_facts(s, goal) -> float:
    return float(len(goal - s))


def h0(s, goal) -> float:
    return 0.0


def misplaced_blocks(s, goal) -> float:
    objs = _objs(s)
    return float(SP.misplaced_compact(SP.from_facts(s, objs), SP.CompiledGoal(objs, goal))[0])


def misplaced_lb(s, goal) -> float:
    objs = _objs(s)
    return float(SP.misplaced_compact(SP.from_facts(s, objs), SP.CompiledGoal(objs, goal))[1])


@lru_cache(maxsize=64)
def _strips(objs: tuple):
    B, _ = BI.load()
    probe = []
    for a in B.ground_actions(list(objs)):
        op = a[0]
        if op == "pick-up":
            x = a[1]
            pre, add = (("clear", x), ("ontable", x), ("handempty",)), (("holding", x),)
        elif op == "put-down":
            x = a[1]
            pre, add = (("holding", x),), (("clear", x), ("handempty",), ("ontable", x))
        elif op == "stack":
            x, y = a[1], a[2]
            pre, add = (("clear", y), ("holding", x)), (("handempty",), ("clear", x), ("on", x, y))
        else:
            x, y = a[1], a[2]
            pre, add = (("on", x, y), ("clear", x), ("handempty",)), (("holding", x), ("clear", y))
        probe.append((pre, add))
    return probe


def h_add(s, goal) -> float:
    """Additive heuristic: cost(f) = 0 for f in s, cost(add) <= 1 + sum cost(pre); fixpoint; h = sum over goal."""
    cost = dict.fromkeys(s, 0.0)
    acts = _strips(tuple(_objs(s)))
    changed = True
    while changed:
        changed = False
        for pre, add in acts:
            c = 0.0
            for p in pre:
                v = cost.get(p)
                if v is None:
                    break
                c += v
            else:
                c += 1.0
                for f in add:
                    if c < cost.get(f, math.inf):
                        cost[f] = c
                        changed = True
    return float(sum(cost.get(f, math.inf) for f in goal))


class OracleH:
    """Exact h*: full-space labels for n <= max_space_n (cached per (objs, goal)), else budgeted A* (None -> raise)."""

    def __init__(self, max_space_n: int = 7, astar_budget: int = 200_000):
        self.max_space_n, self.astar_budget = max_space_n, astar_budget
        self._labels = {}

    def __call__(self, s, goal) -> float:
        objs = _objs(s)
        if len(objs) <= self.max_space_n:
            key = (tuple(objs), frozenset(goal))
            if key not in self._labels:
                self._labels[key] = SP.GoalLabels(objs, goal)
            return self._labels[key].h(s)
        h, _ = SP.astar_hstar(objs, s, goal, self.astar_budget)
        if h is None:
            raise RuntimeError(f"A* budget {self.astar_budget} exceeded for an n={len(objs)} state")
        return h


BASELINES = {"unmet_goal_facts": unmet_goal_facts, "misplaced_blocks": misplaced_blocks, "misplaced_lb": misplaced_lb,
             "h_add": h_add, "h0": h0}
