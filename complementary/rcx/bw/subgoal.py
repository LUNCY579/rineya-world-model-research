"""RCX-E11b: compositional subgoal selection for Blocksworld (RSM-001; CPU mechanism tier; tier B specialist).

Hierarchy. Goals are full on-fact goals (sets of ("on", x, y); a goal tower's bottom block is unconstrained). The skill
library is fixed: one parameterised skill per goal fact, ACHIEVE(on(x, y)) (`achieve`), a deterministic closed-loop
routine executed with BWM's exact dynamics (B.step): from a hand-empty state, move every block above x to the table
(top first), then every block above y to the table, then pick-up/unstack x and stack it on y. It always achieves
on(x, y) and ends hand-empty, but may destroy goal facts achieved earlier. A SELECTOR chooses which unmet goal fact to achieve
next; `execute` alternates selector and achiever until the goal holds (no search anywhere). Selectors:
  learned       SubgoalNet (a relational GNN over encode.graph, scoring every unmet goal fact; `LearnedSelector`)
  bottom_up     BU: goal towers in a fixed order (height desc, then bottom block name); the lowest unmet fact of the
                first tower with an unmet fact
  ready_first   RF (descriptive extra): an unmet fact on(x, y) whose y is done (in its final position, space's
                misplaced-blocks definition) and x, y clear; else a ready fact with the shortest achiever; else BU
  random_order  RND: a fixed random permutation of the goal facts per instance; the first unmet fact in it
Deterministic selectors are stopped when a decision state repeats (`cycle`: the run would loop forever); every selector
is also capped by max_calls subgoal calls and max_steps primitive actions (failure).

Labels (exact, n <= 7, `label_set`). For a hand-empty state s with unmet goal facts U(s), the label set A(s) is the set
of facts f in U(s) such that some optimal plan from s achieves f before any other fact of U(s), where "achieves" means
the plan's step that stacks x on y (the only operator that adds an on-fact). Computed by a DFS over the optimal-move
DAG of space.GoalLabels (h*(v) = h*(u) - 1), stopping each path at its first stack of a fact in U(s). A goal fact that
holds in s, is broken and is re-made along the way does not count (it is not in U(s)). A(s) is never empty for a
non-goal state (every unmet fact is stacked by every plan). A(s) is NOT the achiever-level optimum: see
`achiever_search` / `achiever_label_set` (exact option-level values V_A, Q_A and label sets A_A(s), any n within budget).

Goal structure (`structure`): the multiset of goal-tower heights (singletons included); k = number of towers of
height >= 2, hmax = tallest tower. E11b strata (train_k = 2, train_h = 4 by default): S (k <= 2 and hmax <= 4: the
training structures), T (k >= 3, hmax <= 4: more towers), H (k <= 2, hmax >= 5: taller towers), TH (both).

torch is imported lazily (only by the selector model), so labels, achiever, baselines and execution need no torch.
"""
from __future__ import annotations

import heapq
import itertools
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

from . import bwm_import as BI
from . import encode as EN
from . import space as SP

STRATA = ("S", "T", "H", "TH")


# ----------------------------------------------------------------------------- goal structure and instances

def goal_towers(goal, objs) -> list:
    """Goal towers bottom-up (lists of names, singletons included), ordered by (height desc, bottom name)."""
    below = {f[1]: f[2] for f in goal if f[0] == "on"}
    above = {f[2]: f[1] for f in goal if f[0] == "on"}
    if len(below) != sum(1 for f in goal if f[0] == "on") or len(above) != len(below):
        raise ValueError("goal on-facts do not form disjoint chains")
    towers = []
    for b in sorted(objs):
        if b in below:
            continue
        t = [b]
        while t[-1] in above:
            t.append(above[t[-1]])
        towers.append(t)
    if sum(len(t) for t in towers) != len(objs):
        raise ValueError("goal on-facts contain a cycle or unknown blocks")
    return sorted(towers, key=lambda t: (-len(t), t[0]))


def structure(goal, objs) -> dict:
    heights = sorted((len(t) for t in goal_towers(goal, objs)), reverse=True)
    k = sum(1 for h in heights if h >= 2)
    return {"heights": heights, "k": k, "hmax": heights[0] if heights else 0,
            "key": f"{len(objs)}:" + ",".join(map(str, heights))}


def stratum_of(heights, train_k: int = 2, train_h: int = 4) -> str:
    k = sum(1 for h in heights if h >= 2)
    hmax = max(heights)
    more, taller = k > train_k, hmax > train_h
    return "TH" if more and taller else "T" if more else "H" if taller else "S"


def partitions(n: int, max_part: int | None = None) -> list:
    """All integer partitions of n as descending tuples (deterministic order)."""
    max_part = n if max_part is None else max_part
    if n == 0:
        return [()]
    out = []
    for p in range(min(n, max_part), 0, -1):
        for rest in partitions(n - p, p):
            out.append((p,) + rest)
    return out


def partitions_in(n: int, stratum: str, train_k: int = 2, train_h: int = 4) -> list:
    """Partitions of n in the given stratum that have at least one tower of height >= 2 (a non-empty goal)."""
    return [p for p in partitions(n) if p[0] >= 2 and stratum_of(p, train_k, train_h) == stratum]


def make_goal(rng: random.Random, objs, heights) -> frozenset:
    """Goal towers of the given heights over a random assignment of objs (order of towers random)."""
    perm = list(objs)
    rng.shuffle(perm)
    hs = list(heights)
    rng.shuffle(hs)
    goal, i = set(), 0
    for h in hs:
        t = perm[i:i + h]
        i += h
        for lo, hi in zip(t, t[1:]):
            goal.add(("on", hi, lo))
    return frozenset(goal)


def make_instance(rng: random.Random, n: int, heights):
    """(objs, init, goal) with BWM's name pool and init distribution (B.random_towers, p_stack 0.6) and a goal of the
    given tower heights; None if the goal already holds in init."""
    B, _ = BI.load()
    objs = sorted(rng.sample(B.NAMES, n))
    init = B.towers_to_state(B.random_towers(objs, rng))
    goal = make_goal(rng, objs, heights)
    if not goal or B.goal_met(init, goal):
        return None
    return objs, init, goal


# ----------------------------------------------------------------------------- exact labels (n <= 7)

def unmet_facts(s, goal) -> list:
    return sorted(f for f in goal if f[0] == "on" and f not in s)


def label_set(L: SP.GoalLabels, i: int) -> frozenset:
    """A(s) for state id i of L's space (module doc): goal facts first achieved along some optimal plan."""
    sp, dist, cg, objs = L.space, L.dist, L.cgoal, L.objs
    n = sp.n
    if dist[i] <= 0:
        return frozenset()
    pos0 = sp.states[i]
    unmet = {x for x in range(n) if 0 <= cg.below[x] < n and pos0[x] != cg.below[x]}
    out, seen, stack = set(), set(), [i]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        du = dist[u]
        for e in range(sp.fwd_ptr[u], sp.fwd_ptr[u + 1]):
            v = int(sp.fwd_dst[e])
            if dist[v] != du - 1:
                continue
            op, x, y = sp.decode_action(int(sp.fwd_act[e]))
            if op == SP.STACK and x in unmet and cg.below[x] == y:
                out.add(("on", objs[x], objs[y]))
            else:
                stack.append(v)
    if not out:
        raise AssertionError("empty label set for a non-goal state")
    return frozenset(out)


def handempty_ids(L: SP.GoalLabels) -> list:
    """State ids of L's space with an empty hand that do not satisfy the goal (the selector's decision states)."""
    import numpy as np
    S, n = L.space.states, L.space.n
    return np.flatnonzero(~(S == n + 1).any(axis=1) & (L.dist > 0)).tolist()


def canonical_row_key(s, goal, objs) -> str:
    """Renaming-invariant key of a (state, goal) pair whose goal is disjoint on-chains (every E11b goal).

    Two rows get the same key iff some renaming of blocks maps one (state, goal) onto the other. Different instances
    of one goal class (n, tower heights) share their whole state space up to renaming, so instance-level uniqueness
    does not keep rows apart: this key is what row-level split hygiene (seen-in-training flags) uses. Blocks are coded
    by goal coordinates (tower, level); towers of equal height >= 2 are interchangeable (minimum over their
    permutations); singleton towers carry no goal constraint and share one code (exact, with no permutation)."""
    towers = goal_towers(goal, objs)
    heights = [len(t) for t in towers]
    coord = {b: (ti if len(t) >= 2 else -1, lvl) for ti, t in enumerate(towers) for lvl, b in enumerate(t)}
    groups = []
    for ti, h in enumerate(heights):
        if h < 2:
            continue
        if groups and heights[groups[-1][0]] == h:
            groups[-1].append(ti)
        else:
            groups.append([ti])
    _, above, held = _rel(s)
    stacks = []
    for b in sorted(f[1] for f in s if f[0] == "ontable"):
        st = [b]
        while st[-1] in above:
            st.append(above[st[-1]])
        stacks.append(st)
    if sum(map(len, stacks)) + (held is not None) != len(objs):
        raise ValueError("state does not place every block exactly once")
    best = None
    for perm in itertools.product(*(itertools.permutations(g) for g in groups)):
        rel = {-1: -1}
        for g, p in zip(groups, perm):
            rel.update(zip(g, p))
        code = tuple(sorted(tuple((rel[coord[b][0]], coord[b][1]) for b in st) for st in stacks))
        hv = () if held is None else (rel[coord[held][0]], coord[held][1])
        if best is None or (code, hv) < best:
            best = (code, hv)
    return f"{len(objs)}|{','.join(map(str, heights))}|{best!r}"


# ----------------------------------------------------------------------------- achiever and execution

class AchieverError(RuntimeError):
    pass


def _rel(s):
    on, above, held = {}, {}, None
    for f in s:
        if f[0] == "on":
            on[f[1]] = f[2]
            above[f[2]] = f[1]
        elif f[0] == "holding":
            held = f[1]
    return on, above, held


def achieve(B, s, fact):
    """(primitive actions, final state) of the fixed skill ACHIEVE(on(x, y)) from s (module doc)."""
    _, x, y = fact
    plan, cur = [], s

    def do(a):
        nonlocal cur
        if not B.applicable(cur, a):
            raise AchieverError(f"achiever step {a} inapplicable")
        cur = B.step(cur, a)[0]
        plan.append(a)

    _, _, held = _rel(cur)
    if held is not None:
        do(("put-down", held))
    for b in (x, y):
        while True:
            on, above, _ = _rel(cur)
            if b not in above:
                break
            t = b
            while t in above:
                t = above[t]
            do(("unstack", t, on[t]))
            do(("put-down", t))
    on, _, _ = _rel(cur)
    do(("unstack", x, on[x]) if x in on else ("pick-up", x))
    do(("stack", x, y))
    return plan, cur


def execute(objs, init, goal, select, max_calls: int, max_steps: int, detect_cycles: bool = True) -> dict:
    """Run selector + achiever from init (no search). success = goal reached with at most max_steps actions and the
    plan valid under B.execute. reason: goal | cycle | max_calls | max_steps."""
    B, _ = BI.load()
    t0 = time.perf_counter()
    s, plan, calls, seen, choices, reason = init, [], 0, set(), [], "goal"
    while not B.goal_met(s, goal):
        if calls >= max_calls:
            reason = "max_calls"
            break
        if len(plan) >= max_steps:
            reason = "max_steps"
            break
        if detect_cycles:
            if s in seen:
                reason = "cycle"
                break
            seen.add(s)
        cands = unmet_facts(s, goal)
        f = select(s, goal, objs, cands)
        if f not in cands:
            raise ValueError(f"selector returned {f!r}, not an unmet goal fact")
        acts, s = achieve(B, s, f)
        plan += acts
        calls += 1
        choices.append(" ".join(f))
    found = B.goal_met(s, goal)
    if found and len(plan) > max_steps:
        reason = "max_steps"
    ok = found and len(plan) <= max_steps and B.execute(init, goal, plan)["success"]
    return {"success": bool(ok), "found": found, "reason": reason, "plan_len": len(plan), "calls": calls,
            "plan": [" ".join(a) for a in plan], "choices": choices, "seconds": round(time.perf_counter() - t0, 4)}


# ----------------------------------------------------------------------------- baseline selectors

def done_blocks(s, goal, objs) -> set:
    """Blocks in their final position (space.misplaced_compact's 'done' definition)."""
    objs = list(objs)
    pos = SP.from_facts(s, objs)
    g = SP.CompiledGoal(objs, goal)
    n = len(objs)
    memo = [None] * n

    def is_done(x):
        if memo[x] is not None:
            return memo[x]
        p = pos[x]
        if p == n + 1:
            r = False
        else:
            gb = g.below[x]
            r = gb in (-1, p)
            if r and p < n:
                r = (not g.clear[p]) and all(w == x for w in g.above[p]) and is_done(p)
        memo[x] = r
        return r

    return {objs[x] for x in range(n) if is_done(x)}


class BottomUp:
    name = "bottom_up"

    def __init__(self):
        self._order = {}

    def fact_order(self, goal, objs) -> list:
        key = frozenset(goal)
        if key not in self._order:
            self._order[key] = [("on", hi, lo) for t in goal_towers(goal, objs) for lo, hi in zip(t, t[1:])]
        return self._order[key]

    def __call__(self, s, goal, objs, cands):
        cs = set(cands)
        for f in self.fact_order(goal, objs):
            if f in cs:
                return f
        raise ValueError("no unmet goal fact")


class ReadyFirst:
    name = "ready_first"

    def __init__(self):
        self.bu = BottomUp()

    def __call__(self, s, goal, objs, cands):
        B, _ = BI.load()
        rank = {f: i for i, f in enumerate(self.bu.fact_order(goal, objs))}
        done = done_blocks(s, goal, objs)
        clear = {f[1] for f in s if f[0] == "clear"}
        ready = sorted((f for f in cands if f[2] in done), key=rank.get)
        cheap = [f for f in ready if f[1] in clear and f[2] in clear]
        if cheap:
            return cheap[0]
        if ready:
            return min(ready, key=lambda f: (len(achieve(B, s, f)[0]), rank[f]))
        return self.bu(s, goal, objs, cands)


class RandomOrder:
    """Fixed random priority over the goal facts of one instance (seeded per instance)."""
    name = "random_order"

    def __init__(self, goal, seed):
        facts = sorted(f for f in goal if f[0] == "on")
        random.Random(seed).shuffle(facts)
        self.order = facts

    def __call__(self, s, goal, objs, cands):
        cs = set(cands)
        for f in self.order:
            if f in cs:
                return f
        raise ValueError("no unmet goal fact")


# ----------------------------------------------------------------------------- achiever-level exact values (any n)
#
# The selector's problem is a deterministic shortest path over hand-empty states whose edges are ACHIEVE calls (cost =
# the call's primitive length). V_A(s) is its exact optimum; Q_A(s, f) = |ACHIEVE(s, f)| + V_A(s'); the achiever-level
# label set is A_A(s) = argmin_f Q_A(s, f). A selector that always picks from A_A(s) attains V_A(init)
# (`AchieverOracle`); one that always picks from the optimal-plan label set A(s) (`label_set`) need not: an optimal
# primitive plan may re-make a broken goal fact, or move a goal-tower bottom off a block that must be cleared later,
# before its first unmet stack, and no ACHIEVE call does either (tests: a hand case where A(s) and A_A(s) are disjoint).

def achiever_search(objs, s, goal, max_expansions: int = 200_000) -> dict:
    """Exact V_A(s) by A* over ACHIEVE calls from hand-empty s, with h = 2 x not-done blocks (space.misplaced_compact).
    h is consistent for these edges: a block that is not moved keeps its done status (blocks below it are not moved
    either), and every moved block costs exactly 2 actions. Returns {"value": V_A (None over budget), "choices": an
    optimal fact sequence, "plan": its primitive actions, "expansions"}; value 0 and empty lists at goal states."""
    B, _ = BI.load()
    objs, goal = list(objs), frozenset(goal)
    cg = SP.CompiledGoal(objs, goal)

    def h(u):
        return SP.misplaced_compact(SP.from_facts(u, objs), cg)[1]

    best, parent, tie, exp = {s: 0}, {s: None}, 0, 0
    heap = [(h(s), 0, tie, s)]
    while heap:
        _, g, _, u = heapq.heappop(heap)
        if g > best[u]:
            continue
        if B.goal_met(u, goal):
            choices, plan, v = [], [], u
            while parent[v] is not None:
                v, f, acts = parent[v]
                choices.append(f)
                plan[:0] = acts
            return {"value": g, "choices": choices[::-1], "plan": plan, "expansions": exp}
        exp += 1
        if exp > max_expansions:
            return {"value": None, "choices": None, "plan": None, "expansions": exp}
        for f in unmet_facts(u, goal):
            acts, v = achieve(B, u, f)
            ng = g + len(acts)
            if ng < best.get(v, math.inf):
                best[v], parent[v] = ng, (u, f, acts)
                tie += 1
                heapq.heappush(heap, (ng + h(v), ng, tie, v))
    raise AchieverError("goal unreachable by ACHIEVE calls (goal is not disjoint on-chains)")


def achiever_q(objs, s, goal, max_expansions: int = 200_000, cache: dict | None = None) -> dict:
    """{unmet fact f: Q_A(s, f)} (None where V_A(s') is over budget). cache: optional {state: V_A} for one goal."""
    B, _ = BI.load()
    cache = {} if cache is None else cache
    out = {}
    for f in unmet_facts(s, goal):
        acts, v = achieve(B, s, f)
        if v not in cache:
            cache[v] = achiever_search(objs, v, goal, max_expansions)["value"]
        out[f] = None if cache[v] is None else len(acts) + cache[v]
    return out


def achiever_label_set(objs, s, goal, max_expansions: int = 200_000, cache: dict | None = None):
    """A_A(s) = argmin_f Q_A(s, f) as a frozenset; None if any Q_A is over budget (labels would not be exact)."""
    q = achiever_q(objs, s, goal, max_expansions, cache)
    if not q:
        return frozenset()
    if any(v is None for v in q.values()):
        return None
    m = min(q.values())
    return frozenset(f for f, v in q.items() if v == m)


LABEL_TARGETS = ("plan", "achiever")  # E11b training target: optimal-plan set A(s) or achiever-level set A_A(s)


def row_label_sets(L: SP.GoalLabels, i: int, objs, goal, max_expansions: int = 200_000,
                   cache: dict | None = None) -> dict:
    """Both exact label sets of row i (a hand-empty, non-goal state id of L's space): {"plan": A(s) (label_set),
    "achiever": A_A(s) (achiever_label_set)}. cache: optional {state: V_A} for this goal (one per instance). Raises
    AchieverError if A_A(s) is over budget, so a row never carries an inexact label set."""
    s = L.state(i)
    plan = label_set(L, i)
    ach = achiever_label_set(objs, s, goal, max_expansions, cache)
    if ach is None:
        raise AchieverError(f"achiever-level labels over budget ({max_expansions} expansions)")
    if not plan or not ach:
        raise AchieverError("empty label set for a non-goal row")
    return {"plan": plan, "achiever": ach}


class AchieverOracle:
    """Selector that picks argmin_f Q_A(s, f) (first candidate on ties): the achiever-level optimum (a ceiling, not a
    baseline: it searches). Its execution length equals V_A(init)."""
    name = "achiever_oracle"

    def __init__(self, max_expansions: int = 200_000):
        self.max_expansions, self.cache = max_expansions, {}

    def __call__(self, s, goal, objs, cands):
        key = frozenset(goal)
        cache = self.cache.setdefault(key, {})
        q = achiever_q(objs, s, goal, self.max_expansions, cache)
        if any(v is None for v in q.values()):
            raise AchieverError("achiever oracle over budget")
        return min(cands, key=lambda f: q[f])


# ----------------------------------------------------------------------------- learned selector (lazy torch)

_T: dict = {}


def _torch():
    if "torch" not in _T:
        if "torch" not in sys.modules:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before torch is imported (never '-1')
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from . import models as M  # patches torch.cuda to report no device (RCWM I-21)
        _T.update(torch=torch, nn=nn, F=F, M=M)
    return _T["torch"], _T["nn"], _T["F"], _T["M"]


@dataclass
class SubgoalNetConfig:
    node_in: int = len(EN.NODE_FEATS)
    n_rel: int = len(EN.EDGE_TYPES)
    hidden: int = 48
    rounds: int = 4
    head_hidden: int = 64


def net_class():
    """SubgoalNet: RelGNN-style typed message passing (models.RelGNN's layer, sum aggregation, LayerNorm residual)
    returning node embeddings; candidate on(x, y) logit = MLP([h_x, h_y, h_x * h_y, mean_v h_v of its graph])."""
    if "cls" in _T:
        return _T["cls"]
    torch, nn, F, M = _torch()

    class SubgoalNet(nn.Module):
        kind = "subgoal"

        def __init__(self, cfg: SubgoalNetConfig = SubgoalNetConfig()):
            super().__init__()
            self.cfg = cfg
            H, R = cfg.hidden, 2 * cfg.n_rel
            self.inp = nn.Linear(cfg.node_in, H)
            self.w_rel = nn.ParameterList([nn.Parameter(torch.randn(R, H, H) / H ** 0.5) for _ in range(cfg.rounds)])
            self.w_self = nn.ModuleList([nn.Linear(H, H) for _ in range(cfg.rounds)])
            self.norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(cfg.rounds)])
            self.head = nn.Sequential(nn.Linear(4 * H, cfg.head_hidden), nn.ReLU(), nn.Linear(cfg.head_hidden, 1))

        def embed(self, b):
            h = F.relu(self.inp(b["x"]))
            src, dst, rel = b["src"], b["dst"], b["rel"]
            for k in range(self.cfg.rounds):
                hw = torch.einsum("nh,rhk->rnk", h, self.w_rel[k])
                m = torch.zeros_like(h).index_add_(0, dst, hw[rel, src])
                h = self.norm[k](h + F.relu(self.w_self[k](h) + m))
            G = b["n_graphs"]
            cnt = torch.zeros(G).index_add_(0, b["graph"], torch.ones(h.shape[0]))
            gmean = torch.zeros(G, h.shape[1]).index_add_(0, b["graph"], h) / cnt.unsqueeze(1)
            return h, gmean

        def forward(self, b):
            h, gmean = self.embed(b)
            hx, hy = h[b["cx"]], h[b["cy"]]
            z = torch.cat([hx, hy, hx * hy, gmean[b["cg"]]], dim=1)
            return self.head(z).squeeze(1)

    _T["cls"] = SubgoalNet
    return SubgoalNet


def new_net(cfg: SubgoalNetConfig | None = None, seed: int | None = None):
    torch, _, _, _ = _torch()
    if seed is not None:
        torch.manual_seed(seed)
    return net_class()(cfg or SubgoalNetConfig())


def encode_row(s, goal, objs, cands, labels=()):
    """One selector example: encode.graph + candidate node indices (x, y) + label positions."""
    idx = {o: i for i, o in enumerate(objs)}
    labels = set(labels)
    return {"g": EN.graph(s, goal, objs), "cx": [idx[f[1]] for f in cands], "cy": [idx[f[2]] for f in cands],
            "lab": [k for k, f in enumerate(cands) if f in labels]}


def collate(rows):
    """Batch of encode_row dicts -> model input; candidates flattened, with (row, column) for padding."""
    torch, _, _, M = _torch()
    b = M.collate_graphs([r["g"] for r in rows], len(EN.EDGE_TYPES))
    cx, cy, cg, crow, ccol, lab = [], [], [], [], [], []
    off = 0
    for i, r in enumerate(rows):
        for k, (x, y) in enumerate(zip(r["cx"], r["cy"])):
            cx.append(off + x)
            cy.append(off + y)
            cg.append(i)
            crow.append(i)
            ccol.append(k)
        lab.append(r["lab"])
        off += r["g"]["x"].shape[0]
    b.update(cx=torch.tensor(cx, dtype=torch.long), cy=torch.tensor(cy, dtype=torch.long),
             cg=torch.tensor(cg, dtype=torch.long), crow=torch.tensor(crow, dtype=torch.long),
             ccol=torch.tensor(ccol, dtype=torch.long), lab=lab, nrows=len(rows),
             ncol=max((len(r["cx"]) for r in rows), default=1))
    return b


def _padded(torch, logits, b):
    P = torch.full((b["nrows"], b["ncol"]), -math.inf)
    P[b["crow"], b["ccol"]] = logits
    return P


def set_nll(logits, b):
    """Mean over rows of -log sum_{c in labels} softmax(row logits)_c (marginal likelihood of the label set)."""
    torch, _, _, _ = _torch()
    P = _padded(torch, logits, b)
    mask = torch.zeros_like(P, dtype=torch.bool)
    for i, lab in enumerate(b["lab"]):
        mask[i, lab] = True
    lse_all = torch.logsumexp(P, dim=1)
    lse_lab = torch.logsumexp(P.masked_fill(~mask, -math.inf), dim=1)
    return (lse_all - lse_lab).mean()


def predict_choice(model, rows, batch_size: int = 256) -> list:
    """Argmax candidate position per row (first maximum on ties)."""
    torch, _, _, _ = _torch()
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            b = collate(rows[i:i + batch_size])
            out += _padded(torch, model(b), b).argmax(dim=1).tolist()
    return out


def accuracy(model, rows) -> float:
    ch = predict_choice(model, rows)
    return sum(c in r["lab"] for c, r in zip(ch, rows)) / max(1, len(rows))


def fit(model, rows_tr, rows_va, *, epochs: int, batch_size: int, lr: float, seed: int, time_cap_s: float,
        log=print, courtesy=None) -> dict:
    """Seeded AdamW on set_nll; keeps the epoch with the best validation top-1 accuracy. Returns training info."""
    torch, _, _, _ = _torch()
    import copy
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, best_state, best_ep, hist, step, cap_hit = -1.0, None, -1, [], 0, False
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = list(range(len(rows_tr)))
        rng.shuffle(perm)
        model.train()
        losses, te = [], time.perf_counter()
        for b0 in range(0, len(perm), batch_size):
            b = collate([rows_tr[i] for i in perm[b0:b0 + batch_size]])
            loss = set_nll(model(b), b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            losses.append(float(loss))
            if courtesy:
                courtesy()
            if time.perf_counter() - t0 > time_cap_s:
                cap_hit = True
                break
        acc = accuracy(model, rows_va)
        hist.append({"epoch": ep, "steps": step, "train_loss": float(sum(losses) / max(1, len(losses))),
                     "val_top1": acc, "epoch_s": round(time.perf_counter() - te, 2)})
        log(f"selector seed {seed} ep {ep} steps {step} loss {hist[-1]['train_loss']:.4f} val_top1 {acc:.4f}")
        if acc > best:
            best, best_state, best_ep = acc, copy.deepcopy(model.state_dict()), ep
        if cap_hit:
            log(f"selector seed {seed}: TIME CAP {time_cap_s}s hit at step {step}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"seed": seed, "config": asdict(model.cfg), "n_params": sum(p.numel() for p in model.parameters()),
            "epochs_run": len(hist), "steps": step, "best_epoch": best_ep, "best_val_top1": best,
            "cap_hit": cap_hit, "wall_s": round(time.perf_counter() - t0, 2), "history": hist}


class LearnedSelector:
    """Selector from a trained SubgoalNet: argmax over the unmet facts (first maximum, candidates sorted)."""
    name = "learned"

    def __init__(self, model, threads: int = 1):
        self.model, self.threads = model.eval(), threads

    def __call__(self, s, goal, objs, cands):
        torch, _, _, _ = _torch()
        prev = torch.get_num_threads()
        try:
            if prev != self.threads:
                torch.set_num_threads(self.threads)
            k = predict_choice(self.model, [encode_row(s, goal, list(objs), cands)])[0]
        finally:
            if prev != self.threads:
                torch.set_num_threads(prev)
        return cands[k]
