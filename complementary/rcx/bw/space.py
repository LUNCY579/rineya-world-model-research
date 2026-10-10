"""Exact Blocksworld state spaces and optimal cost-to-go h*(s, g) for small n (BWM 4-op semantics).

Compact state code for n blocks (block i = objs[i]): a tuple `pos` of length n with pos[i] = j (block i is on block j),
TABLE = n (on the table) or HAND = n + 1 (held). The code is name-free: one space per n serves every instance, and
facts are recovered through `objs`. to_facts()/from_facts() map codes to BWM fact sets exactly (clear and handempty
are derived); from_facts() rejects fact sets that are not valid Blocksworld states.

Transitions (`successors`) implement B.applicable/B.step for B.ground_actions(objs): pick-up x (x clear, on table, hand
empty), unstack x y (x on y, x clear, hand empty), put-down x (holding x), stack x y (holding x, y clear, x != y).
Equivalence with B.step over every state and every ground action is checked in tests/test_bw.py (n = 3..5, sampled 6).

StateSpace(n) enumerates all reachable states (including holding states) by BFS from the all-on-table state and stores
the forward graph and its reverse in CSR form. Blocksworld is connected, so the count is a(n) + n*a(n-1) with
a(n) = sum_k Lah(n, k) (OEIS A000262); this is asserted. h*(., goal) for ALL states is a multi-source backward BFS over
the reversed graph from the goal-satisfying states (goal <= s, i.e. B.goal_met); unreachable = inf (a dead end).

For n too large to enumerate (>= 9), `astar_hstar` gives exact h*(s) for single states by A* with the admissible
misplaced-block bound (`misplaced_compact`), with an expansion budget (None if exceeded).
"""
from __future__ import annotations

import heapq
import math
import time
from functools import lru_cache

import numpy as np

PICK, PUT, STACK, UNSTACK = 0, 1, 2, 3
OP_NAMES = ("pick-up", "put-down", "stack", "unstack")


def n_states_formula(n: int) -> int:
    """a(n) + n * a(n-1), a = A000262 (sets of lists): handempty states + holding states."""
    def a(m):
        if m == 0:
            return 1
        return sum(math.factorial(m) // math.factorial(k) * math.comb(m - 1, k - 1) for k in range(1, m + 1))
    return a(n) + n * a(n - 1)


# ----------------------------------------------------------------------------- codec

def to_facts(pos, objs) -> frozenset:
    n = len(objs)
    covered = {p for p in pos if p < n}
    facts, held = set(), False
    for i, p in enumerate(pos):
        x = objs[i]
        if p == n + 1:
            facts.add(("holding", x))
            held = True
            continue
        facts.add(("ontable", x) if p == n else ("on", x, objs[p]))
        if i not in covered:
            facts.add(("clear", x))
    if not held:
        facts.add(("handempty",))
    return frozenset(facts)


def from_facts(s, objs) -> tuple:
    """Compact code of a valid state; ValueError if `s` is not exactly a valid Blocksworld state over `objs`."""
    n = len(objs)
    idx = {o: i for i, o in enumerate(objs)}
    pos = [None] * n
    try:
        for f in s:
            if f[0] == "on":
                k, v = idx[f[1]], idx[f[2]]
            elif f[0] == "ontable":
                k, v = idx[f[1]], n
            elif f[0] == "holding":
                k, v = idx[f[1]], n + 1
            else:
                continue
            if pos[k] is not None:
                raise ValueError(f"block {f[1]} has two positions")
            pos[k] = v
    except KeyError as e:
        raise ValueError(f"unknown object {e}") from None
    if any(p is None for p in pos):
        raise ValueError("block without a position")
    pos = tuple(pos)
    supports = [p for p in pos if p < n]
    if len(set(supports)) != len(supports) or pos.count(n + 1) > 1:
        raise ValueError("two blocks on one block, or two held blocks")
    for i in range(n):
        j, k = i, 0
        while j < n:
            j, k = pos[j], k + 1
            if k > n:
                raise ValueError("cycle of on-facts")
    if to_facts(pos, objs) != frozenset(s):
        raise ValueError("not a valid Blocksworld state (cycle, double support, two held blocks or bad clear facts)")
    return pos


def action_tuple(code: tuple, objs) -> tuple:
    op, x, y = code
    if op in (PICK, PUT):
        return (OP_NAMES[op], objs[x])
    return (OP_NAMES[op], objs[x], objs[y])


def successors(pos: tuple) -> list:
    """[(action code (op, x, y), next pos)] for every applicable ground action (BWM semantics)."""
    n = len(pos)
    hand = n + 1
    held = -1
    covered = [False] * n
    for i, p in enumerate(pos):
        if p == hand:
            held = i
        elif p < n:
            covered[p] = True
    out = []
    if held < 0:
        for x in range(n):
            if not covered[x]:
                p = pos[x]
                nxt = list(pos)
                nxt[x] = hand
                out.append(((PICK, x, x) if p == n else (UNSTACK, x, p), tuple(nxt)))
    else:
        nxt = list(pos)
        nxt[held] = n
        out.append(((PUT, held, held), tuple(nxt)))
        for y in range(n):
            if y != held and not covered[y]:
                nxt = list(pos)
                nxt[held] = y
                out.append(((STACK, held, y), tuple(nxt)))
    return out


# ----------------------------------------------------------------------------- goals in compact form

class CompiledGoal:
    """Goal facts over objs in compact form. below[i]: -1 unconstrained, j / TABLE / HAND as in pos codes."""

    def __init__(self, objs, goal):
        n = len(objs)
        idx = {o: i for i, o in enumerate(objs)}
        self.n, self.below, self.clear = n, [-1] * n, [False] * n
        self.above = [[] for _ in range(n)]
        self.handempty, self.consistent = False, True
        for f in goal:
            if f[0] == "handempty":
                self.handempty = True
                continue
            if f[0] == "clear":
                self.clear[idx[f[1]]] = True
                continue
            k = idx[f[1]]
            v = {"on": lambda: idx[f[2]], "ontable": lambda: n, "holding": lambda: n + 1}[f[0]]()
            if self.below[k] not in (-1, v):
                self.consistent = False
            self.below[k] = v
            if f[0] == "on":
                self.above[v].append(k)
        self.has_holding = any(b == n + 1 for b in self.below)

    def met(self, pos) -> bool:
        n = self.n
        for i, b in enumerate(self.below):
            if b >= 0 and pos[i] != b:
                return False
        if self.handempty and (n + 1) in pos:
            return False
        for i, c in enumerate(self.clear):
            if c and (pos[i] == n + 1 or i in pos):
                return False
        return True


def misplaced_compact(pos, g: CompiledGoal) -> tuple[int, int, bool]:
    """(# blocks not in final position, admissible lower bound on h* in actions, holding?).

    done(x): x not held; goal does not fix x's support or x already sits on it; and if x is on block y: done(y), the goal
    puts no other block on y, and the goal does not ask y to be clear. Every not-done block must be moved in every plan
    (by induction up each tower), costing a pick-up/unstack plus a put-down/stack; a held block costs >= 1 when the goal
    is unmet (with a full hand the only applicable actions place it). Goal states return (0, 0). Goals with `holding`
    facts or inconsistent goals fall back to a bound of 0.
    """
    n = len(pos)
    hand = n + 1
    if g.consistent and g.met(pos):
        return 0, 0, hand in pos
    done = [None] * n

    def is_done(x):
        if done[x] is not None:
            return done[x]
        p = pos[x]
        if p == hand:
            r = False
        else:
            gb = g.below[x]
            r = gb in (-1, p)
            if r and p < n:
                r = (not g.clear[p]) and all(w == x for w in g.above[p]) and is_done(p)
        done[x] = r
        return r

    not_done = sum(1 for x in range(n) if not is_done(x))
    holding = hand in pos
    if g.has_holding or not g.consistent:
        return not_done, 0, holding
    lb = 2 * (not_done - (1 if holding else 0))
    if holding and not g.met(pos):
        lb += 1
    return not_done, lb, holding


# ----------------------------------------------------------------------------- full spaces

class StateSpace:
    """All reachable states for n blocks with forward/reverse CSR graphs (edges carry action codes)."""

    def __init__(self, n: int):
        t0 = time.perf_counter()
        self.n = n
        start = tuple([n] * n)
        index = {start: 0}
        order = [start]
        src, dst, act = [], [], []
        enc = lambda c: (c[0] * n + c[1]) * n + c[2]  # noqa: E731
        head = 0
        while head < len(order):
            pos = order[head]
            for code, nxt in successors(pos):
                j = index.get(nxt)
                if j is None:
                    j = index[nxt] = len(order)
                    order.append(nxt)
                src.append(head)
                dst.append(j)
                act.append(enc(code))
            head += 1
        self.index = index
        self.states = np.array(order, dtype=np.int8).reshape(len(order), n)
        src = np.asarray(src, dtype=np.int32)
        dst = np.asarray(dst, dtype=np.int32)
        act = np.asarray(act, dtype=np.int32)
        N = len(order)
        self.fwd_ptr = np.zeros(N + 1, dtype=np.int64)
        np.add.at(self.fwd_ptr, src + 1, 1)
        self.fwd_ptr = np.cumsum(self.fwd_ptr)
        self.fwd_dst, self.fwd_act = dst, act  # BFS appends edges grouped by src in increasing order
        o = np.argsort(dst, kind="stable")
        self.rev_ptr = np.zeros(N + 1, dtype=np.int64)
        np.add.at(self.rev_ptr, dst + 1, 1)
        self.rev_ptr = np.cumsum(self.rev_ptr)
        self.rev_src = src[o]
        self.n_edges = len(src)
        self.build_s = time.perf_counter() - t0
        expect = n_states_formula(n)
        if N != expect:
            raise AssertionError(f"n={n}: enumerated {N} states, formula says {expect}")

    def __len__(self):
        return len(self.states)

    def decode_action(self, a: int) -> tuple:
        n = self.n
        return (a // (n * n), (a // n) % n, a % n)

    def goal_mask(self, g: CompiledGoal) -> np.ndarray:
        S, n = self.states, self.n
        m = np.ones(len(S), dtype=bool)
        for i, b in enumerate(g.below):
            if b >= 0:
                m &= S[:, i] == b
        if g.handempty:
            m &= ~(S == n + 1).any(axis=1)
        for i, c in enumerate(g.clear):
            if c:
                m &= ~(S == i).any(axis=1) & (S[:, i] != n + 1)
        return m

    def backward_bfs(self, goal_mask: np.ndarray) -> np.ndarray:
        """Distance to the goal set for every state (int32, -1 = unreachable), by BFS on the reversed graph."""
        dist = np.full(len(self.states), -1, dtype=np.int32)
        frontier = np.flatnonzero(goal_mask)
        dist[frontier] = 0
        d = 0
        while frontier.size:
            d += 1
            starts, ends = self.rev_ptr[frontier], self.rev_ptr[frontier + 1]
            lens = ends - starts
            tot = int(lens.sum())
            if tot == 0:
                break
            offs = np.repeat(starts - np.cumsum(lens) + lens, lens) + np.arange(tot)
            pred = self.rev_src[offs]
            pred = np.unique(pred[dist[pred] < 0])
            dist[pred] = d
            frontier = pred
        return dist


@lru_cache(maxsize=8)
def get_space(n: int) -> StateSpace:
    return StateSpace(n)


class GoalLabels:
    """Exact labels for one (objs, goal): h*(s) for all states of the n-block space, optimal-action sets."""

    def __init__(self, objs, goal):
        self.objs = list(objs)
        self.goal = frozenset(goal)
        self.space = get_space(len(self.objs))
        self.cgoal = CompiledGoal(self.objs, self.goal)
        t0 = time.perf_counter()
        self.dist = self.space.backward_bfs(self.space.goal_mask(self.cgoal))
        self.label_s = time.perf_counter() - t0

    def sid(self, s) -> int:
        return self.space.index[from_facts(s, self.objs)]

    def h_id(self, i: int) -> float:
        d = int(self.dist[i])
        return math.inf if d < 0 else float(d)

    def h(self, s) -> float:
        return self.h_id(self.sid(s))

    def hstar_array(self) -> np.ndarray:
        """float64 h* over space.states (inf = dead end)."""
        out = self.dist.astype(np.float64)
        out[self.dist < 0] = math.inf
        return out

    def opt_actions_id(self, i: int) -> list[tuple]:
        """BWM action tuples a with h*(step(s, a)) = h*(s) - 1 (empty at goal states and dead ends)."""
        d = int(self.dist[i])
        if d <= 0:
            return []
        sp = self.space
        lo, hi = sp.fwd_ptr[i], sp.fwd_ptr[i + 1]
        return sorted(action_tuple(sp.decode_action(int(a)), self.objs)
                      for a, j in zip(sp.fwd_act[lo:hi], sp.fwd_dst[lo:hi]) if self.dist[j] == d - 1)

    def opt_actions(self, s) -> list[tuple]:
        return self.opt_actions_id(self.sid(s))

    def state(self, i: int) -> frozenset:
        return to_facts(tuple(int(v) for v in self.space.states[i]), self.objs)


def hstar(objs, s, goal) -> float:
    """Exact h*(s) via the full space (n <= 8)."""
    return GoalLabels(objs, goal).h(s)


# ----------------------------------------------------------------------------- A* for single states (any n)

def astar_hstar(objs, s, goal, max_expansions: int = 200_000):
    """Exact h*(s) by A* with the admissible misplaced bound; returns (h* or None if over budget, expansions).

    Returns (inf, expansions) if the goal is unreachable (inconsistent goal); reopening keeps it exact for any
    admissible bound.
    """
    g = CompiledGoal(list(objs), frozenset(goal))
    start = from_facts(s, list(objs))
    if not g.consistent:
        return math.inf, 0
    best = {start: 0}
    tie = 0
    heap = [(misplaced_compact(start, g)[1], 0, tie, start)]
    exp = 0
    while heap:
        f, neg_g, _, pos = heapq.heappop(heap)
        gc = -neg_g
        if gc > best.get(pos, math.inf):
            continue
        if g.met(pos):
            return float(gc), exp
        exp += 1
        if exp > max_expansions:
            return None, exp
        for _, nxt in successors(pos):
            ng = gc + 1
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                tie += 1
                heapq.heappush(heap, (ng + misplaced_compact(nxt, g)[1], -ng, tie, nxt))
    return math.inf, exp


def astar_optimal(objs, s, goal, max_expansions: int = 300_000):
    """Exact h*(s) AND the set of optimal first actions, by one A* search; returns (h*, actions, expansions).

    After the first goal is popped at cost C = h*(s), every remaining node with f <= C is expanded too (reopening on
    a shorter g), so each node on any optimal path ends with its exact depth g. Optimal-path nodes are then marked
    backwards from the goal nodes at depth C: u precedes a marked v iff u is a neighbour of v with g(u) = g(v) - 1.
    Blocksworld moves are invertible (pick-up/put-down, unstack/stack), so predecessors are the successors. An action
    a is optimal iff step(s, a) is marked. Returns (None, None, expansions) over budget; (inf, [], n) if unreachable.
    """
    objs = list(objs)
    g = CompiledGoal(objs, frozenset(goal))
    start = from_facts(s, objs)
    if not g.consistent:
        return math.inf, [], 0
    if g.met(start):
        return 0.0, [], 0
    best = {start: 0}
    tie, exp, C = 0, 0, None
    heap = [(misplaced_compact(start, g)[1], 0, tie, start)]
    goals = []
    while heap:
        if C is not None and heap[0][0] > C:
            break
        f, neg_g, _, pos = heapq.heappop(heap)
        gc = -neg_g
        if gc > best[pos]:
            continue
        if g.met(pos):
            if C is None:
                C = gc
            goals.append(pos)
            continue
        if exp >= max_expansions:
            return None, None, exp
        exp += 1
        for _, nxt in successors(pos):
            ng = gc + 1
            if ng < best.get(nxt, math.inf):
                best[nxt] = ng
                tie += 1
                heapq.heappush(heap, (ng + misplaced_compact(nxt, g)[1], -ng, tie, nxt))
    if C is None:
        return math.inf, [], exp
    marked = {v for v in goals if best[v] == C}
    frontier = list(marked)
    while frontier:
        nxt_frontier = []
        for v in frontier:
            dv = best[v]
            for _, u in successors(v):
                if u not in marked and best.get(u) == dv - 1:
                    marked.add(u)
                    nxt_frontier.append(u)
        frontier = nxt_frontier
    acts = sorted(action_tuple(code, objs) for code, v in successors(start) if v in marked and best[v] == 1)
    return float(C), acts, exp
