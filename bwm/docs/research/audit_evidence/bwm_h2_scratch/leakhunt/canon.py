"""Exact canonical form of a Blocksworld (state, goal) problem up to block renaming (scratch, read-only audit).

Structure: each block has <= 4 labelled ports (state-below, state-above, goal-below, goal-above) because 'on' is a
partial injective function in a valid state / goal, plus unary labels (state and goal unary facts) and nullary facts.
Canonical code of a connected component = min over roots of a deterministic port-ordered BFS encoding; the problem's
code = (nullary facts, sorted component codes). Equal codes <=> isomorphic (port-ordered BFS from a root fixes the
labelling, so equal codes give an explicit bijection; an isomorphism maps BFS from r to BFS from phi(r)).
"""
from __future__ import annotations


def _ports(facts, nodes, tag):
    below, above, unary, nullary = {}, {}, {}, []
    for f in facts:
        if f[0] == "on":
            x, y = f[1], f[2]
            nodes.add(x)
            nodes.add(y)
            assert x not in below, (tag, "non-functional on", f)
            assert y not in above, (tag, "non-injective on", f)
            below[x] = y
            above[y] = x
        elif len(f) == 1:
            nullary.append(f[0])
        else:
            assert len(f) == 2, f
            nodes.add(f[1])
            unary.setdefault(f[1], []).append(f[0])
    return below, above, {k: tuple(sorted(v)) for k, v in unary.items()}, tuple(sorted(nullary))


def canon(state, goal, objs=()):
    nodes = set(objs)
    sb, sa, su, sn = _ports(state, nodes, "state")
    gb, ga, gu, gn = _ports(goal, nodes, "goal")
    ports = (sb, sa, gb, ga)
    # components (undirected union of both relations)
    parent = {x: x for x in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rel in (sb, gb):
        for x, y in rel.items():
            parent[find(x)] = find(y)
    comps = {}
    for x in nodes:
        comps.setdefault(find(x), []).append(x)
    codes = []
    for members in comps.values():
        best = None
        for r in members:
            lab = {r: 0}
            order = [r]
            i = 0
            while i < len(order):
                x = order[i]
                i += 1
                for P in ports:
                    y = P.get(x)
                    if y is not None and y not in lab:
                        lab[y] = len(order)
                        order.append(y)
            code = tuple((su.get(x, ()), gu.get(x, ()), *(lab[P[x]] if x in P else -1 for P in ports)) for x in order)
            if best is None or code < best:
                best = code
        codes.append(best)
    return (sn, gn, tuple(sorted(codes)))


def goal_canon_n(goal, objs):
    """Goal configuration up to renaming, including blocks the goal does not mention (n-dependent)."""
    return canon(frozenset(), goal, objs)


def goal_canon_chain(goal):
    """Goal configuration up to renaming, only blocks the goal mentions (n-free)."""
    return canon(frozenset(), goal, ())
