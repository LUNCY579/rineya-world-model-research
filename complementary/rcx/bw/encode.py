"""Encodings of a Blocksworld (state, goal) pair for RCX-E07 (all deterministic; states/goals are BWM fact sets).

(a) graph(s, goal, objs): relational graph for RelGNN. Nodes = blocks (in `objs` order) + TABLE + HAND.
    Node features NODE_FEATS: is_block, is_table, is_hand, clear, goal_clear, hand_empty (HAND node), goal_handempty
    (HAND node). Typed directed edges (src -> dst), the model adds the inverse types itself:
      0 on(x,y): x -> y        1 ontable(x): x -> TABLE      2 holding(x): HAND -> x
      3 goal on(x,y): x -> y   4 goal ontable(x): x -> TABLE 5 goal holding(x): HAND -> x
    derived=True adds 6 on-and-goal-on(x,y) and 7 ontable-and-goal-ontable(x) (satisfied goal facts), which 1-WL message
    passing cannot otherwise detect when two edges point at look-alike nodes (an E07 ablation; off by default).
    Nothing depends on block names, so any renaming/reordering of blocks permutes nodes only.

(b) flat(s, goal, objs, n_max=12): padded vector for FlatMLP. Blocks are placed in a name-free CANONICAL ORDER (block
    names from B.generate are random letters, so alphabetical order would be an arbitrary labelling): colour refinement
    (1-WL) over the typed block graph (current below/above, goal below/above; initial colours from table/hand/clear and
    goal-table/goal-clear) with canonical integer relabelling (sorted signatures), then individualisation-refinement to
    split remaining ties: the smallest tied colour class is split by individualising its member that comes first in
    `objs`. Ties left by refinement are almost always automorphism orbits, where the choice does not change the
    encoding; tests measure renaming consistency. Layout (flat_dim = 2*n_max^2 + 6*n_max + 1 = 361 for n_max = 12):
      present[n_max] | on[n_max x n_max] | ontable | clear | holding | handempty[1] | goal_on[n_max x n_max] |
      goal_ontable | goal_clear
(c) text(s, goal): BWM's rendering B.canonical({"goal": B.goal_json(goal), "state": B.state_json(s)}) (the proposer
    prompt body without the chat template and outcome; goal_json keeps only goal on-pairs, as in BWM);
    text_canonical renames blocks by the canonical order to a, b, c, ... first; key_order re-orders state_json keys
    (e.g. BWM D-54 held-out orders H1-H3).
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from . import bwm_import as BI

NODE_FEATS = ("is_block", "is_table", "is_hand", "clear", "goal_clear", "hand_empty", "goal_handempty")
EDGE_TYPES = ("on", "ontable", "holding", "goal_on", "goal_ontable", "goal_holding")
DERIVED_EDGE_TYPES = ("on_sat", "ontable_sat")
N_MAX = 12
D54_HELDOUT_ORDERS = {"H1": ("table", "on", "hand", "clear"), "H2": ("hand", "table", "clear", "on"),
                      "H3": ("on", "clear", "table", "hand")}


def flat_dim(n_max: int = N_MAX) -> int:
    return 2 * n_max * n_max + 6 * n_max + 1


def n_edge_types(derived: bool = False) -> int:
    return len(EDGE_TYPES) + (len(DERIVED_EDGE_TYPES) if derived else 0)


# ----------------------------------------------------------------------------- (a) graph

def graph(s, goal, objs, derived: bool = False) -> dict:
    """{'x': float32 [n+2, F], 'edges': int64 [E, 3] (src, dst, type), 'n_blocks': n}. TABLE = n, HAND = n + 1."""
    n = len(objs)
    idx = {o: i for i, o in enumerate(objs)}
    T, H = n, n + 1
    x = np.zeros((n + 2, len(NODE_FEATS)), dtype=np.float32)
    x[:n, 0] = 1.0
    x[T, 1] = 1.0
    x[H, 2] = 1.0
    edges = []
    for f in s:
        if f[0] == "clear":
            x[idx[f[1]], 3] = 1.0
        elif f[0] == "handempty":
            x[H, 5] = 1.0
        elif f[0] == "on":
            edges.append((idx[f[1]], idx[f[2]], 0))
        elif f[0] == "ontable":
            edges.append((idx[f[1]], T, 1))
        elif f[0] == "holding":
            edges.append((H, idx[f[1]], 2))
    for f in goal:
        if f[0] == "clear":
            x[idx[f[1]], 4] = 1.0
        elif f[0] == "handempty":
            x[H, 6] = 1.0
        elif f[0] == "on":
            edges.append((idx[f[1]], idx[f[2]], 3))
            if derived and f in s:
                edges.append((idx[f[1]], idx[f[2]], 6))
        elif f[0] == "ontable":
            edges.append((idx[f[1]], T, 4))
            if derived and f in s:
                edges.append((idx[f[1]], T, 7))
        elif f[0] == "holding":
            edges.append((H, idx[f[1]], 5))
    e = np.array(sorted(edges), dtype=np.int64).reshape(-1, 3)
    return {"x": x, "edges": e, "n_blocks": n}


# ----------------------------------------------------------------------------- canonical order

def _relations(s, goal, objs):
    n = len(objs)
    idx = {o: i for i, o in enumerate(objs)}
    below, above = [None] * n, [None] * n
    pos_type, clear = [0] * n, [0] * n
    g_below, g_type, g_clear = [None] * n, [-1] * n, [0] * n
    g_above = [[] for _ in range(n)]
    for f in s:
        if f[0] == "on":
            a, b = idx[f[1]], idx[f[2]]
            below[a], above[b], pos_type[a] = b, a, 1
        elif f[0] == "holding":
            pos_type[idx[f[1]]] = 2
        elif f[0] == "clear":
            clear[idx[f[1]]] = 1
    for f in goal:
        if f[0] == "on":
            a, b = idx[f[1]], idx[f[2]]
            g_below[a], g_type[a] = b, 1
            g_above[b].append(a)
        elif f[0] == "ontable":
            g_type[idx[f[1]]] = 0
        elif f[0] == "holding":
            g_type[idx[f[1]]] = 2
        elif f[0] == "clear":
            g_clear[idx[f[1]]] = 1
    init = [(pos_type[i], clear[i], g_type[i], g_clear[i], len(g_above[i])) for i in range(n)]
    return init, below, above, g_below, g_above


def _relabel(sigs):
    rank = {sig: r for r, sig in enumerate(sorted(set(sigs)))}
    return [rank[sig] for sig in sigs]


def _refine(colors, below, above, g_below, g_above):
    k = len(set(colors))
    while True:
        c = colors
        sigs = [(c[i], -1 if below[i] is None else c[below[i]], -1 if above[i] is None else c[above[i]],
                 -1 if g_below[i] is None else c[g_below[i]], tuple(sorted(c[w] for w in g_above[i])))
                for i in range(len(c))]
        new = _relabel(sigs)
        k2 = len(set(new))
        if k2 == k:
            return new
        colors, k = new, k2


def canonical_order(s, goal, objs, stats: dict | None = None) -> list[int]:
    """Indices into `objs` in canonical order (see module doc). stats['individualised'] counts tie splits."""
    init, below, above, g_below, g_above = _relations(s, goal, objs)
    colors = _refine(_relabel(init), below, above, g_below, g_above)
    n_ind = 0
    while len(set(colors)) < len(colors):
        cnt = Counter(colors)
        c = min(col for col, k in cnt.items() if k > 1)
        pick = colors.index(c)
        colors = _refine(_relabel([(col, 0 if i == pick else 1) for i, col in enumerate(colors)]),
                         below, above, g_below, g_above)
        n_ind += 1
    if stats is not None:
        stats["individualised"] = stats.get("individualised", 0) + n_ind
    return sorted(range(len(objs)), key=lambda i: colors[i])


# ----------------------------------------------------------------------------- (b) flat

def flat(s, goal, objs, n_max: int = N_MAX, order: list[int] | None = None) -> np.ndarray:
    n = len(objs)
    if n > n_max:
        raise ValueError(f"{n} blocks > n_max={n_max}")
    order = canonical_order(s, goal, objs) if order is None else order
    slot = {objs[i]: k for k, i in enumerate(order)}
    present = np.zeros(n_max, np.float32)
    present[:n] = 1
    on, gon = np.zeros((n_max, n_max), np.float32), np.zeros((n_max, n_max), np.float32)
    ontable, clear, holding, hempty = (np.zeros(n_max, np.float32) for _ in range(4))
    gtable, gclear = np.zeros(n_max, np.float32), np.zeros(n_max, np.float32)
    for f in s:
        if f[0] == "on":
            on[slot[f[1]], slot[f[2]]] = 1
        elif f[0] == "ontable":
            ontable[slot[f[1]]] = 1
        elif f[0] == "clear":
            clear[slot[f[1]]] = 1
        elif f[0] == "holding":
            holding[slot[f[1]]] = 1
        elif f[0] == "handempty":
            hempty[0] = 1
    for f in goal:
        if f[0] == "on":
            gon[slot[f[1]], slot[f[2]]] = 1
        elif f[0] == "ontable":
            gtable[slot[f[1]]] = 1
        elif f[0] == "clear":
            gclear[slot[f[1]]] = 1
    return np.concatenate([present, on.ravel(), ontable, clear, holding, hempty[:1], gon.ravel(), gtable, gclear])


# ----------------------------------------------------------------------------- (c) text

def text(s, goal, key_order: tuple | None = None) -> str:
    B, _ = BI.load()
    st = B.state_json(s)
    if key_order is None:
        return B.canonical({"goal": B.goal_json(goal), "state": st})
    if sorted(key_order) != sorted(st):
        raise ValueError(f"key_order must permute {sorted(st)}")
    st = {k: st[k] for k in key_order}
    return json.dumps({"goal": B.goal_json(goal), "state": st}, separators=(",", ":"))


def rename(facts, mapping: dict) -> frozenset:
    return frozenset((f[0],) + tuple(mapping[a] for a in f[1:]) for f in facts)


def text_canonical(s, goal, objs, key_order: tuple | None = None) -> str:
    B, _ = BI.load()
    order = canonical_order(s, goal, objs)
    m = {objs[i]: B.NAMES[k] for k, i in enumerate(order)}
    return text(rename(s, m), rename(goal, m), key_order)
