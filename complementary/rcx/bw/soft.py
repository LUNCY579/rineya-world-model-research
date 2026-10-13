"""Exact soft-goal oracle for RCX-E05 on the enumerated Blocksworld spaces of rcx/bw/space.py (PREREG_E05 §3).

Features. phi(s) in {0,1}^F, F = n(n+1): on(i, j) for i != j in (i, j) order (n(n-1) facts), then ontable(i) (n), then
clear(i) (n); exactly the on/ontable/clear facts of space.to_facts (a held block has none). `facts(n)` lists them with
integer block ids; n = 7 gives the 56 facts of the prereg (42 + 7 + 7).

Goals and STOP. A linear goal is (w, c): w over the F facts (small integers, or dyadic rationals), step cost c > 0
(dyadic), w-bar = (w, -c). STOP is always legal and collects stop(s) = w.phi(s); every move costs c. Any terminal reward
vector works in place of w.phi (H-CONJ: R * 1[G <= s]). Values are exact dyadic float64, so ties are exact.

Oracle. V*(s) = max(stop(s), max over legal s' of [-c + V*(s')]) by max-plus value iteration over the forward CSR graph,
started at the STOP values (sweep k gives the best return over paths of <= k moves; converges after <= diameter + 1
sweeps because c > 0 and every state has a move). An optional boolean edge mask (aligned with space.fwd_dst) restricts
the dynamics (H-DYN: `height_limit_mask`); STOP stays legal.

Policies and successor features. pi is greedy on V* with the tie rule: STOP first (iff stop(s) == V*(s)), then the
lowest action code (op, x, y) (space codes, PICK 0 < PUT 1 < STACK 2 < UNSTACK 3). V* strictly increases by c along
every pi move, so pi never loops; endpoints e(s) and lengths T(s) come from pointer doubling, and
psi(s) = (phi(e(s)), T(s)), V*(s) = w.phi(e(s)) - c T(s) = w-bar.psi(s) exactly (`sf_table`).

GPI. For a test goal (w', c') and policy tables C: Q(s, move to s') = max over z in C of w-bar'.[(0, 1) + psi_z(s')]
= -c' + max_z [w'.phi(e_z(s')) - c' T_z(s')] (`gpi_table` for exact tables, `gpi_scores` for any psi array);
Q(s, STOP) = w'.phi(s).

Decision protocol (every arm). Candidates = [STOP] + the legal moves in action-code order; the arm returns the index of
its choice; scored arms take the FIRST maximum (`decide`: STOP first, then lowest action code; exact float compare).
`run_episode`: closed loop from s0 until STOP, `HMAX` = 40 moves (forced STOP) or a move into a state already visited
in this episode (forced STOP at that state, flagged as a loop; the move is counted). J = stop(s_final) - c T and
nreg = clip((V*(s0) - J) / (V*(s0) - stop(s0)), 0, 1). References: ORACLE (greedy on V*, = pi), STOP, GREEDY (one-step
lookahead stop(s') - c), RAND (uniform over [STOP] + moves from a caller-supplied random.Random), exact GPI.

Renaming. sigma is a block permutation (block i -> sigma[i]); `fact_perm` and `state_perm` give its action on fact
indices and state indices (phi(sigma s)[fact_perm[k]] = phi(s)[k]); a weight renames as w_sigma[fact_perm] = w.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, NamedTuple

import numpy as np

from . import space as SP

HMAX = 40  # closed-loop move cap (forced STOP)


# ----------------------------------------------------------------------------- features

@lru_cache(maxsize=None)
def facts(n: int) -> tuple:
    """The F = n(n+1) feature facts with integer block ids, in feature order."""
    out = [("on", i, j) for i in range(n) for j in range(n) if i != j]
    return tuple(out + [("ontable", i) for i in range(n)] + [("clear", i) for i in range(n)])


@lru_cache(maxsize=None)
def fact_index(n: int) -> dict:
    return {f: k for k, f in enumerate(facts(n))}


def phi_table(space) -> np.ndarray:
    """phi for every state: uint8 [N, n(n+1)]."""
    S, n = space.states.astype(np.int64), space.n
    cols = [S[:, i] == j for i in range(n) for j in range(n) if i != j]
    cols += [S[:, i] == n for i in range(n)]
    cols += [(S[:, i] != n + 1) & ~(S == i).any(axis=1) for i in range(n)]
    return np.stack(cols, axis=1).astype(np.uint8)


def get_phi(space) -> np.ndarray:
    """phi_table cached on the space object (read-only use)."""
    p = getattr(space, "_soft_phi", None)
    if p is None:
        p = space._soft_phi = phi_table(space)
    return p


def stop_values(phi: np.ndarray, w) -> np.ndarray:
    """STOP value w.phi(s) for every state (float64, exact for dyadic w)."""
    return phi @ np.asarray(w, dtype=np.float64)


# ----------------------------------------------------------------------------- graph helpers

def _csr(space) -> dict:
    """Forward CSR with each state's edges sorted by action code (cached on the space object)."""
    g = getattr(space, "_soft_csr", None)
    if g is None:
        deg = np.diff(space.fwd_ptr)
        if len(deg) == 0 or (deg <= 0).any():
            raise AssertionError("every state needs a legal move")
        src = np.repeat(np.arange(len(deg)), deg)
        order = np.lexsort((space.fwd_act, src))
        starts = space.fwd_ptr[:-1]
        g = dict(src=src[order], dst=space.fwd_dst[order].astype(np.int64), act=space.fwd_act[order],
                 order=order, starts=starts, ptr=space.fwd_ptr, pos=np.arange(len(src)) - starts[src[order]])
        space._soft_csr = g
    return g


def sorted_mask(space, edge_mask):
    """An edge mask in space edge order -> the same mask in the action-code-sorted edge order (None passes)."""
    if edge_mask is None:
        return None
    m = np.asarray(edge_mask, dtype=bool)
    if m.shape != (space.n_edges,):
        raise ValueError("edge mask must align with space.fwd_dst")
    return m[_csr(space)["order"]]


def moves(space, s: int, keep=None) -> tuple[np.ndarray, np.ndarray]:
    """(action codes, successor indices) of the legal moves at state s in action-code order; `keep` = sorted_mask."""
    g = _csr(space)
    lo, hi = int(g["ptr"][s]), int(g["ptr"][s + 1])
    if keep is None:
        return g["act"][lo:hi], g["dst"][lo:hi]
    k = keep[lo:hi]
    return g["act"][lo:hi][k], g["dst"][lo:hi][k]


def handempty_mask(space) -> np.ndarray:
    return ~(space.states == space.n + 1).any(axis=1)


def tower_levels(space) -> np.ndarray:
    """level[s, i]: 1-based position of block i in its tower (1 = on the table), 0 if held."""
    S, n = space.states.astype(np.int64), space.n
    lev = np.where(S == n, 1, 0)
    below = np.where(S < n, S, 0)
    for _ in range(n):
        lb = np.take_along_axis(lev, below, axis=1)
        new = np.where(S < n, np.where(lb > 0, lb + 1, 0), lev)
        if np.array_equal(new, lev):
            break
        lev = new
    return lev


def height_ok(space, h: int = 3) -> np.ndarray:
    """States whose towers all have <= h blocks."""
    return (tower_levels(space) <= h).all(axis=1)


def height_limit_mask(space, h: int = 3) -> np.ndarray:
    """Edge mask (space edge order) of the height-limit-h dynamics: stack(x, y) is legal only if the tower that x joins
    has <= h blocks afterwards (level(y) + 1 <= h); pick-up, unstack and put-down are unchanged."""
    n = space.n
    lev = tower_levels(space)
    src = np.repeat(np.arange(len(space)), np.diff(space.fwd_ptr))
    op, y = space.fwd_act // (n * n), space.fwd_act % n
    return (op != SP.STACK) | (lev[src, y] + 1 <= h)


# ----------------------------------------------------------------------------- value iteration, policies, SF tables

def max_plus_vi(space, stop, c: float, edge_mask=None) -> np.ndarray:
    """V*(s) = max(stop(s), max over legal s' of [-c + V*(s')]), exact, from the STOP values."""
    if not c > 0:
        raise ValueError("step cost must be > 0")
    g = _csr(space)
    stop = np.asarray(stop, dtype=np.float64)
    keep = sorted_mask(space, edge_mask)
    dst, starts = g["dst"], g["starts"]
    V = stop.copy()
    for _ in range(len(V) + 1):
        cand = V[dst] - c
        if keep is not None:
            cand[~keep] = -np.inf
        nV = np.maximum(stop, np.maximum.reduceat(cand, starts))
        if np.array_equal(nV, V):
            return V
        V = nV
    raise RuntimeError("max-plus value iteration did not converge")


def soft_values(space, w, c: float, edge_mask=None, phi=None) -> np.ndarray:
    """V*_w for a linear goal (w, c) (max-plus VI from the STOP values w.phi)."""
    return max_plus_vi(space, stop_values(get_phi(space) if phi is None else phi, w), c, edge_mask)


def conj_values(space, G, R: float, c: float) -> np.ndarray:
    """H-CONJ values (terminal reward R * 1[G <= s], free dynamics) via backward_bfs:
    V*(s) = max(R * 1[G <= s], R - c h*(s, G)). G: facts with integer block ids, e.g. ("on", 0, 1)."""
    m = space.goal_mask(compile_goal(space.n, G))
    d = space.backward_bfs(m)
    reach = np.where(d >= 0, R - c * d.astype(np.float64), -np.inf)
    return np.maximum(np.where(m, float(R), 0.0), reach)


def compile_goal(n: int, G) -> SP.CompiledGoal:
    cg = SP.CompiledGoal(list(range(n)), frozenset(G))
    if not cg.consistent:
        raise ValueError(f"inconsistent goal {sorted(G)}")
    return cg


def policy_table(space, V, stop, c: float, edge_mask=None) -> np.ndarray:
    """nxt[s] = the successor chosen by pi (nxt[s] = s where pi STOPs): STOP iff stop(s) == V(s), else the lowest
    action code among legal moves with -c + V(s') == V(s)."""
    g = _csr(space)
    V = np.asarray(V, dtype=np.float64)
    opt = (V[g["dst"]] - c) == V[g["src"]]
    keep = sorted_mask(space, edge_mask)
    if keep is not None:
        opt &= keep
    big = len(opt) + 1
    first = np.minimum.reduceat(np.where(opt, g["pos"], big), g["starts"])
    nxt = np.arange(len(V), dtype=np.int64)
    mv = np.asarray(stop, dtype=np.float64) != V
    if (first[mv] >= big).any():
        raise AssertionError("a non-STOP state has no optimal move (V is not the fixed point)")
    nxt[mv] = g["dst"][g["starts"][mv] + first[mv]]
    return nxt


def endpoints_lengths(nxt) -> tuple[np.ndarray, np.ndarray]:
    """(endpoint int32, length int16) of the deterministic policy nxt (absorbing where nxt[s] == s), by pointer
    doubling."""
    P = np.asarray(nxt, dtype=np.int64).copy()
    L = (P != np.arange(len(P))).astype(np.int64)
    for _ in range(64):
        PP = P[P]
        if np.array_equal(PP, P):
            break
        L, P = L + L[P], PP
    if not np.array_equal(np.asarray(nxt)[P], P):
        raise AssertionError("policy has a cycle")
    if L.max(initial=0) > np.iinfo(np.int16).max:
        raise OverflowError("length exceeds int16")
    return P.astype(np.int32), L.astype(np.int16)


class SFTable(NamedTuple):
    """Exact oracle of one goal: STOP values, V*, pi's successor, endpoints and lengths (psi = (phi(end), T))."""
    c: float
    stop: np.ndarray
    V: np.ndarray
    nxt: np.ndarray
    end: np.ndarray
    T: np.ndarray


def sf_table(space, stop, c: float, edge_mask=None) -> SFTable:
    """Solve one goal given its STOP-value vector (w.phi, or R*1[G] for H-CONJ): V*, pi, endpoints, lengths."""
    stop = np.asarray(stop, dtype=np.float64)
    V = max_plus_vi(space, stop, c, edge_mask)
    nxt = policy_table(space, V, stop, c, edge_mask)
    end, T = endpoints_lengths(nxt)
    return SFTable(float(c), stop, V, nxt, end, T)


def psi(phi: np.ndarray, tab: SFTable, idx) -> np.ndarray:
    """psi^{pi}(s) = (phi(end(s)), T(s)) as float64 rows [len(idx), F + 1]."""
    idx = np.asarray(idx, dtype=np.int64)
    return np.concatenate([phi[tab.end[idx]].astype(np.float64), tab.T[idx, None].astype(np.float64)], axis=1)


def wbar_dot(psi_rows: np.ndarray, w, c: float) -> np.ndarray:
    """w-bar . psi = w . psi_phi - c psi_T (any leading shape)."""
    return psi_rows[..., :-1] @ np.asarray(w, dtype=np.float64) - c * psi_rows[..., -1]


def gpi_scores(psi_codes: np.ndarray, w, c: float) -> np.ndarray:
    """Move scores from successor features of the successors: psi_codes[..., m, F + 1] holds psi(s', z) for the m codes
    z in C; returns max over z of w-bar.[(0, 1) + psi(s', z)] = max_z [w.psi_phi - c (1 + psi_T)], shape [...]."""
    return (wbar_dot(psi_codes, w, c) - c).max(axis=-1)


def gpi_table(stop_w: np.ndarray, c: float, tables) -> np.ndarray:
    """Exact GPI continuation values for test STOP values stop_w = w'.phi and cost c': G(s') = max over library tables
    z of [w'.phi(e_z(s')) - c' T_z(s')]; the GPI move score is -c' + G(s')."""
    G = None
    for tab in tables:
        v = stop_w[tab.end] - c * tab.T.astype(np.float64)
        G = v if G is None else np.maximum(G, v)
    if G is None:
        raise ValueError("empty policy set")
    return G


# ----------------------------------------------------------------------------- decision protocol and references

Policy = Callable[[int, np.ndarray, np.ndarray], int]


def decide(stop_score: float, move_scores) -> int:
    """Index of the chosen candidate in [STOP] + moves (action-code order): the first maximum, exact comparison."""
    best, k = stop_score, 0
    for i, v in enumerate(move_scores, 1):
        if v > best:
            best, k = v, i
    return k


def scored_policy(stop, move_scores: Callable[[int, np.ndarray, np.ndarray], np.ndarray]) -> Policy:
    """Arm wrapper: STOP scored exactly as stop[s], moves by move_scores(s, acts, dsts), tie rule of `decide`."""
    return lambda s, acts, dsts: decide(stop[s], move_scores(s, acts, dsts))


def oracle_policy(tab: SFTable) -> Policy:
    """ORACLE = pi*: greedy on V* (move score -c + V*(s'))."""
    return scored_policy(tab.stop, lambda s, a, d: tab.V[d] - tab.c)


def greedy_policy(stop, c: float) -> Policy:
    """GREEDY: one-step lookahead stop(s') - c, plus the STOP rule (the copy-state floor psi(s', z) = (phi(s'), 0))."""
    return scored_policy(stop, lambda s, a, d: stop[d] - c)


def gpi_policy(stop, c: float, G: np.ndarray) -> Policy:
    """GPI over exact tables: move score -c + G(s') with G from gpi_table."""
    return scored_policy(stop, lambda s, a, d: G[d] - c)


def stop_policy() -> Policy:
    return lambda s, acts, dsts: 0


def rand_policy(rng) -> Policy:
    """RAND: uniform over [STOP] + moves, one rng.randrange(1 + #moves) per decision."""
    return lambda s, acts, dsts: rng.randrange(len(dsts) + 1)


class Episode(NamedTuple):
    final: int
    T: int
    loop: bool
    capped: bool
    path: tuple


def run_episode(space, s0: int, policy: Policy, edge_mask=None, hmax: int = HMAX) -> Episode:
    """Closed loop from s0 under the decision protocol (see module docstring)."""
    keep = sorted_mask(space, edge_mask)
    s, T, path, seen = int(s0), 0, [int(s0)], {int(s0)}
    loop = capped = False
    while True:
        if T >= hmax:
            capped = True
            break
        acts, dsts = moves(space, s, keep)
        k = policy(s, acts, dsts)
        if not 0 <= k <= len(dsts):
            raise ValueError(f"policy returned candidate {k} of {len(dsts) + 1}")
        if k == 0:
            break
        s, T = int(dsts[k - 1]), T + 1
        path.append(s)
        if s in seen:
            loop = True
            break
        seen.add(s)
    return Episode(s, T, loop, capped, tuple(path))


def episode_return(stop, c: float, ep: Episode) -> float:
    """J = stop(s_final) - c T."""
    return float(stop[ep.final]) - c * ep.T


def nreg(v0: float, stop0: float, J: float) -> float:
    """clip((V*(s0) - J) / (V*(s0) - stop(s0)), 0, 1); 0 = optimal, 1 = no better than stopping at once."""
    gap = v0 - stop0
    if not gap > 0:
        raise ValueError("nreg needs V*(s0) > stop(s0)")
    return min(1.0, max(0.0, (v0 - J) / gap))


def evaluate(space, tab: SFTable, s0: int, policy: Policy, edge_mask=None, hmax: int = HMAX) -> dict:
    """One (arm, start) cell: nreg against the exact oracle `tab` (true STOP values and V*)."""
    ep = run_episode(space, s0, policy, edge_mask, hmax)
    J = episode_return(tab.stop, tab.c, ep)
    return dict(nreg=nreg(float(tab.V[s0]), float(tab.stop[s0]), J), J=J, T=ep.T, loop=ep.loop, capped=ep.capped,
                final=ep.final)


def q_star(tab: SFTable, s: int, dsts) -> np.ndarray:
    """Exact Q* over [STOP] + moves (action-code order): [stop(s)] + [-c + V*(s')]."""
    return np.concatenate([[tab.stop[s]], tab.V[np.asarray(dsts, dtype=np.int64)] - tab.c])


def regret1(q: np.ndarray, k: int, c: float) -> tuple[float, bool]:
    """One-step bank metrics for choice k: ((max Q* - Q*(k)) / c, top-1 optimal?)."""
    m = float(q.max())
    return (m - float(q[k])) / c, bool(q[k] == m)


# ----------------------------------------------------------------------------- renaming

def fact_perm(n: int, sigma) -> np.ndarray:
    """fp[k] = index of the fact sigma(facts(n)[k]) (block i -> sigma[i])."""
    idx, s = fact_index(n), [int(x) for x in sigma]
    return np.array([idx[(f[0],) + tuple(s[x] for x in f[1:])] for f in facts(n)], dtype=np.int64)


def rename_weight(w, sigma) -> np.ndarray:
    """w_sigma with w_sigma[fact_perm[k]] = w[k], so w_sigma.phi(sigma s) = w.phi(s)."""
    w = np.asarray(w)
    n = int(round((np.sqrt(1 + 4 * len(w)) - 1) / 2))
    out = np.empty_like(w)
    out[fact_perm(n, sigma)] = w
    return out


def _state_keys(space):
    k = getattr(space, "_soft_keys", None)
    if k is None:
        base = (space.n + 2) ** np.arange(space.n, dtype=np.int64)
        keys = space.states.astype(np.int64) @ base
        order = np.argsort(keys)
        k = space._soft_keys = (base, order, keys[order])
    return k


def state_index(space, S) -> np.ndarray:
    """Indices of compact states S [m, n] (each must exist)."""
    base, order, sk = _state_keys(space)
    keys = np.asarray(S, dtype=np.int64) @ base
    pos = np.searchsorted(sk, keys)
    if (pos >= len(sk)).any() or (sk[np.minimum(pos, len(sk) - 1)] != keys).any():
        raise KeyError("state not in space")
    return order[pos]


def state_perm(space, sigma) -> np.ndarray:
    """sp[s] = index of the renamed state sigma(s) (pos'[sigma[i]] = sigma(pos[i]); table and hand fixed)."""
    n = space.n
    sigma = np.asarray(sigma, dtype=np.int64)
    ext = np.concatenate([sigma, [n, n + 1]])
    S = space.states.astype(np.int64)
    R = np.empty_like(S)
    R[:, sigma] = ext[S]
    return state_index(space, R)
