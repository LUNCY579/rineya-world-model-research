"""RCX-E05 ENUM± symbolic soft-goal planners and the non-learned references (PREREG_E05_DRAFT.md §3, §5.2, binding;
DESIGN.md rev 3 §5.3). Pure numpy over the enumerated spaces of rcx/bw/space.py; torch only for GNN-ENUM± (lazy).

Facts use integer block ids: ("on", a, b), ("ontable", a), ("clear", a), ("holding", a); the first three index
rcx.bw.soft.facts(n) (the feature facts). A goal is linear (w, c) over those facts, or conjunctive (H-CONJ: reward
R * 1[G <= s]).

Falsifier menus (L1). M(f) = every single fact that implies not-f:
  M(on(a,b))     = {ontable(a), holding(a), clear(b), holding(b)} + {on(a,x): x not in {a,b}} + {on(x,b): x not in {a,b}}
                   (14 facts at n = 7; version "rev2" keeps only the first three, rev 2's menu)
  M(ontable(a))  = {holding(a)} + {on(a,x): x != a}
  M(clear(a))    = {holding(a)} + {on(x,a): x != a}
Each menu covers not-f (a state where f is false satisfies some menu fact; tested).

Targets (ENUM±, F5). For w' with positive facts P and negative facts N: G = G+ u R, G+ <= P, and for each f in N, R
takes "accept" or one falsifier from M(f); G is kept iff CompiledGoal(G).consistent and goal_mask(G).any(). The raw
count 2^|P| * prod_f (1 + |M(f)|) is logged (n_raw). Utility (persistence rule):
  U(G; s') = sum_{f in G+} w_f + sum_{f in N} w_f * 1[f in phi(s') and G u {f} satisfiable],
  score(s') = max over G of [U(G; s') - c' (1 + h(s', G))],   STOP scored exactly as stop(s) (the true reward).
U depends on (G+, G) only through G+, and (G n P, R) is a valid choice whenever (G+, R) is, with G+ <= G n P; so the
targets are stored once per distinct set G with G+ = G n P, which gives exactly the max over the literal enumeration.
Version "pos" is rev 1's positive-only ENUM (N ignored: no menus, no penalties); for goals without negative weights all
versions coincide. H-CONJ: the single conjunctive target G with U = R (plus the empty target, U = 0), so ENUM-H*± is
ORACLE there (tested).

Heuristics h(s', G):
  mis      ENUM-MIS± as specified (B1): max(misplaced_compact(s', G \\ H)[1], h_hold(s', G)), H = the holding facts of
           G; h_hold = 0 if H is empty or s' holds the block H requires, 1 if the hand is empty, 2 if s' holds another
           block. Admissible for targets without `clear` facts (holding targets included; tested). NOT admissible for
           targets with a `clear` fact: misplaced_compact charges 2 moves for a block x resting on a block that must be
           clear, but x can be lifted as the plan's last move and kept in the hand (1 move); it overestimates h* by at
           most 1 (exhaustive at n = 4; e.g. n = 5, tower 4-0-2-1, G = {clear(4), on(3,2)}: bound 8, h* 7). Such
           targets come from the on-menus (clear(b)) and H-SPAN.
  mis_adm  the same with that correction: max(misplaced_compact(s', G \\ H)[1] - clear_saving, 0) vs h_hold
           (admissible on every target; tested); not the prereg's formula, offered for a coordinator decision
  rev2     rev 2's bound misplaced_compact(s', G)[1] (0 for any G with a holding fact; the B1 defect; control only)
  mis_noh  misplaced_compact(s', G \\ H)[1] without h_hold (the review's minimal fix; control only)
  hstar    ENUM-H*±: exact h*(s', G), one backward_bfs per target (lazily; masked dynamics by max-plus distances)
  gnn      GNN-ENUM±: the frozen R-04 RelGNN on encode.graph(s', G) (all targets of a state in one fixed-composition
           predict call, so values do not depend on call order); 0 where G <= s' (the exact goal test precedes the
           learned h, as in rcx/bw/search.py); raw outputs otherwise (may be < 0, so no pruning)
For h >= 0 (mis, mis_adm, rev2, mis_noh, hstar) the max over targets is computed exactly by branch and bound: targets in
decreasing U order, stopped once U - c' <= best (h >= 0 bounds the rest). Scores are exact dyadic float64.

Diagnostics: per arm, decisions, decisions with >= 2 legal moves ("multi") and those whose move scores are all equal
("all_tied"); heuristic calls; target counts (n_raw, n_kept).

References (all score STOP exactly as `stop`, the parent's true terminal reward, tab.stop; for H-CONJ that is
R * 1[G <= s]): GREEDY, STOP, RAND (uniform over STOP + moves; per-cell stream 20261244:rand:<key>:<s0>), ORACLE,
EXACT-GPI(L) (soft.gpi_table over exact library tables), BEST-SINGLE(L) (the library policy with the best exact value
w'.phi(e_z(s0)) - c' T_z(s0) at s0, first max; followed as tabled; forced STOP if its move is illegal under the parent's
edge mask).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

GNN_FILE = ROOT / "runs" / "RCX-E07" / "main" / "models" / "gnn_s20261276.pt"
GNN_SHA8 = "9cc417a3"  # prereg §2.1 pin (prefix; the full sha is recorded at T2)
NAMES = "abcdefghijkl"
RAND_SEED = 20261244
ARMS = {"ENUM-MIS": ("mis", "full"), "ENUM-HSTAR": ("hstar", "full"), "GNN-ENUM": ("gnn", "full"),
        "ENUM-MIS-rev1": ("mis", "pos"), "ENUM-MIS-rev2": ("rev2", "rev2")}  # ENUM-MIS± etc. (ASCII names)
HEURISTICS = ("mis", "mis_adm", "rev2", "mis_noh", "hstar", "gnn")


def derive(seed: int, tag: str) -> int:
    """App. A: int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big") (= families_e05.derive)."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


# ----------------------------------------------------------------------------- menus and targets

def menu(f: tuple, n: int, version: str = "full") -> tuple:
    """Falsifier menu M(f) of a feature fact f (see module doc)."""
    if f[0] == "on":
        a, b = f[1], f[2]
        m = [("ontable", a), ("holding", a), ("clear", b)]
        if version != "rev2":
            m += [("holding", b)] + [("on", a, x) for x in range(n) if x not in (a, b)]
            m += [("on", x, b) for x in range(n) if x not in (a, b)]
        return tuple(m)
    a = f[1]
    if f[0] == "ontable":
        return (("holding", a),) + tuple(("on", a, x) for x in range(n) if x != a)
    if f[0] == "clear":
        return (("holding", a),) + tuple(("on", x, a) for x in range(n) if x != a)
    raise ValueError(f"no menu for {f}")


def holds(f: tuple, pos) -> bool:
    """f is true in compact state pos (feature facts and holding)."""
    n, k = len(pos), f[0]
    if k == "on":
        return pos[f[1]] == f[2]
    if k == "ontable":
        return pos[f[1]] == n
    if k == "holding":
        return pos[f[1]] == n + 1
    if k == "clear":
        return pos[f[1]] != n + 1 and f[1] not in pos
    raise ValueError(f)


def compile_goal(n: int, G) -> SP.CompiledGoal:
    return SP.CompiledGoal(list(range(n)), frozenset(G))


class Targets:
    """Distinct satisfiable targets of one goal: goals[t] (frozensets), U0[t] = sum of w over G n P, the negative facts
    neg (feature ids) with weights wneg, sat[t, j] = 1.0 iff G u {neg_j} is satisfiable, the required held block
    hold[t] (-1 if none), compiled goals full[t] (G) and rest[t] (G \\ H), n_raw (before pruning and dedup)."""

    def __init__(self, n, goals, U0, neg, wneg, sat, n_raw):
        self.n, self.goals, self.n_raw = n, list(goals), int(n_raw)
        self.U0 = np.asarray(U0, dtype=np.float64)
        self.neg, self.wneg = list(neg), np.asarray(wneg, dtype=np.float64)
        self.sat = np.asarray(sat, dtype=np.float64).reshape(len(self.goals), len(self.neg))
        self.hold = np.array([next((f[1] for f in G if f[0] == "holding"), -1) for G in self.goals], dtype=np.int64)
        self.full = [compile_goal(n, G) for G in self.goals]
        self.rest = [compile_goal(n, [f for f in G if f[0] != "holding"]) for G in self.goals]

    def __len__(self):
        return len(self.goals)

    @property
    def n_kept(self) -> int:
        return len(self.goals)


class Satisfiable:
    """Exact satisfiability cache: CompiledGoal(G).consistent and space.goal_mask(G).any()."""

    def __init__(self, space):
        self.space, self.cache = space, {}

    def __call__(self, G) -> bool:
        G = frozenset(G)
        r = self.cache.get(G)
        if r is None:
            cg = compile_goal(self.space.n, G)
            r = self.cache[G] = bool(cg.consistent and self.space.goal_mask(cg).any())
        return r


def linear_targets(space, w, version: str = "full", sat: Satisfiable | None = None) -> Targets:
    """ENUM± targets of a linear goal w over soft.facts(n) (version 'full', 'rev2' menus, or 'pos' = rev 1)."""
    n, fs = space.n, SO.facts(space.n)
    w = np.asarray(w, dtype=np.float64)
    sat = Satisfiable(space) if sat is None else sat
    P = [fs[k] for k in np.flatnonzero(w > 0)]
    N = [] if version == "pos" else [int(k) for k in np.flatnonzero(w < 0)]
    wP = {fs[k]: float(w[k]) for k in np.flatnonzero(w > 0)}
    choices = [(None,) + menu(fs[k], n, version) for k in N]
    goals, seen, n_raw = [], set(), 0
    for bits in itertools.product((0, 1), repeat=len(P)):
        gp = [f for f, b in zip(P, bits) if b]
        for R in itertools.product(*choices):
            n_raw += 1
            G = frozenset(gp + [f for f in R if f is not None])
            if G in seen:
                continue
            seen.add(G)
            if sat(G):
                goals.append(G)
    U0 = [sum(wP.get(f, 0.0) for f in G) for G in goals]
    S = [[1.0 if sat(G | {fs[k]}) else 0.0 for k in N] for G in goals]
    return Targets(n, goals, U0, N, [w[k] for k in N], S, n_raw)


def conj_targets(space, G, R: float) -> Targets:
    """H-CONJ: the empty target (U = 0) and the conjunctive target G (U = R); G must be satisfiable."""
    G = frozenset(G)
    if not Satisfiable(space)(G):
        raise ValueError(f"unsatisfiable H-CONJ goal {sorted(G)}")
    return Targets(space.n, [frozenset(), G], [0.0, float(R)], [], [], np.zeros((2, 0)), 2)


def parent_targets(space, p, version: str = "full") -> Targets:
    """Targets of a families_e05.Parent-like object (rtype, w, G as feature ids, R)."""
    if p.rtype == "conj":
        fs = SO.facts(space.n)
        return conj_targets(space, [fs[k] for k in p.G], p.R)
    return linear_targets(space, p.w, version)


# ----------------------------------------------------------------------------- heuristics h(s', G)

def h_hold(pos, a: int) -> int:
    """0 if no holding target (a < 0) or s' holds a; 1 if the hand is empty; 2 if s' holds another block."""
    n = len(pos)
    if a < 0 or pos[a] == n + 1:
        return 0
    return 2 if (n + 1) in pos else 1


def clear_saving(pos, g: SP.CompiledGoal, a: int = -1) -> int:
    """1 if a block x can end in the hand as the plan's last move although misplaced_compact charges it 2 moves: x sits
    on y with clear(y) required, x's own position is free, clear(x) is not required, handempty is not required, and
    (if a holding target requires block a) x == a; else 0. misplaced_compact(s, g)[1] - clear_saving <= h*(s, g)."""
    n = len(pos)
    if g.handempty:
        return 0
    for x in range(n):
        y = pos[x]
        if y < n and g.clear[y] and g.below[x] == -1 and not g.clear[x] and (a < 0 or x == a):
            return 1
    return 0


class _Heuristic:
    prune = True  # h >= 0 everywhere (enables exact branch and bound)

    def __init__(self, space, T: Targets):
        self.space, self.T, self.calls, self._pos = space, T, 0, {}

    def pos(self, s: int) -> tuple:
        p = self._pos.get(s)
        if p is None:
            p = self._pos[s] = tuple(int(v) for v in self.space.states[s])
        return p

    def value(self, s: int, t: int) -> float:
        raise NotImplementedError

    def vector(self, s: int) -> np.ndarray:
        return np.array([self.value(s, t) for t in range(len(self.T))], dtype=np.float64)


class HMis(_Heuristic):
    """misplaced-based bounds: variant 'mis' (ENUM-MIS± as specified, B1), 'mis_adm' (the same with the clear-goal
    correction), 'rev2' or 'mis_noh' (controls)."""

    def __init__(self, space, T: Targets, variant: str = "mis"):
        super().__init__(space, T)
        if variant not in ("mis", "mis_adm", "rev2", "mis_noh"):
            raise ValueError(variant)
        self.variant = variant

    def value(self, s: int, t: int) -> float:
        self.calls += 1
        pos, T = self.pos(s), self.T
        if self.variant == "rev2":
            return float(SP.misplaced_compact(pos, T.full[t])[1])
        a = int(T.hold[t])
        lb = SP.misplaced_compact(pos, T.rest[t])[1]
        if self.variant == "mis_noh":
            return float(lb)
        if self.variant == "mis_adm":
            lb = max(0, lb - clear_saving(pos, T.rest[t], a))
        return float(max(lb, h_hold(pos, a)))


class HStar(_Heuristic):
    """Exact h*(s', G) per target: space.backward_bfs (free dynamics) or max-plus distances under an edge mask."""

    def __init__(self, space, T: Targets, edge_mask=None):
        super().__init__(space, T)
        self.edge_mask, self.dist, self.bfs = edge_mask, {}, 0

    def table(self, t: int) -> np.ndarray:
        d = self.dist.get(t)
        if d is None:
            m = self.space.goal_mask(self.T.full[t])
            if self.edge_mask is None:
                b = self.space.backward_bfs(m)
                d = np.where(b >= 0, b.astype(np.float64), np.inf)
            else:
                d = -SO.max_plus_vi(self.space, np.where(m, 0.0, -np.inf), 1.0, self.edge_mask)
            self.dist[t], self.bfs = d, self.bfs + 1
        return d

    def value(self, s: int, t: int) -> float:
        self.calls += 1
        return float(self.table(t)[s])


class HGnn(_Heuristic):
    """GNN-ENUM±: h(s', G) = model value on encode.graph (0 where G <= s'); one predict per state over all targets."""
    prune = False

    def __init__(self, space, T: Targets, model, batch_size: int = 512):
        super().__init__(space, T)
        from rcx.bw import encode as EN
        from rcx.bw import models as M
        self.EN, self.M, self.model, self.bs = EN, M, model, batch_size
        self.derived = model.cfg.n_rel > len(EN.EDGE_TYPES)
        self.objs = list(NAMES[:space.n])
        self.goals = [self._name(G) for G in T.goals]
        self._vec, self.forwards = {}, 0

    def _name(self, facts) -> frozenset:
        return frozenset((f[0],) + tuple(self.objs[x] for x in f[1:]) for f in facts)

    def vector(self, s: int) -> np.ndarray:
        v = self._vec.get(s)
        if v is None:
            pos = self.pos(s)
            st = SP.to_facts(pos, self.objs)
            graphs = [self.EN.graph(st, g, self.objs, derived=self.derived) for g in self.goals]
            v = self.M.predict(self.model, graphs, batch_size=self.bs).astype(np.float64)
            met = np.array([cg.met(pos) for cg in self.T.full])
            v[met] = 0.0
            self._vec[s] = v
            self.calls += len(graphs)
            self.forwards += 1
        return v

    def value(self, s: int, t: int) -> float:
        return float(self.vector(s)[t])


def make_heuristic(kind: str, space, T: Targets, edge_mask=None, model=None) -> _Heuristic:
    if kind in ("mis", "mis_adm", "rev2", "mis_noh"):
        return HMis(space, T, kind)
    if kind == "hstar":
        return HStar(space, T, edge_mask)
    if kind == "gnn":
        if model is None:
            raise ValueError("GNN-ENUM needs a model (load_gnn)")
        return HGnn(space, T, model)
    raise ValueError(kind)


# ----------------------------------------------------------------------------- the ENUM± arm

class EnumArm:
    """One ENUM± arm on one parent: move score(s') = max over targets of [U(G; s') - c (1 + h(s', G))]; STOP is
    stop[s] exactly (soft.decide tie rule). Scores are cached per successor (they do not depend on the current state)."""

    def __init__(self, space, phi, T: Targets, h: _Heuristic, stop, c: float, name: str = "ENUM"):
        self.space, self.T, self.h, self.stop, self.c, self.name = space, T, h, stop, float(c), name
        self.phi_neg = phi[:, T.neg].astype(np.float64) if T.neg else None
        self.stats, self._score = Counter(), {}

    def U(self, s: int) -> np.ndarray:
        """U(G; s') for every target (persistence rule)."""
        if self.phi_neg is None:
            return self.T.U0
        return self.T.U0 + self.T.sat @ (self.T.wneg * self.phi_neg[s])

    def score(self, s: int) -> float:
        v = self._score.get(s)
        if v is not None:
            return v
        U, c = self.U(s), self.c
        if self.h.prune:
            best = -math.inf
            for t in np.argsort(-U, kind="stable"):
                if U[t] - c <= best:
                    break
                best = max(best, U[t] - c * (1.0 + self.h.value(s, int(t))))
        else:
            best = float((U - c * (1.0 + self.h.vector(s))).max())
        self._score[s] = best = float(best)
        return best

    def move_scores(self, s: int, acts, dsts) -> np.ndarray:
        sc = np.array([self.score(int(d)) for d in dsts], dtype=np.float64)
        self.stats["decisions"] += 1
        if len(sc) >= 2:
            self.stats["multi"] += 1
            self.stats["all_tied"] += int((sc == sc[0]).all())
        return sc

    def policy(self) -> SO.Policy:
        return SO.scored_policy(self.stop, self.move_scores)

    def report(self) -> dict:
        st = dict(self.stats)
        return dict(arm=self.name, n_raw=self.T.n_raw, n_kept=self.T.n_kept, h_calls=self.h.calls,
                    tie_share=tie_share(st), **st)


def tie_share(stats) -> float | None:
    """Share of decisions with >= 2 legal moves whose move scores are all equal (None if there were none)."""
    m = stats.get("multi", 0)
    return stats.get("all_tied", 0) / m if m else None


def enum_arm(space, phi, p, tab: SO.SFTable, kind: str = "mis", version: str = "full", edge_mask=None, model=None,
             name: str | None = None) -> EnumArm:
    """ENUM± arm for a parent (STOP = tab.stop, cost p.c); kind in HEURISTICS, version 'full' / 'rev2' / 'pos'."""
    T = parent_targets(space, p, version)
    return EnumArm(space, phi, T, make_heuristic(kind, space, T, edge_mask, model), tab.stop, p.c,
                   name or f"ENUM[{kind},{version}]")


def named_arm(arm: str, space, phi, p, tab: SO.SFTable, edge_mask=None, model=None) -> EnumArm:
    """One of ARMS by name (ENUM-MIS, ENUM-HSTAR, GNN-ENUM, ENUM-MIS-rev1, ENUM-MIS-rev2)."""
    kind, version = ARMS[arm]
    return enum_arm(space, phi, p, tab, kind, version, edge_mask, model, arm)


def load_gnn(path=GNN_FILE, expect_sha8: str | None = GNN_SHA8):
    """The frozen R-04 RelGNN (inference only): (model in eval mode, provenance). Architecture from the sidecar
    <stem>.json 'config' (written by run_e07.phase_train), else RelGNNConfig(); strict load; sha256 prefix checked."""
    import torch

    from rcx.bw import models as M
    path = Path(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expect_sha8 and not sha.startswith(expect_sha8):
        raise ValueError(f"{path.name}: sha256 {sha[:8]} != pinned {expect_sha8}")
    side = path.with_suffix(".json")
    cfg = dict(json.loads(side.read_text(encoding="utf-8")).get("config") or {}) if side.exists() else {}
    names = set(M.RelGNNConfig.__dataclass_fields__)
    model = M.RelGNN(M.RelGNNConfig(**{k: v for k, v in cfg.items() if k in names}))
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    if "n_params" in cfg and int(cfg["n_params"]) != M.n_params(model):
        raise ValueError("n_params mismatch")
    return model, dict(file=str(path), sha256=sha, config=cfg)


# ----------------------------------------------------------------------------- references

def greedy(stop, c: float) -> SO.Policy:
    return SO.greedy_policy(stop, c)


def stop_arm() -> SO.Policy:
    return SO.stop_policy()


def oracle(tab: SO.SFTable) -> SO.Policy:
    return SO.oracle_policy(tab)


def rand_rng(key: str, s0: int) -> random.Random:
    """RAND's per-cell stream 20261244:rand:<parent key>:<start>."""
    return random.Random(derive(RAND_SEED, f"rand:{key}:{int(s0)}"))


def rand(key: str, s0: int) -> SO.Policy:
    return SO.rand_policy(rand_rng(key, s0))


def exact_gpi(stop, c: float, lib) -> SO.Policy:
    """EXACT-GPI(L): GPI over the exact library tables, evaluated with the test reward `stop` and cost c."""
    return SO.gpi_policy(stop, c, SO.gpi_table(np.asarray(stop, dtype=np.float64), c, lib))


def best_single(stop, c: float, lib, s0: int) -> tuple:
    """BEST-SINGLE(L): (policy following the library table with the best exact value at s0, its index z)."""
    stop = np.asarray(stop, dtype=np.float64)
    vals = [float(stop[t.end[s0]]) - c * float(t.T[s0]) for t in lib]
    z = int(np.argmax(vals))
    nxt = lib[z].nxt

    def pol(s, acts, dsts):
        t = int(nxt[s])
        if t == s:
            return 0
        hit = np.flatnonzero(np.asarray(dsts) == t)
        return int(hit[0]) + 1 if len(hit) else 0

    return pol, z
