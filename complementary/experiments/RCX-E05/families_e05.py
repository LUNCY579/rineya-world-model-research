"""RCX-E05 goal families, parent and support keys, splits, renaming, the exclusion adaptor and the seeds
(PREREG_E05_DRAFT.md §3-4 and App. A, binding; DESIGN.md rev 3 §4).

Seeds. derive(seed, tag) = int.from_bytes(sha256(f"{seed}:{tag}")[:8], "big"); `stream(seed, tag)` is
random.Random(derive(seed, tag)). Streams used here (App. A; tags not fixed by the prereg are marked *):
  20261242:pilot*               TRAIN, DEV-ID, DEV-OOD goals (one stream, drawn in that order)
  20261242:test:<attempt>        TEST goals (one stream, families in TEST_ORDER)
  20261242:pgv-subset            the 48-parent k-curve subset;   20261242:dev-kcurve*  the 8 DEV-ID k-curve parents
  20261244:starts:<split>:<key>* and 20261244:bank:<split>:<key>*   starts and one-step banks (per split, per parent)
  20261243:train:<key>*          TRAIN label rows;   20261249:labels:<key>*  per-goal labels (nested prefixes)

Keys (§4.1, M6). Facts are those of rcx.bw.soft.facts(n) (on, ontable, clear; integer block ids). A renaming sigma
maps block i -> sigma[i]; canonical forms are the lexicographic minimum of the renamed integer vector (fact order) over
all n! renamings.
  parent key  "<rtype>|<dyn>|c<k>|<i:v,...>": v = the primitive integer vector of 8*w-bar = (8w, -8c) (divided by
              the gcd of its entries; ValueError if 8*w-bar is not integral), canonical under renaming; c<k> is its
              cost entry. (w, c) and (lambda w, lambda c) share one key. H-CONJ: the vector of (8R*1_G, -8c), i.e.
              canonical G with the primitive (8R, -8c). rtype "lin"/"conj", dyn "free"/"height-3".
  support key "supp|i,j,..." = the canonical unweighted support of w (H-CONJ: of 1_G); signed families (H-NEG, H-KN)
              use "ssupp|+i,-j,..." = canonical (fact, sign). Seen = equal to the support key of a library goal.

Families (§4.2). TRAIN-like: K facts drawn one at a time (on with probability 3/4, uniform within on / ontable,
redrawn if repeated), weights iid {1, 2, 3}, c = 0.25; "TRAIN-like (K a-b)" = the TRAIN K law {1: .1, 2: .3, 3: .3,
4: .3} restricted to a..b. H-NEG/H-KN: the number of negatives (1 at K = 2, else 1 or 2) and their positions are drawn
first, negatives iid {-1, -2}. H-SPAN: plus 1-2 distinct clear facts weighted {1, 2}. H-COST: c in {0.125, 0.75}.
H-CONJ: 2-3 distinct on facts G, kept only if satisfiable (CompiledGoal consistent and goal_mask non-empty), R = 3,
c = 0.25; arms get w = (3/|G|) 1_G. H-DYN: TRAIN-like weights (K law of TRAIN) under height-limit-3 dynamics.
Generation is rejection sampling from one stream. Rejections (logged per family): unsatisfiable, quota (stratum full or
support in neither stratum), duplicate support (within a family), duplicate class (parent key used anywhere in the
draw), too few starts, exclusion (< 16 non-excluded starts), too few bank states. H-ID and DEV-ID are stratified: S =
support in supports(L) (H-ID: min(40, available) parents, where a seen class with K in {2, 3, 4} is available until it
is used or fails 50 times), U = support not in the blocked set (DEV-ID: supports(L); H-ID: supports(L) U
supports(DEV-ID)), U gets the remaining count. Supports are unique within every family except TRAIN.

Starts and banks (§3). Starts: hand-empty, V* - stop >= 1.0 (H-DYN: towers <= 3), not excluded, uniform without
replacement, 16 per parent. Bank: 64 states with V* > stop. Labels: uniform over all states. Every draw is a partial
Fisher-Yates over the pool, so a draw of k is the prefix of a draw of k' > k from the same stream (400 < 1600 < 6400).

Exclusions (§4.4 item 3). Rows (state, goal+) - starts, bank states, label rows, goal+ = the positive on/ontable facts
(H-CONJ: G) - are checked with excl.row_exclusion_reason, H-CONJ starts (start, G) with excl.exclusion_reason, on
the `ex` passed in (with_selection_sets(load_exclusions()) in the experiment). `Exclusions.prefilter` removes matches
from the pool before drawing: the excl decision depends on the state only through VL.towers_of (the invariant) unless
the signature branch can run, so it is evaluated once per (tower class, hand-empty) and applied to the class; states
whose decision needs a signature (rows: hand-empty with n in test_signature_block_counts or a selection invariant;
H-CONJ starts: every state not already excluded by the invariant) are checked one by one while drawing and skipped if
excluded. At n = 7 no row needs a signature, so rows are filtered completely before drawing. Counts are logged.

Import by path with registration (dataclasses need the module in sys.modules):
  spec = importlib.util.spec_from_file_location("families_e05", path); mod = importlib.util.module_from_spec(spec)
  sys.modules["families_e05"] = mod; spec.loader.exec_module(mod)
"""
from __future__ import annotations

import hashlib
import itertools
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rcx.bw import soft as SO  # noqa: E402
from rcx.bw import space as SP  # noqa: E402

SEEDS = {"families": 20261242, "train_labels": 20261243, "starts": 20261244, "selection": 20261245,
         "replicates": (20261246, 20261247, 20261248), "pergoal": 20261249, "boot": 20261250}
C_TRAIN, R_CONJ, GAP_MIN, H_LIMIT = 0.25, 3.0, 1.0, 3
SIZES = {"train": 64, "train_labels": 400, "train_labels_fallback": 800, "dev_id_s": 16, "dev_id_u": 16,
         "dev_ood": 12, "starts": 16, "bank": 64, "pergoal": 400, "kcurve": (400, 1600, 6400), "dev_kcurve": 8,
         "hid_s_max": 40, "n_id": (96, 152), "n_k": (56, 72), "subset_per_family": 16, "subset_per_stratum": 8,
         "H-COST": 40, "H-SPAN": 40, "H-KN": 24, "H-CONJ": 40, "H-DYN": 40, "hmax": SO.HMAX}
DEV_OOD = ("H-K", "H-NEG", "H-COST", "H-SPAN")
PRIMARY = ("H-ID", "H-K", "H-NEG")
SECONDARY = ("H-COST", "H-SPAN", "H-KN", "H-CONJ", "H-DYN")
TEST_ORDER = PRIMARY + SECONDARY
TRAIN_K = ((1, 1), (2, 3), (3, 3), (4, 3))  # K law of TRAIN, integer weights (exact)


def _restrict(dist, lo, hi):
    return tuple((k, p) for k, p in dist if lo <= k <= hi)


SPECS = {
    "TRAIN": dict(kind="pos", K=TRAIN_K),
    "ID": dict(kind="pos", K=_restrict(TRAIN_K, 2, 4)),  # = K uniform on {2, 3, 4}
    "H-K": dict(kind="pos", K=((5, 1), (6, 1))),
    "H-NEG": dict(kind="neg", K=((2, 1), (3, 1), (4, 1))),
    "H-COST": dict(kind="pos", K=_restrict(TRAIN_K, 2, 4), costs=(0.125, 0.75)),
    "H-SPAN": dict(kind="span", K=_restrict(TRAIN_K, 1, 3)),
    "H-KN": dict(kind="neg", K=((5, 1), (6, 1))),
    "H-CONJ": dict(kind="conj", K=((2, 1), (3, 1))),
    "H-DYN": dict(kind="pos", K=TRAIN_K, dyn="height-3"),
}
SIGNED = frozenset({"H-NEG", "H-KN"})


def base_family(family: str) -> str:
    """'DEV-ID'/'H-ID' -> 'ID'; 'DEV-OOD:H-K' -> 'H-K'; otherwise the name itself."""
    if family in ("DEV-ID", "H-ID"):
        return "ID"
    return family.split(":", 1)[1] if family.startswith("DEV-OOD:") else family


# ----------------------------------------------------------------------------- seeds

def derive(seed: int, tag: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")


def stream(seed: int, tag: str) -> random.Random:
    return random.Random(derive(seed, tag))


# ----------------------------------------------------------------------------- renaming and keys

@lru_cache(maxsize=None)
def perms(n: int) -> np.ndarray:
    """All n! renamings sigma (rows; block i -> sigma[i]), in itertools order."""
    return np.array(list(itertools.permutations(range(n))), dtype=np.int64)


@lru_cache(maxsize=None)
def _gather(n: int) -> np.ndarray:
    """G[r] with (v[G[r]]) = v renamed by perms(n)[r] (i.e. G[r][fact_perm[k]] = k)."""
    P, F = perms(n), n * (n + 1)
    out = np.empty((len(P), F), dtype=np.int64)
    ar = np.arange(F)
    for r, sigma in enumerate(P):
        out[r, SO.fact_perm(n, sigma)] = ar
    return out


def renamings(v, n: int) -> np.ndarray:
    """[n!, F]: row r = v renamed by perms(n)[r]."""
    return np.asarray(v)[_gather(n)]


def canon_vec(v, n: int) -> tuple:
    """Lexicographically smallest renamed copy of the integer vector v (fact order)."""
    W = renamings(np.asarray(v, dtype=np.int64), n)
    rows = np.arange(len(W))
    for col in range(W.shape[1]):
        x = W[rows, col]
        rows = rows[x == x.min()]
        if len(rows) == 1:
            break
    return tuple(int(t) for t in W[rows[0]])


def primitive_vector(w, c: float) -> tuple:
    """(8w, -8c) divided by the gcd of its entries (ValueError unless 8*w-bar is integral and c > 0)."""
    if not c > 0:
        raise ValueError("c must be > 0")
    ints = []
    for x in list(np.asarray(w, dtype=np.float64)) + [-float(c)]:
        y = 8.0 * float(x)
        if y != math.floor(y):
            raise ValueError(f"8 * w-bar is not integral ({x})")
        ints.append(int(y))
    g = 0
    for t in ints:
        g = math.gcd(g, abs(t))
    return tuple(t // g for t in ints)


def indicator(n: int, G) -> np.ndarray:
    v = np.zeros(n * (n + 1), dtype=np.int64)
    v[list(G)] = 1
    return v


def canon_key(w, c: float, n: int, rtype: str = "lin", dyn: str = "free", G=(), R: float = 0.0) -> str:
    """Parent key (M6): canonical primitive integer vector of 8*w-bar, with reward type and dynamics."""
    if rtype == "conj":
        w = R * indicator(n, G)
    elif rtype != "lin":
        raise ValueError(rtype)
    if dyn not in ("free", f"height-{H_LIMIT}"):
        raise ValueError(dyn)
    prim = primitive_vector(w, c)
    v = canon_vec(prim[:-1], n)
    return f"{rtype}|{dyn}|c{prim[-1]}|" + ",".join(f"{k}:{x}" for k, x in enumerate(v) if x)


def support_key(w, n: int, signed: bool = False) -> str:
    """Canonical support class of w: unweighted, or (fact, sign) if signed."""
    w = np.asarray(w)
    v = canon_vec(np.sign(w).astype(np.int64) if signed else (w != 0).astype(np.int64), n)
    if signed:
        return "ssupp|" + ",".join(f"{'+' if x > 0 else '-'}{k}" for k, x in enumerate(v) if x)
    return "supp|" + ",".join(str(k) for k, x in enumerate(v) if x)


def support_fingerprint(w, n: int) -> tuple:
    """Cheap renaming-invariant summary of the unweighted support (equal for equal support classes)."""
    fs = SO.facts(n)
    sup = [fs[k] for k in np.flatnonzero(np.asarray(w))]
    prof = [[0, 0, 0, 0] for _ in range(n)]
    for f in sup:
        if f[0] == "on":
            prof[f[1]][0] += 1
            prof[f[2]][1] += 1
        else:
            prof[f[1]][2 if f[0] == "ontable" else 3] += 1
    prof = [tuple(p) for p in prof]
    return tuple(sorted(prof)), tuple(sorted((prof[f[1]], prof[f[2]]) for f in sup if f[0] == "on"))


def stabilizer(w, n: int) -> np.ndarray:
    """All renamings sigma with sigma(w) = w (rows of perms(n))."""
    v = np.asarray(w)
    return perms(n)[(renamings(v, n) == v).all(axis=1)]


def mentioned_blocks(w, n: int) -> set:
    fs = SO.facts(n)
    return {b for k in np.flatnonzero(np.asarray(w)) for b in fs[k][1:]}


rename_weight = SO.rename_weight  # w_sigma with w_sigma . phi(sigma s) = w . phi(s)
rename_states = SO.state_perm  # index map s -> sigma(s)


def random_sigma(rng: random.Random, n: int) -> list:
    return rng.sample(range(n), n)


def n_support_classes(n: int, K: int, fact_ids=None) -> int:
    """Number of K-subsets of `fact_ids` (default on + ontable; must be renaming-closed) up to renaming (Burnside)."""
    fact_ids = list(range(n * n)) if fact_ids is None else list(fact_ids)
    inside, total = set(fact_ids), 0
    for sigma in perms(n):
        fp = SO.fact_perm(n, sigma)
        poly, seen = [1] + [0] * K, set()
        for k in fact_ids:
            if k in seen:
                continue
            L, j = 0, k
            while j not in seen:
                if j not in inside:
                    raise ValueError("fact set is not closed under renaming")
                seen.add(j)
                j, L = int(fp[j]), L + 1
            for d in range(K, L - 1, -1):
                poly[d] += poly[d - L]
        total += poly[K]
    return total // len(perms(n))


# ----------------------------------------------------------------------------- goals

def _pick(rng: random.Random, dist) -> int:
    u, acc = rng.randrange(sum(p for _, p in dist)), 0
    for k, p in dist:
        acc += p
        if u < acc:
            return k
    raise AssertionError


def _train_facts(rng: random.Random, n: int, K: int) -> list:
    n_on, out = n * (n - 1), []
    while len(out) < K:
        k = rng.randrange(n_on) if rng.randrange(4) < 3 else n_on + rng.randrange(n)
        if k not in out:
            out.append(k)
    return out


def draw_goal(base: str, rng: random.Random, n: int) -> dict:
    """One raw goal of family `base` (see module docstring): dict(w int array or conj linearization, c, rtype, dyn,
    G, R, K, n_neg)."""
    spec = SPECS[base]
    F, K = n * (n + 1), _pick(rng, spec["K"])
    if spec["kind"] == "conj":
        G = tuple(sorted(rng.sample(range(n * (n - 1)), K)))
        return dict(w=None, c=C_TRAIN, rtype="conj", dyn="free", G=G, R=R_CONJ, K=K, n_neg=0)
    fs = _train_facts(rng, n, K)
    n_neg = 0
    if spec["kind"] == "neg":
        n_neg = 1 if K == 2 else rng.choice((1, 2))
    neg = set(rng.sample(range(K), n_neg)) if n_neg else set()
    w = np.zeros(F, dtype=np.int64)
    for i, k in enumerate(fs):
        w[k] = rng.choice((-1, -2)) if i in neg else rng.choice((1, 2, 3))
    if spec["kind"] == "span":
        m = rng.choice((1, 2))
        for k in rng.sample(range(n * n, F), m):
            w[k] = rng.choice((1, 2))
    c = rng.choice(spec["costs"]) if "costs" in spec else C_TRAIN
    return dict(w=w, c=c, rtype="lin", dyn=spec.get("dyn", "free"), G=(), R=0.0, K=K, n_neg=n_neg)


def conj_satisfiable(space, G) -> bool:
    fs = SO.facts(space.n)
    cg = SP.CompiledGoal(list(range(space.n)), frozenset(fs[k] for k in G))
    return cg.consistent and bool(space.goal_mask(cg).any())


@dataclass
class Parent:
    """One goal (= one support-class cluster in TEST). w: integer weights over facts(n) (H-CONJ: the linearization
    (R/|G|) 1_G given to the learned arms; the true reward is R * 1[G <= s])."""
    family: str
    w: tuple
    c: float
    rtype: str = "lin"
    dyn: str = "free"
    G: tuple = ()
    R: float = 0.0
    K: int = 0
    n_neg: int = 0
    stratum: str = ""
    key: str = ""
    supp: str = ""
    supp_u: str = ""
    stab: int = 0
    starts: list = field(default_factory=list)
    bank: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def weights(self) -> np.ndarray:
        return np.asarray(self.w, dtype=np.float64)

    def to_json(self) -> dict:
        d = asdict(self)
        d["w"], d["G"] = list(self.w), list(self.G)
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Parent":
        d = dict(d)
        d["w"], d["G"] = tuple(d["w"]), tuple(d["G"])
        return cls(**d)


def make_parent(family: str, n: int, g: dict, stratum: str = "") -> Parent:
    """Parent with keys and |Stab| from a raw goal dict (draw_goal format)."""
    if g["rtype"] == "conj":
        G = tuple(sorted(g["G"]))
        w = tuple(g["R"] / len(G) if k in G else 0.0 for k in range(n * (n + 1)))
        sv = indicator(n, G)
    else:
        G, w = (), tuple(int(x) for x in g["w"])
        sv = np.asarray(w, dtype=np.int64)
    supp_u = support_key(sv, n)
    supp = support_key(sv, n, signed=True) if base_family(family) in SIGNED else supp_u
    return Parent(family=family, w=w, c=float(g["c"]), rtype=g["rtype"], dyn=g["dyn"], G=G, R=float(g["R"]),
                  K=int(g["K"]), n_neg=int(g["n_neg"]), stratum=stratum,
                  key=canon_key(w, g["c"], n, g["rtype"], g["dyn"], G, g["R"]), supp=supp, supp_u=supp_u,
                  stab=len(stabilizer(sv, n)))


def edge_mask(space, dyn: str):
    if dyn == "free":
        return None
    m = getattr(space, "_e05_hmask", None)
    if m is None:
        m = space._e05_hmask = SO.height_limit_mask(space, H_LIMIT)
    return m


def _height_ok(space):
    m = getattr(space, "_e05_hok", None)
    if m is None:
        m = space._e05_hok = SO.height_ok(space, H_LIMIT)
    return m


def parent_stop(space, phi, p: Parent) -> np.ndarray:
    """True STOP values: w.phi (lin) or R * 1[G <= s] (H-CONJ)."""
    if p.rtype == "conj":
        fs = SO.facts(space.n)
        return p.R * space.goal_mask(SO.compile_goal(space.n, [fs[k] for k in p.G])).astype(np.float64)
    return SO.stop_values(phi, p.weights())


def parent_table(space, phi, p: Parent) -> SO.SFTable:
    """Exact oracle of a parent (its dynamics' edge mask for H-DYN)."""
    return SO.sf_table(space, parent_stop(space, phi, p), p.c, edge_mask(space, p.dyn))


def goal_plus(p: Parent, n: int) -> frozenset:
    """Facts (integer ids) of the BWM instance view: positive on/ontable facts (H-CONJ: G)."""
    fs = SO.facts(n)
    if p.rtype == "conj":
        return frozenset(fs[k] for k in p.G)
    return frozenset(fs[k] for k, x in enumerate(p.w) if x > 0 and fs[k][0] in ("on", "ontable"))


def start_pool(space, tab: SO.SFTable, dyn: str = "free") -> np.ndarray:
    """Hand-empty states with V* - stop >= 1.0 (H-DYN: towers <= 3)."""
    m = SO.handempty_mask(space) & (tab.V - tab.stop >= GAP_MIN)
    if dyn != "free":
        m &= _height_ok(space)
    return np.flatnonzero(m)


def bank_pool(tab: SO.SFTable) -> np.ndarray:
    """States where STOP is not optimal."""
    return np.flatnonzero(tab.V > tab.stop)


# ----------------------------------------------------------------------------- exclusions and sampling

_DEFER = object()


class Exclusions:
    """Exact compact-state adaptor to rcx/bw/excl.py for one space and one `ex` dict (see module docstring)."""

    def __init__(self, space, ex: dict):
        from rcx.bw import bwm_import as BI
        from rcx.bw import excl as EX
        self.EX, (self.B, self.VL) = EX, BI.load()
        self.space, self.ex, self.n = space, ex, space.n
        self.objs = list(self.B.NAMES[:space.n])
        self.he = SO.handempty_mask(space)
        self.sig_rows = space.n in set(ex["raw"]["test_signature_block_counts"])
        self.sel_inv = ex.get("selection_invariants", frozenset())
        self._cid = None
        self.calls = Counter()

    def name(self, goal) -> frozenset:
        return frozenset((f[0],) + tuple(self.objs[x] for x in f[1:]) for f in goal)

    def facts_of(self, i: int) -> frozenset:
        return SP.to_facts(tuple(int(v) for v in self.space.states[i]), self.objs)

    def class_ids(self) -> np.ndarray:
        """Per state: id of (VL.towers_of(state), hand-empty), the only state data the invariant and the
        signature gate read (computed once per space with BWM's own towers_of)."""
        if self._cid is None:
            keys, cid = {}, np.empty(len(self.space), dtype=np.int64)
            for i in range(len(self.space)):
                k = (tuple(self.VL.towers_of(self.facts_of(i), self.objs)), bool(self.he[i]))
                cid[i] = keys.setdefault(k, len(keys))
            self._cid = cid
        return self._cid

    def row_reason(self, i: int, goal) -> str | None:
        """excl.row_exclusion_reason for the row (state i, goal), goal in integer ids."""
        self.calls["row"] += 1
        return self.EX.row_exclusion_reason(self.objs, self.facts_of(i), self.name(goal), self.ex)

    def instance_reason(self, i: int, goal) -> str | None:
        """excl.exclusion_reason for the instance (init = state i, goal)."""
        self.calls["instance"] += 1
        return self.EX.exclusion_reason(self.objs, self.facts_of(i), self.name(goal), self.ex)

    def _class_decision(self, i: int, goal, named, kind: str):
        s = self.facts_of(i)
        inv = self.VL.invariant(self.objs, s, named)
        if inv in self.ex["invariants"]:
            return self.row_reason(i, goal) if kind == "row" else self.instance_reason(i, goal)  # 'invariant'
        if kind == "row" and not (self.he[i] and (self.sig_rows or inv in self.sel_inv)):
            return self.row_reason(i, goal)  # no signature test runs: the decision is the class's
        return _DEFER

    def prefilter(self, idx, goal, kind: str = "row"):
        """(kept indices in pool order, per-state check for deferred classes or None, Counter of class-level
        exclusions). Each class present in the pool is decided on its first pool member."""
        idx = np.asarray(idx, dtype=np.int64)
        named, cid = self.name(goal), self.class_ids()
        c = cid[idx]
        drop, cnt, defer = np.zeros(len(idx), dtype=bool), Counter(), set()
        for k, j in zip(*(a.tolist() for a in np.unique(c, return_index=True))):
            r = self._class_decision(int(idx[j]), goal, named, kind)
            if r is _DEFER:
                defer.add(k)
            elif r:
                m = c == k
                drop |= m
                cnt[r] += int(m.sum())
        fn = self.row_reason if kind == "row" else self.instance_reason
        check = (lambda i: fn(i, goal) if int(cid[i]) in defer else None) if defer else None
        return idx[~drop], check, cnt


def draw_checked(pool, k: int, rng: random.Random, check=None) -> tuple[list, Counter]:
    """Up to k distinct pool members in draw order (partial Fisher-Yates); members for which check(i) is truthy are
    skipped and counted by reason. Prefix-consistent across k for one rng state."""
    a, out, cnt, j = [int(x) for x in pool], [], Counter(), 0
    while len(out) < k and j < len(a):
        r = j + rng.randrange(len(a) - j)
        a[j], a[r] = a[r], a[j]
        i, j = a[j], j + 1
        why = check(i) if check is not None else None
        if why:
            cnt[why] += 1
        else:
            out.append(i)
    return out, cnt


def sample_states(pool, k: int, rng: random.Random, goal, excl: Exclusions | None = None, kind: str = "row"):
    """k states from `pool`, excluded (state, goal) rows/instances removed; returns (list, Counter of exclusions)."""
    check, cnt = None, Counter()
    if excl is not None:
        pool, check, cnt = excl.prefilter(pool, goal, kind)
    out, c2 = draw_checked(pool, k, rng, check)
    cnt.update(c2)
    return out, cnt


def draw_labels(space, p: Parent, k: int, excl: Exclusions | None = None, seed: int = SEEDS["pergoal"],
                tag: str = "labels") -> tuple[list, Counter]:
    """k labelled states (uniform over all states, excluded rows removed) from stream seed:<tag>:<key>; nested in k.
    TRAIN: seed = SEEDS['train_labels'], tag = 'train'."""
    return sample_states(np.arange(len(space)), k, stream(seed, f"{tag}:{p.key}"), goal_plus(p, space.n), excl)


# ----------------------------------------------------------------------------- generation

class Registry:
    """All parents of one draw sequence (pilot, then TEST) in generation order, with the rejection log."""

    def __init__(self, n: int):
        self.n, self.parents, self.keys = n, defaultdict(list), {}
        self.rejections, self.notes = defaultdict(Counter), defaultdict(dict)

    def add(self, p: Parent) -> None:
        if p.key in self.keys:
            raise AssertionError(f"duplicate class {p.key} ({self.keys[p.key]}, {p.family})")
        self.keys[p.key] = p.family
        self.parents[p.family].append(p)

    def supports(self, family: str, signed: bool = False) -> set:
        return {p.supp if signed else p.supp_u for p in self.parents.get(family, ())}

    def all_parents(self) -> list:
        return [p for ps in self.parents.values() for p in ps]

    @classmethod
    def from_parents(cls, n: int, parents) -> "Registry":
        reg = cls(n)
        for p in parents:
            reg.add(p)
        return reg


def draw_family(reg: Registry, family: str, count: int, rng: random.Random, space, phi, *, split: str,
                excl: Exclusions | None = None, n_starts: int = SIZES["starts"], n_bank: int = SIZES["bank"],
                quota_s: int | None = None, seen=frozenset(), blocked=frozenset(),
                unique_support: bool = True, max_draws: int = 1_000_000, fail_cap: int = 50) -> list:
    """Draw `count` parents of `family` into reg (rejection sampling from rng). Stratified (quota_s given): S = support
    in `seen` (at most quota_s; closes when full or no available seen class with K in the family's range is left),
    U = support not in `blocked`, U count = count - n_S. Starts/banks from 20261244 streams per split and parent."""
    n, base = space.n, base_family(family)
    rej, out, used = reg.rejections[family], [], set()
    strat = quota_s is not None
    Ks = {k for k, _ in SPECS[base]["K"]}
    lib = reg.parents.get("TRAIN", [])
    avail = {p.supp_u for p in lib if p.supp_u in seen and p.K in Ks} if strat else set()
    seen_fp = {support_fingerprint(p.w, n) for p in lib if p.supp_u in avail}
    failed, fails, n_s = set(), Counter(), 0
    s_open = strat and quota_s > 0 and bool(avail)
    draws = guard = 0
    while True:
        if strat:
            if s_open and (n_s >= quota_s or not (avail - used - failed)):
                s_open = False
            u_cap = count - (min(quota_s, len(avail)) if s_open else n_s)
            if not s_open and len(out) - n_s >= u_cap:
                break
        elif len(out) >= count:
            break
        draws, guard = draws + 1, guard + 1
        if guard > max_draws:
            if strat and s_open:
                s_open, guard = False, 0
                reg.notes[family]["s_closed_by_draw_guard"] = True
                continue
            raise RuntimeError(f"{family}: only {len(out)} of {count} parents after {draws} draws")
        g = draw_goal(base, rng, n)
        if g["rtype"] == "conj" and not conj_satisfiable(space, g["G"]):
            rej["unsatisfiable"] += 1
            continue
        stratum = ""
        if strat:  # (ID families only: linear goals)
            u_full = len(out) - n_s >= u_cap
            if u_full and support_fingerprint(g["w"], n) not in seen_fp:
                rej["quota"] += 1  # not a seen class (cheap renaming-invariant test), and U is full
                continue
            su = support_key(g["w"], n)
            stratum = "S" if su in seen else ("" if su in blocked else "U")
            if not stratum or (stratum == "S" and (not s_open or su in failed)) or (stratum == "U" and u_full):
                rej["quota"] += 1
                continue
        p = make_parent(family, n, g, stratum)
        if unique_support and p.supp in used:
            rej["duplicate support"] += 1
            continue

        def fail(why):
            rej[why] += 1
            if stratum == "S":
                fails[p.supp_u] += 1
                if fails[p.supp_u] >= fail_cap:
                    failed.add(p.supp_u)

        if p.key in reg.keys:
            fail("duplicate class")
            continue
        if n_starts or n_bank:
            tab = parent_table(space, phi, p)
            gp = goal_plus(p, n)
            if n_starts:
                pool = start_pool(space, tab, p.dyn)
                if len(pool) < n_starts:
                    fail("too few starts")
                    continue
                p.starts, cs = sample_states(pool, n_starts, stream(SEEDS["starts"], f"starts:{split}:{p.key}"),
                                             gp, excl, "instance" if p.rtype == "conj" else "row")
                p.counts.update(start_pool=len(pool), starts_excluded=dict(cs))
                if len(p.starts) < n_starts:
                    fail("exclusion")
                    continue
            if n_bank:
                bpool = bank_pool(tab)
                p.bank, cb = sample_states(bpool, n_bank, stream(SEEDS["starts"], f"bank:{split}:{p.key}"), gp, excl)
                p.counts.update(bank_pool=len(bpool), bank_excluded=dict(cb))
                if len(p.bank) < n_bank:
                    fail("too few bank states")
                    continue
        reg.add(p)
        out.append(p)
        used.add(p.supp)
        n_s += stratum == "S"
    reg.notes[family].update(draws=draws, accepted=len(out), n_s=n_s, n_u=len(out) - n_s if strat else 0,
                             available_s=len(avail), failed_s=len(failed))
    return out


def draw_pilot(space, phi=None, excl: Exclusions | None = None, sizes: dict | None = None,
               rng: random.Random | None = None) -> Registry:
    """TRAIN (64), DEV-ID (16 S + 16 U, banks), DEV-OOD (12 each of H-K, H-NEG, H-COST, H-SPAN; starts only), from
    stream 20261242:pilot in that order."""
    sz = dict(SIZES, **(sizes or {}))
    phi = SO.get_phi(space) if phi is None else phi
    rng = stream(SEEDS["families"], "pilot") if rng is None else rng
    reg = Registry(space.n)
    kw = dict(split="pilot", excl=excl, n_starts=sz["starts"])
    draw_family(reg, "TRAIN", sz["train"], rng, space, phi, split="pilot", excl=excl, n_starts=0, n_bank=0,
                unique_support=False)
    supp_l = reg.supports("TRAIN")
    draw_family(reg, "DEV-ID", sz["dev_id_s"] + sz["dev_id_u"], rng, space, phi, n_bank=sz["bank"],
                quota_s=sz["dev_id_s"], seen=supp_l, blocked=supp_l, **kw)
    for fam in DEV_OOD:
        draw_family(reg, f"DEV-OOD:{fam}", sz["dev_ood"], rng, space, phi, n_bank=0, **kw)
    return reg


def draw_test(space, reg: Registry, attempt: int, n_id: int, n_k: int, n_neg: int | None = None, phi=None,
              excl: Exclusions | None = None, sizes: dict | None = None, rng: random.Random | None = None) -> Registry:
    """TEST families (TEST_ORDER) into a registry holding the pilot parents (class disjointness across all splits),
    from stream 20261242:test:<attempt>. H-ID: S = min(hid_s_max, available) seen supports, U = n_id - n_S unseen
    (not in supports(L) U supports(DEV-ID))."""
    sz = dict(SIZES, **(sizes or {}))
    phi = SO.get_phi(space) if phi is None else phi
    rng = stream(SEEDS["families"], f"test:{attempt}") if rng is None else rng
    kw = dict(split=f"test:{attempt}", excl=excl, n_starts=sz["starts"], n_bank=sz["bank"])
    supp_l, supp_dev = reg.supports("TRAIN"), reg.supports("DEV-ID")
    draw_family(reg, "H-ID", n_id, rng, space, phi, quota_s=sz["hid_s_max"], seen=supp_l, blocked=supp_l | supp_dev,
                **kw)
    draw_family(reg, "H-K", n_k, rng, space, phi, **kw)
    draw_family(reg, "H-NEG", n_k if n_neg is None else n_neg, rng, space, phi, **kw)
    for fam in SECONDARY:
        draw_family(reg, fam, sz[fam], rng, space, phi, **kw)
    return reg


def pgv_subset(reg: Registry, per_family: int = SIZES["subset_per_family"],
               per_stratum: int = SIZES["subset_per_stratum"], rng: random.Random | None = None) -> list:
    """The k-curve subset (keys, in draw order): per_stratum of each H-ID stratum, per_family of H-K and of H-NEG,
    from stream 20261242:pgv-subset."""
    rng = stream(SEEDS["families"], "pgv-subset") if rng is None else rng
    hid = reg.parents["H-ID"]
    out = []
    for st in ("S", "U"):
        out += rng.sample([p.key for p in hid if p.stratum == st], per_stratum)
    for fam in ("H-K", "H-NEG"):
        out += rng.sample([p.key for p in reg.parents[fam]], per_family)
    return out


def dev_kcurve_subset(reg: Registry, per_stratum: int = SIZES["dev_kcurve"] // 2,
                      rng: random.Random | None = None) -> list:
    """The 8 DEV-ID k-curve parents (keys): per_stratum per DEV-ID stratum, stream 20261242:dev-kcurve."""
    rng = stream(SEEDS["families"], "dev-kcurve") if rng is None else rng
    dev = reg.parents["DEV-ID"]
    return [k for st in ("S", "U") for k in rng.sample([p.key for p in dev if p.stratum == st], per_stratum)]


# ----------------------------------------------------------------------------- split hygiene (Q8)

def check_splits(reg: Registry) -> dict:
    """Split hygiene on the M6 key: {check: (ok, detail)} (assert_splits raises on any failure)."""
    n, res = reg.n, {}
    ps = reg.all_parents()
    bad = [p.key for p in ps if p.key != canon_key(p.w if p.rtype == "lin" else (), p.c, n, p.rtype, p.dyn, p.G, p.R)]
    res["keys_recomputed"] = (not bad, bad[:5])
    cnt = Counter(p.key for p in ps)
    dup = [k for k, v in cnt.items() if v > 1]
    res["class_disjoint"] = (not dup, dup[:5])
    dsup = {f: [s for s, v in Counter(p.supp for p in reg.parents[f]).items() if v > 1]
            for f in reg.parents if f != "TRAIN"}
    res["support_unique"] = (not any(dsup.values()), {f: v[:3] for f, v in dsup.items() if v})
    supp_l, supp_dev = reg.supports("TRAIN"), reg.supports("DEV-ID")
    for fam, block in (("DEV-ID", supp_l), ("H-ID", supp_l | supp_dev)):
        if fam in reg.parents:
            s_bad = [p.key for p in reg.parents[fam] if p.stratum == "S" and p.supp_u not in supp_l]
            u_bad = [p.key for p in reg.parents[fam] if p.stratum == "U" and p.supp_u in block]
            other = [p.key for p in reg.parents[fam] if p.stratum not in ("S", "U")]
            res[f"{fam}_strata"] = (not (s_bad or u_bad or other), dict(s_unseen=s_bad[:3], u_seen=u_bad[:3],
                                                                      no_stratum=other[:3]))
    return res


def assert_splits(reg: Registry) -> dict:
    res = check_splits(reg)
    bad = {k: v[1] for k, v in res.items() if not v[0]}
    if bad:
        raise AssertionError(f"split hygiene failed: {bad}")
    return res


def verify_parent(space, phi, p: Parent, excl: Exclusions | None = None, labels=()) -> dict:
    """Re-check a drawn parent against the oracle: starts distinct, hand-empty, gap >= 1 (H-DYN: towers <= 3); bank
    states distinct with V* > stop; and (with excl) no start, bank state or label row excluded now."""
    tab = parent_table(space, phi, p)
    st, bk = np.asarray(p.starts, dtype=np.int64), np.asarray(p.bank, dtype=np.int64)
    ok_st = set(start_pool(space, tab, p.dyn).tolist())
    out = dict(starts_ok=len(set(p.starts)) == len(p.starts) and all(int(s) in ok_st for s in st),
               bank_ok=len(set(p.bank)) == len(p.bank) and bool((tab.V[bk] > tab.stop[bk]).all()))
    if excl is not None:
        gp = goal_plus(p, space.n)
        sf = excl.instance_reason if p.rtype == "conj" else excl.row_reason
        out["excluded_starts"] = sum(bool(sf(int(i), gp)) for i in st)
        out["excluded_bank"] = sum(bool(excl.row_reason(int(i), gp)) for i in bk)
        out["excluded_labels"] = sum(bool(excl.row_reason(int(i), gp)) for i in labels)
    return out
