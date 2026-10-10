"""T2b-register screen, revision 6: the EXTENDED SHORTCUT FAMILY H and the design of the adversarial identification
stratum cond_adv. Torch-free (numpy only). Imported by t2br_make_data (the cond_adv draw and its target check) and by
t2br_analysis (the enumeration over H, the report-only shortcut-agreement analysis).

Every member of H is a predicate 'add iff P(x, t)' of x = reg[other] and t = threshold alone (the env's branch is
'x > t', STRICT). H is evaluated on the DOMAIN of this kit's register rows: x in 0..96 (registers are integers modulo
97) and t in -40..136 (the in-range thresholds 0..96 and cond_oor_thr's out-of-range -40..-1 / 97..136).

H = the ENUMERATED family (revisions 4-5; 3,483 predicates, unchanged; enumerated_family)
    + the EXTENSION of revision 6 (36 predicates; extension):
  lexicographic   str(x) OP str(t) with sign handling (sign first: a non-negative number is above a negative one; two
                  non-negative numbers compare their digit strings; two negative ones the reversed comparison of their
                  magnitudes' strings; on the domain this is Python's str(x) OP str(t), because '-' sorts before every
                  digit) and without it (str(|x|) OP str(|t|)); OP in {>, >=}                                       (4)
  digit features  f(|x|) OP f(|t|) for f = the units digit, the tens digit ((|n| // 10) % 10) and the digit sum of the
                  magnitude, each plain, sign-aware (sign first, as above), length-aware (a magnitude with more digits
                  is larger; equal lengths compare f) and sign+length-aware (sign, then length, then f; both negative:
                  every step reversed); OP in {>, >=}                                                                (24)
  magnitude       |x| OP |t|, OP in {>, >=}                                                                         (2)
  exceptions      the comparison with ONE isolated distance flipped: 'x > t' XOR (x - t == k) for k in -4..-2, +2..+4
                  (k < 0: 'x > t or x == t + k'; k > 0: 'x > t and x != t + k')                                    (6)
  The branch-blind constants TRUE / FALSE are members of the enumerated family (its atoms), as are the floor-semantics
  units / tens comparisons (Python // and %: tens(-3) = -1, units(-3) = 7).
Members equal to the true rule on the whole domain (e.g. the digit-wise lexicographic comparison of the enumerated
family) compute the comparison and are not shortcuts.

cond_adv (100 rows, 50 ok-true / 50 ok-false) is drawn by seeded rejection sampling (t2br_make_data.build_adv): a
proposal is a uniformly random valid conditional row of the domain; it is accepted iff its (x, t) cell lies in one of
the 15 design types ADV_TYPES with quota left (a new cell of the type while the type has unused cells, then the
least-used ones), neither branch wraps and the row is fresh against the name-skipping harvest registry. Each type is a
SIGNATURE CLASS of the domain (all cells on which exactly the same members of H are wrong; 13 classes), one class split
by digit lengths (1 vs 2, 1 vs 3 and 2 vs 3 digits) so that every cross-length kind is drawn; every member's number of
wrong rows on cond_adv is therefore fixed by the quotas, and it is checked after the draw (adv_target_report).

What the operator asked (DECISIONS D-42 (2)): every member of H wrong on >= 40 of the 100 rows. That is impossible on
ANY 100-row branch-balanced stratum (proofs in adv_target_report / DESIGN.md section 15):
  * members with pairwise-disjoint error sets share the 100 rows: five such members of H exist (DISJOINT_FIVE), so some
    member of H is wrong on at most 20 rows of any 100;
  * three members that pass revision 5's G1 against a weak plain arm have pairwise-disjoint error sets
    (DISJOINT_SURVIVOR_TRIPLE), so at least one of them is wrong on at most 33 rows;
  * the isolated exceptions at d = -4, -3, -2 are wrong only on ok-false rows at their own distance; failing the 0.80
    predicate bar needs >= 21 such rows each, and 3 x 21 > 50 ok-false rows, so one of them always passes.
The design therefore meets, in this order (an integer program over the signature classes found it; the bounds above
certify that (1) and (2) are the maxima):
  (1) every member of H that passes revision 5's G1 against the weak plain arm (the most permissive baseline), is not
      the comparison itself and is not an isolated distance exception is wrong on >= ADV_MIN_BROAD = 33 rows (predicate
      accuracy <= 0.67);
  (2) the surviving distance exceptions at d = -3, -2, +3, +4 are wrong on >= ADV_MIN_EXCEPTION = 21 rows each
      (accuracy <= 0.79 < 0.80); the one at d = -4 (ADV_EXCEPTION_NOT_CAUGHT) cannot also be caught and passes G1;
  (3) the fewest rows on any one (x, t) cell under (1)-(2): 5 (at least 25 ok-true rows must be sign-crossing cells at
      d = +3 / +4, of which only five exist with x <= |t|), and then the most distinct cells: 67 (also the most without
      (3)'s first part).
How many members of H reach the operator's 40 is reported (adv_target_report), not optimized further.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import itertools  # noqa: E402
from collections import OrderedDict  # noqa: E402
from fractions import Fraction  # noqa: E402

import numpy as np  # noqa: E402

DOMAIN = {"other": (0, 96), "threshold": (-40, 136)}      # every (other, threshold) of this kit's register rows
_CMP = {">": np.greater, ">=": np.greater_equal, "<": np.less, "<=": np.less_equal, "==": np.equal, "!=": np.not_equal}
OBO_SHIFTS = tuple(range(-3, 4))                    # off-by-one family bases 'x > t + c'
POINTS = (("t-1", -1), ("t", 0), ("t+1", 1))       # the modifier points x == t - 1 / t / t + 1
EXCEPTION_DISTANCES = (-4, -3, -2, 2, 3, 4)        # the isolated single-distance exceptions of the extension


def _modifier_name(states) -> str:
    return ", ".join(("or x==" if s == 1 else "and x!=") + p for (p, _), s in zip(POINTS, states) if s)


def enumerated_family(x, t) -> dict:
    """Every predicate of the enumerated family (revisions 4-5), as a boolean array over the (other, threshold) arrays
    x, t ('add iff ...'). Digits use floor semantics (numpy // and % equal Python's, also for negative thresholds). Family:
      atoms        TRUE, FALSE, tens(x) OP tens(t), units(x) OP units(t)         OP in > >= < <= == !=   (14)
      bases        every atom; tens(x) OP1 tens(t) AND/OR units(x) OP2 units(t) (72); the lexicographic form
                   tens(x) OP1 tens(t) or (tens(x) == tens(t) and units(x) OP2 units(t)) (36); the off-by-one family
                   x > t + c for c = -3..+3 (7: x>t-2 = x >= t-1, x>t-1 = x >= t, x>t, x>t+1, and the off-by-two /
                   -three shifts)
      modifiers    at each of x == t-1, x == t, x == t+1 independently: unchanged / forced true ('or x==p') / forced
                   false ('and x!=p'): 3^3 = 27 (revision 4 had the 9 on t and t+1 only)
    3,483 predicates in all (many coincide). With the base x>t the modifiers give the env's rule and every off-by-one
    variant (each non-empty flip set of x - t in {-1, 0, +1})."""
    x, t = np.asarray(x), np.asarray(t)
    tx, tt, ux, ut = x // 10, t // 10, x % 10, t % 10
    atoms = {"TRUE": np.ones(x.shape, bool), "FALSE": np.zeros(x.shape, bool)}
    for o, f in _CMP.items():
        atoms[f"tens(x){o}tens(t)"] = f(tx, tt)
    for o, f in _CMP.items():
        atoms[f"units(x){o}units(t)"] = f(ux, ut)
    bases = dict(atoms)
    for o1, f1 in _CMP.items():
        for o2, f2 in _CMP.items():
            bases[f"tens(x){o1}tens(t) and units(x){o2}units(t)"] = f1(tx, tt) & f2(ux, ut)
            bases[f"tens(x){o1}tens(t) or units(x){o2}units(t)"] = f1(tx, tt) | f2(ux, ut)
    for o1, f1 in _CMP.items():
        for o2, f2 in _CMP.items():
            bases[f"tens(x){o1}tens(t) or (tens(x)==tens(t) and units(x){o2}units(t))"] = f1(tx, tt) | ((tx == tt) & f2(ux, ut))
    for c in OBO_SHIFTS:
        bases["x>t" + (f"{c:+d}" if c else "")] = x > t + c
    eq = {p: x == t + k for p, k in POINTS}
    out = {}
    for states in itertools.product((0, 1, 2), repeat=3):          # 0 unchanged, 1 forced true, 2 forced false
        mn = _modifier_name(states)
        for bn, b in bases.items():
            v = b.copy()
            for (p, _), s in zip(POINTS, states):
                if s == 1:
                    v = v | eq[p]
                elif s == 2:
                    v = v & ~eq[p]
            out[bn + (" | " + mn if mn else "")] = v
    return out


def _ndigits(a):
    return np.where(a >= 100, 3, np.where(a >= 10, 2, 1))


def _digit_sum(a):
    return a // 100 + (a // 10) % 10 + a % 10


def _strs(a) -> np.ndarray:
    a = np.asarray(a)
    return np.array([str(int(v)) for v in a.ravel()], dtype=object).reshape(a.shape)


def extension(x, t) -> dict:
    """Revision 6's 36 members (module docstring), as boolean arrays over x, t. Magnitudes |n| for every digit feature
    (domain magnitudes have 1-3 digits)."""
    x, t = np.asarray(x), np.asarray(t)
    ax, at = np.abs(x), np.abs(t)
    sx, st = x < 0, t < 0
    both_neg, sign_differs = sx & st, sx != st
    lx, lt = _ndigits(ax), _ndigits(at)
    out = OrderedDict()
    sxs, sts = _strs(ax), _strs(at)
    for o in (">", ">="):
        fwd = (sxs > sts) if o == ">" else (sxs >= sts)
        rev = (sts > sxs) if o == ">" else (sts >= sxs)
        out[f"lex signed: str(x){o}str(t)"] = np.where(sign_differs, ~sx, np.where(both_neg, rev, fwd)).astype(bool)
        out[f"lex unsigned: str(|x|){o}str(|t|)"] = fwd.astype(bool)
    feats = OrderedDict([("units", (ax % 10, at % 10)), ("tens", ((ax // 10) % 10, (at // 10) % 10)),
                         ("digitsum", (_digit_sum(ax), _digit_sum(at)))])
    for fn, (fx, ft) in feats.items():
        for o in (">", ">="):
            f = _CMP[o]
            plain = f(fx, ft)
            same = np.where(both_neg, f(ft, fx), plain)
            out[f"{fn} plain: {fn}(|x|){o}{fn}(|t|)"] = plain
            out[f"{fn} sign-aware: {o}"] = np.where(sign_differs, ~sx, same)
            out[f"{fn} length-aware: {o}"] = np.where(lx != lt, lx > lt, plain)
            out[f"{fn} sign+length-aware: {o}"] = np.where(
                sign_differs, ~sx, np.where(lx != lt, np.where(both_neg, lx < lt, lx > lt), same))
    for o in (">", ">="):
        out[f"magnitude: |x|{o}|t|"] = _CMP[o](ax, at)
    for k in EXCEPTION_DISTANCES:
        form = f"x>t or x==t{k:+d}" if k < 0 else f"x>t and x!=t{k:+d}"
        out[f"exception d={k:+d}: {form}"] = (x > t) != (x - t == k)
    return out


def family(x, t) -> OrderedDict:
    """H: the enumerated family followed by the extension (names are unique across the two)."""
    e = enumerated_family(x, t)
    x_ = extension(x, t)
    if set(e) & set(x_):
        raise AssertionError("name collision between the enumerated family and the extension")
    return OrderedDict(list(e.items()) + list(x_.items()))


def family_rule(name: str):
    """One member of H as a scalar rule (x, t) -> bool (for synthetic answers and single rows)."""
    part = extension if name in extension(np.array([0]), np.array([0])) else enumerated_family
    if name not in part(np.array([0]), np.array([0])):
        raise KeyError(name)

    def rule(x, t, name=name, part=part):
        return bool(part(np.array([x]), np.array([t]))[name][0])
    return rule


def domain_grid() -> tuple:
    """(x, t) arrays over every cell of the domain (97 x 177 = 17,169 cells), x-major."""
    X, T = np.meshgrid(np.arange(DOMAIN["other"][0], DOMAIN["other"][1] + 1),
                       np.arange(DOMAIN["threshold"][0], DOMAIN["threshold"][1] + 1), indexing="ij")
    return X.ravel(), T.ravel()


# ---------------------------------------------------------------------------------------------- the design of cond_adv
ADV_N = 100
# The 15 design types: name -> (representative cell (x, t), rows, gold ok, what the cells are, digit lengths). A type
# is the SIGNATURE CLASS of its representative (every domain cell on which exactly the same members of H are wrong;
# adv_type_cells), restricted to the given (digits of x, digits of |t|) when that is not None, so a type's cells are
# interchangeable for every member of H. The quotas are an optimum of the integer program described in the module
# docstring (with, last, the fewest rows on any one cell: 5); the one class split by digit lengths (len_lead) keeps its
# total of 8 rows, so no member's count depends on the split.
ADV_TYPES = OrderedDict([
    ("sc_far", ((0, -9), 8, True, "sign-crossing (t < 0 <= x), x and |t| in one decade, units(x) < units(|t|), "
                                  "units(x) + units(|t|) <= 9 (i.e. units(x) < t % 10 under floor semantics), "
                                  "x - t >= 5", None)),
    ("sc_d4", ((0, -4), 10, True, "sign-crossing at x - t = +4 with x < |t|: (0, -4), (1, -3)", None)),
    ("sc_d3", ((0, -3), 10, True, "sign-crossing at x - t = +3 with x < |t|: (0, -3), (1, -2)", None)),
    ("sc_d4_eq", ((2, -2), 5, True, "sign-crossing at x - t = +4 with x = |t|: (2, -2)", None)),
    ("len21_d4", ((10, 6), 6, True, "two-digit x, one-digit t, x - t = +4: (10, 6) .. (13, 9)", None)),
    ("len21_d3", ((10, 7), 6, True, "two-digit x, one-digit t, x - t = +3: (10, 7) .. (12, 9)", None)),
    ("dec_d3", ((20, 17), 5, True, "across a decade at x - t = +3 with units(x) < units(t): x = 10k + 0..2, k = 2..9",
                None)),
    ("dec0_m2", ((0, 2), 1, False, "x - t = -2 inside one decade: x = 10k + 0..7 (x <= 96)", None)),
    ("len_lead_12", ((3, 22), 3, False, "one-digit x, two-digit t, str(x) > str(t), units(x) > units(t) and "
                                        "digitsum(x) < digitsum(t) (the class of len_lead_13 / _23)", (1, 2))),
    ("len_lead_13", ((2, 111), 2, False, "one-digit x, three-digit t, the same class", (1, 3))),
    ("len_lead_23", ((13, 122), 3, False, "two-digit x, three-digit t, the same class", (2, 3))),
    ("len12_m3", ((7, 10), 3, False, "one-digit x, two-digit t, x - t = -3: (7, 10), (8, 11), (9, 12)", None)),
    ("len12_m2", ((8, 10), 10, False, "one-digit x, two-digit t, x - t = -2: (8, 10), (9, 11)", None)),
    ("dec_m3", ((17, 20), 18, False, "across a decade at x - t = -3: x = 10k + 7..9, k = 1..8", None)),
    ("dec_m2", ((18, 20), 10, False, "across a decade at x - t = -2: x = 10k + 8..9, k = 1..8", None)),
])
ADV_MIN_BROAD = 33          # (1): wrong rows of every non-exception member of H that passes revision 5's G1 (weak plain)
ADV_MIN_EXCEPTION = 21      # (2): wrong rows of each caught isolated distance exception (79/100 < 4/5)
ADV_EXCEPTIONS_CAUGHT = (-3, -2, 3, 4)
ADV_EXCEPTION_NOT_CAUGHT = -4
OPERATOR_TARGET = 40        # DECISIONS D-42 (2): every member of H wrong on >= 40 of the 100 rows (infeasible; see above)
# certificates (pairwise-disjoint error sets on the domain; checked by adv_target_report)
DISJOINT_FIVE = ("tens(x)>tens(t) | or x==t+1",
                 "tens(x)>tens(t) or (tens(x)==tens(t) and units(x)!=units(t)) | and x!=t-1",
                 "magnitude: |x|>|t|", "digitsum sign+length-aware: >", "lex signed: str(x)>str(t)")
DISJOINT_SURVIVOR_TRIPLE = ("magnitude: |x|>|t|", "lex signed: str(x)>str(t)", "digitsum sign+length-aware: >")
REV5_G1_STRATA = OrderedDict([("conditional", ("cond_oor_val", "cond_oor_thr", "cond_far")),
                              ("cond_oor_val", ("cond_oor_val",)), ("cond_oor_edge", ("cond_oor_edge",)),
                              ("cond_sametens", ("cond_sametens",))])
REV5_PRED_STRATA = ("cond_oor_edge", "cond_sametens")
GAIN = Fraction(1, 4)
PRED_MIN = Fraction(4, 5)
_CACHE = {}


def domain_family() -> tuple:
    """(x, t, truth, H on the domain, error matrix [member x cell], member names); cached."""
    if "dom" not in _CACHE:
        X, T = domain_grid()
        H = family(X, T)
        names = list(H)
        truth = X > T
        E = np.array([H[n] != truth for n in names])
        _CACHE["dom"] = (X, T, truth, H, E, names)
    return _CACHE["dom"]


def true_rule_equivalents() -> list:
    """Members of H equal to the env's predicate on the whole domain (they compute the comparison)."""
    _, _, _, _, E, names = domain_family()
    return [n for n, e in zip(names, E) if not e.any()]


def distance_only(name: str):
    """The flip distances of a member that is a function of d = x - t alone on the domain (wrong on EVERY cell of each
    flipped distance and nowhere else), else None."""
    X, T, _, _, E, names = domain_family()
    e = E[names.index(name)]
    D = X - T
    ds = sorted({int(d) for d in np.unique(D[e])})
    return ds if ds and all(bool(e[D == d].all()) for d in ds) else None


def adv_type_cells() -> OrderedDict:
    """type -> sorted cells (x, t) of its signature class over H on the domain."""
    if "types" not in _CACHE:
        X, T, truth, _, E, _ = domain_family()
        packed = np.packbits(E, axis=0).T
        idx = {(int(x), int(t)): i for i, (x, t) in enumerate(zip(X, T))}
        out = OrderedDict()
        for name, (cell, _n, ok, _w, digits) in ADV_TYPES.items():
            sel = [i for i in np.where((packed == packed[idx[cell]]).all(axis=1))[0]
                   if digits is None or (len(str(abs(int(X[i])))), len(str(abs(int(T[i]))))) == digits]
            if idx[cell] not in sel:
                raise AssertionError(f"cond_adv type {name}: the representative is not in its own type")
            cells = sorted((int(X[i]), int(T[i])) for i in sel)
            if any(bool(truth[i]) != ok for i in sel):
                raise AssertionError(f"cond_adv type {name}: its gold ok differs from the design")
            out[name] = cells
        seen = [c for cs in out.values() for c in cs]
        if len(seen) != len(set(seen)):
            raise AssertionError("cond_adv types overlap")
        if sum(v[1] for v in ADV_TYPES.values()) != ADV_N or \
                2 * sum(v[1] for v in ADV_TYPES.values() if v[2]) != ADV_N:
            raise AssertionError("cond_adv quotas are not 100 rows, 50 / 50")
        _CACHE["types"] = out
    return _CACHE["types"]


def adv_type_of(x: int, t: int):
    """The design type of a cell, or None."""
    if "type_of" not in _CACHE:
        _CACHE["type_of"] = {c: k for k, cs in adv_type_cells().items() for c in cs}
    return _CACHE["type_of"].get((int(x), int(t)))


def wrong_counts(xs, ts) -> OrderedDict:
    """member of H -> number of the given (x, t) rows on which it differs from the env's predicate."""
    xs, ts = np.asarray(xs), np.asarray(ts)
    truth = xs > ts
    return OrderedDict((n, int((v != truth).sum())) for n, v in family(xs, ts).items())


def g1_rev5_survivors(rows: list, baseline: Fraction = Fraction(0)) -> list:
    """Members of H (not equal to the true rule on the domain) that pass revision 5's G1 on these dev rows when a trace
    side answers every conditional with the member's branch and exact arithmetic (after right <=> predicate right <=>
    branch right) against a plain arm at `baseline` on every relative part (0 = the weak plain arm, the most permissive):
    trace - plain >= 1/4 on the pooled conditional, cond_oor_val, cond_oor_edge and cond_sametens, and the predicate
    accuracy >= 4/5 on cond_oor_edge and cond_sametens (the held-out surface has the same rows, so the same values)."""
    rs = [r for r in rows if r["stratum"] in {s for ss in REV5_G1_STRATA.values() for s in ss}]
    x = np.array([r["before"][r["event"]["other"]] for r in rs])
    t = np.array([r["event"]["threshold"] for r in rs])
    gold = np.array([bool(r["gold_ok"]) for r in rs])
    st = np.array([r["stratum"] for r in rs])
    trivial = set(true_rule_equivalents())
    out = []
    for n, v in family(x, t).items():
        if n in trivial:
            continue
        right = v == gold
        acc = {m: Fraction(int(right[np.isin(st, ss)].sum()), int(np.isin(st, ss).sum()))
               for m, ss in REV5_G1_STRATA.items()}
        if all(a - baseline >= GAIN for a in acc.values()) and all(acc[s] >= PRED_MIN for s in REV5_PRED_STRATA):
            out.append(n)
    return out


def _disjoint(names: tuple) -> bool:
    _, _, _, _, E, all_names = domain_family()
    es = [E[all_names.index(n)] for n in names]
    return all(not (a & b).any() for a, b in itertools.combinations(es, 2)) and all(bool(e.any()) for e in es)


def type_error_profile() -> OrderedDict:
    """Per design type: which members of the EXTENSION are wrong on its cells (the enumerated family's members are in
    the per-member counts)."""
    X, T, _, _, E, names = domain_family()
    idx = {(int(x), int(t)): i for i, (x, t) in enumerate(zip(X, T))}
    ext = list(extension(np.array([0]), np.array([0])))
    return OrderedDict((k, [n for n in ext if bool(E[names.index(n)][idx[ADV_TYPES[k][0]]])]) for k in ADV_TYPES)


def adv_target_report(dev_rows: list) -> dict:
    """Revision 6: cond_adv against H (data only). dev_rows: the dev bank's rows (with 'stratum', 'before', 'event',
    'gold_ok'); the cond_adv rows are those of stratum 'cond_adv', the rest are revision 5's bank. Targets (all must
    hold): 100 rows, 50 / 50, every row in a design type, the type quotas, cells balanced inside each type; (1) every
    member of H that passes revision 5's G1 against the weak plain arm and is not a function of x - t alone is wrong on
    >= ADV_MIN_BROAD rows; (2) the distance exceptions among them at ADV_EXCEPTIONS_CAUGHT on >= ADV_MIN_EXCEPTION rows
    each, and only the one at ADV_EXCEPTION_NOT_CAUGHT is not caught. Also: every member's count, the operator's 40-row
    target and its infeasibility certificates."""
    adv = [r for r in dev_rows if r["stratum"] == "cond_adv"]
    rest = [r for r in dev_rows if r["stratum"] != "cond_adv"]
    xs = [r["before"][r["event"]["other"]] for r in adv]
    ts = [r["event"]["threshold"] for r in adv]
    types = [adv_type_of(x, t) for x, t in zip(xs, ts)]
    cells = adv_type_cells()
    want_n = {k: v[1] for k, v in ADV_TYPES.items()}
    got_n = {k: types.count(k) for k in ADV_TYPES}
    use = {k: {} for k in ADV_TYPES}
    for ty, x, t in zip(types, xs, ts):
        if ty is not None:
            use[ty][(x, t)] = use[ty].get((x, t), 0) + 1
    balanced = all(len(use[k]) == min(want_n[k], len(cells[k]))
                   and (max(use[k].values()) - min(use[k].values()) <= 1 if use[k] else want_n[k] == 0)
                   for k in ADV_TYPES)
    counts = wrong_counts(xs, ts)
    trivial = set(true_rule_equivalents())
    surv = g1_rev5_survivors(rest)
    exc = {n: distance_only(n) for n in surv}
    broad = [n for n in surv if exc[n] is None]
    excs = {n: d for n, d in exc.items() if d is not None}
    caught = {n: d for n, d in excs.items() if len(d) == 1 and d[0] in ADV_EXCEPTIONS_CAUGHT}
    not_caught = {n: d for n, d in excs.items() if n not in caught}
    nontrue = [n for n in counts if n not in trivial]
    _, _, _, _, E, names = domain_family()
    fn = {}
    for n, e in zip(names, E):
        if n not in trivial:
            fn.setdefault(e.tobytes(), n)
    distinct = list(fn.values())
    gold_ok = [bool(r["gold_ok"]) for r in adv]
    certs = OrderedDict([
        ("five_members_with_pairwise_disjoint_error_sets", list(DISJOINT_FIVE)),
        ("five_disjoint", _disjoint(DISJOINT_FIVE)),
        ("bound", "some member of H is wrong on at most 100 / 5 = 20 rows of ANY 100 rows (fewer than the 21 needed to "
                  "fail the 0.80 bar there)"),
        ("three_rev5_survivors_with_pairwise_disjoint_error_sets", list(DISJOINT_SURVIVOR_TRIPLE)),
        ("three_disjoint_and_survivors", _disjoint(DISJOINT_SURVIVOR_TRIPLE)
         and all(n in surv for n in DISJOINT_SURVIVOR_TRIPLE)),
        ("bound_survivors", "one of them is wrong on at most floor(100 / 3) = 33 rows of any 100 rows: target (1) is the "
                            "maximum"),
        ("exceptions", "the isolated exceptions at x - t = -4, -3, -2 are wrong only on ok-false rows at their own "
                       "distance; failing the 0.80 bar needs >= 21 such rows each and 3 x 21 = 63 > 50 ok-false rows, "
                       "so one of them passes on any 100-row branch-balanced stratum (here d = -4, the farthest from the "
                       "boundary); none of them is caught by another G1 part (cond_sametens has 18 / 12 / 12 rows at "
                       "d = -2 / -3 / -4)")])
    res = OrderedDict()
    res["n_rows"] = len(adv)
    res["branch"] = {"ok_true": sum(gold_ok), "ok_false": len(gold_ok) - sum(gold_ok)}
    res["types"] = OrderedDict((k, {"rows": got_n[k], "quota": want_n[k], "gold_ok": ADV_TYPES[k][2],
                                    "representative": list(ADV_TYPES[k][0]), "what": ADV_TYPES[k][3],
                                    "digits": ADV_TYPES[k][4],
                                    "class_cells": len(cells[k]), "cells_used": len(use[k]),
                                    "max_rows_on_a_cell": max(use[k].values()) if use[k] else 0,
                                    "cells": [list(c) for c in cells[k]]}) for k in ADV_TYPES)
    res["type_error_profile_extension"] = type_error_profile()
    res["distinct_cells"] = len(set(zip(xs, ts)))
    res["max_rows_on_a_cell"] = max((max(u.values()) for u in use.values() if u), default=0)
    res["every_row_in_a_type"] = None not in types
    res["quotas_met"] = got_n == want_n
    res["cells_balanced_within_types"] = balanced
    res["h"] = {"members": len(counts), "members_equal_to_the_true_rule_on_the_domain": sorted(trivial),
                "distinct_non_true_functions_on_the_domain": len(distinct)}
    res["rev5_g1_survivors_weak_plain"] = OrderedDict((n, {"wrong_on_cond_adv": counts[n],
                                                           "flip_distances": exc[n]}) for n in surv)
    res["target_1_broad_survivors"] = {"min_wrong": ADV_MIN_BROAD, "n": len(broad),
                                       "wrong_min": min(counts[n] for n in broad) if broad else None,
                                       "below_target": sorted(n for n in broad if counts[n] < ADV_MIN_BROAD)}
    res["target_2_exceptions"] = {"min_wrong": ADV_MIN_EXCEPTION, "caught_distances": list(ADV_EXCEPTIONS_CAUGHT),
                                  "caught": {n: {"flip_distances": d, "wrong": counts[n]} for n, d in caught.items()},
                                  "not_caught": {n: {"flip_distances": d, "wrong": counts[n]}
                                                 for n, d in not_caught.items()},
                                  "below_target": sorted(n for n in caught if counts[n] < ADV_MIN_EXCEPTION)}
    res["operator_target_40"] = {
        "target": f"every member of H wrong on >= {OPERATOR_TARGET} of the {ADV_N} rows (DECISIONS D-42 (2))",
        "met": all(counts[n] >= OPERATOR_TARGET for n in nontrue),
        "predicates_meeting_it": sum(counts[n] >= OPERATOR_TARGET for n in nontrue), "predicates": len(nontrue),
        "distinct_functions_meeting_it": sum(counts[n] >= OPERATOR_TARGET for n in distinct),
        "distinct_functions": len(distinct),
        "predicates_wrong_on_at_least_21": sum(counts[n] >= ADV_MIN_EXCEPTION for n in nontrue),
        "min_wrong_over_H": min(counts[n] for n in nontrue),
        "infeasible_on_any_100_row_branch_balanced_stratum": certs}
    res["wrong_counts"] = counts
    caught_d = sorted(d[0] for d in caught.values())
    res["ok"] = bool(res["n_rows"] == ADV_N and res["branch"] == {"ok_true": ADV_N // 2, "ok_false": ADV_N // 2}
                     and res["every_row_in_a_type"] and res["quotas_met"] and balanced
                     and not res["target_1_broad_survivors"]["below_target"]
                     and not res["target_2_exceptions"]["below_target"]
                     and set(caught_d) == set(ADV_EXCEPTIONS_CAUGHT)
                     and all(d == [ADV_EXCEPTION_NOT_CAUGHT] for d in not_caught.values())
                     and certs["five_disjoint"] and certs["three_disjoint_and_survivors"])
    return res
