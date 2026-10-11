"""RCX-E11a: parameterised macro-operators (open-loop options) mined from verified Blocksworld plans, validated
symbolically against BWM's exact dynamics, and used inside GBFS / beam search (RSM-001; CPU mechanism tier; a macro
library is an external specialist, tier B).

Macro = (steps, goal_on), over variables 0..V-1 numbered by first appearance:
  steps    2-4 primitive action patterns, e.g. (("pick-up", 0), ("stack", 0, 1)). Distinct variables bind to distinct
           blocks (injective binding), so `stack ?i ?i` / `unstack ?i ?i` patterns are rejected outright.
  goal_on  static initiation condition: pairs (i, j) such that ("on", ?i, ?j) must be a goal fact. Mining records
           every goal on-relation among a segment's blocks in its source problem's goal (`abstract`), so a mined skill
           is goal-directed ("stack ?0 where the goal wants it"); the relation is a filter on bindings only and never
           enters the dynamics.
  pre/add/dele  derived by STRIPS regression (`regress`) over SCHEMAS (bwmn/blocks.py applicable()/step(), pinned
           b1e28ed5...): walking the steps, a precondition already produced by an earlier step is satisfied; one deleted
           by an earlier step (and not re-produced) makes the macro INCONSISTENT; otherwise it joins pre. Effects
           compose as add' = (add - D) | A, dele' = (dele - A) | D, so that executing the steps from s gives
           (s - dele) | add whenever pre <= s (a theorem of STRIPS sequential composition, given SCHEMAS = B.step;
           SCHEMAS are checked against B.applicable/B.step on every ground action of every state for n <= 4 in tests).
Initiation set I(s, goal) = {injective beta: pre[beta] <= s and goal_on[beta] <= goal} (`Macro.bindings`, a join over
the facts of s); policy = the ground step sequence; termination after the last step.

Validation (`validate`, the gate for every macro used anywhere): bounded exhaustive check against BWM's simulator on
EVERY valid state of n_check blocks (n_check = min(V + 2, n_check_max), at least 3; space.get_space):
  * consistent   regression succeeded (no step precondition destroyed by an earlier step; no self-pairs);
  * sound        for every (state, binding) with pre[beta] <= s, replaying the ground steps with B.applicable/B.step
                 succeeds and ends in exactly (s - dele[beta]) | add[beta];
  * complete     every binding under which the steps are executable (enumerated independently by walking B.step) has
                 pre[beta] <= s, i.e. pre is exactly the applicability condition;
  * non-vacuous  at least one (state, binding) is executable (rejects mutex preconditions such as two held blocks or
                 two blocks on one block, which STRIPS regression alone cannot see);
  * non-identity some execution changes the state (rejects e.g. pick-up ?0; put-down ?0 and stack ?0 ?1; unstack ?0 ?1).
accepted = all five. A macro built with `claimed=(pre, add, dele)` is checked against its claim instead (tests use
this to show that unsound claims are rejected). Validation ignores goal_on (dynamics only), so results are cached by
step key.

Search (`gbfs_macro`, `beam_macro`): identical to rcx.bw.search.gbfs / beam_search (same successor order, duplicate
detection, generation-time goal test, tie-breaking and budget semantics) plus, after the primitive children of an
expanded state, the end states of every macro binding (library order, bindings sorted), each replayed step by step with
B.step. One node expansion generates all primitive and macro children, so a macro edge costs the same single expansion
of its parent as a primitive edge; prim_steps (B.step calls during search) and h_calls are reported as the compute-side
cost. If an intermediate state of a macro satisfies the goal, the plan stops there (goal_mid_macro). With an empty
library both functions return exactly what search.gbfs / search.beam_search return (tested). Both are budget-monotone
in the same sense as search.gbfs (run_e11.at_budget derives smaller budgets from the largest-budget run).
"""
from __future__ import annotations

import heapq
import itertools
import math
import random
import time
from collections import Counter, defaultdict

from . import bwm_import as BI
from . import search as S
from . import space as SP

OPS = ("pick-up", "put-down", "stack", "unstack")
ARITY = {"pick-up": 1, "put-down": 1, "stack": 2, "unstack": 2}
HE = ("handempty",)
N_CHECK_MAX = 6
_PRED_COST = {"handempty": 0, "holding": 1, "on": 2, "goal_on": 2, "clear": 5, "ontable": 7}


class MacroError(RuntimeError):
    """A validated macro failed to apply (never expected; signals a bug)."""


# ----------------------------------------------------------------------------- action schemas and regression

def schema(op: str, args: tuple):
    """(pre, add, dele) frozensets of one BWM action over arbitrary argument symbols (names or variable ids).

    Mirrors bwmn/blocks.py: pick-up x needs clear x, ontable x, handempty; put-down x needs holding x; stack x y needs
    clear y, holding x (and x != y); unstack x y needs on x y, clear x, handempty. step() = (s - dele) | add.
    """
    if op == "pick-up":
        (x,) = args
        p = frozenset({("clear", x), ("ontable", x), HE})
        return p, frozenset({("holding", x)}), p
    if op == "put-down":
        (x,) = args
        return frozenset({("holding", x)}), frozenset({("clear", x), HE, ("ontable", x)}), frozenset({("holding", x)})
    if op == "stack":
        x, y = args
        p = frozenset({("clear", y), ("holding", x)})
        return p, frozenset({HE, ("clear", x), ("on", x, y)}), p
    if op == "unstack":
        x, y = args
        p = frozenset({("on", x, y), ("clear", x), HE})
        return p, frozenset({("holding", x), ("clear", y)}), p
    raise ValueError(f"unknown operator {op!r}")


def regress(steps):
    """((pre, add, dele), None) for a consistent step pattern, else (None, reason)."""
    pre, add, dele = set(), set(), set()
    for k, st in enumerate(steps):
        op, args = st[0], tuple(st[1:])
        if op not in ARITY or len(args) != ARITY[op]:
            return None, f"step {k}: malformed {st!r}"
        if len(args) == 2 and args[0] == args[1]:
            return None, f"step {k}: {op} with identical arguments"
        P, A, D = schema(op, args)
        for p in sorted(P, key=repr):
            if p in add:
                continue
            if p in dele:
                return None, f"step {k}: precondition {p} was deleted by an earlier step"
            pre.add(p)
        add = (add - D) | A
        dele = (dele - A) | D
    return (frozenset(pre), frozenset(add), frozenset(dele)), None


def canonical_steps(steps) -> tuple:
    """Renumber variables by first appearance (the canonical form used for keys)."""
    m = {}
    return tuple((st[0],) + tuple(m.setdefault(v, len(m)) for v in st[1:]) for st in steps)


def _fact_str(f) -> str:
    return f[0] if len(f) == 1 else f"{f[0]}(" + ",".join(f"?{v}" for v in f[1:]) + ")"


def step_key(steps) -> str:
    return "; ".join(" ".join([st[0]] + [f"?{v}" for v in st[1:]]) for st in steps)


def macro_key(steps, goal_on) -> str:
    g = ",".join(f"on(?{i},?{j})" for i, j in goal_on)
    return step_key(steps) + " | " + (g or "-")


def abstract(seq, goal) -> tuple:
    """(steps, goal_on, names) of a ground action segment: variables by first appearance; goal_on = every goal
    on-fact between two of the segment's blocks, as variable pairs (sorted)."""
    m = {}
    steps = tuple((a[0],) + tuple(m.setdefault(x, len(m)) for x in a[1:]) for a in seq)
    gon = tuple(sorted((m[f[1]], m[f[2]]) for f in goal if f[0] == "on" and f[1] in m and f[2] in m))
    return steps, gon, sorted(m, key=m.get)


# ----------------------------------------------------------------------------- indices for binding enumeration

class StateIndex:
    """Facts of one state grouped by predicate (sorted argument tuples) for Macro.bindings."""
    __slots__ = ("facts", "rel")

    def __init__(self, s):
        rel = defaultdict(list)
        for f in s:
            rel[f[0]].append(tuple(f[1:]))
        for v in rel.values():
            v.sort()
        self.facts, self.rel = s, rel


def goal_index(goal) -> list:
    """Sorted (x, y) pairs of the goal's on-facts (the relation goal_on literals are matched against)."""
    return sorted((f[1], f[2]) for f in goal if f[0] == "on")


def _join_order(lits, nvars):
    order, bound, rest = [], set(), sorted(lits, key=repr)
    while rest:
        def cost(lit):
            free = len(set(lit[1]) - bound)
            if free == 0:
                return (-1, repr(lit))
            c = _PRED_COST[lit[0]]
            if lit[0] in ("on", "goal_on") and free == 1:
                c = 1
            return (c * free, repr(lit))
        best = min(rest, key=cost)
        rest.remove(best)
        order.append(best)
        bound |= set(best[1])
    return tuple(order)


# ----------------------------------------------------------------------------- the macro

class Macro:
    """A parameterised macro (see module doc). `claimed` overrides the derived (pre, add, dele) (tests only)."""

    def __init__(self, steps, goal_on=(), claimed=None, meta: dict | None = None):
        steps = tuple(tuple(st) for st in steps)
        if canonical_steps(steps) != steps:
            raise ValueError(f"steps are not in canonical variable order: {steps}")
        self.steps = steps
        self.goal_on = tuple(sorted(tuple(p) for p in goal_on))
        self.L = len(steps)
        self.nvars = len({v for st in steps for v in st[1:]})
        if any(not 0 <= v < self.nvars for p in self.goal_on for v in p) or any(i == j for i, j in self.goal_on):
            raise ValueError(f"bad goal_on {self.goal_on} for {self.nvars} variables")
        self.step_key = step_key(steps)
        self.key = macro_key(steps, self.goal_on)
        self.meta = dict(meta or {})
        derived, why = regress(steps)
        self.derived = derived
        self.claimed = claimed is not None
        eff = tuple(frozenset(x) for x in claimed) if claimed is not None else derived
        self.consistent = derived is not None
        self.why = why
        if eff is None:
            self.pre = self.add = self.dele = frozenset()
            self._order = self._order_dyn = ()
            return
        self.pre, self.add, self.dele = eff
        if {v for f in self.pre for v in f[1:]} != set(range(self.nvars)):
            if not self.claimed:  # every variable of a BW action occurs in its precondition, so this cannot happen
                raise MacroError(f"{self.key}: a variable does not occur in the derived precondition")
            self.consistent, self.why = False, "claimed precondition leaves a variable unbound"
            self._order = self._order_dyn = ()
            return
        pre_lits = [(f[0], tuple(f[1:])) for f in self.pre]
        self._order_dyn = _join_order(pre_lits, self.nvars)
        self._order = _join_order(pre_lits + [("goal_on", p) for p in self.goal_on], self.nvars)

    def __repr__(self):
        return f"Macro({self.key})"

    # -- binding enumeration and application

    def bindings(self, sidx: StateIndex, gidx=None) -> list:
        """Sorted injective bindings (tuples of block names, index = variable) with pre[beta] <= s and, if `gidx`
        (goal_index) is given, goal_on[beta] <= goal. gidx=None ignores goal_on (dynamics only)."""
        if not self.consistent:
            return []
        order = self._order_dyn if gidx is None else self._order
        beta = [None] * self.nvars
        used, out, rel, facts = set(), [], sidx.rel, sidx.facts

        def rec(k):
            if k == len(order):
                out.append(tuple(beta))
                return
            pred, args = order[k]
            if pred == "handempty":
                if HE in facts:
                    rec(k + 1)
                return
            for tup in (gidx if pred == "goal_on" else rel.get(pred, ())):
                newly, ok = [], True
                for v, val in zip(args, tup):
                    b = beta[v]
                    if b is None:
                        if val in used:
                            ok = False
                            break
                        beta[v] = val
                        used.add(val)
                        newly.append(v)
                    elif b != val:
                        ok = False
                        break
                if ok:
                    rec(k + 1)
                for v in newly:
                    used.discard(beta[v])
                    beta[v] = None

        rec(0)
        out.sort()
        return out

    def ground(self, beta) -> tuple:
        return tuple((st[0],) + tuple(beta[v] for v in st[1:]) for st in self.steps)

    def _sub(self, facts, beta):
        return {(f[0],) + tuple(beta[v] for v in f[1:]) for f in facts}

    def apply_symbolic(self, s, beta) -> frozenset:
        return frozenset((set(s) - self._sub(self.dele, beta)) | self._sub(self.add, beta))

    def replay(self, B, s, beta):
        """States after each ground step via B.applicable/B.step, or None if some step is inapplicable."""
        out, cur = [], s
        for a in self.ground(beta):
            if not B.applicable(cur, a):
                return None
            cur = B.step(cur, a)[0]
            out.append(cur)
        return out

    # -- serialisation

    def to_json(self) -> dict:
        fl = lambda fs: sorted([list(f) for f in fs], key=repr)  # noqa: E731
        return {"key": self.key, "step_key": self.step_key, "steps": [list(st) for st in self.steps],
                "goal_on": [list(p) for p in self.goal_on], "L": self.L, "nvars": self.nvars,
                "consistent": self.consistent, "why": self.why, "pre": fl(self.pre), "add": fl(self.add),
                "dele": fl(self.dele), "pre_str": sorted(_fact_str(f) for f in self.pre), "meta": self.meta}

    @classmethod
    def from_json(cls, d: dict) -> "Macro":
        m = cls([tuple(st) for st in d["steps"]], [tuple(p) for p in d["goal_on"]], meta=d.get("meta"))
        if m.key != d["key"]:
            raise ValueError(f"macro key mismatch: {m.key} vs {d['key']}")
        fl = lambda fs: sorted([list(f) for f in fs], key=repr)  # noqa: E731
        if m.consistent and (fl(m.pre), fl(m.add), fl(m.dele)) != (d["pre"], d["add"], d["dele"]):
            raise ValueError(f"{m.key}: stored pre/add/dele differ from the regression of its steps")
        return m


# ----------------------------------------------------------------------------- exhaustive validation

def default_n_check(nvars: int, n_check_max: int = N_CHECK_MAX) -> int:
    return max(3, min(nvars + 2, n_check_max), nvars)


def executable_bindings(B, s, steps, nvars, objs) -> dict:
    """{beta: final state} for every injective binding under which the ground steps are executable from s, found by
    walking B.applicable/B.step (independent of the regression)."""
    out = {}
    beta = [None] * nvars
    used = set()

    def rec(k, cur):
        if k == len(steps):
            out[tuple(beta)] = cur
            return
        op, args = steps[k][0], steps[k][1:]
        free = [v for v in dict.fromkeys(args) if beta[v] is None]
        for vals in itertools.permutations([o for o in objs if o not in used], len(free)):
            for v, o in zip(free, vals):
                beta[v] = o
            a = (op,) + tuple(beta[v] for v in args)
            if B.applicable(cur, a):
                used.update(vals)
                rec(k + 1, B.step(cur, a)[0])
                used.difference_update(vals)
            for v in free:
                beta[v] = None

    rec(0, s)
    return out


def validate(m: Macro, n_check: int | None = None, n_check_max: int = N_CHECK_MAX, max_examples: int = 3) -> dict:
    """Bounded exhaustive check of `m` against BWM's exact dynamics on every valid state of n_check blocks."""
    res = {"key": m.key, "step_key": m.step_key, "consistent": m.consistent, "why": m.why, "claimed": m.claimed}
    if not m.consistent:
        res.update(accepted=False, reason="inconsistent")
        return res
    B, _ = BI.load()
    n = n_check if n_check is not None else default_n_check(m.nvars, n_check_max)
    if n < m.nvars:
        raise ValueError(f"n_check={n} < {m.nvars} variables")
    objs = B.NAMES[:n]
    sp = SP.get_space(n)
    t0 = time.perf_counter()
    cnt = Counter()
    examples = defaultdict(list)
    for row in sp.states:
        s = SP.to_facts(tuple(int(v) for v in row), objs)
        sat = set(m.bindings(StateIndex(s)))
        exe = executable_bindings(B, s, m.steps, m.nvars, objs)
        cnt["states"] += 1
        for beta in sat - set(exe):
            cnt["pre_but_not_executable"] += 1
            if len(examples["unsound_pre"]) < max_examples:
                examples["unsound_pre"].append({"state": sorted(map(list, s)), "beta": list(beta)})
        for beta in set(exe) - sat:
            cnt["executable_but_not_pre"] += 1
            if len(examples["incomplete_pre"]) < max_examples:
                examples["incomplete_pre"].append({"state": sorted(map(list, s)), "beta": list(beta)})
        for beta, fin in exe.items():
            cnt["executable_pairs"] += 1
            if m.apply_symbolic(s, beta) != fin:
                cnt["effect_mismatch"] += 1
                if len(examples["effect_mismatch"]) < max_examples:
                    examples["effect_mismatch"].append({"state": sorted(map(list, s)), "beta": list(beta)})
            if fin != s:
                cnt["state_changing_pairs"] += 1
    sound = cnt["pre_but_not_executable"] == 0 and cnt["effect_mismatch"] == 0
    complete = cnt["executable_but_not_pre"] == 0
    vacuous = cnt["executable_pairs"] == 0
    identity = not vacuous and cnt["state_changing_pairs"] == 0
    accepted = sound and complete and not vacuous and not identity
    reason = None if accepted else ("unsound" if not sound else "incomplete_pre" if not complete else
                                    "vacuous" if vacuous else "identity")
    res.update(accepted=accepted, reason=reason, sound=sound, complete=complete, vacuous=vacuous, identity=identity,
               n_check=n, counts=dict(cnt), examples=dict(examples), seconds=round(time.perf_counter() - t0, 3))
    return res


class ValidationCache:
    """validate() results cached by step key (validation ignores goal_on). A consistent pattern whose precondition is
    mutex (pre_mutex) is rejected as vacuous without enumeration (marked 'shortcut'); everything else is validated
    exhaustively."""

    def __init__(self, n_check_max: int = N_CHECK_MAX, courtesy=None):
        self.n_check_max, self.courtesy, self.cache = n_check_max, courtesy, {}

    def __call__(self, m: Macro) -> dict:
        if m.claimed:
            return validate(m, n_check_max=self.n_check_max)
        if m.step_key not in self.cache:
            base = Macro(m.steps)
            why = pre_mutex(base.pre) if base.consistent else None
            if why is not None:
                self.cache[m.step_key] = {"key": base.key, "step_key": base.step_key, "consistent": True,
                                          "why": None, "claimed": False, "accepted": False, "reason": "vacuous",
                                          "shortcut": f"mutex_pre:{why}"}
            else:
                self.cache[m.step_key] = validate(base, n_check_max=self.n_check_max)
            if self.courtesy:
                self.courtesy()
        return self.cache[m.step_key]

    def summary(self) -> dict:
        c = Counter(v["reason"] or "accepted" for v in self.cache.values())
        return {"patterns_checked": len(self.cache), "outcomes": dict(sorted(c.items())),
                "mutex_shortcuts": sum(1 for v in self.cache.values() if "shortcut" in v),
                "seconds": round(sum(v.get("seconds", 0.0) for v in self.cache.values()), 2)}


# ----------------------------------------------------------------------------- mining and library selection

def mine(plans, lengths=(2, 3, 4), max_vars: int = 4) -> tuple[dict, dict]:
    """Count abstract segments of every plan. plans: dicts with 'id', 'plan' (BWM action tuples), 'goal' (frozenset).

    Returns ({key: candidate}, stats). A candidate: key, step_key, steps, goal_on, L, nvars, occurrences (all offsets,
    overlapping), support (number of distinct plans), score = occurrences * (L - 1) (primitive steps saved per use,
    summed over uses). Segments with more than max_vars variables are skipped (counted)."""
    table, plans_of, stats = {}, defaultdict(set), Counter()
    for p in plans:
        seq = [tuple(a) for a in p["plan"]]
        for L in lengths:
            for i in range(len(seq) - L + 1):
                steps, gon, names = abstract(seq[i:i + L], p["goal"])
                stats["segments"] += 1
                if len(names) > max_vars:
                    stats[f"skipped_L{L}_over_{max_vars}_vars"] += 1
                    continue
                key = macro_key(steps, gon)
                e = table.get(key)
                if e is None:
                    e = table[key] = {"key": key, "step_key": step_key(steps), "steps": steps, "goal_on": gon,
                                      "L": L, "nvars": len(names), "occurrences": 0}
                e["occurrences"] += 1
                plans_of[key].add(p["id"])
    for k, e in table.items():
        e["support"] = len(plans_of[k])
        e["score"] = e["occurrences"] * (e["L"] - 1)
    stats["plans"] = len(plans)
    stats["distinct_keys"] = len(table)
    return table, dict(stats)


def rank(table: dict) -> list:
    """Candidates by score desc, then support desc, then key (deterministic)."""
    return sorted(table.values(), key=lambda e: (-e["score"], -e["support"], e["key"]))


def select_library(ranked, K: int, min_support: int, vcache, exclude_step_keys=frozenset(),
                   fill_min_support: int | None = None) -> tuple[list, list]:
    """Top-K accepted macros in rank order with support >= min_support; if fewer than K, optionally a second pass with
    support >= fill_min_support. Returns (library, log of every candidate examined)."""
    lib, log, chosen = [], [], set()
    passes = [min_support] + ([fill_min_support] if fill_min_support is not None and
                              fill_min_support < min_support else [])
    for pi, ms in enumerate(passes):
        for c in ranked:
            if len(lib) >= K:
                break
            if c["support"] < ms or c["key"] in chosen:
                continue
            entry = {"key": c["key"], "score": c["score"], "occurrences": c["occurrences"], "support": c["support"],
                     "pass": pi}
            if c["step_key"] in exclude_step_keys:
                log.append(dict(entry, decision="excluded_step_pattern"))
                continue
            m = Macro(c["steps"], c["goal_on"], meta={k: c[k] for k in ("occurrences", "support", "score")} |
                      {"pass": pi})
            v = vcache(m)
            log.append(dict(entry, decision="accepted" if v["accepted"] else f"rejected_{v['reason']}"))
            if v["accepted"]:
                lib.append(m)
                chosen.add(c["key"])
        if len(lib) >= K:
            break
    return lib, log


MATCH_TIERS = ("same_L_min_support", "same_L_fill_support", "any_L_min_support", "any_L_fill_support")


def select_matched_library(ranked, templates, min_support: int, vcache, exclude_step_keys=frozenset(),
                           fill_min_support: int = 1) -> tuple[list, list]:
    """A library matched to the templates' length mix and disjoint from an excluded library (the shuffled-plan control,
    CX decision on E11 Q7). For each template, in order, take the highest-ranked candidate that validate() accepts, is
    not already chosen and whose abstract step signature (step_key) is not in exclude_step_keys. Candidates sharing an
    excluded step signature are skipped whatever their goal_on, so the library shares no action pattern with the
    excluded one. The first tier with a candidate is used:
      tier 0  same L, support >= min_support      tier 1  same L, support >= fill_min_support
      tier 2  any L, support >= min_support       tier 3  any L, support >= fill_min_support
    Returns (library, log); every macro's meta records its tier, template, rank, occurrences, support and score."""
    lib, log, chosen = [], [], set()
    skipped_excluded, rejected = set(), {}
    for t in templates:
        pick = None
        for tier, name in enumerate(MATCH_TIERS):
            ms = min_support if tier in (0, 2) else fill_min_support
            for rank_i, c in enumerate(ranked):
                if c["key"] in chosen or c["support"] < ms or (tier < 2 and c["L"] != t.L):
                    continue
                if c["step_key"] in exclude_step_keys:
                    skipped_excluded.add(c["key"])
                    continue
                v = vcache(Macro(c["steps"], c["goal_on"]))
                if not v["accepted"]:
                    rejected[c["key"]] = v["reason"]
                    continue
                pick = Macro(c["steps"], c["goal_on"], meta={"occurrences": c["occurrences"],
                                                              "support": c["support"], "score": c["score"],
                                                              "rank": rank_i, "tier": name, "template": t.key})
                break
            if pick is not None:
                break
        if pick is None:
            raise RuntimeError(f"no accepted, non-excluded candidate for template {t.key} in any tier")
        lib.append(pick)
        chosen.add(pick.key)
        log.append({"template": t.key, "template_L": t.L, "chosen": pick.key, "L": pick.L, "tier": pick.meta["tier"],
                    "rank": pick.meta["rank"], "occurrences": pick.meta["occurrences"],
                    "support": pick.meta["support"]})
    log.append({"summary": True, "skipped_excluded_keys": sorted(skipped_excluded),
                "rejected_keys": dict(sorted(rejected.items()))})
    return lib, log


def shuffled_plans(plans, rng: random.Random) -> list:
    """The shuffled-plan control: every plan's actions in a random order (same multiset, same goal and id)."""
    out = []
    for p in plans:
        seq = [tuple(a) for a in p["plan"]]
        rng.shuffle(seq)
        out.append(dict(p, plan=seq))
    return out


def pre_mutex(pre) -> str | None:
    """A reason if the (variable-level) precondition cannot hold in any valid Blocksworld state under an injective
    binding (two held blocks, held and handempty, a held block with a position or clear, a block on a held block, two
    positions, two blocks on one, on(x, y) with clear(y), a cycle of on-facts), else None. Necessary conditions only:
    `validate` stays the authority; ValidationCache uses this as a shortcut to reject vacuous patterns."""
    held = [f[1] for f in pre if f[0] == "holding"]
    on = [(f[1], f[2]) for f in pre if f[0] == "on"]
    table = {f[1] for f in pre if f[0] == "ontable"}
    clear = {f[1] for f in pre if f[0] == "clear"}
    if len(held) > 1:
        return "two_held"
    if held and HE in pre:
        return "held_and_handempty"
    h = set(held)
    if h & (table | clear | {x for x, _ in on}) or h & {y for _, y in on}:
        return "held_block_placed_or_supporting"
    tops, bots = [x for x, _ in on], [y for _, y in on]
    if len(set(tops)) != len(tops) or set(tops) & table:
        return "two_positions"
    if len(set(bots)) != len(bots):
        return "two_on_one"
    if set(bots) & clear:
        return "covered_and_clear"
    below = dict(on)
    for start in below:
        seen, x = set(), start
        while x in below:
            if x in seen:
                return "cycle"
            seen.add(x)
            x = below[x]
    return None


def _rgs(slots: int, V: int):
    """Restricted growth strings of length `slots` using exactly V values (canonical variable assignments)."""
    def rec(prefix, m):
        if len(prefix) == slots:
            if m == V:
                yield tuple(prefix)
            return
        if V - m > slots - len(prefix):
            return
        for v in range(min(m + 1, V)):
            yield from rec(prefix + [v], max(m, v + 1))
    yield from rec([], 0)


def pattern_universe(L: int, V: int) -> list:
    """Every canonical step pattern of length L with exactly V variables that has no self-pair, is STRIPS-consistent
    and has a mutex-free precondition (cheap filters; not yet exhaustively validated), sorted by step key."""
    out = []
    for ops in itertools.product(OPS, repeat=L):
        slots = sum(ARITY[o] for o in ops)
        for vals in _rgs(slots, V):
            it = iter(vals)
            steps = tuple((o,) + tuple(next(it) for _ in range(ARITY[o])) for o in ops)
            eff, _ = regress(steps)
            if eff is not None and pre_mutex(eff[0]) is None:
                out.append(steps)
    return sorted(out, key=step_key)


def random_goal_on(rng: random.Random, V: int, G: int, max_tries: int = 2000):
    """G distinct ordered pairs over V variables forming a consistent goal fragment (each variable on at most one and
    under at most one other, no cycle), drawn uniformly by rejection; None if none found."""
    if G == 0:
        return ()
    pairs = [(i, j) for i in range(V) for j in range(V) if i != j]
    if G > len(pairs):
        return None
    for _ in range(max_tries):
        pick = sorted(rng.sample(pairs, G))
        tops, bots = [p[0] for p in pick], [p[1] for p in pick]
        if len(set(tops)) != G or len(set(bots)) != G:
            continue
        below = dict(pick)
        cyclic = False
        for start in below:
            seen, x = set(), start
            while x in below:
                if x in seen:
                    cyclic = True
                    break
                seen.add(x)
                x = below[x]
            if cyclic:
                break
        if not cyclic:
            return tuple(pick)
    return None


RANDOM_TIERS = ("same_L_V_G_new_pattern", "same_L_V_G_any_pattern", "same_L_G_any_V")


def random_library(templates, rng: random.Random, vcache, exclude_keys=frozenset(), exclude_step_keys=frozenset(),
                   max_vars: int = 4) -> tuple[list, list]:
    """The random-macro control: for every template macro (in order), a uniformly random ACCEPTED macro matched to
    it, drawn by the first tier that has one:
      tier 0  same (L, V, G); step pattern neither in exclude_step_keys (the mined library) nor already drawn;
      tier 1  same (L, V, G); step pattern not already drawn (a mined pattern may recur with a different goal_on);
      tier 2  same (L, G), any V <= max_vars; step pattern not already drawn.
    Within a tier, candidate patterns (pattern_universe) are visited in a seeded random order and the first one that
    validate() accepts and that can carry G goal literals is taken (uniform over accepted patterns of the tier);
    goal_on = random_goal_on(rng, V, G); the full key must not be in exclude_keys (the mined library's keys).
    Small (L, V) classes are tiny (for L = 2, V = 2 only three sound non-identity patterns exist), hence the tiers.
    Returns (library, per-template log)."""
    lib, log, used = [], [], set()
    universes = {}

    def universe(L, V):
        if (L, V) not in universes:
            universes[(L, V)] = pattern_universe(L, V)
        return universes[(L, V)]

    for t in templates:
        L, V, G = t.L, t.nvars, len(t.goal_on)
        stat, pick = Counter(), None
        for tier, name in enumerate(RANDOM_TIERS):
            pool = universe(L, V) if tier < 2 else [p for v in range(1, max_vars + 1) for p in universe(L, v)]
            order = list(pool)
            rng.shuffle(order)
            for steps in order:
                sk = step_key(steps)
                if sk in used or (tier == 0 and sk in exclude_step_keys):
                    continue
                m0 = Macro(steps)
                if G > m0.nvars - 1:
                    stat["too_few_vars_for_G"] += 1
                    continue
                v = vcache(m0)
                if not v["accepted"]:
                    stat[f"rejected_{v['reason']}"] += 1
                    continue
                gon = random_goal_on(rng, m0.nvars, G)
                if gon is None or macro_key(steps, gon) in exclude_keys:
                    stat["no_goal_fragment_or_mined_key"] += 1
                    continue
                pick = Macro(steps, gon, meta={"template": t.key, "tier": name})
                break
            if pick is not None:
                break
        if pick is None:
            raise RuntimeError(f"no random macro for template {t.key} in any tier: {dict(stat)}")
        lib.append(pick)
        used.add(pick.step_key)
        log.append({"template": t.key, "drawn": pick.key, "tier": pick.meta["tier"], "skipped": dict(stat)})
    return lib, log


# ----------------------------------------------------------------------------- search with macros

_ST_KEYS = ("exp", "gen", "calls", "prim", "mkids", "mbind", "mid", "symbad")


def _path(parent, s):
    segs = []
    while parent[s] is not None:
        prev, seq, mi = parent[s]
        segs.append((seq, mi))
        s = prev
    segs.reverse()
    return [a for seq, _ in segs for a in seq], [mi for _, mi in segs]


def _result(B, init, goal, found, reason, plan, edges, st, t0) -> dict:
    plan = plan or []
    ex = B.execute(init, goal, plan) if found else {"success": False}
    uses = Counter(mi for mi in edges if mi >= 0)
    return {"success": bool(ex["success"]), "found": found, "plan": plan, "plan_len": len(plan) if found else None,
            "macro_edges": sum(uses.values()) if found else None,
            "macro_uses": {str(k): v for k, v in sorted(uses.items())} if found else {},
            "expansions": st["exp"], "generated": st["gen"], "h_calls": st["calls"], "prim_steps": st["prim"],
            "macro_children": st["mkids"], "macro_bindings": st["mbind"], "goal_mid_macro": st["mid"],
            "sym_mismatch": st["symbad"], "reason": reason, "seconds": round(time.perf_counter() - t0, 4)}


def _children(B, s, goal, acts, library, gidx, parent, st, check_sym):
    """(new children in generation order, goal state or None); registers parents exactly like search.gbfs."""
    kids = []
    for a in acts:
        if B.applicable(s, a):
            st["prim"] += 1
            ns = B.step(s, a)[0]
            if ns in parent:
                continue
            parent[ns] = (s, (a,), -1)
            st["gen"] += 1
            if B.goal_met(ns, goal):
                return kids, ns
            kids.append(ns)
    if not library:
        return kids, None
    sidx = StateIndex(s)
    for mi, m in enumerate(library):
        for beta in m.bindings(sidx, gidx):
            st["mbind"] += 1
            seq = m.ground(beta)
            cur = s
            for k, a in enumerate(seq):
                if not B.applicable(cur, a):
                    raise MacroError(f"{m.key} with {beta}: step {k} {a} inapplicable")
                cur = B.step(cur, a)[0]
                st["prim"] += 1
                if k < len(seq) - 1 and B.goal_met(cur, goal):
                    parent[cur] = (s, seq[:k + 1], mi)
                    st["gen"] += 1
                    st["mid"] += 1
                    return kids, cur
            if check_sym and m.apply_symbolic(s, beta) != cur:
                st["symbad"] += 1
            if cur in parent:
                continue
            parent[cur] = (s, seq, mi)
            st["gen"] += 1
            st["mkids"] += 1
            if B.goal_met(cur, goal):
                return kids, cur
            kids.append(cur)
    return kids, None


def gbfs_macro(objs, init, goal, h, library=(), max_expansions: int = 10_000, check_sym: bool = True) -> dict:
    """search.gbfs plus macro children (module doc). library=() reproduces search.gbfs exactly."""
    B, _ = BI.load()
    t0 = time.perf_counter()
    hb = S.as_batch(h)
    st = dict.fromkeys(_ST_KEYS, 0)
    if B.goal_met(init, goal):
        st["gen"] = 1
        return _result(B, init, goal, True, "goal", [], [], st, t0)
    acts = B.ground_actions(list(objs))
    gidx = goal_index(goal)
    parent = {init: None}
    st["calls"], st["gen"] = 1, 1
    tie = 0
    h0 = hb([init], goal)[0]
    heap = [(h0, 0, init)] if h0 < math.inf else []
    while heap:
        if st["exp"] >= max_expansions:
            return _result(B, init, goal, False, "budget", None, [], st, t0)
        _, _, s = heapq.heappop(heap)
        st["exp"] += 1
        kids, g = _children(B, s, goal, acts, library, gidx, parent, st, check_sym)
        if g is not None:
            plan, edges = _path(parent, g)
            return _result(B, init, goal, True, "goal", plan, edges, st, t0)
        if kids:
            hs = hb(kids, goal)
            st["calls"] += len(kids)
            for ns, v in zip(kids, hs):
                if v < math.inf:
                    tie += 1
                    heapq.heappush(heap, (v, tie, ns))
    return _result(B, init, goal, False, "exhausted", None, [], st, t0)


def beam_macro(objs, init, goal, h, library=(), width: int = 4, max_expansions: int = 10_000,
               check_sym: bool = True) -> dict:
    """search.beam_search plus macro children (module doc). library=() reproduces search.beam_search exactly."""
    B, _ = BI.load()
    t0 = time.perf_counter()
    hb = S.as_batch(h)
    st = dict.fromkeys(_ST_KEYS, 0)
    if B.goal_met(init, goal):
        st["gen"] = 1
        return _result(B, init, goal, True, "goal", [], [], st, t0)
    acts = B.ground_actions(list(objs))
    gidx = goal_index(goal)
    parent = {init: None}
    beam = [init]
    st["gen"] = 1
    while beam:
        kids = []
        for s in beam:
            if st["exp"] >= max_expansions:
                return _result(B, init, goal, False, "budget", None, [], st, t0)
            st["exp"] += 1
            new, g = _children(B, s, goal, acts, library, gidx, parent, st, check_sym)
            if g is not None:
                plan, edges = _path(parent, g)
                return _result(B, init, goal, True, "goal", plan, edges, st, t0)
            kids += new
        if not kids:
            break
        hs = hb(kids, goal)
        st["calls"] += len(kids)
        ranked = sorted((v, i) for i, v in enumerate(hs) if v < math.inf)
        beam = [kids[i] for _, i in ranked[:width]]
    return _result(B, init, goal, False, "exhausted", None, [], st, t0)


def library_json(lib) -> list:
    return [m.to_json() for m in lib]


def library_from_json(rows) -> list:
    return [Macro.from_json(d) for d in rows]
