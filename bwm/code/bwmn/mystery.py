"""Mystery Blocksworld under BWM contracts: arbitrary lifted STRIPS domains whose rules are READ FROM THE PROMPT.

PlanBench Mystery = Blocksworld with deceptive renaming (clear->province, ontable->planet, handempty->harmony,
holding->pain, on->craves; pick-up->attack, put-down->succumb, stack->overcome, unstack->feast). A model that
memorised Blocksworld words cannot transfer; it has to read the action schemas given in the prompt.

Conventions (shared with bwmn/blocks.py): state = frozenset of ground facts (pred, *objs); ground action =
(name, *objs); an inapplicable action is illegal and leaves the state unchanged; effects = delete then add (PDDL);
distinct parameters denote distinct objects (in Blocksworld/Mystery a repeated-object action is never applicable in a
reachable state, so this agrees with VAL there); chat-frame prompts with canonical JSON user content; JSON targets.
Prompt rule format (identical for training and for PlanBench Mystery evaluation), one string per action, sorted:
    name(X,Y): pre[a X, b X Y] add[...] del[...]
Typing: single object type; every atom is arity-consistent, uses only the schema's parameters (no repeated variable)
and every parameter occurs in a precondition.

Training regimes (our generator only; PlanBench instances are test-only):
  A  Blocksworld structure with fresh nonsense names for predicates/actions/objects per instance and random argument
     order for the binary predicate/actions; instances isomorphic (up to block renaming) to any PlanBench
     Blocksworld/Blocksworld-Hard (<= 7 blocks)/Mystery instance are excluded by blocks.signature, and so are proposer
     rows whose intermediate (state, goal) is such a renaming (suffix-signature guard, B-02).
  B  random lifted STRIPS domains (2-5 schemas, 1-2 params, 2-4 predicates of arity 0-2, del <= pre, add disjoint
     from pre), 3-6 objects, BFS-optimal plan length 2-12. Domains isomorphic to Blocksworld - or to any
     predicate/action projection of it - are rejected via a canonical schema hash (renaming of predicates, actions,
     parameters and argument order of binary predicates); so is any domain that CONTAINS a Blocksworld fragment with
     >= 3 predicates and >= 3 actions (a projection of the domain equal to such a projection of Blocksworld).
Every training predicate/action name is a pronounceable nonsense word that occurs nowhere in the vendored PlanBench
assets, contains no deceptive Mystery word and no 4+ letter Blocksworld word as a substring, and (given a tokenizer)
is not a single token. Objects: nonsense words, or - in 2 of every 5 instance slots (40%) - single letters a..o as in
PlanBench. Every instance slot has its own RNG (instance_seed), so data can be generated, resumed or sharded per slot.
The exact executor / BFS / VAL are data-collection and evaluation instruments only.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from . import blocks as B
from . import stats as S

BENCH = B.BENCH
MYSTERY_DIR = BENCH / "instances_mystery" / "mystery"
MYSTERY_DOMAIN_PDDL = MYSTERY_DIR / "generated_domain.pddl"
MYSTERY_SUBDIRS = ("generated_basic", "generated_basic_3")
HARD_DIR = BENCH / "blocksworld_hard" / "generated"
VARS = tuple("XYZUVW")
LETTERS = tuple(B.NAMES)  # single-letter object names a..o (PlanBench uses a..e)
canonical = B.canonical
render_chat = B.render_chat


# ----------------------------------------------------------------------------- domain representation

@dataclass(frozen=True)
class Schema:
    params: tuple  # ("X", "Y")
    pre: tuple     # sorted atoms, atom = (pred, *vars)
    add: tuple
    dele: tuple


@dataclass(eq=False)
class Domain:
    name: str
    preds: dict    # pred -> arity
    actions: dict  # action name -> Schema
    _g: dict = field(default_factory=dict, repr=False)  # grounding cache

    def to_json(self) -> dict:
        L = lambda xs: [list(a) for a in xs]
        return {"name": self.name, "predicates": dict(sorted(self.preds.items())),
                "actions": {n: {"params": list(s.params), "pre": L(s.pre), "add": L(s.add), "del": L(s.dele)}
                            for n, s in sorted(self.actions.items())}}

    @staticmethod
    def from_json(d: dict) -> "Domain":
        T = lambda xs: tuple(sorted(tuple(a) for a in xs))
        return Domain(d["name"], dict(d["predicates"]),
                      {n: Schema(tuple(s["params"]), T(s["pre"]), T(s["add"]), T(s["del"]))
                       for n, s in d["actions"].items()})

    def same_as(self, other: "Domain") -> bool:
        return self.preds == other.preds and self.actions == other.actions


# ----------------------------------------------------------------------------- PDDL I/O (generic STRIPS)

def _sexp(text: str) -> list:
    text = re.sub(r";[^\n]*", "", text).lower()
    stack = [[]]
    for t in re.findall(r"\(|\)|[^\s()]+", text):
        if t == "(":
            stack.append([])
        elif t == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(t)
    return next(x for x in stack[0] if isinstance(x, list) and x and x[0] == "define")


def _conj(f) -> list:
    if not f:
        return []
    if f[0] == "and":
        return [g for x in f[1:] for g in _conj(x)]
    return [f]


def _untyped(lst) -> list:
    out, skip = [], False
    for t in lst:
        if skip:
            skip = False
        elif t == "-":
            skip = True
        else:
            out.append(t)
    return out


def parse_domain(path) -> Domain:
    form = _sexp(Path(path).read_text())
    preds, acts = {}, {}
    for sec in form[2:]:
        if sec[0] == ":predicates":
            for p in sec[1:]:
                preds[p[0]] = len(_untyped(p[1:]))
        elif sec[0] == ":action":
            kv = dict(zip(sec[2::2], sec[3::2]))
            ps = _untyped(kv.get(":parameters", []))
            vm = {v: VARS[i] for i, v in enumerate(ps)}
            at = lambda a: (a[0],) + tuple(vm[x] for x in a[1:])
            pre = {at(a) for a in _conj(kv.get(":precondition", []))}
            add, dele = set(), set()
            for e in _conj(kv.get(":effect", [])):
                if e[0] == "not":
                    dele.add(at(e[1]))
                else:
                    add.add(at(e))
            acts[sec[1]] = Schema(VARS[:len(ps)], tuple(sorted(pre)), tuple(sorted(add)), tuple(sorted(dele)))
    return Domain(form[1][1], preds, acts)


def parse_problem(path):
    form = _sexp(Path(path).read_text())
    objs, init, goal = [], [], []
    for sec in form[2:]:
        if sec[0] == ":objects":
            objs = _untyped(sec[1:])
        elif sec[0] == ":init":
            init = [tuple(a) for a in sec[1:]]
        elif sec[0] == ":goal":
            goal = [tuple(a) for a in _conj(sec[1])]
            if any(g[0] == "not" for g in goal):
                raise ValueError("negative goals unsupported")
    return sorted(objs), frozenset(init), frozenset(goal)


def write_problem(dom: Domain, objs, init, goal, name="bwm") -> str:
    f = lambda fs: "\n".join("(" + " ".join(x) + ")" for x in sorted(fs))
    return (f"(define (problem {name})\n(:domain {dom.name})\n(:objects {' '.join(objs)} )\n(:init\n{f(init)}\n)\n"
            f"(:goal\n(and\n{f(goal)})\n)\n)\n")


def write_domain(dom: Domain) -> str:
    """Untyped STRIPS PDDL of `dom` (parameters ?x ?y ...), for VAL; parse_domain(write_domain(d)).same_as(d)."""
    v = lambda x: "?" + x.lower()
    at = lambda a: "(" + " ".join((a[0],) + tuple(v(x) for x in a[1:])) + ")"
    preds = " ".join("(" + " ".join([p] + [v(x) for x in VARS[:k]]) + ")" for p, k in sorted(dom.preds.items()))
    acts = []
    for n, sc in sorted(dom.actions.items()):
        eff = [at(a) for a in sc.add] + [f"(not {at(a)})" for a in sc.dele]
        acts.append(f"(:action {n}\n  :parameters ({' '.join(v(x) for x in sc.params)})\n"
                    f"  :precondition (and {' '.join(at(a) for a in sc.pre)})\n  :effect (and {' '.join(eff)}))")
    return f"(define (domain {dom.name})\n(:requirements :strips)\n(:predicates {preds})\n" + "\n".join(acts) + "\n)\n"


@lru_cache(maxsize=1)
def bw_domain() -> Domain:
    return parse_domain(B.DOMAIN_PDDL)


@lru_cache(maxsize=1)
def mystery_domain() -> Domain:
    return parse_domain(MYSTERY_DOMAIN_PDDL)


# ----------------------------------------------------------------------------- executor (any STRIPS domain)

def _subst(atoms, b) -> frozenset:
    return frozenset((a[0],) + tuple(b[x] for x in a[1:]) for a in atoms)


def ground(dom: Domain, a: tuple):
    """(pre, add, del) of a ground action, or None if malformed (unknown name, wrong arity, repeated objects)."""
    g = dom._g.get(a)
    if g is None:
        sc = dom.actions.get(a[0]) if a else None
        if sc is None or len(a) - 1 != len(sc.params) or len(set(a[1:])) != len(a) - 1:
            g = False
        else:
            b = dict(zip(sc.params, a[1:]))
            g = (_subst(sc.pre, b), _subst(sc.add, b), _subst(sc.dele, b))
        dom._g[a] = g
    return g or None


def applicable(dom: Domain, s: frozenset, a: tuple) -> bool:
    g = ground(dom, a)
    return g is not None and g[0] <= s


def step(dom: Domain, s: frozenset, a: tuple) -> tuple[frozenset, bool]:
    g = ground(dom, a)
    if g is None or not g[0] <= s:
        return s, False
    return (s - g[2]) | g[1], True


def ground_actions(dom: Domain, objs) -> list[tuple]:
    objs = sorted(objs)
    return [(n,) + p for n in sorted(dom.actions) for p in itertools.permutations(objs, len(dom.actions[n].params))]


def all_facts(dom: Domain, objs) -> list[tuple]:
    objs = sorted(objs)
    return [(p,) + t for p in sorted(dom.preds) for t in itertools.permutations(objs, dom.preds[p])]


def _compiled(dom: Domain, objs) -> list:
    key = ("__compiled__", tuple(sorted(objs)))
    c = dom._g.get(key)
    if c is None:
        c = [(a,) + ground(dom, a) for a in ground_actions(dom, objs)]
        dom._g[key] = c
    return c


def goal_met(s: frozenset, goal: frozenset) -> bool:
    return goal <= s


def bfs_plan(dom: Domain, init: frozenset, goal: frozenset, objs, max_states: int = 200000):
    """Shortest plan or None. Teacher/oracle only."""
    if goal <= init:
        return []
    acts = _compiled(dom, objs)
    prev = {init: None}
    q = deque([init])
    while q and len(prev) < max_states:
        s = q.popleft()
        for a, pre, add, dele in acts:
            if not pre <= s:
                continue
            ns = (s - dele) | add
            if ns in prev:
                continue
            prev[ns] = (s, a)
            if goal <= ns:
                plan, cur = [], ns
                while prev[cur] is not None:
                    cur, pa = prev[cur]
                    plan.append(pa)
                return plan[::-1]
            q.append(ns)
    return None


def execute(dom: Domain, init: frozenset, goal: frozenset, plan) -> dict:
    """Exact execution; success = every action applicable and goal met at the end (VAL semantics)."""
    s = init
    for i, a in enumerate(plan):
        if not applicable(dom, s, a):
            return {"success": False, "valid": False, "fail_step": i, "final": s}
        s, _ = step(dom, s, a)
    return {"success": goal <= s, "valid": True, "fail_step": None, "final": s}


def val_check_mystery(problem_path, plan) -> bool:
    return B.val_check(Path(problem_path), plan, MYSTERY_DOMAIN_PDDL)


# ----------------------------------------------------------------------------- renaming / isomorphism

def _conv_schema(sc: Schema, pm: dict, flip, p: tuple) -> Schema:
    """New param j = old param p[j]; predicates renamed by pm; binary predicates in `flip` get swapped args."""
    vm = {sc.params[p[j]]: VARS[j] for j in range(len(p))}

    def conv(atoms):
        out = []
        for a in atoms:
            args = [vm.get(x, x) for x in a[1:]]
            if a[0] in flip:
                args.reverse()
            out.append((pm[a[0]],) + tuple(args))
        return tuple(sorted(out))

    return Schema(VARS[:len(p)], conv(sc.pre), conv(sc.add), conv(sc.dele))


@dataclass
class Renaming:
    pred: dict
    act: dict
    obj: dict
    flip: frozenset = frozenset()
    aperm: dict = field(default_factory=dict)

    def fact(self, f: tuple) -> tuple:
        args = [self.obj.get(x, x) for x in f[1:]]
        if f[0] in self.flip:
            args.reverse()
        return (self.pred[f[0]],) + tuple(args)

    def facts(self, fs) -> frozenset:
        return frozenset(self.fact(f) for f in fs)

    def action(self, a: tuple) -> tuple:
        p = self.aperm.get(a[0], tuple(range(len(a) - 1)))
        return (self.act[a[0]],) + tuple(self.obj.get(a[1 + p[j]], a[1 + p[j]]) for j in range(len(p)))

    def domain(self, dom: Domain, name: str | None = None) -> Domain:
        acts = {self.act[n]: _conv_schema(sc, self.pred, self.flip, self.aperm.get(n, tuple(range(len(sc.params)))))
                for n, sc in dom.actions.items()}
        return Domain(name or dom.name, {self.pred[p]: k for p, k in dom.preds.items()}, acts)

    def inverse(self) -> "Renaming":
        """R.inverse().facts(R.facts(fs)) == fs and R.inverse().action(R.action(a)) == a (maps must be injective)."""
        for m in (self.pred, self.act, self.obj):
            assert len(set(m.values())) == len(m), "renaming is not injective"
        inv = lambda m: {v: k for k, v in m.items()}
        ap = {}
        for a, p in self.aperm.items():
            q = [0] * len(p)
            for j, k in enumerate(p):
                q[k] = j
            ap[self.act[a]] = tuple(q)
        return Renaming(inv(self.pred), inv(self.act), inv(self.obj), frozenset(self.pred[p] for p in self.flip), ap)

    def to_json(self) -> dict:
        return {"pred": dict(sorted(self.pred.items())), "act": dict(sorted(self.act.items())),
                "obj": dict(sorted(self.obj.items())), "flip": sorted(self.flip),
                "aperm": {a: list(p) for a, p in sorted(self.aperm.items())}}

    @staticmethod
    def from_json(d: dict) -> "Renaming":
        return Renaming(dict(d["pred"]), dict(d["act"]), dict(d["obj"]), frozenset(d["flip"]),
                        {a: tuple(p) for a, p in d["aperm"].items()})


def find_isomorphism(src: Domain, dst: Domain) -> Renaming | None:
    """Renaming R (objects identity) with R.domain(src) == dst structurally, or None."""
    if sorted(src.preds.values()) != sorted(dst.preds.values()) or len(src.actions) != len(dst.actions):
        return None
    sp = sorted(src.preds)
    bins = [p for p in sp if src.preds[p] == 2]
    for perm in itertools.permutations(sorted(dst.preds)):
        pm = dict(zip(sp, perm))
        if any(src.preds[p] != dst.preds[pm[p]] for p in sp):
            continue
        for fl in itertools.product((False, True), repeat=len(bins)):
            flip = frozenset(p for p, f in zip(bins, fl) if f)
            am, ap, used = {}, {}, set()
            for an in sorted(src.actions):
                sc, hit = src.actions[an], None
                for bn in sorted(dst.actions):
                    if bn in used or len(dst.actions[bn].params) != len(sc.params):
                        continue
                    for p in itertools.permutations(range(len(sc.params))):
                        if _conv_schema(sc, pm, flip, p) == dst.actions[bn]:
                            hit = (bn, p)
                            break
                    if hit:
                        break
                if hit is None:
                    break
                am[an], ap[an] = hit
                used.add(hit[0])
            else:
                return Renaming(pm, am, {}, flip, ap)
    return None


def _enc_action(sc: Schema, pm: dict, flip) -> tuple:
    best = None
    for p in itertools.permutations(range(len(sc.params))):
        vm = {sc.params[p[j]]: j for j in range(len(p))}

        def enc(atoms):
            out = []
            for a in atoms:
                args = [vm[x] for x in a[1:]]
                if a[0] in flip:
                    args.reverse()
                out.append((a[0] if pm is None else pm[a[0]],) + tuple(args))
            return tuple(sorted(out))

        e = (len(p), enc(sc.pre), enc(sc.add), enc(sc.dele))
        if best is None or e < best:
            best = e
    return best


def schema_key(dom: Domain) -> tuple:
    """Canonical structure (predicate arities + multiset-free set of action pre/add/del encodings), minimised over
    predicate renamings, argument-order flips of binary predicates and per-action parameter permutations. Names are
    discarded; duplicate actions collapse."""
    preds = sorted(dom.preds)
    bins = [p for p in preds if dom.preds[p] == 2]
    best = None
    for perm in itertools.permutations(range(len(preds))):
        pm = dict(zip(preds, perm))
        ar = tuple(dom.preds[p] for p in sorted(preds, key=pm.get))
        if best is not None and ar > best[0]:
            continue
        for fl in itertools.product((False, True), repeat=len(bins)):
            flip = {p for p, f in zip(bins, fl) if f}
            key = (ar, tuple(sorted({_enc_action(sc, pm, flip) for sc in dom.actions.values()})))
            if best is None or key < best:
                best = key
    return best


def schema_hash(dom: Domain) -> str:
    return hashlib.sha256(repr(schema_key(dom)).encode()).hexdigest()[:20]


def _project(dom: Domain, keep: set, acts) -> Domain | None:
    new = {}
    for an in acts:
        sc = dom.actions[an]
        f = lambda xs: tuple(a for a in xs if a[0] in keep)
        pre, add, dele = f(sc.pre), f(sc.add), f(sc.dele)
        if not add and not dele:
            continue
        used = [v for v in sc.params if any(v in a[1:] for a in pre + add + dele)]
        vm = {v: VARS[i] for i, v in enumerate(used)}
        g = lambda xs: tuple(sorted((a[0],) + tuple(vm[x] for x in a[1:]) for a in xs))
        new[an] = Schema(VARS[:len(used)], g(pre), g(add), g(dele))
    return Domain("bw-projection", {p: dom.preds[p] for p in keep}, new) if new else None


@lru_cache(maxsize=1)
def bw_hash() -> str:
    return schema_hash(bw_domain())


@lru_cache(maxsize=1)
def bw_family_hashes() -> frozenset:
    """Hashes of Blocksworld and of every projection of it onto a predicate subset x action subset."""
    bw = bw_domain()
    P, A = sorted(bw.preds), sorted(bw.actions)
    out = set()
    for r in range(1, len(P) + 1):
        for keep in itertools.combinations(P, r):
            for m in range(1, len(A) + 1):
                for acts in itertools.combinations(A, m):
                    d = _project(bw, set(keep), acts)
                    if d is not None:
                        out.add(schema_hash(d))
    return frozenset(out)


def is_blocksworld(dom: Domain) -> bool:
    return schema_hash(dom) == bw_hash()


def is_blocksworld_family(dom: Domain) -> bool:
    return schema_hash(dom) in bw_family_hashes()


FRAGMENT_MIN = 3  # a Blocksworld fragment worth excluding has >= 3 predicates and >= 3 actions


def _fragment_inv(d: Domain) -> tuple:
    """Cheap function of schema_key (arity multiset, set of per-action atom counts): a prefilter for the hash."""
    return (tuple(sorted(d.preds.values())),
            frozenset((len(sc.params), len(sc.pre), len(sc.add), len(sc.dele)) for sc in d.actions.values()))


def _fragments(dom: Domain, k_min: int = FRAGMENT_MIN):
    """Projections of `dom` onto predicate subsets (>= k_min, every kept predicate used) x action subsets (>= k_min
    actions that still have an effect)."""
    P, A = sorted(dom.preds), sorted(dom.actions)
    for r in range(k_min, len(P) + 1):
        for keep in itertools.combinations(P, r):
            for m in range(k_min, len(A) + 1):
                for acts in itertools.combinations(A, m):
                    d = _project(dom, set(keep), acts)
                    if d is None or len(d.actions) < k_min:
                        continue
                    if {a[0] for sc in d.actions.values() for a in sc.pre + sc.add + sc.dele} != set(keep):
                        continue
                    yield d


@lru_cache(maxsize=1)
def bw_fragment_hashes() -> dict:
    """_fragment_inv -> schema hashes of every Blocksworld fragment with >= 3 predicates and >= 3 actions."""
    out = {}
    for d in _fragments(bw_domain()):
        out.setdefault(_fragment_inv(d), set()).add(schema_hash(d))
    return {k: frozenset(v) for k, v in out.items()}


def contains_blocksworld_fragment(dom: Domain) -> bool:
    """True if some projection of `dom` onto >= 3 predicates x >= 3 actions equals, up to renaming (predicates,
    actions, parameters, binary argument order), a projection of Blocksworld onto >= 3 predicates x >= 3 actions.
    Extra predicates / actions / atoms on other predicates do not hide the fragment (containment, not equality)."""
    frag = bw_fragment_hashes()
    if len(dom.preds) < FRAGMENT_MIN or len(dom.actions) < FRAGMENT_MIN:
        return False
    for d in _fragments(dom):
        hs = frag.get(_fragment_inv(d))
        if hs and schema_hash(d) in hs:
            return True
    return False


# ----------------------------------------------------------------------------- vocabulary

FORMAT_WORDS = frozenset(("pre add del goal state states action actions steps legal outcome success failure rules "
                          "objects object true false null im start end system user assistant think json plan").split())
BW_WORDS = frozenset(("pick up put down stack unstack clear ontable table handempty hand empty holding hold on block "
                      "blocks top arm tower").split())
_VOCAB_SUFFIXES = (".pddl", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv", ".tsv", ".py")
_WORDS_MEMO: dict = {}


def vocab_sources() -> list[Path]:
    """Every text asset of the vendored PlanBench (Blocksworld, Mystery, published prompts/responses, any randomized-
    vocabulary file); the VAL tool sources are excluded."""
    return [p for p in sorted(BENCH.rglob("*")) if p.is_file() and p.suffix.lower() in _VOCAB_SUFFIXES
            and "VAL" not in p.relative_to(BENCH).parts]


def randomized_vocab_files() -> list[Path]:
    return [p for p in vocab_sources() if re.search(r"random|obfusc|shuffl", str(p.relative_to(BENCH)), re.I)
            and "results" not in p.relative_to(BENCH).parts]


def extract_words(text: str) -> set:
    t = re.sub(r"\\[nrt]", " ", text.lower())
    return set(re.findall(r"[a-z]+", t))


def forbidden_words(cache: Path | None = None) -> frozenset:
    """All alphabetic runs found anywhere in the PlanBench assets + format/Blocksworld words. Optional disk cache
    keyed by a fingerprint of the source files."""
    files = vocab_sources()
    fp = hashlib.sha256(json.dumps([(str(p.relative_to(BENCH)), p.stat().st_size, p.stat().st_mtime_ns)
                                    for p in files]).encode()).hexdigest()
    if fp in _WORDS_MEMO:
        return _WORDS_MEMO[fp]
    if cache is not None and Path(cache).exists():
        d = json.loads(Path(cache).read_text())
        if d.get("fingerprint") == fp:
            _WORDS_MEMO[fp] = frozenset(d["words"])
            return _WORDS_MEMO[fp]
    W = set(FORMAT_WORDS | BW_WORDS)
    for p in files:
        W |= extract_words(p.read_text(encoding="utf-8", errors="ignore"))
    W = frozenset(W)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps({"fingerprint": fp, "n_files": len(files), "n_words": len(W),
                                           "words": sorted(W)}))
    _WORDS_MEMO[fp] = W
    return W


_ON = "b d f g k l m n p r s t v z br dr fr gr kr pr tr bl fl gl kl pl sl sk sp st sn sm zh sh ch th j w y h".split()
_NU = "a e i o u ai au ei oi ou oo ee ia io ua".split()
_CO = ["", "", "", "", "n", "r", "l", "s", "k", "m", "x", "th", "rn", "nd", "sk", "lt", "p", "g", "b", "d", "z", "v",
       "sh"]


# PDDL keywords VAL must not see as names (most are PlanBench words already; listed so write_domain stays parseable)
PDDL_WORDS = frozenset(("define domain problem requirements types constants predicates action parameters "
                        "precondition effect and or not imply exists forall when either object objects init goal "
                        "strips typing equality at over start end all number").split())


@lru_cache(maxsize=1)
def banned_substrings() -> tuple:
    """Deceptive Mystery names and every 4+ letter Blocksworld word: no generated name may CONTAIN one ('zhand',
    'spainor', 'tabled' are rejected although they are not PlanBench words themselves)."""
    myst, bw = mystery_domain(), bw_domain()
    words = set(myst.preds) | set(myst.actions)
    for w in BW_WORDS | set(bw.preds) | set(bw.actions):
        words |= {x for x in w.split("-") if len(x) >= 4}
    return tuple(sorted(words))


class NameGen:
    """Pronounceable nonsense words; never returns a forbidden word, a PDDL keyword, a word shorter than 2 letters, a
    word containing a banned substring (banned_substrings()) or - if `single_token(w)` is given - a word the LM
    tokenizer encodes as one token (bare or after a space): single-token words are English (or other real) words such
    as 'spoon', 'hound', 'drip' that the PlanBench vocabulary does not happen to contain."""

    def __init__(self, rng: random.Random, forbidden: frozenset, banned=None, single_token=None):
        if not forbidden:
            raise ValueError("NameGen needs the forbidden vocabulary")
        self.rng = rng
        self.forbidden = forbidden if isinstance(forbidden, frozenset) else frozenset(forbidden)
        self.banned = tuple(banned_substrings() if banned is None else banned)
        self.single_token = single_token

    def ok(self, w: str) -> bool:
        return (2 <= len(w) <= 12 and w.isalpha() and w not in self.forbidden and w not in PDDL_WORDS
                and not any(b in w for b in self.banned)
                and not (self.single_token is not None and self.single_token(w)))

    def word(self, min_syl: int = 1, max_syl: int = 3) -> str:
        r = self.rng
        while True:
            k = r.randint(min_syl, max_syl)
            w = "".join((r.choice(_ON) if (i > 0 or r.random() < 0.85) else "") + r.choice(_NU) + r.choice(_CO)
                        for i in range(k))
            if self.ok(w):
                return w

    def sample(self, k: int, min_syl: int = 1, max_syl: int = 3, exclude=()) -> list[str]:
        out, seen = [], set(exclude)
        while len(out) < k:
            w = self.word(min_syl, max_syl)
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out


def instance_names(inst: dict) -> list[str]:
    d = inst["domain"]
    return list(d.preds) + list(d.actions) + list(inst["objs"])


# ----------------------------------------------------------------------------- test exclusion (instance level)

@lru_cache(maxsize=1)
def mystery_to_bw() -> Renaming:
    r = find_isomorphism(mystery_domain(), bw_domain())
    assert r is not None, "Mystery domain is not isomorphic to Blocksworld?"
    return r


def mystery_instances(subdirs=MYSTERY_SUBDIRS) -> list[dict]:
    dom, out = mystery_domain(), []
    for sub in subdirs:
        files = sorted((MYSTERY_DIR / sub).glob("instance-*.pddl"), key=lambda q: int(re.findall(r"\d+", q.stem)[-1]))
        for p in files:
            objs, init, goal = parse_problem(p)
            out.append({"id": f"{sub}/{p.stem}", "path": str(p), "domain": dom, "objs": objs, "init": init,
                        "goal": goal})
    return out


def mystery_test_signatures() -> set:
    R = mystery_to_bw()
    return {B.signature(it["objs"], R.facts(it["init"]), R.facts(it["goal"])) for it in mystery_instances()
            if len(it["objs"]) <= 7}


def hard_test_signatures(max_objs: int = 7) -> set:
    """Blocksworld-Hard instances with <= max_objs blocks (blocks.test_signatures covers basic/basic_3/unsolvable
    only). Parse failures raise: a test file that cannot be read must not silently drop out of the exclusion."""
    sigs = set()
    for p in sorted(HARD_DIR.glob("*.pddl")):
        objs, init, goal = parse_problem(p)
        if len(objs) <= max_objs:
            sigs.add(B.signature(objs, init, goal))
    return sigs


def test_signatures() -> set:
    """Blocksworld-space signatures of every PlanBench Blocksworld (incl. unsolvable), Blocksworld-Hard (<= 7 blocks)
    and Mystery instance."""
    return B.test_signatures() | hard_test_signatures() | mystery_test_signatures()


# ----------------------------------------------------------------------------- evaluation (PlanBench Mystery)

def mystery_eval_set(subdirs=MYSTERY_SUBDIRS) -> list[dict]:
    """PlanBench Mystery instances + Blocksworld-space renaming signature `sig` (the de-duplication cluster: 603
    instances are 262 distinct problems) and `n_blocks`."""
    R, out = mystery_to_bw(), mystery_instances(subdirs)
    for it in out:
        it["sig"] = B.signature(it["objs"], R.facts(it["init"]), R.facts(it["goal"]))
        it["n_blocks"] = len(it["objs"])
    return out


def _accuracy(sigs: list, succ: list, n_boot: int, seed: int) -> dict:
    cl = {}
    for s, x in zip(sigs, succ):
        cl.setdefault(s, []).append(float(x))
    groups = [cl[s] for s in sorted(cl)]
    n, k = len(succ), sum(float(x) for x in succ)
    means = [sum(g) / len(g) for g in groups]
    rng, per, ded = random.Random(seed), [], []
    for _ in range(n_boot):  # cluster bootstrap: resample distinct problems with all their renamings
        pick = [rng.randrange(len(groups)) for _ in groups]
        per.append(sum(sum(groups[j]) for j in pick) / sum(len(groups[j]) for j in pick))
        ded.append(sum(means[j] for j in pick) / len(pick))
    q = lambda xs: [xs[int(0.025 * n_boot)], xs[int(0.975 * n_boot) - 1]] if n_boot else None
    integral = all(float(x) in (0.0, 1.0) for x in succ)
    return {"per_instance": {"n": n, "successes": k, "acc": k / n,
                             "wilson95_iid": list(S.wilson(int(k), n)) if integral else None,
                             "cluster_boot95": q(sorted(per))},
            "dedup_signature": {"n_distinct": len(groups), "acc": sum(means) / len(groups),
                                "solved_all_renamings": sum(m == 1.0 for m in means),
                                "solved_any_renaming": sum(m > 0.0 for m in means),
                                "cluster_boot95": q(sorted(ded))}}


def eval_report(items: list[dict], success, n_boot: int = 2000, seed: int = 0) -> dict:
    """Accuracy (a) per instance and (b) de-duplicated by renaming signature - each distinct problem scores the mean
    success over its renamings and weighs 1 - overall and by block count. items: dicts with "sig" and "objs" (e.g.
    mystery_eval_set()); success: 0/1 (or fractions) aligned with items. CIs: Wilson (instances as i.i.d.; optimistic)
    and a bootstrap that resamples signature clusters (valid for both)."""
    assert len(items) == len(success) and items
    sigs = [it["sig"] for it in items]
    by = {}
    for i, it in enumerate(items):
        by.setdefault(len(it["objs"]), []).append(i)
    return {"all": _accuracy(sigs, list(success), n_boot, seed),
            "by_blocks": {nb: _accuracy([sigs[i] for i in ix], [success[i] for i in ix], n_boot, seed)
                          for nb, ix in sorted(by.items())}}


# ----------------------------------------------------------------------------- Regime A (Blocksworld, fresh names)

def regime_a(rng: random.Random, ng: NameGen, test_sigs: set, n_blocks: int | None = None,
             stats=None, letters: bool = False) -> dict | None:
    """letters: objects are single letters a..o (PlanBench style) instead of nonsense words."""
    n = n_blocks or rng.choice([3, 4, 4, 5, 5, 5, 6])
    g = B.generate(rng, n)
    why = None
    if g is None:
        why = "generate"
    else:
        objs, init, goal = g
        sig = B.signature(objs, init, goal)
        if sig in test_sigs:
            why = "planbench_signature"
        else:
            plan = B.bfs_plan(init, goal, objs)
            if not plan:
                why = "no_plan"
    if why:
        if stats is not None:
            stats[why] += 1
        return None
    bw = bw_domain()
    P, A = sorted(bw.preds), sorted(bw.actions)
    pn = ng.sample(len(P), 1, 3)
    an = ng.sample(len(A), 1, 3, exclude=pn)
    on = rng.sample(LETTERS, n) if letters else ng.sample(n, 1, 2, exclude=pn + an)
    R = Renaming(dict(zip(P, pn)), dict(zip(A, an)), dict(zip(objs, on)),
                 frozenset(p for p in P if bw.preds[p] == 2 and rng.random() < 0.5),
                 {a: tuple(rng.sample(range(k), k)) for a in A for k in [len(bw.actions[a].params)]})
    return {"regime": "A", "domain": R.domain(bw, "regime-a"), "objs": sorted(on), "init": R.facts(init),
            "goal": R.facts(goal), "plan": [R.action(a) for a in plan], "sig": sig, "n_blocks": n,
            "bw": {"objs": objs, "init": init, "goal": goal, "plan": plan}, "renaming": R}


# ----------------------------------------------------------------------------- Regime B (random STRIPS domains)

def _atoms_over(preds: dict, params: tuple) -> list[tuple]:
    out = []
    for p in sorted(preds):
        k = preds[p]
        if k == 0:
            out.append((p,))
        elif k == 1:
            out += [(p, v) for v in params]
        elif k == 2 and len(params) >= 2:
            out += [(p,) + t for t in itertools.permutations(params, 2)]
    return out


def _local_key(sc: Schema) -> tuple:
    """Schema identity up to parameter renaming (predicate names kept)."""
    return _enc_action(sc, None, ())


def domain_rejection(dom: Domain) -> str | None:
    """None if `dom` is a valid Regime-B domain, else the reason. Blocksworld-family check first; the (costlier)
    Blocksworld-fragment containment check last, on otherwise valid domains."""
    if is_blocksworld_family(dom):
        return "blocksworld_family"
    if not 2 <= len(dom.preds) <= 4:
        return "n_predicates"
    if not 2 <= len(dom.actions) <= 5:
        return "n_actions"
    if any(k > 2 for k in dom.preds.values()):
        return "arity"
    used, keys = set(), set()
    for sc in dom.actions.values():
        if not 1 <= len(sc.params) <= 2:
            return "n_params"
        atoms = sc.pre + sc.add + sc.dele
        if any(a[0] not in dom.preds or len(a) - 1 != dom.preds[a[0]] or len(set(a[1:])) != len(a) - 1
               or any(v not in sc.params for v in a[1:]) for a in atoms):
            return "typing"
        if any(not any(v in a[1:] for a in sc.pre) for v in sc.params):
            return "unbound_param"
        if not set(sc.dele) <= set(sc.pre) or set(sc.add) & set(sc.pre):
            return "effects"
        if not sc.add and not sc.dele:
            return "no_effect"
        used |= {a[0] for a in atoms}
        k = _local_key(sc)
        if k in keys:
            return "duplicate_action"
        keys.add(k)
    if used != set(dom.preds):
        return "unused_predicate"
    if not any(sc.add for sc in dom.actions.values()):
        return "no_add"
    if contains_blocksworld_fragment(dom):
        return "blocksworld_fragment"
    return None


def random_domain(rng: random.Random, ng: NameGen, stats=None) -> Domain | None:
    n_pred, n_act = rng.randint(2, 4), rng.randint(2, 5)
    ar = [rng.choices((0, 1, 2), (0.15, 0.45, 0.40))[0] for _ in range(n_pred)]
    if ar.count(0) > 1 or max(ar) == 0:
        if stats is not None:
            stats["arity_mix"] += 1
        return None
    names = ng.sample(n_pred + n_act, 1, 3)
    preds = dict(zip(names[:n_pred], ar))
    acts = {}
    for an in names[n_pred:]:
        params = VARS[:rng.randint(1, 2)]
        cands = _atoms_over(preds, params)
        sc = None
        for _ in range(20):
            if not cands:
                break
            pre = rng.sample(cands, rng.randint(1, min(3, len(cands))))
            if not all(any(v in a[1:] for a in pre) for v in params):
                continue
            rest = [c for c in cands if c not in pre]
            add = rng.sample(rest, min(len(rest), rng.choice((1, 1, 2, 2, 3))))
            dele = rng.sample(pre, rng.randint(0, len(pre)))
            if add or dele:
                sc = Schema(params, tuple(sorted(pre)), tuple(sorted(add)), tuple(sorted(dele)))
                break
        if sc is None:
            if stats is not None:
                stats["schema_sampling"] += 1
            return None
        acts[an] = sc
    dom = Domain("regime-b", preds, acts)
    why = domain_rejection(dom)
    if why:
        if stats is not None:
            stats[why] += 1
        return None
    return dom


def _layered_bfs(dom: Domain, init: frozenset, objs, max_depth: int, max_states: int):
    """BFS by layers; a layer that would exceed max_states is discarded, so every kept layer is complete."""
    acts = _compiled(dom, objs)
    parent, depth, order, layer = {init: None}, {init: 0}, [init], [init]
    for d in range(1, max_depth + 1):
        nxt = {}
        for s in layer:
            for a, pre, add, dele in acts:
                if pre <= s:
                    ns = (s - dele) | add
                    if ns not in parent and ns not in nxt:
                        nxt[ns] = (s, a)
            if len(parent) + len(nxt) > max_states:
                return order, parent, depth, d - 1
        if not nxt:
            return order, parent, depth, max_depth
        for ns, pa in nxt.items():
            parent[ns], depth[ns] = pa, d
            order.append(ns)
        layer = list(nxt)
    return order, parent, depth, max_depth


def regime_b(rng: random.Random, ng: NameGen, max_states: int = 20000, min_len: int = 2, max_len: int = 12,
             want_len: int | None = None, tol: int | None = None, goal_tries: int = 40, max_goal: int = 6,
             stats=None, letters: bool = False) -> dict | None:
    """Random domain + instance. Goals are 1..max_goal positive facts of a reachable state; among `goal_tries`
    candidate goals the one whose optimal plan length is closest to `want_len` is kept, and the instance is rejected
    if it misses by more than `tol` (spreads lengths over min_len..max_len; random domains are mostly shallow).
    letters: objects are single letters a..o instead of nonsense words."""
    dom = random_domain(rng, ng, stats)
    if dom is None:
        return None
    k = rng.randint(3, 6)
    objs = sorted(rng.sample(LETTERS, k) if letters else ng.sample(k, 1, 2, exclude=set(dom.preds) | set(dom.actions)))
    p = rng.uniform(0.1, 0.45)
    init = frozenset(f for f in all_facts(dom, objs) if rng.random() < p)
    order, parent, depth, complete = _layered_bfs(dom, init, objs, max_len, max_states)
    by_depth = {}
    for s in order:
        by_depth.setdefault(depth[s], []).append(s)
    ks = [k for k in range(min_len, min(max_len, complete) + 1) if by_depth.get(k)]
    if not ks:
        if stats is not None:
            stats["no_deep_states"] += 1
        return None
    want = want_len if want_len is not None else rng.randint(min_len, max_len)
    if tol is not None and ks[-1] < want - tol:
        if stats is not None:
            stats["too_shallow_for_want"] += 1
        return None
    best = None
    for i in range(goal_tries):
        k = rng.choice([x for x in ks if x >= want] or ks[-1:]) if rng.random() < 0.7 else rng.choice(ks)
        tgt = rng.choice(by_depth[k])
        new = sorted(tgt - init)
        if not new:
            continue
        if i % 4 == 0 and len(new) <= max_goal:
            goal = set(new)
        else:
            goal = set(rng.sample(new, rng.randint(1, min(max_goal, len(new)))))
        kept = sorted(tgt & init)
        if kept and len(goal) < max_goal and rng.random() < 0.3:
            goal.add(rng.choice(kept))
        goal = frozenset(goal)
        hit = next(s for s in order if goal <= s)  # BFS order over complete layers -> optimal
        L = depth[hit]
        if L < min_len:
            continue
        if best is None or abs(L - want) < best[0]:
            best = (abs(L - want), goal, hit)
            if best[0] == 0:
                break
    if best is None or (tol is not None and best[0] > tol):
        if stats is not None:
            stats["goal_too_shallow" if best is None else "length_miss"] += 1
        return None
    _, goal, hit = best
    plan, cur = [], hit
    while parent[cur] is not None:
        cur, a = parent[cur]
        plan.append(a)
    return {"regime": "B", "domain": dom, "objs": objs, "init": init, "goal": goal, "plan": plan[::-1],
            "explored": len(order), "complete_depth": complete, "want_len": want}


B_LEN_WEIGHTS = {k: (1.0 if k <= 8 else 0.5) for k in range(2, 13)}


def regime_b_sample(rng: random.Random, ng: NameGen, max_states: int = 20000, tol: int = 1, max_tries: int = 400,
                    stats=None, letters: bool = False) -> dict:
    """One Regime-B instance whose optimal length is within `tol` of a length drawn from B_LEN_WEIGHTS; the drawn
    length is kept until filled (re-drawn after max_tries), so the length histogram follows the weights."""
    while True:
        want = rng.choices(list(B_LEN_WEIGHTS), list(B_LEN_WEIGHTS.values()))[0]
        for _ in range(max_tries):
            inst = regime_b(rng, ng, max_states=max_states, want_len=want, tol=tol, stats=stats, letters=letters)
            if inst is not None:
                return inst
        if stats is not None:
            stats["want_redrawn"] += 1


# ----------------------------------------------------------------------------- rendering for the LM

SYS_PROPOSER = (
    "Planner for an unfamiliar domain. Each rule reads name(params): pre[...] add[...] del[...]; params are distinct "
    "objects. An action can be taken only when all its pre facts hold; it then deletes its del facts and adds its add "
    "facts. Reach every goal fact. Reply with a JSON plan."
)
SYS_FORWARD = (
    "Simulator for an unfamiliar domain. Each rule reads name(params): pre[...] add[...] del[...]; params are distinct "
    "objects. An action is legal only when all its pre facts hold; it then deletes its del facts and adds its add "
    "facts. An illegal action changes nothing. Given the state and one action, reply with the resulting JSON state."
)


def atom_str(a: tuple) -> str:
    return " ".join(a)


def rule_str(name: str, sc: Schema) -> str:
    f = lambda xs: "[" + ", ".join(atom_str(a) for a in sorted(xs)) + "]"
    return f"{name}({','.join(sc.params)}): pre{f(sc.pre)} add{f(sc.add)} del{f(sc.dele)}"


def rules(dom: Domain) -> list[str]:
    return [rule_str(n, dom.actions[n]) for n in sorted(dom.actions)]


def state_list(s) -> list[str]:
    return sorted(atom_str(f) for f in s)


def action_str(a: tuple) -> str:
    return " ".join(a)


def proposer_prompt(dom: Domain, objs, s: frozenset, goal: frozenset, outcome: str = "success") -> str:
    return render_chat(SYS_PROPOSER, canonical({"goal": state_list(goal), "objects": sorted(objs), "outcome": outcome,
                                                "rules": rules(dom), "state": state_list(s)}))


def forward_prompt(dom: Domain, objs, s: frozenset, a: tuple) -> str:
    """Goal-blind transition query."""
    return render_chat(SYS_FORWARD, canonical({"action": action_str(a), "objects": sorted(objs), "rules": rules(dom),
                                               "state": state_list(s)}))


def forward_target(ns: frozenset, legal: bool) -> str:
    return canonical({"legal": legal, "state": state_list(ns)})


def target_actions(plan) -> str:
    return canonical({"actions": [action_str(a) for a in plan]})


def target_states(steps) -> str:
    """steps: list of (action, next_state)."""
    return canonical({"steps": [{"action": action_str(a), "state": state_list(ns)} for a, ns in steps]})


def mystery_prompt(inst: dict, outcome: str = "success") -> str:
    """PlanBench Mystery instance (from mystery_instances) -> the identical proposer prompt format (evaluation)."""
    return proposer_prompt(inst["domain"], inst["objs"], inst["init"], inst["goal"], outcome)


# ----------------------------------------------------------------------------- parsing LM output

def prompt_user(prompt: str) -> dict:
    """The canonical-JSON user content of a rendered prompt (data audits: recover state / goal / action)."""
    return json.loads(prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0])


def parse_fact(t, dom: Domain, objs) -> tuple | None:
    p = t.strip().split() if isinstance(t, str) else None
    if not p or p[0] not in dom.preds or len(p) - 1 != dom.preds[p[0]] or any(x not in objs for x in p[1:]):
        return None
    return tuple(p)


def state_from_list(lst, dom: Domain, objs) -> frozenset | None:
    if not isinstance(lst, list):
        return None
    fs = [parse_fact(t, dom, objs) for t in lst]
    return None if any(f is None for f in fs) else frozenset(fs)


def parse_action(t, dom: Domain, objs) -> tuple | None:
    p = t.strip().split() if isinstance(t, str) else None
    if not p or p[0] not in dom.actions or len(p) - 1 != len(dom.actions[p[0]].params) \
            or any(x not in objs for x in p[1:]):
        return None
    return tuple(p)


def parse_forward(text: str, dom: Domain, objs):
    try:
        d = json.loads(text.strip())
        if set(d) != {"legal", "state"} or type(d["legal"]) is not bool:
            return None
        s = state_from_list(d["state"], dom, objs)
        return None if s is None else (s, d["legal"])
    except Exception:
        return None


def parse_plan(text: str, dom: Domain, objs, mode: str = "actions"):
    try:
        d = json.loads(text.strip())
        if mode == "actions":
            acts, states = [parse_action(t, dom, objs) for t in d["actions"]], None
        else:
            acts = [parse_action(st["action"], dom, objs) for st in d["steps"]]
            states = [state_from_list(st["state"], dom, objs) for st in d["steps"]]
    except Exception:
        return None
    if not acts or any(a is None for a in acts) or len(acts) > 60:
        return None
    return acts, states


# ----------------------------------------------------------------------------- rows (same recipe as blocks_data)

def bw_suffix_signatures(inst: dict) -> list[str]:
    """Regime A: Blocksworld-space renaming signature of (state_t, goal) for t = 0..len(plan)-1."""
    bw = inst["bw"]
    s, out = bw["init"], [inst["sig"]]
    for a in bw["plan"][:-1]:
        s, ok = B.step(s, a)
        assert ok
        out.append(B.signature(bw["objs"], s, bw["goal"]))
    return out


def instance_rows(inst: dict, rng: random.Random, n_forward: int = 2, test_sigs: set | None = None,
                  stats=None) -> dict:
    """Proposer rows: every suffix of the optimal plan (success) + one noisy plan with its achieved outcome.
    Forward rows: goal-blind transitions from plan states / short random walks, alternating legal and illegal.
    Suffix leakage guard (B-02, exactly as scripts/blocks_data.py): an intermediate (state_t, goal) of a Regime-A
    instance can be a renaming of a PlanBench test problem even when its (init, goal) is not; such rows (t > 0) are
    dropped and counted (stats["suffix_test_signature"], rows["suffix_dropped"]); Regime-A rows carry their
    Blocksworld-space signature `sig`. The noisy plan is resampled until it differs from the optimal plan (else it
    duplicates the t = 0 row); a repeated forward query is resampled (dropped after 20 tries)."""
    dom, objs, init, goal, plan = inst["domain"], inst["objs"], inst["init"], inst["goal"], inst["plan"]
    states = [init]
    for a in plan:
        states.append(step(dom, states[-1], a)[0])
    sigs = None
    if inst["regime"] == "A":
        if test_sigs is None:
            raise ValueError("Regime-A rows need test_sigs (suffix-signature guard)")
        sigs = bw_suffix_signatures(inst)
    prop, dropped = [], 0
    for t in range(len(plan)):
        if t > 0 and sigs is not None and sigs[t] in test_sigs:
            dropped += 1
            continue
        row = {"prompt": proposer_prompt(dom, objs, states[t], goal, "success"), "target_actions": target_actions(plan[t:]),
               "target_states": target_states(list(zip(plan[t:], states[t + 1:]))), "outcome": "success", "t": t}
        if sigs is not None:
            row["sig"] = sigs[t]
        prop.append(row)
    if stats is not None and dropped:
        stats["suffix_test_signature"] += dropped
    ga = ground_actions(dom, objs)
    assert len(ga) > 1
    noisy = list(plan)
    j = rng.randrange(len(noisy))
    while noisy == plan:  # resample: an unchanged "noisy" plan would duplicate the t = 0 success row
        noisy[j] = rng.choice(ga)
    s, seq = init, []
    for a in noisy:
        s, _ = step(dom, s, a)
        seq.append(s)
    oc = "success" if execute(dom, init, goal, noisy)["success"] else "failure"
    row = {"prompt": proposer_prompt(dom, objs, init, goal, oc), "target_actions": target_actions(noisy),
           "target_states": target_states(list(zip(noisy, seq))), "outcome": oc, "t": -1}
    if sigs is not None:
        row["sig"] = sigs[0]
    prop.append(row)
    fwd, asked = [], set()
    for i in range(n_forward):
        for _ in range(20):
            s = rng.choice(states)
            for _ in range(rng.randrange(0, 8)):
                legal = [a for a in ga if applicable(dom, s, a)]
                if not legal:
                    break
                s, _ = step(dom, s, rng.choice(legal))
            legal = [a for a in ga if applicable(dom, s, a)]
            illegal = [a for a in ga if not applicable(dom, s, a)]
            pool = legal if (i % 2 == 0 and legal) or not illegal else illegal
            a = rng.choice(pool)
            if (s, a) not in asked:
                break
        if (s, a) in asked:
            continue
        asked.add((s, a))
        ns, ok = step(dom, s, a)
        fwd.append({"prompt": forward_prompt(dom, objs, s, a), "target": forward_target(ns, ok), "legal": ok})
    return {"proposer": prop, "forward": fwd, "suffix_dropped": dropped}


def row_token_counts(rows: dict, count) -> int:
    """Annotate every row with n_tokens = prompt + target + EOS, tokenized separately as bwmn/lm.encode_pair does
    (proposer: {"actions", "states"} for the two target forms); returns the largest count."""
    m = 0
    for r in rows["proposer"]:
        p = count(r["prompt"])
        r["n_tokens"] = {"actions": p + count(r["target_actions"]) + 1, "states": p + count(r["target_states"]) + 1}
        m = max(m, *r["n_tokens"].values())
    for r in rows["forward"]:
        r["n_tokens"] = count(r["prompt"]) + count(r["target"]) + 1
        m = max(m, r["n_tokens"])
    return m


# ----------------------------------------------------------------------------- per-slot generation

def instance_seed(seed: int, regime: str, i: int) -> int:
    return int(hashlib.sha256(f"{seed}/{regime}/{i}".encode()).hexdigest()[:16], 16)


def letter_objects(i: int) -> bool:
    """Slot i uses single-letter object names iff i % 5 in (1, 3): 2 of every 5 consecutive slots (40%), so any
    prefix of n >= 2 slots has a share in [1/3, 1/2]."""
    return i % 5 in (1, 3)


def make_instance(regime: str, seed: int, i: int, forbidden: frozenset, test_sigs: set, seen=frozenset(), *,
                  single_token=None, count_tokens=None, max_tokens: int | None = None, max_states: int = 20000,
                  n_forward: int = 2, max_attempts: int = 1000):
    """Instance slot i of regime "A"/"B": a pure function of (seed, regime, i, test_sigs, seen), drawn from its own
    RNG (generation can stream, resume or shard per slot). Regime-A candidates whose signature is in `seen` (earlier
    slots) are redrawn. Rows obey the suffix-signature guard; with count_tokens, every row must fit max_tokens
    (prompt + target + EOS, both target forms) or the candidate is redrawn: the budget is enforced here, so training
    never has to drop an overlength row. Returns (inst, rows, rejection counts)."""
    if (count_tokens is None) != (max_tokens is None):
        raise ValueError("count_tokens and max_tokens go together")
    sd = instance_seed(seed, regime, i)
    rng = random.Random(sd)
    ng = NameGen(rng, forbidden, single_token=single_token)
    letters = letter_objects(i)
    rej = Counter()
    for _ in range(max_attempts):
        if regime == "A":
            inst = regime_a(rng, ng, test_sigs, stats=rej, letters=letters)
            if inst is None:
                continue
            if inst["sig"] in seen:
                rej["duplicate_in_sample"] += 1
                continue
        elif regime == "B":
            inst = regime_b_sample(rng, ng, max_states=max_states, stats=rej, letters=letters)
        else:
            raise ValueError(regime)
        rows = instance_rows(inst, rng, n_forward, test_sigs=test_sigs, stats=rej)
        if count_tokens is not None and row_token_counts(rows, count_tokens) > max_tokens:
            rej["token_budget"] += 1
            continue
        inst.update(letter_objects=letters, seed=sd, suffix_rows_dropped=rows["suffix_dropped"])
        return inst, rows, rej
    raise RuntimeError(f"slot {regime}{i}: no instance in {max_attempts} attempts {dict(rej)}")


def facts_json(fs) -> list:
    return sorted(list(f) for f in fs)


def instance_record(inst: dict, idx) -> dict:
    d = {"id": idx, "regime": inst["regime"], "domain": inst["domain"].to_json(), "rules": rules(inst["domain"]),
         "schema_hash": schema_hash(inst["domain"]), "objs": inst["objs"], "init": facts_json(inst["init"]),
         "goal": facts_json(inst["goal"]), "plan": [list(a) for a in inst["plan"]], "opt": len(inst["plan"])}
    for k in ("sig", "n_blocks", "explored", "complete_depth", "want_len", "letter_objects", "seed",
              "suffix_rows_dropped"):
        if k in inst:
            d[k] = inst[k]
    if "bw" in inst:  # Regime A: Blocksworld-space instance + renaming (audits map rows back to signatures)
        bw = inst["bw"]
        d["bw"] = {"objs": bw["objs"], "init": facts_json(bw["init"]), "goal": facts_json(bw["goal"]),
                   "plan": [list(a) for a in bw["plan"]]}
        d["renaming"] = inst["renaming"].to_json()
    return d


def instance_from_record(d: dict) -> dict:
    """Inverse of instance_record (domain, facts, plan; Regime A also bw + renaming)."""
    F = lambda xs: frozenset(tuple(x) for x in xs)
    inst = {k: v for k, v in d.items() if k not in ("domain", "init", "goal", "plan", "bw", "renaming")}
    inst.update(domain=Domain.from_json(d["domain"]), init=F(d["init"]), goal=F(d["goal"]),
                plan=[tuple(a) for a in d["plan"]])
    if "bw" in d:
        bw = d["bw"]
        inst["bw"] = {"objs": bw["objs"], "init": F(bw["init"]), "goal": F(bw["goal"]),
                      "plan": [tuple(a) for a in bw["plan"]]}
        inst["renaming"] = Renaming.from_json(d["renaming"])
    return inst
