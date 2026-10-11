"""PlanBench Logistics (6-op untyped STRIPS, `logistics-strips`) under BWM contracts: exact VAL semantics, PDDL I/O,
typed grounding, A* teacher, PlanBench-shape generator, renaming signature, JSON rendering, VAL.

State = the full PDDL fact frozenset, static facts included, all names lower case (as parsed by blocks.parse_problem):
  type facts ('obj',p) ('truck',t) ('airplane',a) ('location',l) ('city',c) ('airport',l); ('in-city',l,c);
  dynamic facts ('at',x,l) for packages and vehicles, ('in',p,v) for a package inside a vehicle.
Actions = PDDL operator tuples: ('load-truck',p,t,l) ('load-airplane',p,a,l) ('unload-truck',p,t,l)
  ('unload-airplane',p,a,l) ('drive-truck',t,l1,l2,c) ('fly-airplane',a,l1,l2).
Semantics follow VAL on bench/planbench/logistics/generated_domain.pddl: every precondition, the type facts included,
must hold; effects are applied delete-then-add, so a self-move (drive-truck t l l c, fly-airplane a l l) whose
preconditions hold is a legal no-op; a type-wrong action (e.g. load-truck p0 a1 l1-0) is illegal. An inapplicable action
is illegal and leaves the state unchanged (BWM convention); PlanBench/VAL scoring requires every action to be
applicable and the goal to hold at the end. Repeated objects are allowed (mystery.ground rejects them; VAL does not).
Raw plan texts (e.g. published LLM plans) are read as VAL reads them (read_val_plan / execute_text: skipped
characters, stop at prose, bad-plan errors, timestamps and concurrent happenings, extra arguments ignored);
plan_from_pddl is the lenient '(...)'-group extractor for well-formed texts. An action tuple must have exactly its
operator's arity (parse_action's contract): step() treats any other tuple as illegal, whereas VAL, reading a text,
ignores extra arguments (val_bind) and crashes on missing ones.
The exact simulator, the A* teacher and VAL are data-collection / evaluation instruments only.
"""
from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import re
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

from . import blocks as B

BENCH = B.BENCH
LG_DIR = BENCH / "logistics"
DOMAIN_PDDL = LG_DIR / "generated_domain.pddl"
INSTANCE_DIR = LG_DIR / "generated_basic"
PROMPTS_T1 = LG_DIR / "prompts" / "task_1_plan_generation.json"
RESULTS_DIR = BENCH / "results" / "logistics"
VAL_EXE = B.VAL_EXE
DOMAIN_NAME = "logistics-strips"
TYPES = ("airplane", "airport", "city", "location", "obj", "truck")
STATIC = frozenset(TYPES) | {"in-city"}
ARITY = {"load-truck": 3, "load-airplane": 3, "unload-truck": 3, "unload-airplane": 3, "drive-truck": 4,
         "fly-airplane": 3}
# operator -> parameter types (unary type predicates of the domain file, in parameter order)
PARAM_TYPES = {"load-truck": ("obj", "truck", "location"), "load-airplane": ("obj", "airplane", "location"),
               "unload-truck": ("obj", "truck", "location"), "unload-airplane": ("obj", "airplane", "location"),
               "drive-truck": ("truck", "location", "location", "city"),
               "fly-airplane": ("airplane", "airport", "airport")}
MAX_PLAN_ACTIONS = 96  # parse cap (B2: 60), rescaled to ~14 tokens per action
canonical = B.canonical
render_chat = B.render_chat


# ----------------------------------------------------------------------------- semantics

@lru_cache(maxsize=1 << 14)
def ground(a: tuple):
    """(pre, add, del) fact sets of a ground action, or None if the operator is unknown or the arity is wrong.
    Mirrors generated_domain.pddl literally (tests compare with mystery.parse_domain of that file). The cache holds
    about 1.6 KB per entry; 1 << 14 entries (about 26 MB) is over 4x the typed actions of the whole generator name
    grammar (wrong-type random actions are one-off misses). ground.cache_clear() frees it."""
    if not a or a[0] not in ARITY or len(a) - 1 != ARITY[a[0]]:
        return None
    op = a[0]
    if op in ("load-truck", "load-airplane", "unload-truck", "unload-airplane"):
        p, v, l = a[1:]
        vt = "truck" if op.endswith("truck") else "airplane"
        typ = {("obj", p), (vt, v), ("location", l), ("at", v, l)}
        if op.startswith("load"):
            return frozenset(typ | {("at", p, l)}), frozenset({("in", p, v)}), frozenset({("at", p, l)})
        return frozenset(typ | {("in", p, v)}), frozenset({("at", p, l)}), frozenset({("in", p, v)})
    if op == "drive-truck":
        t, l1, l2, c = a[1:]
        pre = {("truck", t), ("location", l1), ("location", l2), ("city", c), ("at", t, l1), ("in-city", l1, c),
               ("in-city", l2, c)}
        return frozenset(pre), frozenset({("at", t, l2)}), frozenset({("at", t, l1)})
    x, l1, l2 = a[1:]
    pre = {("airplane", x), ("airport", l1), ("airport", l2), ("at", x, l1)}
    return frozenset(pre), frozenset({("at", x, l2)}), frozenset({("at", x, l1)})


def applicable(s: frozenset, a: tuple) -> bool:
    g = ground(a) if isinstance(a, tuple) else None
    return g is not None and g[0] <= s


def step(s: frozenset, a: tuple) -> tuple[frozenset, bool]:
    g = ground(a) if isinstance(a, tuple) else None
    if g is None or not g[0] <= s:
        return s, False
    return (s - g[2]) | g[1], True  # delete then add (VAL): a self-move is a no-op


def is_self_loop(a: tuple) -> bool:
    return (a[0] == "drive-truck" and len(a) == 5 and a[2] == a[3]) or (
        a[0] == "fly-airplane" and len(a) == 4 and a[2] == a[3])


def goal_met(s: frozenset, goal: frozenset) -> bool:
    return goal <= s


def unmet(s: frozenset, goal: frozenset) -> int:
    """plan_beam hook: number of unsatisfied goal facts (equals blocks_planner's default)."""
    return len(goal - s)


def execute(init: frozenset, goal: frozenset, plan: list[tuple]) -> dict:
    """Exact execution; success = every action applicable and goal met at the end (VAL semantics)."""
    s = init
    for i, a in enumerate(plan):
        if not applicable(s, a):
            return {"success": False, "valid": False, "fail_step": i, "final": s}
        s, _ = step(s, a)
    return {"success": goal_met(s, goal), "valid": True, "fail_step": None, "final": s}


# ----------------------------------------------------------------------------- objects and typed grounding

def objs_of(s: frozenset) -> dict:
    """Objects by type predicate ({'airplane','airport','city','location','obj','truck'} -> sorted names)."""
    out = {t: set() for t in TYPES}
    for f in s:
        if len(f) == 2 and f[0] in out:
            out[f[0]].add(f[1])
    return {t: sorted(v) for t, v in out.items()}


def all_objects(objs) -> list[str]:
    """Every object name of a by-type dict (or any iterable of names)."""
    if isinstance(objs, dict):
        return sorted({x for v in objs.values() for x in v})
    return sorted(set(objs))


def ground_actions(objs: dict, init: frozenset, self_loops: bool = False) -> list[tuple]:
    """Typed grounding: every action whose type and static preconditions (in-city, airport) hold in `init`.
    Loads/unloads range over all locations; drives over location pairs of one city; flies over airport pairs.
    Self-moves (legal no-ops) are included only on request; 14-720 actions per PlanBench instance without them."""
    city_locs = {}
    for f in sorted(init):
        if f[0] == "in-city":
            city_locs.setdefault(f[2], []).append(f[1])
    P, T, A, L, AP = objs["obj"], objs["truck"], objs["airplane"], objs["location"], objs["airport"]
    acts = [(op, p, v, l) for op, vs in (("load-truck", T), ("unload-truck", T), ("load-airplane", A),
                                         ("unload-airplane", A)) for p in P for v in vs for l in L]
    acts += [("drive-truck", t, l1, l2, c) for t in T for c in sorted(city_locs) if c in objs["city"]
             for l1 in city_locs[c] for l2 in city_locs[c] if self_loops or l1 != l2]
    acts += [("fly-airplane", a, l1, l2) for a in A for l1 in AP for l2 in AP if self_loops or l1 != l2]
    return acts


def applicable_actions(s: frozenset, self_loops: bool = True) -> list[tuple]:
    """The typed ground actions applicable in s (objects and static facts read from s). Self-moves are legal no-ops
    and are a large share of them (about 36% on prereg-mix F start states, tests/test_logistics.py); pass
    self_loops=False to draw legal actions without them."""
    return [a for a in ground_actions(objs_of(s), s, self_loops) if applicable(s, a)]


# ----------------------------------------------------------------------------- PDDL I/O

def parse_problem(path: Path):
    """(objects by type, init frozenset, goal frozenset). Facts are lower case (blocks.parse_problem)."""
    names, init, goal = B.parse_problem(path)
    objs = objs_of(init)
    untyped = set(names) - set(all_objects(objs))
    if untyped:
        raise ValueError(f"untyped objects in {path}: {sorted(untyped)}")
    return objs, init, goal


def write_problem(objs, init, goal, name="bwm") -> str:
    f = lambda fs: "\n".join("(" + " ".join(x) + ")" for x in sorted(fs))
    return (f"(define (problem {name})\n(:domain {DOMAIN_NAME})\n(:objects {' '.join(all_objects(objs))} )\n"
            f"(:init\n{f(init)}\n)\n(:goal\n(and\n{f(goal)})\n)\n)\n")


def plan_to_pddl(plan) -> str:
    return B.plan_to_pddl(plan)


# VAL's plan lexer, probed on the local binary with every printable ASCII character inside a group, between groups,
# before and after a plan: any character other than an ASCII letter or digit, whitespace and _ - . : ; ( ) [ ] * + /
# < = > ? is skipped like a blank (e.g. , " ' \ { } | ! # $ % & @ ^ ` ~, control and non-ASCII characters), except
# '#t' / '#T' (PDDL+'s #t), which is a token.
VAL_SKIPPED = re.compile(r"#(?![tT])|[^A-Za-z0-9_\-.:;()\[\]*+/<=>?# \t\n\r\f\v]")
_VAL_TOKEN = re.compile(r"[ \t\n\r\f\v]+|;[^\n]*|(?P<illegal>[0-9]+[a-z])|(?P<num>-?[0-9]+(?:\.[0-9]*)?)"
                        r"|(?P<name>[a-z_][a-z0-9_\-]*)|(?P<sym>#t|.)", re.S)


def val_bind(a: tuple) -> tuple:
    """An action as VAL binds it when reading a plan text: a known operator keeps its declared parameters and VAL
    ignores extra arguments (even undeclared names; its -v listing prints the bound action, and the plan can be
    valid). Fewer arguments are kept (VAL crashes on them: not valid). step() itself stays strict on tuples (BWM's
    action contract, enforced by parse_action), so only the text readers apply this."""
    return a[:ARITY[a[0]] + 1] if a and a[0] in ARITY and len(a) - 1 > ARITY[a[0]] else a


def plan_from_pddl(text) -> list[tuple]:
    """Every '(...)' group of a PDDL plan text (or list of lines), lower case, bound as VAL binds it (val_bind);
    ';' comments dropped; characters VAL skips (VAL_SKIPPED, e.g. ',') read as blanks. A lenient extractor for
    well-formed texts (ground-truth plans, renderings): it ignores text outside the groups, unbalanced parentheses
    and timestamps. read_val_plan and execute_text give VAL's own reading of a raw (e.g. LLM-written) text."""
    if isinstance(text, (list, tuple)):
        text = "\n".join(text)
    text = VAL_SKIPPED.sub(" ", re.sub(r";[^\n]*", "", text or "")).lower()
    return [val_bind(tuple(m.split())) for m in re.findall(r"\(([^()]*)\)", text)]


def _val_tokens(text: str) -> list[tuple[str, str]]:
    text = VAL_SKIPPED.sub(" ", text).lower()  # skip first: lower() could turn a skipped character into a letter
    return [(m.lastgroup, m.group(m.lastgroup)) for m in _VAL_TOKEN.finditer(text) if m.lastgroup]


def read_val_plan(text) -> list[tuple[float, tuple]] | None:
    """VAL's reading of a raw plan text (or list of lines): [(time, action)] in text order, or None where VAL prints
    'Bad plan description!' or 'Illegal symbol' (the plan is then not valid). Probed on the local binary:
      - a step is [NUMBER ':'] '(' NAME+ ')' ['[' NUMBER ']']; 'time NUMBER' between steps only records a run time;
      - reading stops at the first token that cannot start a step (prose, a stray ')' or ':', a '-' or '*' list
        marker, ...) and VAL validates the steps read so far: leading prose gives an empty plan, trailing prose and
        everything after a stop are ignored;
      - an error inside a step (unclosed or nested group, a number or symbol among the names, a number without ':',
        a malformed duration, 'time' without a number) or a digit run glued to a letter makes the text a bad plan;
      - untimed steps get times 1, 2, 3, ... in text order (a counter over the untimed steps only);
      - actions are bound as VAL binds them (val_bind: extra arguments of a known operator are ignored).
    Names are lower case. A PDDL keyword inside a group is kept as a name (VAL rejects some as a bad plan; the
    verdict is the same, as no Logistics operator or object is named like one)."""
    if isinstance(text, (list, tuple)):
        text = "\n".join(text)
    toks = _val_tokens(text or "")
    tok = lambda k: toks[k] if k < len(toks) else ("eof", "")
    i, steps, untimed = 0, [], 0
    while True:
        kind, v = tok(i)
        if kind == "illegal":
            return None
        if (kind, v) == ("name", "time"):
            if tok(i + 1)[0] != "num":
                return None
            i += 2
            continue
        t = None
        if kind == "num":
            if tok(i + 1) != ("sym", ":") or tok(i + 2) != ("sym", "("):
                return None
            t, i = float(v), i + 2
        elif (kind, v) != ("sym", "("):
            return steps  # end of text, or a token that cannot start a step
        i += 1
        act = []
        while tok(i)[0] == "name":
            act.append(tok(i)[1])
            i += 1
        if not act or tok(i) != ("sym", ")"):
            return None
        i += 1
        if tok(i) == ("sym", "["):
            if tok(i + 1)[0] != "num" or tok(i + 2) != ("sym", "]"):
                return None
            i += 3
        if t is None:
            untimed += 1
            t = float(untimed)
        steps.append((t, val_bind(tuple(act))))


def execute_text(init: frozenset, goal: frozenset, text) -> dict:
    """VAL's verdict on a raw plan text, re-implemented on the exact semantics: read_val_plan, then the happenings in
    time order (stable). Steps with equal times form one concurrent happening, as in VAL: every precondition holds
    before it, no step's effects touch another step's preconditions or deletes (VAL's mutex check), deletes then
    adds. Returns execute()'s keys plus 'read' ('ok' or 'bad') and 'actions' (the steps read, in time order, or
    None). fail_step indexes happenings, which is the action index when the text has no timestamps."""
    steps = read_val_plan(text)
    if steps is None:
        return {"success": False, "valid": False, "fail_step": None, "final": init, "read": "bad", "actions": None}
    hs = []
    for t, a in sorted(steps, key=lambda x: x[0]):
        if hs and hs[-1][0] == t:
            hs[-1][1].append(a)
        else:
            hs.append((t, [a]))
    acts = [a for _, h in hs for a in h]
    s = init
    for i, (_, h) in enumerate(hs):
        gs = [ground(a) for a in h]
        ok = all(g is not None and g[0] <= s for g in gs) and not any(
            (gs[x][1] | gs[x][2]) & gs[y][0] or gs[x][2] & (gs[y][1] | gs[y][2])
            for x in range(len(gs)) for y in range(len(gs)) if x != y)
        if not ok:
            return {"success": False, "valid": False, "fail_step": i, "final": s, "read": "ok", "actions": acts}
        s = (s - frozenset().union(*(g[2] for g in gs))) | frozenset().union(*(g[1] for g in gs))
    return {"success": goal_met(s, goal), "valid": True, "fail_step": None, "final": s, "read": "ok", "actions": acts}


def val_check(problem_path: Path, plan, domain_path: Path = DOMAIN_PDDL) -> bool:
    """VAL on a list of action tuples (blocks.val_check with the vendored Logistics domain). Equals execute() on
    tuples of the right arity (VAL would bind an over-long tuple with val_bind; step() rejects it)."""
    return B.val_check(Path(problem_path), plan, domain_path)


def val_run(problem_path: Path, plan_text: str, domain_path: Path = DOMAIN_PDDL, timeout: int = 60,
            verbose: bool = False) -> dict:
    """VAL on a raw plan text, written as UTF-8 bytes (VAL skips non-ASCII bytes). Returns {'valid', 'returncode',
    'stdout'}; a crash (VAL segfaults on a wrong-arity action) or a timeout counts as not valid. verbose adds '-v'
    (VAL then lists the plan it read and every happening)."""
    with tempfile.TemporaryDirectory() as d:
        pp = Path(d) / "plan.txt"
        pp.write_bytes((plan_text or "").encode("utf-8"))
        cmd = [str(VAL_EXE)] + (["-v"] if verbose else []) + [str(domain_path), str(problem_path), str(pp)]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
            out, rc = (r.stdout or b"").decode("utf-8", "replace"), r.returncode
        except subprocess.TimeoutExpired:
            out, rc = "TIMEOUT", None
    return {"valid": rc == 0 and "Plan valid" in out, "returncode": rc, "stdout": out}


_VAL_LISTING_END = ("Plan Validation details", "Plan executed", "Plan failed", "Goal not satisfied", "Plan Repair",
                    "Plan valid", "Plan invalid", "Failed plans", "Successful plans")


def val_listing(stdout: str):
    """The steps VAL read, from `validate -v` output ('Plan to validate'), as a sorted [(time, action)] list
    (comparable with sorted(read_val_plan(text))); None if VAL reported 'Bad plan description!' or 'Illegal symbol';
    'n/a' if VAL printed no listing (it stops on an unknown operator and crashes on a wrong arity)."""
    if "Bad plan description" in stdout or "Illegal symbol" in stdout:
        return None
    m = re.search(r"Plan size: (\d+)", stdout)
    if not m:
        return "n/a"
    sec = stdout[m.end():]
    sec = sec[:min([sec.find(x) for x in _VAL_LISTING_END if x in sec] + [len(sec)])]
    steps, t = [], None
    for ln in sec.splitlines():
        ln = ln.strip()
        if re.fullmatch(r"-?[0-9.]+:", ln):
            t = float(ln[:-1])
        elif ln.startswith("("):
            steps.append((t, tuple(ln.strip("()").split())))
    return sorted(steps)


def planbench_path(i: int) -> Path:
    return INSTANCE_DIR / f"instance-{i}.pddl"


def planbench_gt() -> dict:
    """Task-1 ground-truth plans (Fast Downward astar(lmcut)), instance id -> list of action tuples (ids 2-286)."""
    d = json.loads(PROMPTS_T1.read_text())
    return {int(x["instance_id"]): plan_from_pddl(x["ground_truth_plan"]) for x in d["instances"]}


def planbench_instances(with_gt: bool = True) -> list[dict]:
    """All 286 generated_basic instances: id, path, objs, init, goal, n_packages and (ids 2-286) the GT plan."""
    gt = planbench_gt() if with_gt else {}
    out = []
    for i in range(1, 287):
        objs, init, goal = parse_problem(planbench_path(i))
        out.append({"id": i, "path": planbench_path(i), "objs": objs, "init": init, "goal": goal,
                    "n_packages": len(objs["obj"]), "gt": gt.get(i)})
    return out


# ----------------------------------------------------------------------------- A* teacher

def _compile(init: frozenset, goal: frozenset):
    """Compact integer encoding for search. Package position: location index, or n_locs + vehicle index."""
    objs = objs_of(init)
    L = objs["location"]
    li = {l: i for i, l in enumerate(L)}
    P, T, A = objs["obj"], objs["truck"], objs["airplane"]
    V = T + A
    vi = {v: i for i, v in enumerate(V)}
    nL, nP = len(L), len(P)
    cities_of = {l: sorted(f[2] for f in init if f[0] == "in-city" and f[1] == l) for l in L}
    city_locs = {c: [li[l] for l in L if c in cities_of[l]] for c in objs["city"]}
    air = [l in set(objs["airport"]) for l in L]
    pos = {}
    for f in init:
        if f[0] in ("at", "in") and len(f) == 3:
            pos.setdefault(f[1], []).append(f)
    st = []
    for p in P:
        fs = pos.get(p, [])
        if len(fs) != 1:
            raise ValueError(f"package {p} needs exactly one position")
        f = fs[0]
        if f[0] == "at" and f[2] in li:
            st.append(li[f[2]])
        elif f[0] == "in" and f[2] in vi:
            st.append(nL + vi[f[2]])
        else:
            raise ValueError(f"bad package position {f}")
    for v in V:
        fs = pos.get(v, [])
        if len(fs) != 1 or fs[0][0] != "at" or fs[0][2] not in li:
            raise ValueError(f"vehicle {v} needs exactly one location")
        st.append(li[fs[0][2]])
    pi = {p: i for i, p in enumerate(P)}
    gpk, gveh = [], []
    for f in sorted(goal):
        if f[0] == "at" and len(f) == 3 and f[1] in pi and f[2] in li:
            gpk.append((pi[f[1]], li[f[2]]))
        elif f[0] == "at" and len(f) == 3 and f[1] in vi and f[2] in li:
            gveh.append((nP + vi[f[1]], li[f[2]]))
        elif f[0] == "in" and len(f) == 3 and f[1] in pi and f[2] in vi:
            gpk.append((pi[f[1]], nL + vi[f[2]]))
        elif f[0] in STATIC:
            if f not in init:
                return None  # a static goal fact that does not hold: unsolvable
        else:
            raise ValueError(f"unsupported goal fact {f}")
    # heuristic needs the PlanBench structure: each location in one city, one airport per city, planes at airports
    one_city = all(len(cities_of[l]) == 1 for l in L)
    aps = {c: [i for i in city_locs[c] if air[i]] for c in city_locs}
    shaped = (one_city and all(len(v) == 1 for v in aps.values()) and all(air[st[nP + vi[a]]] for a in A)
              and not gveh and all(g < nL for _, g in gpk))
    city = [cities_of[l][0] if cities_of[l] else None for l in L]
    return {"L": L, "P": P, "V": V, "nL": nL, "nP": nP, "is_truck": [v in set(T) for v in V], "air": air,
            "city": city, "city_locs": city_locs, "ap": {c: aps[c][0] for c in aps if aps[c]},
            "cities_of": cities_of, "state": tuple(st), "gpk": gpk, "gveh": gveh, "shaped": shaped}


def _heuristic(C):
    """Admissible and consistent lower bound (PlanBench structure): the per-package minimum number of
    load/unload actions (each touches one package) + the number of distinct airports some flight must arrive at
    + the number of locations a truck must visit that no truck occupies now. The three action classes are
    disjoint, and every action lowers the bound by at most 1."""
    if not C["shaped"]:
        return lambda st: 0
    nL, nP, air, city, ap = C["nL"], C["nP"], C["air"], C["city"], C["ap"]
    is_truck, gpk = C["is_truck"], C["gpk"]
    trucks = [nP + j for j, t in enumerate(is_truck) if t]
    planes = [nP + j for j, t in enumerate(is_truck) if not t]

    def h(st):
        lu, fly, req = 0, set(), set()
        plane_at = {st[k] for k in planes}
        for i, g in gpk:
            x = st[i]
            if x == g:
                continue
            gc, g_air = city[g], air[g]
            if x < nL:
                c = city[x]
                if c == gc:
                    lu += 2
                    req.update((x, g))
                    continue
                lu += 2
                fly.add(ap[gc])
                if ap[c] not in plane_at:
                    fly.add(ap[c])
                if not air[x]:
                    lu += 2
                    req.update((x, ap[c]))
                if not g_air:
                    lu += 2
                    req.update((ap[gc], g))
            else:
                v = x - nL
                c = city[st[nP + v]]
                if is_truck[v]:
                    if c == gc:
                        lu += 1
                        req.add(g)
                        continue
                    lu += 3
                    req.add(ap[c])
                    fly.add(ap[gc])
                    if ap[c] not in plane_at:
                        fly.add(ap[c])
                    if not g_air:
                        lu += 2
                        req.update((ap[gc], g))
                else:
                    if c != gc:
                        fly.add(ap[gc])
                    if g_air:
                        lu += 1
                    else:
                        lu += 3
                        req.update((ap[gc], g))
        return lu + len(fly) + len(req - {st[k] for k in trucks})

    return h


def _successors(C, st):
    """Applicable non-self-loop actions of a compact state, in a fixed order: (action tuple, next compact state)."""
    L, P, V, nL, nP = C["L"], C["P"], C["V"], C["nL"], C["nP"]
    is_truck, air = C["is_truck"], C["air"]
    at_loc = {}
    for j in range(len(V)):
        at_loc.setdefault(st[nP + j], []).append(j)
    out = []
    for i in range(nP):
        x = st[i]
        if x < nL:
            for j in at_loc.get(x, ()):
                op = "load-truck" if is_truck[j] else "load-airplane"
                out.append(((op, P[i], V[j], L[x]), st[:i] + (nL + j,) + st[i + 1:]))
        else:
            j = x - nL
            l = st[nP + j]
            op = "unload-truck" if is_truck[j] else "unload-airplane"
            out.append(((op, P[i], V[j], L[l]), st[:i] + (l,) + st[i + 1:]))
    for j in range(len(V)):
        k = nP + j
        l = st[k]
        if is_truck[j]:
            for c in C["cities_of"][L[l]]:
                for l2 in C["city_locs"][c]:
                    if l2 != l:
                        out.append((("drive-truck", V[j], L[l], L[l2], c), st[:k] + (l2,) + st[k + 1:]))
        elif air[l]:
            for l2 in range(nL):
                if air[l2] and l2 != l:
                    out.append((("fly-airplane", V[j], L[l], L[l2]), st[:k] + (l2,) + st[k + 1:]))
    return out


def astar_plan(init: frozenset, goal: frozenset, max_expansions: int = 1_000_000, max_seconds: float | None = None,
               stats: dict | None = None):
    """Optimal plan (list of actions) or None. Teacher/oracle only. A* over the compact state with the admissible,
    consistent _heuristic (uniform-cost search if the instance lacks the PlanBench structure); self-moves are never
    generated. Ties on f prefer deeper nodes, then insertion order (deterministic). Stops with None after
    `max_expansions` expansions or `max_seconds` wall seconds (wall-clock caps make failures machine-dependent).
    `stats` (optional dict) receives status, expansions (nodes actually expanded: at most max_expansions, and a
    capped run reports exactly the cap), generated, seconds, h0."""
    t0 = time.time()
    st = stats if stats is not None else {}
    C = _compile(init, goal)
    if C is None:
        st.update(status="unsolvable", expansions=0, generated=0, seconds=0.0, h0=None)
        return None
    h = _heuristic(C)
    s0 = C["state"]
    targets = C["gpk"] + C["gveh"]
    done = lambda x: all(x[i] == g for i, g in targets)
    h0 = h(s0)
    st.update(h0=h0)
    best_g = {s0: 0}
    parent = {s0: None}
    closed = set()
    cnt = itertools.count()
    heap = [(h0, h0, next(cnt), s0)]
    exp = gen = 0
    status = "exhausted"
    while heap:
        f, hv, _, s = heapq.heappop(heap)
        if s in closed:
            continue
        if done(s):
            plan = []
            while parent[s] is not None:
                s, a = parent[s]
                plan.append(a)
            plan.reverse()
            st.update(status="solved", expansions=exp, generated=gen, seconds=round(time.time() - t0, 3))
            assert execute(init, goal, plan)["success"], "A* plan failed exact execution"
            return plan
        closed.add(s)
        if exp >= max_expansions:
            status = "cap"
            break
        if max_seconds is not None and exp % 512 == 0 and time.time() - t0 > max_seconds:
            status = "timeout"
            break
        exp += 1
        g = best_g[s] + 1
        for a, ns in _successors(C, s):
            if ns in closed or g >= best_g.get(ns, 1 << 30):
                continue
            best_g[ns] = g
            parent[ns] = (s, a)
            hn = h(ns)
            gen += 1
            heapq.heappush(heap, (g + hn, hn, next(cnt), ns))
    st.update(status=status, expansions=exp, generated=gen, seconds=round(time.time() - t0, 3))
    return None


# ----------------------------------------------------------------------------- generator (training only)

def build_instance(n_cities: int, city_size: int, trucks_at: list, planes_at: list, pkg_from: list, pkg_to: list,
                   pkg_names: list | None = None):
    """PlanBench names (logistics.c): cities c<i>, locations l<i>-<j> with airport l<i>-0, truck t<i> in c<i>,
    planes a<k>, packages p<k>. trucks_at[i] = j (location index in city i); planes_at[k] = city index;
    pkg_from/pkg_to[k] = (city, location index)."""
    loc = lambda i, j: f"l{i}-{j}"
    s = set()
    for k, ci in enumerate(planes_at):
        s |= {("airplane", f"a{k}"), ("at", f"a{k}", loc(ci, 0))}
    for i in range(n_cities):
        s |= {("city", f"c{i}"), ("truck", f"t{i}"), ("airport", loc(i, 0)), ("at", f"t{i}", loc(i, trucks_at[i]))}
        for j in range(city_size):
            s |= {("location", loc(i, j)), ("in-city", loc(i, j), f"c{i}")}
    names = pkg_names or [f"p{k}" for k in range(len(pkg_from))]
    goal = set()
    for p, (ci, j), (gi, gj) in zip(names, pkg_from, pkg_to):
        s |= {("obj", p), ("at", p, loc(ci, j))}
        goal.add(("at", p, loc(gi, gj)))
    init = frozenset(s)
    return objs_of(init), init, frozenset(goal)


def generate(rng, n_packages: int, n_cities: int | None = None, city_size: int | None = None,
             n_planes: int | None = None, rename: float = 0.0):
    """PlanBench shape (logistics.c): n_cities cities of city_size locations each, one airport and one truck per
    city (truck at a random location of its city), n_planes planes each at a random airport, package origins and
    goals uniform over all locations, one goal per package. Unset shape parameters are drawn: cities from {2, 3},
    city size from {1, 2, 3}, planes from 1..cities. With probability `rename` the names are randomised within the
    PlanBench name grammar (rename_instance); at rename == 0 the random stream does not depend on this option.
    Returns (objs, init, goal), or None if the goal already holds."""
    n_cities = rng.choice((2, 3)) if n_cities is None else n_cities
    city_size = rng.choice((1, 2, 3)) if city_size is None else city_size
    n_planes = rng.randint(1, n_cities) if n_planes is None else n_planes
    trucks_at = [rng.randrange(city_size) for _ in range(n_cities)]
    pkg_from = [(rng.randrange(n_cities), rng.randrange(city_size)) for _ in range(n_packages)]
    pkg_to = [(rng.randrange(n_cities), rng.randrange(city_size)) for _ in range(n_packages)]
    planes_at = [rng.randrange(n_cities) for _ in range(n_planes)]
    objs, init, goal = build_instance(n_cities, city_size, trucks_at, planes_at, pkg_from, pkg_to)
    if rename > 0 and rng.random() < rename:
        objs, init, goal = rename_instance(objs, init, goal, rng)
    if goal_met(init, goal):
        return None
    return objs, init, goal


def apply_renaming(m: dict, init: frozenset, goal: frozenset):
    """Rename objects by the injective map m (names not in m are kept); predicates are untouched."""
    r = lambda fs: frozenset((f[0],) + tuple(m.get(x, x) for x in f[1:]) for f in fs)
    ni = r(init)
    return objs_of(ni), ni, r(goal)


def rename_instance(objs: dict, init: frozenset, goal: frozenset, rng):
    """Random names inside the PlanBench grammar that break its conventions: city, truck, plane and location-prefix
    indices are drawn independently (so l<x>-<y> need not lie in c<x> and t<i> need not sit in c<i>), the airport's
    second index is random, and packages are drawn from p0..p11. A within-type renaming: the signature is unchanged."""
    city_locs = {}
    for f in sorted(init):
        if f[0] == "in-city":
            city_locs.setdefault(f[2], []).append(f[1])
    cities = objs["city"]
    m = {}
    pool = max(3, len(cities))
    for c, k in zip(cities, rng.sample(range(pool), len(cities))):
        m[c] = f"c{k}"
    for c, x in zip(cities, rng.sample(range(pool), len(cities))):
        locs = city_locs.get(c, [])
        for l, y in zip(locs, rng.sample(range(max(3, len(locs))), len(locs))):
            m[l] = f"l{x}-{y}"
    for typ, pre, lo in (("truck", "t", 3), ("airplane", "a", 3), ("obj", "p", 12)):
        xs = objs[typ]
        for o, k in zip(xs, rng.sample(range(max(lo, len(xs))), len(xs))):
            m[o] = f"{pre}{k}"
    rest = [l for l in objs["location"] if l not in m]  # locations outside every city (not in PlanBench)
    for l, k in zip(rest, rng.sample(range(100, 100 + len(rest)), len(rest))):
        m[l] = f"l{k}-0"
    if len(set(m.values())) != len(m):
        raise ValueError("renaming is not injective")
    return apply_renaming(m, init, goal)


# ----------------------------------------------------------------------------- renaming signature

def _structure(init: frozenset, goal: frozenset):
    objs = objs_of(init)
    for f in init:
        if f[0] not in STATIC and f[0] not in ("at", "in"):
            raise ValueError(f"unsupported fact {f}")
    prim = [x for t in ("obj", "truck", "airplane", "location", "city") for x in objs[t]]
    if len(prim) != len(set(prim)):
        raise ValueError("an object has two primary types")
    cities_of = {}
    for f in init:
        if f[0] == "in-city":
            cities_of.setdefault(f[1], []).append(f[2])
    if any(len(cities_of.get(l, [])) != 1 for l in objs["location"]) or any(
            l not in set(objs["location"]) or c not in set(objs["city"]) for l, cs in cities_of.items() for c in cs):
        raise ValueError("every location must lie in exactly one city")
    if not set(objs["airport"]) <= set(objs["location"]):
        raise ValueError("airport that is not a location")
    pos = {}
    for f in init:
        if f[0] in ("at", "in"):
            pos.setdefault(f[1], []).append(f)
    veh = set(objs["truck"]) | set(objs["airplane"])
    for x in objs["obj"]:
        fs = pos.get(x, [])
        if len(fs) != 1 or not (fs[0][0] == "at" and fs[0][2] in cities_of or fs[0][0] == "in" and fs[0][2] in veh):
            raise ValueError(f"package {x} needs exactly one position")
    for v in veh:
        fs = pos.get(v, [])
        if len(fs) != 1 or fs[0][0] != "at" or fs[0][2] not in cities_of:
            raise ValueError(f"vehicle {v} needs exactly one location")
    if set(pos) - set(objs["obj"]) - veh:
        raise ValueError("position of an object that is neither a package nor a vehicle")
    gpos = {}
    for f in goal:
        if f[0] == "at" and len(f) == 3 and (f[1] in set(objs["obj"]) or f[1] in veh) and f[2] in cities_of:
            gpos.setdefault(f[1], []).append(f[2])
        else:
            raise ValueError(f"unsupported goal fact {f}")
    return objs, cities_of, pos, gpos


def canonical_form(init: frozenset, goal: frozenset) -> str:
    """Exact canonical form up to renaming within each type. Enumerates the orderings of cities and, within each
    city, of its airports and of its other locations (<= 48 labelings in PlanBench); for each labeling, vehicles are
    the multiset of (type, location, goal, multiset of carried packages' goals) and free packages the multiset of
    (location, goal); the lexicographically smallest key wins. Raises ValueError outside the supported fact
    vocabulary (types, in-city, at, in; goals 'at' of packages or vehicles)."""
    objs, cities_of, pos, gpos = _structure(init, goal)
    airports = set(objs["airport"])
    groups = {c: ([l for l in objs["location"] if cities_of[l][0] == c and l in airports],
                  [l for l in objs["location"] if cities_of[l][0] == c and l not in airports]) for c in objs["city"]}
    trucks = set(objs["truck"])
    vehicles = objs["truck"] + objs["airplane"]
    inside = {}
    free = []
    for p in objs["obj"]:
        f = pos[p][0]
        if f[0] == "in":
            inside.setdefault(f[2], []).append(p)
        else:
            free.append(p)
    best = None
    for cperm in itertools.permutations(objs["city"]):
        per_city = [list(itertools.product(itertools.permutations(groups[c][0]), itertools.permutations(groups[c][1])))
                    for c in cperm]
        shape = [[len(groups[c][0]), len(groups[c][1])] for c in cperm]
        for choice in itertools.product(*per_city):
            lid = {}
            for ci, (aps, rest) in enumerate(choice):
                for k, l in enumerate(aps):
                    lid[l] = [ci, 0, k]
                for k, l in enumerate(rest):
                    lid[l] = [ci, 1, k]
            gd = lambda x: sorted(lid[l] for l in gpos.get(x, []))
            vd = sorted([0 if v in trucks else 1, lid[pos[v][0][2]], gd(v), sorted(gd(p) for p in inside.get(v, []))]
                        for v in vehicles)
            pd = sorted([lid[pos[p][0][2]], gd(p)] for p in free)
            key = canonical([shape, vd, pd])
            if best is None or key < best:
                best = key
    return best


def signature(objs, init: frozenset, goal: frozenset) -> str:
    """sha256 of canonical_form (objs is accepted for blocks.signature parity; types come from the facts)."""
    return hashlib.sha256(canonical_form(init, goal).encode()).hexdigest()


def planbench_signatures() -> dict:
    """Instance id -> signature for all 286 PlanBench Logistics generated_basic instances."""
    out = {}
    for i in range(1, 287):
        objs, init, goal = parse_problem(planbench_path(i))
        out[i] = signature(objs, init, goal)
    return out


# ----------------------------------------------------------------------------- rendering for the LM

def _places(s: frozenset) -> dict:
    out = {}
    for f in s:
        if f[0] in ("at", "in") and len(f) == 3:
            out.setdefault(f[1], set()).add(f[2])
    return out


def _place(pl: dict, x: str):
    v = sorted(pl.get(x, ()))
    return v[0] if len(v) == 1 else (None if not v else v)  # deterministic even on malformed imagined states


def state_json(s: frozenset) -> dict:
    """Grouped JSON: {"map":{"airports":[..],"cities":{c:[locs]}},"packages":{p:place},"planes":{a:loc},
    "trucks":{t:loc}}; a package's place is a location or the vehicle carrying it. Types come from the group."""
    ty = objs_of(s)
    pl = _places(s)
    cities = {c: [] for c in ty["city"]}
    for f in s:
        if f[0] == "in-city" and len(f) == 3 and f[2] in cities:
            cities[f[2]].append(f[1])
    return {"map": {"airports": ty["airport"], "cities": {c: sorted(v) for c, v in cities.items()}},
            "packages": {p: _place(pl, p) for p in ty["obj"]},
            "planes": {a: _place(pl, a) for a in ty["airplane"]},
            "trucks": {t: _place(pl, t) for t in ty["truck"]}}


def state_from_json(d: dict) -> frozenset | None:
    try:
        if set(d) != {"map", "packages", "planes", "trucks"} or set(d["map"]) != {"airports", "cities"}:
            return None
        s = set()
        for l in d["map"]["airports"]:
            s.add(("airport", l))
        for c, locs in d["map"]["cities"].items():
            s.add(("city", c))
            for l in locs:
                s |= {("location", l), ("in-city", l, c)}
        as_list = lambda x: [] if x is None else ([x] if isinstance(x, str) else list(x))
        for typ, key in (("truck", "trucks"), ("airplane", "planes")):
            for v, x in d[key].items():
                s.add((typ, v))
                s |= {("at", v, l) for l in as_list(x)}
        veh = set(d["trucks"]) | set(d["planes"])
        for p, x in d["packages"].items():
            s.add(("obj", p))
            s |= {("in" if l in veh else "at", p, l) for l in as_list(x)}
        if not all(isinstance(y, str) for f in s for y in f):
            return None
        return frozenset(s)
    except Exception:
        return None


def goal_json(goal: frozenset) -> dict:
    return {f[1]: f[2] for f in sorted(goal) if f[0] == "at" and len(f) == 3}


def action_str(a: tuple) -> str:
    return " ".join(a)


fact_str = B.fact_str


def _names(objs) -> set | frozenset | None:
    """Object names of a by-type dict, a set or an iterable of names; None if objs is missing or malformed."""
    if isinstance(objs, (set, frozenset)):
        return objs
    try:
        return set(all_objects(objs))
    except TypeError:
        return None


def parse_action(t: str, objs) -> tuple | None:
    """Operator name, arity (3/3/3/3/4/3) and problem objects are checked; types are NOT (type preconditions are
    the simulator's job, as in the PDDL). None also when objs is missing (blocks.parse_plan parity)."""
    p = t.strip().split() if isinstance(t, str) else []
    names = _names(objs)
    if not p or names is None or p[0] not in ARITY or len(p) - 1 != ARITY[p[0]] or not all(x in names for x in p[1:]):
        return None
    return tuple(p)


# Final prompt texts (the synthesis drafts measured at 164 / 84 Titan tokens; the module SHA pins them).
SYS_FORWARD_DELTA = (
    "Logistics simulator. Types come from the groups: packages, trucks, planes, locations (map.cities), airports. "
    "load-truck P T L needs truck T and package P at L; P goes into T. unload-truck P T L needs T at L and P in T; "
    "P goes to L. load-airplane P A L and unload-airplane P A L are the same for plane A. drive-truck T L1 L2 C needs "
    "T at L1 and L1, L2 both in city C. fly-airplane A L1 L2 needs A at L1 and L1, L2 airports. An action whose "
    "requirements fail is illegal and changes nothing. Given the state and one action, reply with JSON: whether it "
    "is legal and the facts it adds and deletes.")
SYS_PROPOSER = (
    "Logistics planner. Actions: load-truck P T L, unload-truck P T L, load-airplane P A L, unload-airplane P A L "
    "(vehicle and package at L / package in vehicle), drive-truck T L1 L2 C (within city C), fly-airplane A L1 L2 "
    "(between airports). Bring every package to its goal location. Reply with a JSON plan.")
SYS_FORWARD_DELTA_LG = SYS_FORWARD_DELTA
SYS_PROPOSER_LG = SYS_PROPOSER


def proposer_prompt(s: frozenset, goal: frozenset, outcome: str = "success") -> str:
    return render_chat(SYS_PROPOSER, canonical({"goal": goal_json(goal), "outcome": outcome, "state": state_json(s)}))


def target_actions(plan) -> str:
    return canonical({"actions": [action_str(a) for a in plan]})


proposer_target = target_actions


def forward_prompt_delta(s: frozenset, a: tuple) -> str:
    return render_chat(SYS_FORWARD_DELTA, canonical({"action": action_str(a), "state": state_json(s)}))


forward_target_delta = B.forward_target_delta  # {"add":[..],"del":[..],"legal":b}; a self-move: add = del = []
parse_forward_delta = B.parse_forward_delta


def parse_plan(text: str, objs, mode: str = "actions"):
    """(actions, states-or-None) or None; blocks.parse_plan with the Logistics parser and a 96-action cap. Bad input
    (not JSON, objs missing or malformed) gives None, never an exception."""
    try:
        d = json.loads(text.strip())
    except Exception:
        return None
    names = _names(objs)
    if names is None:
        return None
    try:
        if mode == "actions":
            acts = [parse_action(t, names) for t in d["actions"]]
            states = None
        else:
            acts = [parse_action(st["action"], names) for st in d["steps"]]
            states = [state_from_json(st["state"]) for st in d["steps"]]
    except Exception:
        return None
    if not acts or any(a is None for a in acts) or len(acts) > MAX_PLAN_ACTIONS:
        return None
    return acts, states


def parse_plan_prefix(text: str, objs):
    """Leading complete actions of a possibly truncated action-only plan (blocks.parse_plan_prefix); None on bad
    input (text not a string, objs missing or malformed)."""
    full = parse_plan(text, objs, "actions")
    if full is not None:
        return full
    names = _names(objs)
    if not isinstance(text, str) or names is None:
        return None
    head = text.split('"actions"', 1)
    if len(head) < 2:
        return None
    acts = []
    for m in re.finditer(r'"([^"]*)"', head[1]):
        a = parse_action(m.group(1), names)
        if a is None or len(acts) >= MAX_PLAN_ACTIONS:
            break
        acts.append(a)
    return (acts, None) if acts else None
