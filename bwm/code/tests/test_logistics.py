"""Unit tests for the PlanBench Logistics domain module (bwmn/logistics.py; item 9 CPU-2). CPU only: the local VAL
binary and (optionally) the standalone tokenizers package; never torch/transformers/peft. About 1.5 min; the A* checks
on the 4-7-package PlanBench instances dominate (expansion-capped, so the outcome is machine-independent). VAL runs
are sequential. Problem and plan files for VAL go to system temp dirs that are removed.

  python -B -m pytest tests/test_logistics.py -q -s -p no:cacheprovider
"""
from __future__ import annotations

import itertools
import json
import random
import re
import sys
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bwmn import logistics as L  # noqa: E402
from bwmn import mystery as M  # noqa: E402

TOKENIZER = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                 "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze") / "tokenizer.json"
SEED = 4242
N_RENAMINGS = 1000


@pytest.fixture(scope="module")
def pb():
    return L.planbench_instances()


@pytest.fixture(scope="module")
def pb_gt(pb):
    return [x for x in pb if x["gt"] is not None]


@pytest.fixture(scope="module")
def pb_sigs():
    return L.planbench_signatures()


@pytest.fixture(scope="module")
def tmpdir_mod():
    with tempfile.TemporaryDirectory(prefix="bwm_lg_test_") as d:
        yield Path(d)


def legal_actions(s):
    return [a for a in L.ground_actions(L.objs_of(s), s, self_loops=True) if L.applicable(s, a)]


def bfs_len(init, goal):
    """Plain BFS on fact sets with the reference semantics (optimality cross-check for A*)."""
    acts = L.ground_actions(L.objs_of(init), init)
    if L.goal_met(init, goal):
        return 0
    seen, q = {init}, deque([(init, 0)])
    while q:
        s, d = q.popleft()
        for a in acts:
            ns, ok = L.step(s, a)
            if ok and ns not in seen:
                if L.goal_met(ns, goal):
                    return d + 1
                seen.add(ns)
                q.append((ns, d + 1))
    return None


# ----------------------------------------------------------------------------- domain file and PlanBench assets

def test_schemas_match_domain_file():
    dom = M.parse_domain(L.DOMAIN_PDDL)
    assert dom.name == L.DOMAIN_NAME
    assert set(dom.actions) == set(L.ARITY)
    for op, sc in dom.actions.items():
        assert len(sc.params) == L.ARITY[op]
        args = tuple(f"o{i}" for i in range(len(sc.params)))
        b = dict(zip(sc.params, args))
        sub = lambda atoms: frozenset((a[0],) + tuple(b[x] for x in a[1:]) for a in atoms)
        assert L.ground((op,) + args) == (sub(sc.pre), sub(sc.add), sub(sc.dele)), op
        types = tuple(next(a[0] for a in sc.pre if len(a) == 2 and a[1] == v) for v in sc.params)
        assert types == L.PARAM_TYPES[op], op


def test_planbench_shape(pb):
    assert len(pb) == 286 and sum(x["gt"] is not None for x in pb) == 285
    assert [x["id"] for x in pb if x["gt"] is None] == [1]
    for x in pb:
        o, s, g = x["objs"], x["init"], x["goal"]
        nc = len(o["city"])
        assert nc in (2, 3) and len(o["truck"]) == nc and 1 <= len(o["airplane"]) <= nc
        sizes = Counter(f[2] for f in s if f[0] == "in-city")
        assert len(set(sizes.values())) == 1 and set(sizes.values()) <= {1, 2, 3}
        assert sorted(o["airport"]) == sorted(f"l{c[1:]}-0" for c in o["city"])
        for t in o["truck"]:
            loc = next(f[2] for f in s if f[0] == "at" and f[1] == t)
            assert loc.startswith(f"l{t[1:]}-")
        assert all(next(f[2] for f in s if f[0] == "at" and f[1] == a) in o["airport"] for a in o["airplane"])
        assert sorted(f[1] for f in g) == o["obj"] and all(f[0] == "at" for f in g)
    print(f"\n[shape] package counts ids 2-286: {sorted(Counter(x['n_packages'] for x in pb if x['gt']).items())}")


# ----------------------------------------------------------------------------- semantics vs VAL

def test_gt_plans_valid_executor(pb_gt):
    ok = sum(L.execute(x["init"], x["goal"], x["gt"])["success"] for x in pb_gt)
    print(f"\n[GT] executor: {ok}/{len(pb_gt)} valid")
    assert ok == len(pb_gt) == 285


def test_gt_plans_valid_val(pb_gt):
    ok = sum(L.val_check(x["path"], x["gt"]) for x in pb_gt)
    print(f"\n[GT] local VAL: {ok}/{len(pb_gt)} valid")
    assert ok == 285


def test_write_problem_roundtrip_and_val(pb_gt, tmpdir_mod):
    for x in pb_gt[::20]:
        txt = L.write_problem(x["objs"], x["init"], x["goal"], name=f"rt{x['id']}")
        assert "(:domain logistics-strips)" in txt
        p = tmpdir_mod / f"rt{x['id']}.pddl"
        p.write_text(txt)
        assert L.parse_problem(p) == (x["objs"], x["init"], x["goal"])
        assert L.val_check(p, x["gt"])
        assert not L.val_check(p, x["gt"][:-1])


def test_self_loops_are_legal_noops(pb):
    x = next(y for y in pb if y["id"] == 2)  # a0, a1 at l1-0; t1 at l1-0 (c1)
    s = x["init"]
    for a in (("fly-airplane", "a0", "l1-0", "l1-0"), ("drive-truck", "t1", "l1-0", "l1-0", "c1")):
        assert L.is_self_loop(a)
        assert L.step(s, a) == (s, True)
        assert L.forward_target_delta(s, s, True) == '{"add":[],"del":[],"legal":true}'
    plan = [("fly-airplane", "a0", "l1-0", "l1-0"), ("drive-truck", "t1", "l1-0", "l1-0", "c1")] + x["gt"]
    assert L.execute(x["init"], x["goal"], plan)["success"]
    assert L.val_check(x["path"], plan)
    # a self-move whose preconditions fail stays illegal
    for a in (("fly-airplane", "a0", "l0-0", "l0-0"), ("drive-truck", "t1", "l1-0", "l1-0", "c0"),
              ("drive-truck", "t0", "l1-0", "l1-0", "c1")):
        assert L.step(s, a) == (s, False)
        assert not L.val_check(x["path"], [a] + x["gt"])


def test_type_wrong_and_malformed_actions_rejected(pb):
    x = next(y for y in pb if y["id"] == 2)
    s, names = x["init"], set(L.all_objects(x["objs"]))
    bad = [("load-truck", "p0", "a1", "l1-0"), ("load-airplane", "p0", "t1", "l1-0"), ("load-truck", "t1", "t1", "l1-0"),
           ("fly-airplane", "t1", "l1-0", "l0-0"), ("drive-truck", "a0", "l1-0", "l1-0", "c1"),
           ("drive-truck", "t1", "l1-0", "l0-0", "c1"), ("unload-airplane", "p0", "a1", "c0")]
    for a in bad:
        assert L.step(s, a) == (s, False), a
        assert L.parse_action(L.action_str(a), names) == a  # the parser does not check types
    assert not L.val_check(x["path"], [bad[0]] + x["gt"])
    for a in (("load-truck", "p0", "t1"), ("fly-airplane", "a1", "l1-0", "l0-0", "c0"), ("pick-up", "p0"), ()):
        assert L.ground(a) is None and L.step(s, a) == (s, False)
    assert L.parse_action("load-truck p0 t1", names) is None
    assert L.parse_action("load-truck p0 t9 l1-0", names) is None
    assert L.parse_action("stack p0 t1", names) is None
    assert L.parse_action("drive-truck t1 l1-0 l0-0 c1", names) == ("drive-truck", "t1", "l1-0", "l0-0", "c1")


def test_ground_action_counts_and_completeness(pb, pb_gt):
    n = {x["id"]: len(L.ground_actions(x["objs"], x["init"])) for x in pb}
    n_sl = {x["id"]: len(L.ground_actions(x["objs"], x["init"], self_loops=True)) for x in pb}
    print(f"\n[ground] typed (no self-moves) min {min(n.values())} max {max(n.values())}; "
          f"with self-moves min {min(n_sl.values())} max {max(n_sl.values())}")
    assert all(14 <= v <= 720 for v in n.values())
    assert min(n.values()) == 14 and max(n.values()) == 720
    rng = random.Random(SEED)
    for x in pb_gt[::10]:
        ga = set(L.ground_actions(x["objs"], x["init"], self_loops=True))
        C = L._compile(x["init"], x["goal"])
        s = x["init"]
        for a in x["gt"] + [None]:
            app = {g for g in ga if L.applicable(s, g)}
            # A* successor generator == applicable non-self-move ground actions (reference semantics)
            st = _compact(C, s)
            assert {b for b, _ in L._successors(C, st)} == {g for g in app if not L.is_self_loop(g)}
            for b, nst in L._successors(C, st):
                assert nst == _compact(C, L.step(s, b)[0])
            # completeness: random untyped actions over all objects are applicable only if grounded
            objs = L.all_objects(x["objs"])
            for _ in range(200):
                op = rng.choice(sorted(L.ARITY))
                b = (op,) + tuple(rng.choice(objs) for _ in range(L.ARITY[op]))
                assert not L.applicable(s, b) or b in ga
            if a is not None:
                s = L.step(s, a)[0]


def _compact(C, s):
    li = {l: i for i, l in enumerate(C["L"])}
    vi = {v: i for i, v in enumerate(C["V"])}
    pos = {f[1]: f for f in s if f[0] in ("at", "in")}
    out = []
    for p in C["P"]:
        f = pos[p]
        out.append(li[f[2]] if f[0] == "at" else C["nL"] + vi[f[2]])
    return tuple(out + [li[pos[v][2]] for v in C["V"]])


# ----------------------------------------------------------------------------- A* teacher

def test_astar_optimal_on_small_planbench(pb_gt):
    small = [x for x in pb_gt if x["n_packages"] <= 3]
    solved, worst = 0, 0
    for x in small:
        st = {}
        p = L.astar_plan(x["init"], x["goal"], max_expansions=200_000, stats=st)
        worst = max(worst, st["expansions"])
        if p is None:
            continue
        solved += 1
        assert len(p) == len(x["gt"]), x["id"]
        assert L.execute(x["init"], x["goal"], p)["success"]
        assert st["h0"] <= len(p)
    print(f"\n[A*] <=3 packages: solved {solved}/{len(small)} within 200k expansions, "
          f"length == FD GT on all solved; max expansions {worst}")
    assert len(small) == 73 and solved == 73


def test_astar_optimal_on_large_planbench_capped(pb_gt):
    big = [x for x in pb_gt if x["n_packages"] >= 4]
    solved, worst, by = 0, 0, Counter()
    for x in big:
        st = {}
        p = L.astar_plan(x["init"], x["goal"], max_expansions=100_000, stats=st)
        worst = max(worst, st["expansions"])
        if p is None:
            continue
        solved += 1
        by[x["n_packages"]] += 1
        assert len(p) == len(x["gt"]), x["id"]
    print(f"\n[A*] 4-7 packages: solved {solved}/{len(big)} within 100k expansions {dict(sorted(by.items()))}, "
          f"length == FD GT on all solved; max expansions {worst}")
    assert len(big) == 212 and solved == 212


def test_astar_cap_counts_real_expansions(pb, monkeypatch):
    """stats['expansions'] == nodes actually expanded (calls of _successors); a capped run reports exactly the cap."""
    x = next(y for y in pb if y["id"] == 186)  # 7 packages, 5102 expansions uncapped
    calls = [0]
    orig = L._successors

    def counted(C, s):
        calls[0] += 1
        return orig(C, s)

    monkeypatch.setattr(L, "_successors", counted)

    def run(cap):
        calls[0], st = 0, {}
        p = L.astar_plan(x["init"], x["goal"], max_expansions=cap, stats=st)
        assert st["expansions"] == calls[0], (cap, st, calls[0])
        return p, st

    p, st = run(10 ** 6)
    full = st["expansions"]
    assert p is not None and st["status"] == "solved" and full == 5102
    for cap in (0, 1, 7, 100, full - 1):
        p, st = run(cap)
        assert p is None and st["status"] == "cap" and st["expansions"] == cap
    p, st = run(full)
    assert p is not None and st["status"] == "solved" and st["expansions"] == full


def test_astar_matches_bfs_on_generated():
    rng = random.Random(SEED)
    n = 0
    while n < 30:
        g = L.generate(rng, rng.choice((1, 2)), n_cities=2, city_size=rng.choice((1, 2)), rename=0.5)
        if g is None:
            continue
        objs, init, goal = g
        p = L.astar_plan(init, goal)
        assert p is not None and L.execute(init, goal, p)["success"]
        assert len(p) == bfs_len(init, goal)
        n += 1


# ----------------------------------------------------------------------------- renaming signature

def _arbitrary_renaming(objs, init, goal, rng):
    """A random within-type bijection onto fresh names (no PlanBench grammar)."""
    m = {}
    for t, pre in (("city", "k"), ("location", "x"), ("truck", "u"), ("airplane", "w"), ("obj", "q")):
        xs = objs[t]
        for o, k in zip(xs, rng.sample(range(1000), len(xs))):
            m[o] = f"{pre}{k}"
    return L.apply_renaming(m, init, goal)


def test_signature_invariant_under_renamings(pb, pb_sigs):
    rng = random.Random(SEED)
    done = 0
    while done < N_RENAMINGS:
        x = pb[done % len(pb)]
        if done % 2:
            o2, i2, g2 = L.rename_instance(x["objs"], x["init"], x["goal"], rng)
        else:
            o2, i2, g2 = _arbitrary_renaming(x["objs"], x["init"], x["goal"], rng)
        assert L.signature(o2, i2, g2) == pb_sigs[x["id"]]
        done += 1
    classes = defaultdict(list)
    for i, s in pb_sigs.items():
        classes[s].append(i)
    dups = sorted(v for v in classes.values() if len(v) > 1)
    print(f"\n[signature] {N_RENAMINGS} renamings invariant; {len(classes)} distinct signatures over "
          f"{len(pb_sigs)} generated_basic files; collisions {dups}")
    assert len(pb_sigs) == 286 and len(classes) == 285 and dups == [[47, 48]]


def test_signature_separates_changes(pb, pb_sigs):
    x = next(y for y in pb if y["id"] == 150)
    s, g = x["init"], x["goal"]
    p, l = sorted(g)[0][1:]
    other = next(z for z in x["objs"]["location"] if z != l)
    assert L.signature(None, s, (g - {("at", p, l)}) | {("at", p, other)}) != pb_sigs[150]
    a = next(b for b in legal_actions(s) if not L.is_self_loop(b))
    assert L.signature(None, L.step(s, a)[0], g) != pb_sigs[150]


def _bf_canon(init, goal):
    """Brute-force canonical form: min over every within-type bijection of the renamed sorted fact lists."""
    o = L.objs_of(init)
    types = ("city", "location", "truck", "airplane", "obj")
    best = None
    for perms in itertools.product(*[itertools.permutations(o[t]) for t in types]):
        m = {x: f"{t}{i}" for t, perm in zip(types, perms) for i, x in enumerate(perm)}
        key = repr((sorted((f[0],) + tuple(m[y] for y in f[1:]) for f in init),
                    sorted((f[0],) + tuple(m[y] for y in f[1:]) for f in goal)))
        if best is None or key < best:
            best = key
    return best


def test_signature_exact_on_small_states():
    """sig(x) == sig(y) iff x and y are within-type renamings (brute force), on small instances and walk states."""
    rng = random.Random(SEED)
    items = []
    while len(items) < 160:
        g = L.generate(rng, rng.choice((1, 2)), n_cities=2, city_size=rng.choice((1, 2)), n_planes=rng.choice((1, 2)),
                       rename=0.5)
        if g is None:
            continue
        objs, s, goal = g
        for _ in range(rng.randrange(4)):
            s = L.step(s, rng.choice(legal_actions(s)))[0]
        items.append((s, goal))
        if rng.random() < 0.3:
            items.append(_arbitrary_renaming(L.objs_of(s), s, goal, rng)[1:])
    sig2bf, bf2sig = defaultdict(set), defaultdict(set)
    for s, goal in items:
        sg, bf = L.signature(None, s, goal), _bf_canon(s, goal)
        sig2bf[sg].add(bf)
        bf2sig[bf].add(sg)
    print(f"\n[signature] exactness: {len(items)} small states, {len(sig2bf)} classes, "
          f"{len(items) - len(sig2bf)} isomorphic pairs merged")
    assert all(len(v) == 1 for v in sig2bf.values()) and all(len(v) == 1 for v in bf2sig.values())
    assert len(items) - len(sig2bf) > 20


def _structural_canon(init, goal):
    """Independent brute force for city size 3: min over city orders x orders of each city's non-airport locations
    (its airport fixed: one per city) x truck, plane and package orders of the renamed sorted fact lists."""
    o = L.objs_of(init)
    city_of = {f[1]: f[2] for f in init if f[0] == "in-city"}
    aps = set(o["airport"])
    best = None
    for cp in itertools.permutations(o["city"]):
        per = [[([l for l in o["location"] if city_of[l] == c and l in aps], q) for q in itertools.permutations(
            [l for l in o["location"] if city_of[l] == c and l not in aps])] for c in cp]
        for ch in itertools.product(*per):
            m = {c: f"C{i}" for i, c in enumerate(cp)}
            for ci, (ap, q) in enumerate(ch):
                m.update({l: f"L{ci}a{k}" for k, l in enumerate(ap)})
                m.update({l: f"L{ci}n{k}" for k, l in enumerate(q)})
            for tp, ap_, pp in itertools.product(*(itertools.permutations(o[t]) for t in ("truck", "airplane", "obj"))):
                mm = dict(m)
                for pre, perm in (("T", tp), ("A", ap_), ("P", pp)):
                    mm.update({x: f"{pre}{i}" for i, x in enumerate(perm)})
                key = repr((sorted((f[0],) + tuple(mm[y] for y in f[1:]) for f in init),
                            sorted((f[0],) + tuple(mm[y] for y in f[1:]) for f in goal)))
                if best is None or key < best:
                    best = key
    return best


def test_signature_exact_city_size_3():
    """Exactness where within-city permutations of non-airport locations matter (city size 3, the dominant PB-OUT
    shape): sig(x) == sig(y) iff the structural brute force agrees; isomorphic twins permute those locations."""
    rng = random.Random(SEED + 3)
    items, n_twins = [], 0
    while len(items) < 200:
        g = L.generate(rng, rng.choice((1, 2, 3)), n_cities=rng.choice((2, 3)), city_size=3,
                       n_planes=rng.choice((1, 2)), rename=0.5)
        if g is None:
            continue
        objs, s, goal = g
        for _ in range(rng.randrange(5)):
            s = L.step(s, rng.choice(L.applicable_actions(s, self_loops=False)))[0]
        items.append((s, goal))
        if rng.random() < 0.5:  # an isomorphic twin: shuffle non-airport locations within each city and the objects
            o = L.objs_of(s)
            city_of = {f[1]: f[2] for f in s if f[0] == "in-city"}
            m = {}
            for c in o["city"]:
                na = [l for l in o["location"] if city_of[l] == c and l not in set(o["airport"])]
                m.update(zip(na, rng.sample(na, len(na))))
            for t in ("truck", "airplane", "obj"):
                m.update(zip(o[t], rng.sample(o[t], len(o[t]))))
            n_twins += any(k != v for k, v in m.items() if k in o["location"])
            items.append(L.apply_renaming(m, s, goal)[1:])
    sig2bf, bf2sig = defaultdict(set), defaultdict(set)
    for s, goal in items:
        sg, bf = L.signature(None, s, goal), _structural_canon(s, goal)
        sig2bf[sg].add(bf)
        bf2sig[bf].add(sg)
    print(f"\n[signature] city size 3: {len(items)} states, {len(sig2bf)} classes, {n_twins} twins with permuted "
          f"non-airport locations")
    assert all(len(v) == 1 for v in sig2bf.values()) and all(len(v) == 1 for v in bf2sig.values())
    assert n_twins > 30 and len(items) - len(sig2bf) >= n_twins


def test_signature_excludes_planbench_renamings(pb, pb_sigs):
    rng = random.Random(SEED + 1)
    sigset = set(pb_sigs.values())
    for x in pb[::10]:
        o2, i2, g2 = L.rename_instance(x["objs"], x["init"], x["goal"], rng)
        assert L.signature(o2, i2, g2) in sigset
    # a generated instance equal to a PlanBench one up to names is caught; instance 2 is c2-s1-p1-a2
    objs, init, goal = L.build_instance(2, 1, [0, 0], [1, 1], [(1, 0)], [(0, 0)], pkg_names=["p7"])
    assert L.signature(objs, init, goal) == pb_sigs[2]


# ----------------------------------------------------------------------------- rendering, delta and parsers

def test_state_json_roundtrip_and_delta_on_random_walks(pb):
    rng = random.Random(SEED)
    starts = [(x["init"], x["goal"]) for x in pb[::15]]
    while len(starts) < 60:
        g = L.generate(rng, rng.choice((1, 2, 3)), rename=0.5)
        if g is not None:
            starts.append(g[1:])
    n_steps, kinds = 0, Counter()
    for s, goal in starts:
        objs = L.objs_of(s)
        ga = L.ground_actions(objs, s, self_loops=True)
        for _ in range(25):
            r = rng.random()
            if r < 0.5:
                a = rng.choice(legal_actions(s))
            elif r < 0.85:
                a = rng.choice(ga)
            else:  # right arity, random (often wrong-type) arguments
                op = rng.choice(sorted(L.ARITY))
                a = (op,) + tuple(rng.choice(L.all_objects(objs)) for _ in range(L.ARITY[op]))
            ns, legal = L.step(s, a)
            kinds["legal" if legal else "illegal"] += 1
            kinds["self-move"] += legal and L.is_self_loop(a)
            assert L.state_from_json(json.loads(L.canonical(L.state_json(s)))) == s
            tgt = L.forward_target_delta(s, ns, legal)
            assert L.parse_forward_delta(tgt, s) == (ns, legal)
            d = json.loads(tgt)
            if legal and not L.is_self_loop(a):
                assert len(d["add"]) == 1 and len(d["del"]) == 1
            else:
                assert d["add"] == [] and d["del"] == []
            prompt = L.forward_prompt_delta(s, a)
            assert prompt.startswith("<|im_start|>system\n" + L.SYS_FORWARD_DELTA) and L.action_str(a) in prompt
            s = ns
            n_steps += 1
    print(f"\n[delta] {n_steps} random-walk steps round-tripped {dict(kinds)}")
    assert kinds["legal"] > 300 and kinds["illegal"] > 300 and kinds["self-move"] > 5


def test_state_json_format(pb):
    x = next(y for y in pb if y["id"] == 2)
    assert L.state_json(x["init"]) == {"map": {"airports": ["l0-0", "l1-0"], "cities": {"c0": ["l0-0"], "c1": ["l1-0"]}},
                                       "packages": {"p0": "l1-0"}, "planes": {"a0": "l1-0", "a1": "l1-0"},
                                       "trucks": {"t0": "l0-0", "t1": "l1-0"}}
    s = L.step(x["init"], ("load-airplane", "p0", "a1", "l1-0"))[0]
    assert L.state_json(s)["packages"] == {"p0": "a1"}
    assert L.goal_json(x["goal"]) == {"p0": "l0-0"}
    pp = L.proposer_prompt(x["init"], x["goal"])
    assert pp.startswith("<|im_start|>system\n" + L.SYS_PROPOSER)
    assert '{"goal":{"p0":"l0-0"},"outcome":"success","state":{"map":' in pp
    assert L.unmet(x["init"], x["goal"]) == 1 and L.unmet(L.execute(x["init"], x["goal"], x["gt"])["final"],
                                                          x["goal"]) == 0
    for bad in ({}, {"map": {}, "packages": {}, "planes": {}, "trucks": {}}, {"map": {"airports": [1], "cities": {}},
                                                                             "packages": {}, "planes": {}, "trucks": {}}):
        assert L.state_from_json(bad) is None


def test_plan_parser_roundtrip(pb_gt):
    for x in pb_gt:
        names = L.all_objects(x["objs"])
        txt = L.target_actions(x["gt"])
        assert L.parse_plan(txt, x["objs"]) == (x["gt"], None)
        assert L.plan_from_pddl(L.plan_to_pddl(x["gt"])) == x["gt"]
        # truncated generations: the leading complete actions
        ends, pos = [], len('{"actions":[')
        for a in x["gt"]:
            pos += len(L.action_str(a)) + 2
            ends.append(pos)
            pos += 1
        for cut in sorted({len(txt) - 1, len(txt) - 3, len(txt) // 2, 15, ends[0] - 1, ends[0]}):
            k = sum(e <= cut for e in ends)
            got = L.parse_plan_prefix(txt[:cut], names)
            assert got == ((x["gt"][:k], None) if k else None), (x["id"], cut)
    x = pb_gt[0]
    steps = [{"action": L.action_str(a), "state": L.state_json(L.step(x["init"], a)[0])} for a in x["gt"][:1]]
    out = L.parse_plan(L.canonical({"steps": steps}), x["objs"], "steps")
    assert out[0] == x["gt"][:1] and out[1] == [L.step(x["init"], x["gt"][0])[0]]
    a0 = L.action_str(x["gt"][0])
    assert L.parse_plan(L.canonical({"actions": [a0] * L.MAX_PLAN_ACTIONS}), x["objs"]) is not None
    assert L.parse_plan(L.canonical({"actions": [a0] * (L.MAX_PLAN_ACTIONS + 1)}), x["objs"]) is None
    assert L.parse_plan(L.canonical({"actions": [a0, "load-truck p0 t9 l0-0"]}), x["objs"]) is None
    assert L.parse_plan(L.canonical({"actions": []}), x["objs"]) is None
    assert L.parse_plan("not json", x["objs"]) is None
    assert L.parse_plan(L.canonical({"actions": ["load-truck p0 a1 l1-0"]}), x["objs"]) == (
        [("load-truck", "p0", "a1", "l1-0")], None)


def test_plan_from_pddl_published_forms():
    assert L.plan_from_pddl("(LOAD-TRUCK p0 t0 l0-0)\n; cost = 1 (unit cost)\n") == [("load-truck", "p0", "t0", "l0-0")]
    assert L.plan_from_pddl(["(fly-airplane a0 l0-0 l1-0)", "(unload-airplane p0 a0 l1-0)"]) == [
        ("fly-airplane", "a0", "l0-0", "l1-0"), ("unload-airplane", "p0", "a0", "l1-0")]
    assert L.plan_from_pddl("") == [] and L.plan_from_pddl(None) == []


def _published_raw(model_file: str, iid: int) -> str:
    d = json.loads((L.RESULTS_DIR / model_file).read_text(encoding="utf-8"))
    x = next(y for y in d["instances"] if int(y["instance_id"]) == iid)
    p = x["extracted_llm_plan"]
    return "\n".join(p) if isinstance(p, list) else p


def test_val_skipped_characters(pb):
    """Characters VAL's lexer skips (e.g. ',') read as blanks in plan_from_pddl and read_val_plan (review SEM-1)."""
    assert L.plan_from_pddl("(unload-truck p2 t0, l0-0)") == [("unload-truck", "p2", "t0", "l0-0")]
    x = next(y for y in pb if y["id"] == 2)
    G = L.plan_to_pddl(x["gt"])
    for ch in ",\"'\\{}|!#$%&@^`~\t\r\x7f\u00a0\u2192":
        t = G.replace("p0 a1", f"p0{ch}a1", 1)  # glued, as in the published 't0,l0-0'
        assert L.plan_from_pddl(t) == x["gt"], repr(ch)
        assert [a for _, a in L.read_val_plan(t)] == x["gt"], repr(ch)
        assert L.execute_text(x["init"], x["goal"], t)["success"], repr(ch)
    assert L.val_run(x["path"], G.replace("p0 a1", "p0,a1", 1))["valid"]
    # published gpt-4 one-shot NL id 29: action 4 is '(unload-truck p2 t0, l0-0)'; VAL fails at happening 23
    raw = _published_raw("gpt-4_chat/task_1_plan_generation.json", 29)
    assert "(unload-truck p2 t0, l0-0)" in raw
    _, init, goal = L.parse_problem(L.planbench_path(29))
    assert L.execute(init, goal, L.plan_from_pddl(raw))["fail_step"] == 22
    assert L.execute_text(init, goal, raw)["fail_step"] == 22
    out = L.val_run(L.planbench_path(29), raw, verbose=True)
    assert not out["valid"] and "Plan failed to execute" in out["stdout"]
    assert re.findall(r"Checking next happening \(time (\d+)\)", out["stdout"])[-1] == "23"


def test_read_val_plan_cases_match_val(pb):
    """Pinned raw-text forms (probed on the local VAL): steps read (None = 'Bad plan description!') and verdict;
    VAL's own -v listing and verdict are re-checked on every case."""
    x = next(y for y in pb if y["id"] == 2)
    G = L.plan_to_pddl(x["gt"])
    a, b, c = G.strip().split("\n")
    cases = {
        "clean": (G, 3, True), "comma": (G.replace("p0 a1", "p0, a1", 1), 3, True), "upper": (G.upper(), 3, True),
        "comments": ("; plan\n" + G + "; cost = 3 (unit cost)\n", 3, True), "empty": ("", 0, False),
        "leading prose": ("Here is my plan:\n" + G, 0, False), "trailing prose": (G + "This plan works.\n", 3, True),
        "word between": (f"{a} and\n{b}\n{c}\n", 1, False), "stray ) between": (f"{a}\n)\n{b}\n{c}\n", 1, False),
        "stray ) after": (G + ")\n", 3, True), "unclosed last": (G + "(fly-airplane a0 l0-0 l1-0\n", None, False),
        "nested": (f"({a})\n{b}\n{c}\n", None, False), "empty group": (G + "()\n", None, False),
        "numbered 1.": (f"1. {a}\n2. {b}\n3. {c}\n", None, False), "dash list": (f"- {a}\n- {b}\n- {c}\n", 0, False),
        "md fence": ("```\n" + G + "```\n", 3, True), "md fence pddl": ("```pddl\n" + G + "```\n", 0, False),
        "ts increasing": (f"0: {a}\n1: {b}\n2: {c}\n", 3, True),
        "ts float + durations": (f"0.000: {a} [1.000]\n0.5: {b} [1]\n7.25: {c} [0]\n", 3, True),
        "ts unsorted": (f"1: {b}\n0: {a}\n2: {c}\n", 3, True), "ts decreasing": (f"2: {a}\n1: {b}\n0: {c}\n", 3, False),
        "ts equal (mutex)": (f"0: {a}\n0: {b}\n1: {c}\n", 3, False), "ts mixed": (f"{a}\n{b}\n9: {c}\n", 3, True),
        "ts mixed early": (f"{a}\n{b}\n0: {c}\n", 3, False), "durations only": (f"{a} [1]\n{b} [1]\n{c} [1]\n", 3, True),
        "time N": ("time 3\n" + G, 3, True), "bare time": (G + "time\n", None, False),
        "illegal symbol": ("1e0: " + G, None, False), "#t in group": (G.replace("a1 l1-0)", "a1 #t)", 1), None, False),
        "#t between": (f"{a}\n#t\n{b}\n{c}\n", 1, False), "digit after": (G + "2\n", None, False),
        # VAL binds the declared parameters and ignores extra arguments (even undeclared names)
        "extra argument": (G.replace("a1 l1-0)", "a1 l1-0 zzz)", 1), 3, True),
        "extra arguments everywhere": ("".join(ln[:-1] + " c0 p0)\n" for ln in (a, b, c)), 3, True),
        "extra argument binds wrong": (G.replace("a1 l1-0)", "a1 a0 l1-0)", 1), 3, False),
        # no listing: VAL stops on an unknown operator and crashes on a missing argument
        "unknown operator": (G + "(pick-up p0)\n", 4, False), "missing argument": (G.replace("a1 l1-0)", "a1)", 1), 3, False),
    }
    no_listing = {"unknown operator", "missing argument"}
    y = next(z for z in pb if z["id"] == 4)  # GT starts with two loads into a0 at l1-0
    tl = lambda ts: "".join(f"{t}: ({' '.join(act)})\n" for t, act in zip(ts, y["gt"]))
    conc = {"concurrent loads": (tl([0, 0, 2, 3, 4]), 5, True), "duplicate step": (tl([0, 1, 2, 3, 4]) + "0: ("
                                                                                   + " ".join(y["gt"][0]) + ")\n", 6, False)}
    for name, (text, n_read, valid) in list(cases.items()) + [(k, v) for k, v in conc.items()]:
        z = y if name in conc else x
        steps = L.read_val_plan(text)
        assert (None if steps is None else len(steps)) == n_read, name
        assert L.execute_text(z["init"], z["goal"], text)["success"] == valid, name
        out = L.val_run(z["path"], text, verbose=True)
        assert out["valid"] == valid, name
        listing = L.val_listing(out["stdout"])
        assert listing == ("n/a" if name in no_listing else (None if steps is None else sorted(steps))), name
    assert L.plan_from_pddl(cases["extra argument"][0]) == x["gt"]
    assert L.val_bind(("fly-airplane", "a1", "l1-0", "l0-0", "c0")) == ("fly-airplane", "a1", "l1-0", "l0-0")
    assert L.val_bind(("load-truck", "p0")) == ("load-truck", "p0") and L.val_bind(("x", "y")) == ("x", "y")
    s = x["init"]  # tuples stay strict (parse_action's contract): an over-long tuple is illegal for step()
    assert L.step(s, ("fly-airplane", "a1", "l1-0", "l0-0", "c0")) == (s, False)


_SKIPS = list(",\"'\\{}|!#$%&@^`~\r\t\u00a0\u2192") + ["#", "#"]
_JUNK = ["and", "then", "at", "Step", "time", "time 3", "done.", "1.", "-", "*", ")", ":", "]", "[", "2", "1e0", "; note",
         "(", "()", "?", "=", ".5:", "1.:", "Here is the plan:", "```", "```pddl", "1)", "-1:", "#t", "x1-0"]


def _raw_text(plan, rng):
    """A random raw rendering of an action list: case, timestamps (increasing, float, equal, permuted, mixed),
    durations, numbering, separators, skipped characters, junk lines, prose, broken parentheses."""
    lines = ["(" + " ".join(a) + ")" for a in plan]
    if rng.random() < 0.15:
        lines = [ln.upper() if rng.random() < 0.5 else ln for ln in lines]
    n, r = len(lines), rng.random()
    ts = [None] * n
    if r < 0.12:
        ts = list(range(n))
    elif r < 0.2:
        ts = [round(k * 0.5 + rng.random() * 0.4, 3) for k in range(n)]
    elif r < 0.3 and n >= 2:
        ts = list(range(n))
        k = rng.randrange(n - 1)
        ts[k + 1] = ts[k]
    elif r < 0.36:
        ts = rng.sample(range(n), n)
    elif r < 0.42:
        ts = [rng.randrange(n + 2) if rng.random() < 0.5 else None for _ in range(n)]
    lines = [(f"{t}: " if t is not None else "") + ln for t, ln in zip(ts, lines)]
    if rng.random() < 0.12:
        lines = [ln + f" [{rng.choice(['1', '1.000', '0'])}]" for ln in lines]
    if rng.random() < 0.08:
        lines = [f"{k + 1}{rng.choice(['.', ')', ':'])} {ln}" for k, ln in enumerate(lines)]
    text = rng.choice(["\n", "\n", " ", ", ", "\r\n"]).join(lines) + rng.choice(["", "\n"])
    for _ in range(rng.choice([0, 0, 1, 2, 4])):
        k = rng.randrange(len(text) + 1)
        text = text[:k] + rng.choice(_SKIPS) + text[k:]
    if rng.random() < 0.2 and ")" in text:  # extra arguments (VAL ignores them)
        k = rng.choice([m.start() for m in re.finditer(r"\)", text)])
        text = text[:k] + " " + rng.choice(["c0", "zzz", "p0 t0", "l0-0", "a1"]) + text[k:]
    if rng.random() < 0.15:
        ms = [m.start() + 1 for m in re.finditer(r"[ (][a-z]", text)]
        if ms:
            k = rng.choice(ms)
            text = text[:k] + "#" + text[k:]
    if rng.random() < 0.25:
        parts = text.split("\n")
        parts.insert(rng.randrange(len(parts) + 1), rng.choice(_JUNK))
        text = "\n".join(parts)
    if rng.random() < 0.06:
        text = rng.choice(["Here is my plan:\n", "# plan\n", "```\n"]) + text
    if rng.random() < 0.06:
        text += rng.choice(["\nThis plan works.", "\n```", "\n; end", "\n)", "\n(", "\n2"])
    r = rng.random()
    if r < 0.05:
        text = text.rstrip().rstrip(")")
    elif r < 0.08 and ")" in text:
        k = rng.choice([m.start() for m in re.finditer(r"\)", text)])
        text = text[:k] + text[k + 1:]
    elif r < 0.1:
        text = "(" + text + ")"
    return text


def test_execute_text_matches_val_on_random_raw_texts(pb_gt):
    """Random raw texts of GT, self-loop-padded, corrupted and truncated plans: execute_text's verdict == VAL's, and
    read_val_plan == VAL's -v listing wherever VAL prints one (review SEM-1; 6,000 such cases, 5,181 with a listing,
    agreed in development)."""
    rng = random.Random(SEED + 9)
    kinds = Counter()
    for _ in range(250):
        x = rng.choice(pb_gt)
        plan, r = list(x["gt"]), rng.random()
        if r < 0.15:  # self-moves interleaved (concurrency and no-op cases)
            s, new = x["init"], []
            for a in plan:
                if rng.random() < 0.3:
                    new += [b for b in L.applicable_actions(s) if L.is_self_loop(b)][:1]
                new.append(a)
                s = L.step(s, a)[0]
            plan = new
        elif r < 0.3:
            plan[rng.randrange(len(plan))] = rng.choice(L.ground_actions(x["objs"], x["init"], self_loops=True))
        elif r < 0.4:
            plan = plan[:rng.randrange(len(plan))]
        text = _raw_text(plan[:12] if rng.random() < 0.5 else plan, rng)
        out = L.val_run(x["path"], text, verbose=True)
        steps = L.read_val_plan(text)
        ex = L.execute_text(x["init"], x["goal"], text)
        assert ex["success"] == out["valid"], (x["id"], text)
        listing = L.val_listing(out["stdout"])
        if listing != "n/a":
            assert listing == (None if steps is None else sorted(steps)), (x["id"], text)
        kinds["valid" if out["valid"] else "invalid"] += 1
        kinds["bad plan" if steps is None else ("no listing" if listing == "n/a" else "listed")] += 1
    print(f"\n[raw texts] 250 random texts agree with VAL: {dict(kinds)}")
    assert kinds["valid"] > 15 and kinds["bad plan"] > 50 and kinds["listed"] > 50


@pytest.mark.skipif(not TOKENIZER.exists(), reason="Titan tokenizer not available")
def test_token_table(pb_gt):
    tokenizers = pytest.importorskip("tokenizers")
    tk = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    n = lambda t: len(tk.encode(t, add_special_tokens=False).ids)
    dmax, frow, pa, full, per_act = 0, 0, 0, defaultdict(int), []
    for x in pb_gt:
        s = x["init"]
        pp = n(L.proposer_prompt(s, x["goal"]))
        tg = n(L.target_actions(x["gt"]))
        pa = max(pa, pp)
        full[x["n_packages"] <= 3] = max(full[x["n_packages"] <= 3], pp + tg + 1)
        per_act.append((tg - n('{"actions":[]}')) / len(x["gt"]))
        for a in x["gt"]:
            ns, ok = L.step(s, a)
            t = n(L.forward_target_delta(s, ns, ok)) + 1
            dmax = max(dmax, t)
            frow = max(frow, n(L.forward_prompt_delta(s, a)) + t)
            s = ns
    illegal = n(L.forward_target_delta(s, s, False)) + 1
    mean_act = round(sum(per_act) / len(per_act), 2)
    print(f"\n[tokens] sys F {n(L.SYS_FORWARD_DELTA)} Pa {n(L.SYS_PROPOSER)}; delta target max {dmax} (+EOS), "
          f"illegal {illegal}; Pa prompt max {pa}; F row max {frow}; full GT row max 1-3 pk {full[True]}, "
          f"4-7 pk {full[False]}; tokens/action mean {mean_act} max {round(max(per_act), 2)}")
    assert (n(L.SYS_FORWARD_DELTA), n(L.SYS_PROPOSER)) == (164, 84)
    assert dmax <= 26 and illegal == 12
    assert (pa, frow, full[True], full[False]) == (348, 410, 588, 968)
    assert mean_act == 13.0


@pytest.mark.skipif(not TOKENIZER.exists(), reason="Titan tokenizer not available")
def test_generated_row_lengths():
    """Generated (A*-teacher) rows vs the draft max-len 640 (review LG-SPEC-1). Full Pa rows (prompt + whole optimal
    plan + EOS) of 3-package 3x3 maps exceed 640 (the prereg's 'Expected count: 0' does not hold) but stay under 768;
    optimal plans reach 29 steps here (30 in larger samples); legal F rows stay far below 640."""
    tokenizers = pytest.importorskip("tokenizers")
    tk = tokenizers.Tokenizer.from_file(str(TOKENIZER))
    n = lambda t: len(tk.encode(t, add_special_tokens=False).ids)

    def measure(rng, k, mix, **shape):
        res = Counter()
        while res["n"] < k:
            g = L.generate(rng, rng.choice(mix), rename=0.5, **shape)
            if g is None:
                continue
            objs, init, goal = g
            p = L.astar_plan(init, goal)
            row = n(L.proposer_prompt(init, goal)) + n(L.target_actions(p)) + 1
            res["n"] += 1
            res["over640"] += row > 640
            res["len"], res["pa"] = max(res["len"], len(p)), max(res["pa"], row)
            s = init
            for a in p:
                ns = L.step(s, a)[0]
                res["f"] = max(res["f"], n(L.forward_prompt_delta(s, a)) + n(L.forward_target_delta(s, ns, True)) + 1)
                s = ns
        return dict(res)

    worst = measure(random.Random(SEED), 300, (3,), n_cities=3, city_size=3)
    mix = measure(random.Random(SEED), 300, (1, 2, 2, 3, 3, 3))
    print(f"\n[rows] 3 packages on 3x3 maps: {worst}; prereg mix: {mix}")
    assert worst == {"n": 300, "over640": 7, "len": 29, "pa": 682, "f": 388}
    assert mix == {"n": 300, "over640": 0, "len": 24, "pa": 600, "f": 386}


def test_self_loop_share_of_applicable_actions():
    """Under the draft F recipe (start = a teacher-plan state + 0-7 random legal steps), about 36% of the applicable
    typed actions are self-moves (review LG-SPEC-2); applicable_actions(s, self_loops=False) drops them."""
    rng = random.Random(SEED)
    shares, k = [], 0
    while k < 200:
        g = L.generate(rng, rng.choice((1, 2, 2, 3, 3, 3)), rename=0.5)
        if g is None:
            continue
        k += 1
        objs, init, goal = g
        states = [init]
        for a in L.astar_plan(init, goal):
            states.append(L.step(states[-1], a)[0])
        s = rng.choice(states)
        for _ in range(rng.randrange(8)):
            s = L.step(s, rng.choice(L.applicable_actions(s)))[0]
        app = L.applicable_actions(s)
        assert app == legal_actions(s)
        assert L.applicable_actions(s, self_loops=False) == [a for a in app if not L.is_self_loop(a)]
        shares.append(sum(map(L.is_self_loop, app)) / len(app))
    share = sum(shares) / len(shares)
    print(f"\n[F] mean self-move share among applicable typed actions: {share:.3f}")
    assert 0.30 < share < 0.42


def test_parse_forward_delta_rejects_absent_delete(pb):
    x = next(y for y in pb if y["id"] == 2)  # p0 at l1-0
    s = x["init"]
    bad = '{"add":["in p0 a1"],"del":["at p0 l0-0"],"legal":true}'
    good = '{"add":["in p0 a1"],"del":["at p0 l1-0"],"legal":true}'
    assert L.parse_forward_delta(bad, s) is None
    assert L.parse_forward_delta(good, s) == L.step(s, ("load-airplane", "p0", "a1", "l1-0"))


def test_parsers_return_none_on_missing_objs(pb):
    """parse_plan / parse_plan_prefix / parse_action return None (never raise) when objs is missing (LG-SPEC-4)."""
    x = next(y for y in pb if y["id"] == 2)
    txt = L.target_actions(x["gt"])
    for objs in (None, 5, {"obj": None}):
        assert L.parse_plan(txt, objs) is None
        assert L.parse_plan(txt, objs, "steps") is None
        assert L.parse_plan_prefix(txt, objs) is None
        assert L.parse_plan_prefix(txt[:40], objs) is None
        assert L.parse_action("load-airplane p0 a1 l1-0", objs) is None
    assert L.parse_plan_prefix(None, x["objs"]) is None and L.parse_plan(None, x["objs"]) is None
    assert L.parse_plan('{"actions":["x"]}', None) is None
    assert L.parse_plan(txt, x["objs"]) == (x["gt"], None)
    assert L.parse_plan(txt, set(L.all_objects(x["objs"]))) == (x["gt"], None)


def test_ground_cache_bounded():
    """ground()'s LRU cache is bounded to 1 << 14 entries, about 26 MB when full (LG-SPEC-5: 1 << 18 was ~400 MB)."""
    import tracemalloc
    assert L.ground.cache_info().maxsize == 1 << 14
    names = ([f"p{k}" for k in range(12)] + [f"t{k}" for k in range(3)] + [f"a{k}" for k in range(3)]
             + [f"l{i}-{j}" for i in range(3) for j in range(3)] + [f"c{k}" for k in range(3)])
    rng = random.Random(SEED)
    L.ground.cache_clear()
    tracemalloc.start()
    try:
        ops = sorted(L.ARITY)
        for _ in range(3 * (1 << 14)):
            op = rng.choice(ops)
            L.ground((op,) + tuple(rng.choice(names) for _ in range(L.ARITY[op])))
        cur = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    info = L.ground.cache_info()
    L.ground.cache_clear()
    print(f"\n[ground cache] {info.currsize} entries, {cur / 2 ** 20:.1f} MB")
    assert info.currsize == 1 << 14 and cur < 40 * 2 ** 20


# ----------------------------------------------------------------------------- generator

def test_generator_deterministic_by_seed():
    def draw(seed, k=60, rename=0.5):
        rng = random.Random(seed)
        return [L.generate(rng, rng.choice((1, 2, 2, 3, 3, 3)), rename=rename) for _ in range(k)]
    assert draw(7) == draw(7)
    assert draw(7) != draw(8)
    # at rename == 0 the stream does not depend on the option
    a = random.Random(3)
    b = random.Random(3)
    assert [L.generate(a, 2) for _ in range(30)] == [L.generate(b, 2, rename=0.0) for _ in range(30)]
    assert a.random() == b.random()


def test_generator_shape_and_renaming(tmpdir_mod):
    rng = random.Random(SEED)
    shapes, n_none, n_ren = Counter(), 0, 0
    grammar = re.compile(r"^(c\d+|t\d+|a\d+|p\d+|l\d+-\d+)$")
    for k in range(400):
        npk = rng.choice((1, 2, 2, 3, 3, 3))
        g = L.generate(rng, npk, rename=0.5)
        if g is None:
            n_none += 1
            continue
        objs, init, goal = g
        nc, size = len(objs["city"]), len(objs["location"]) // len(objs["city"])
        shapes[(nc, size, len(objs["airplane"]))] += 1
        assert nc in (2, 3) and size in (1, 2, 3) and len(objs["location"]) == nc * size
        assert len(objs["truck"]) == nc and 1 <= len(objs["airplane"]) <= nc and len(objs["obj"]) == npk
        city_of = {f[1]: f[2] for f in init if f[0] == "in-city"}
        assert Counter(city_of[a] for a in objs["airport"]) == Counter({c: 1 for c in objs["city"]})
        assert sorted(city_of[next(f[2] for f in init if f[0] == "at" and f[1] == t)] for t in objs["truck"]) == \
            objs["city"]
        assert all(next(f[2] for f in init if f[0] == "at" and f[1] == a) in objs["airport"] for a in objs["airplane"])
        assert sorted(f[1] for f in goal) == objs["obj"] and all(f[0] == "at" for f in goal)
        assert not L.goal_met(init, goal)
        assert all(grammar.match(x) for x in L.all_objects(objs))
        assert all(int(p[1:]) < 12 for p in objs["obj"])
        renamed = any(f[0] == "airport" and not f[1].endswith("-0") for f in init) or any(
            not city_of[next(f[2] for f in init if f[0] == "at" and f[1] == t)].endswith(t[1:]) for t in objs["truck"])
        n_ren += renamed
        if k < 10:
            p = L.astar_plan(init, goal)
            path = tmpdir_mod / f"gen{k}.pddl"
            path.write_text(L.write_problem(objs, init, goal, name=f"gen{k}"))
            assert L.val_check(path, p)
    print(f"\n[generator] 400 draws: {n_none} goal-already-met, {n_ren} visibly renamed, "
          f"{len(shapes)} (cities, size, planes) shapes")
    assert len(shapes) == 15  # (2|3 cities) x (size 1-3) x (planes 1..cities)
    assert 100 < n_ren < 250
    # renaming is within-type: same signature as the PlanBench-named draw from the same stream
    for seed in range(20):
        a = L.generate(random.Random(seed), 3, rename=0.0)
        b = L.generate(random.Random(seed), 3, rename=1.0)
        if a is not None and b is not None:
            assert L.signature(*a) == L.signature(*b)


def test_no_model_libraries_loaded():
    assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
