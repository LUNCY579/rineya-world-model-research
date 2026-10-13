"""Verify that the repo's B.signature is a renaming-invariant canonical form (and that VL.invariant / the part keys are
renaming-invariant), against a definitional brute-force isomorphism test. Read-only."""
import itertools
import json
import random
import sys
import time
from collections import Counter

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from audit_common import B, VL, state_key, goal_key, find_iso, rename, valid_state  # noqa: E402


def brute_iso(A, Bp):
    oa, ia, ga = A
    ob, ib, gb = Bp
    if len(oa) != len(ob):
        return False
    for perm in itertools.permutations(ob):
        m = dict(zip(oa, perm))
        if rename(ia, m) == ib and rename(ga, m) == gb:
            return True
    return False


def rand_names(rng, n):
    pool = [chr(c) for c in range(97, 123)] + [f"b{i}" for i in range(30)] + ["o0", "o1", "o2", "o10", "zz"]
    return rng.sample(pool, n)


def random_rename(P, rng):
    objs, init, goal = P
    new = rand_names(rng, len(objs))
    m = dict(zip(objs, new))
    return sorted(new), rename(init, m), rename(goal, m)


def perturb(P, rng):
    """One-fact-level structural perturbation keeping a valid state and an on-only linear-forest goal."""
    objs, init, goal = P
    kind = rng.choice(["goal_retarget", "goal_drop", "goal_add", "init_move"])
    if kind == "init_move":
        on = {f[1]: f[2] for f in init if f[0] == "on"}
        below = set(on.values())
        tops = [b for b in objs if b not in below]
        x = rng.choice(tops)
        dests = [t for t in tops if t != x] + ["TABLE"]
        d = rng.choice(dests)
        s = set(init)
        if x in on:
            s.discard(("on", x, on[x]))
            s.add(("clear", on[x]))
        else:
            s.discard(("ontable", x))
        if d == "TABLE":
            s.add(("ontable", x))
        else:
            s.add(("on", x, d))
            s.discard(("clear", d))
        return kind, (objs, frozenset(s), goal)
    g = set(goal)
    if kind == "goal_drop" and len(g) > 1:
        g.discard(rng.choice(sorted(g)))
        return kind, (objs, init, frozenset(g))
    if kind == "goal_retarget" and g:
        f = rng.choice(sorted(g))
        g.discard(f)
        z = rng.choice([o for o in objs if o != f[1]])
        g.add(("on", f[1], z))
    else:
        x, y = rng.sample(objs, 2)
        g.add(("on", x, y))
    return kind, (objs, init, frozenset(g))


def main():
    rng = random.Random(12345)
    t0 = time.time()
    # 1) invariance under random renaming
    inv_fail = Counter()
    n_inv = 0
    pool = []
    for _ in range(600):
        n = rng.choice([3, 4, 5, 6, 7])
        g = B.generate(rng, n, full_goal_frac=rng.choice([0.0, 0.5, 1.0]))
        if g is None:
            continue
        pool.append(g)
        for _ in range(2):
            R = random_rename(g, rng)
            n_inv += 1
            if B.signature(*g) != B.signature(*R):
                inv_fail["signature"] += 1
            if VL.invariant(*g) != VL.invariant(*R):
                inv_fail["VL.invariant"] += 1
            if state_key(g[1], g[0]) != state_key(R[1], R[0]) or state_key(g[1], g[0]) is None:
                inv_fail["state_key"] += 1
            if goal_key(g[2], g[0]) != goal_key(R[2], R[0]) or goal_key(g[2], g[0]) is None:
                inv_fail["goal_key"] += 1
            if find_iso(g, R) is None:
                inv_fail["find_iso"] += 1
    print(f"[1] renaming invariance: {n_inv} (problem, random renaming) pairs, n=3..7; failures {dict(inv_fail)}")

    # 2) single-fact perturbations: signature must change unless the perturbed problem is (definitionally) isomorphic
    stats = Counter()
    for P in pool[:400]:
        if len(P[0]) > 6:
            continue
        kind, Q = perturb(P, rng)
        if not valid_state(Q[1], Q[0]):
            stats["invalid_perturbation_skipped"] += 1
            continue
        truth = brute_iso(P, Q)
        s_eq = B.signature(*P) == B.signature(*Q)
        try:
            fi = find_iso(P, Q) is not None
        except ValueError:
            fi = None
        stats[(kind, "iso" if truth else "non-iso", "sig_equal" if s_eq else "sig_changed")] += 1
        if s_eq != truth:
            stats["SIGNATURE_DISAGREES_WITH_BRUTE"] += 1
        if fi is not None and fi != truth:
            stats["FIND_ISO_DISAGREES_WITH_BRUTE"] += 1
        # renamed perturbation too (perturb, then rename): signature must still equal brute truth
        R = random_rename(Q, rng)
        if (B.signature(*P) == B.signature(*R)) != truth:
            stats["SIGNATURE_DISAGREES_AFTER_RENAME"] += 1
    print("[2] perturbations:", json.dumps({str(k): v for k, v in sorted(stats.items(), key=str)}, indent=0))

    # 3) collision-prone pairs: same part keys (state_key, goal_key) but possibly different alignment
    byK = {}
    for P in pool:
        if len(P[0]) <= 6:
            byK.setdefault((state_key(P[1], P[0]), goal_key(P[2], P[0])), []).append(P)
    c = Counter()
    for K, L in byK.items():
        for P, Q in itertools.combinations(L[:12], 2):
            truth = brute_iso(P, Q)
            s_eq = B.signature(*P) == B.signature(*Q)
            fi = find_iso(P, Q) is not None
            c[("iso" if truth else "non-iso")] += 1
            if s_eq != truth:
                c["SIGNATURE_DISAGREES"] += 1
            if fi != truth:
                c["FIND_ISO_DISAGREES"] += 1
    print("[3] same-K pairs (keys necessary only):", dict(c))
    # small hand-made examples
    objs = ["a", "b", "c"]
    init = B.towers_to_state([["a", "b"], ["c"]])  # b on a; c on table
    goal = frozenset({("on", "a", "c")})
    P = (objs, init, goal)
    Q = (["x", "y", "z"], B.towers_to_state([["z", "x"], ["y"]]), frozenset({("on", "z", "y")}))  # a->z, b->x, c->y
    Q2 = (["x", "y", "z"], B.towers_to_state([["z", "x"], ["y"]]), frozenset({("on", "y", "z")}))  # goal reversed
    Q3 = (["x", "y", "z"], B.towers_to_state([["z", "x"], ["y"]]), frozenset({("on", "x", "y")}))  # top on singleton
    print("[4] hand examples: renamed copy sig equal:", B.signature(*P) == B.signature(*Q), find_iso(P, Q),
          "| reversed goal sig equal:", B.signature(*P) == B.signature(*Q2), find_iso(P, Q2),
          "| top-block goal sig equal:", B.signature(*P) == B.signature(*Q3), find_iso(P, Q3),
          "| keys equal P,Q2:", (state_key(init, objs), goal_key(goal, objs)) == (state_key(Q2[1], Q2[0]), goal_key(Q2[2], Q2[0])))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
