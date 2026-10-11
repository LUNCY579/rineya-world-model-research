"""Signature review: generic backtracking isomorphism (independent of the Logistics structure) vs L.signature."""
import sys, time, random, itertools, json
from collections import Counter, defaultdict
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L

rng = random.Random(int(sys.argv[1]) if len(sys.argv) > 1 else 5)


def profile(init, goal):
    """object -> (type set, multiset of (section, pred, position, arity))"""
    objs = set(x for f in init | goal for x in f[1:])
    types = defaultdict(set)
    deg = defaultdict(Counter)
    for sec, fs in (("i", init), ("g", goal)):
        for f in fs:
            if len(f) == 2 and sec == "i":
                types[f[1]].add(f[0])
            for k, x in enumerate(f[1:]):
                deg[x][(sec, f[0], k, len(f))] += 1
    return {o: (tuple(sorted(types[o])), tuple(sorted(deg[o].items()))) for o in objs}


def isomorphic(a, b):
    (ia, ga), (ib, gb) = a, b
    if len(ia) != len(ib) or len(ga) != len(gb):
        return False
    pa, pb = profile(ia, ga), profile(ib, gb)
    if Counter(pa.values()) != Counter(pb.values()):
        return False
    cand = defaultdict(list)
    for o, pr in pb.items():
        cand[pr].append(o)
    # order: objects that appear in in-city first (locations/cities), then vehicles, then packages
    order = sorted(pa, key=lambda o: (len(cand[pa[o]]), o))
    facts_of = defaultdict(list)
    for sec, fs in (("i", ia), ("g", ga)):
        for f in fs:
            for x in f[1:]:
                facts_of[x].append((sec, f))
    tgt = {"i": ib, "g": gb}
    m, used = {}, set()

    def ok(o):
        for sec, f in facts_of[o]:
            if all(x in m for x in f[1:]):
                if (f[0],) + tuple(m[x] for x in f[1:]) not in tgt[sec]:
                    return False
        return True

    def bt(k):
        if k == len(order):
            return True
        o = order[k]
        for c in cand[pa[o]]:
            if c in used:
                continue
            m[o] = c
            used.add(c)
            if ok(o) and bt(k + 1):
                return True
            del m[o]
            used.discard(c)
        return False

    return bt(0)


def rand_rename(init, goal):
    objs = L.objs_of(init)
    m = {}
    # bijection within exact type-set classes (airports among airports, plain locations among plain locations)
    ap = set(objs["airport"])
    classes = [objs["city"], [l for l in objs["location"] if l in ap], [l for l in objs["location"] if l not in ap],
               objs["truck"], objs["airplane"], objs["obj"]]
    for cl in classes:
        names = [f"z{rng.randrange(10**6)}_{i}" for i in range(len(cl))]
        rng.shuffle(names)
        m.update(zip(cl, names))
    return L.apply_renaming(m, init, goal)[1:]


def legal(s):
    return [a for a in L.ground_actions(L.objs_of(s), s, self_loops=True) if L.applicable(s, a)]


t0 = time.time()
stats = Counter()
bad = []
# 1. PlanBench: invariance under arbitrary within-type renaming
pb = L.planbench_instances(with_gt=False)
sigs = {x["id"]: L.signature(x["objs"], x["init"], x["goal"]) for x in pb}
for x in pb:
    for _ in range(3):
        i2, g2 = rand_rename(x["init"], x["goal"])
        if L.signature(None, i2, g2) != sigs[x["id"]]:
            bad.append(("pb not invariant", x["id"]))
        stats["pb_renamings"] += 1
# 2. PlanBench: pairwise exactness within same profile multiset
groups = defaultdict(list)
for x in pb:
    groups[tuple(sorted(Counter(profile(x["init"], x["goal"]).values()).items()))].append(x)
for key, xs in groups.items():
    for a, b in itertools.combinations(xs, 2):
        iso = isomorphic((a["init"], a["goal"]), (b["init"], b["goal"]))
        same = sigs[a["id"]] == sigs[b["id"]]
        stats["pb_pairs_checked"] += 1
        stats["pb_iso_pairs"] += iso
        if iso != same:
            bad.append(("pb pair", a["id"], b["id"], iso, same))
print("[pb] groups", len(groups), dict(stats), "bad", bad[:5], round(time.time() - t0, 1), flush=True)

# 3. small generated instances + walk states (packages in vehicles), dense collisions
items = []
while len(items) < 1500:
    nc = rng.choice((2, 2, 3))
    g = L.generate(rng, rng.choice((1, 2, 3)), n_cities=nc, city_size=rng.choice((1, 2, 2, 3)),
                   n_planes=rng.randint(1, nc), rename=0.5)
    if g is None:
        continue
    objs, s, goal = g
    for _ in range(rng.randrange(6)):
        s = L.step(s, rng.choice(legal(s)))[0]
    if rng.random() < 0.3:  # partial goal (drop some package goals) -> packages without goals
        gl = sorted(goal)
        goal = frozenset(rng.sample(gl, rng.randint(1, len(gl))))
    items.append((s, goal))
    if rng.random() < 0.4:
        items.append(rand_rename(s, goal))
sig = [L.signature(None, s, g) for s, g in items]
by_prof = defaultdict(list)
for k, (s, g) in enumerate(items):
    by_prof[tuple(sorted(Counter(profile(s, g).values()).items()))].append(k)
n_pairs = n_iso = 0
for key, ks in by_prof.items():
    for a, b in itertools.combinations(ks, 2):
        iso = isomorphic(items[a], items[b])
        n_pairs += 1
        n_iso += iso
        if iso != (sig[a] == sig[b]):
            bad.append(("gen pair", a, b, iso, sig[a] == sig[b], items[a], items[b]))
# signature collisions across different profiles
prof_of = {}
for key, ks in by_prof.items():
    for k in ks:
        prof_of[k] = key
sig_prof = defaultdict(set)
for k, sg in enumerate(sig):
    sig_prof[sg].add(prof_of[k])
cross = [sg for sg, ps in sig_prof.items() if len(ps) > 1]
print("[gen] items", len(items), "pairs", n_pairs, "iso", n_iso, "distinct sigs", len(set(sig)),
      "cross-profile collisions", len(cross), "bad", len(bad), round(time.time() - t0, 1))
for b in bad[:5]:
    print(json.dumps(b, default=lambda x: sorted(map(list, x)) if isinstance(x, frozenset) else str(x))[:2000])
