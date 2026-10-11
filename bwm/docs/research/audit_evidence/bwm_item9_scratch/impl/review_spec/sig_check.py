"""Independent canonical form (structural labelings incl. within-city permutations at city size 3) vs L.signature."""
import sys, random, itertools
from collections import defaultdict
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L

def canon2(init, goal):
    o = L.objs_of(init)
    city_of = {f[1]: f[2] for f in init if f[0] == "in-city"}
    aps = set(o["airport"])
    best = None
    for cp in itertools.permutations(o["city"]):
        cidx = {c: i for i, c in enumerate(cp)}
        per = []
        for c in cp:
            nonap = [l for l in o["location"] if city_of[l] == c and l not in aps]
            ap = [l for l in o["location"] if city_of[l] == c and l in aps]
            per.append([(ap, q) for q in itertools.permutations(nonap)])
        for ch in itertools.product(*per):
            m = {c: f"C{cidx[c]}" for c in cp}
            for ci, (ap, q) in enumerate(ch):
                for k, l in enumerate(ap): m[l] = f"L{ci}a{k}"
                for k, l in enumerate(q): m[l] = f"L{ci}n{k}"
            for tp in itertools.permutations(o["truck"]):
                for apn in itertools.permutations(o["airplane"]):
                    for pp in itertools.permutations(o["obj"]):
                        mm = dict(m)
                        mm.update({x: f"T{i}" for i, x in enumerate(tp)})
                        mm.update({x: f"A{i}" for i, x in enumerate(apn)})
                        mm.update({x: f"P{i}" for i, x in enumerate(pp)})
                        key = repr((sorted((f[0],) + tuple(mm[y] for y in f[1:]) for f in init),
                                    sorted((f[0],) + tuple(mm[y] for y in f[1:]) for f in goal)))
                        if best is None or key < best: best = key
    return best

rng = random.Random(99)
items = []
while len(items) < 220:
    g = L.generate(rng, rng.choice((1, 2, 3)), n_cities=rng.choice((2, 3)), city_size=3, n_planes=rng.choice((1, 2)), rename=0.5)
    if g is None: continue
    objs, s, goal = g
    for _ in range(rng.randrange(5)):
        acts = [a for a in L.ground_actions(L.objs_of(s), s, self_loops=False) if L.applicable(s, a)]
        s = L.step(s, rng.choice(acts))[0]
    items.append((s, goal))
    # an isomorphic twin: permute non-airport locations within a city + vehicles/packages
    if rng.random() < 0.5:
        o = L.objs_of(s)
        city_of = {f[1]: f[2] for f in s if f[0] == "in-city"}
        m = {}
        for c in o["city"]:
            na = [l for l in o["location"] if city_of[l] == c and l not in set(o["airport"])]
            sh = na[:]; rng.shuffle(sh); m.update(zip(na, sh))
        for t in ("truck", "airplane", "obj"):
            xs = o[t]; sh = xs[:]; rng.shuffle(sh); m.update(zip(xs, sh))
        items.append(L.apply_renaming(m, s, goal)[1:])
    # a near-twin: swap the goals of two non-airport locations within one city (often NOT isomorphic)
s2b, b2s = defaultdict(set), defaultdict(set)
for s, g in items:
    a, b = L.signature(None, s, g), canon2(s, g)
    s2b[a].add(b); b2s[b].add(a)
print("items", len(items), "sig classes", len(s2b), "bf classes", len(b2s),
      "false merges", sum(len(v) > 1 for v in s2b.values()), "false splits", sum(len(v) > 1 for v in b2s.values()))
