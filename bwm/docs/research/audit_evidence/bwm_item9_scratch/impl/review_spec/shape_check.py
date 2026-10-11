import sys, random
from collections import Counter
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
pb = L.planbench_instances()
sh = lambda o: (len(o["city"]), len(o["location"]) // len(o["city"]), len(o["airplane"]))
print("PB shapes all:", sorted(Counter(sh(x["objs"]) for x in pb).items()))
print("PB shapes 1-3 pk:", sorted(Counter(sh(x["objs"]) for x in pb if x["n_packages"] <= 3).items()))
print("PB shapes 4-7 pk:", sorted(Counter(sh(x["objs"]) for x in pb if x["n_packages"] >= 4).items()))
# truck position: uniform in city? plane position: uniform over airports?
rng = random.Random(1); c = Counter(); nn = 0
for _ in range(20000):
    g = L.generate(rng, 2)
    if g is None: nn += 1; continue
    c[sh(g[0])] += 1
print("gen shapes (n=2 pk):", sorted((k, round(v / sum(c.values()), 3)) for k, v in c.items()), "none", nn)
# goal-already-met rate by package count for size-1 shapes etc
