import sys, random, tracemalloc
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import logistics as L
names = [f"p{k}" for k in range(12)] + [f"t{k}" for k in range(3)] + [f"a{k}" for k in range(3)] + [f"l{x}-{y}" for x in range(3) for y in range(3)] + [f"c{k}" for k in range(3)]
rng = random.Random(0)
tracemalloc.start()
ops = sorted(L.ARITY)
for _ in range(300000):
    op = rng.choice(ops)
    a = (op,) + tuple(rng.choice(names) for _ in range(L.ARITY[op]))
    L.ground(a)
cur, peak = tracemalloc.get_traced_memory()
print("cache", L.ground.cache_info(), "MB", round(cur / 2**20, 1))
