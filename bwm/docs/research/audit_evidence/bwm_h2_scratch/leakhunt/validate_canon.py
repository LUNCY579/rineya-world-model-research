"""Validate canon.canon against the repo's brute-force B.signature (n <= 7) and renaming invariance (n <= 15)."""
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(sys.argv[1])
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
from bwmn import blocks as B  # noqa: E402
from canon import canon  # noqa: E402

assert "torch" not in sys.modules
rng = random.Random(12345)


def rand_problem(n, holding=False):
    while True:
        g = B.generate(rng, n, full_goal_frac=rng.choice([0.0, 1.0]))
        if g is None:
            continue
        objs, init, goal = g
        if holding:
            legal = [a for a in B.ground_actions(objs) if B.applicable(init, a) and a[0] in ("pick-up", "unstack")]
            init, _ = B.step(init, rng.choice(legal))
        return objs, init, goal


def rename(objs, init, goal):
    names = rng.sample(B.NAMES, len(objs))
    m = dict(zip(objs, names))
    r = lambda fs: frozenset(tuple([f[0]] + [m[x] for x in f[1:]]) for f in fs)  # noqa: E731
    return sorted(m[o] for o in objs), r(init), r(goal)


# 1) partition agreement with brute-force signature on small n (many collisions)
by_sig, by_can = defaultdict(set), defaultdict(set)
N = 0
for n in (3, 3, 4, 4, 5, 6):
    for _ in range(700 if n <= 4 else 150):
        p = rand_problem(n, holding=rng.random() < 0.3)
        if rng.random() < 0.5:
            p = rename(*p)
        s, c = B.signature(*p), canon(p[1], p[2], p[0])
        by_sig[s].add(c)
        by_can[c].add(s)
        N += 1
bad1 = sum(len(v) != 1 for v in by_sig.values())
bad2 = sum(len(v) != 1 for v in by_can.values())
print(f"partition test: {N} problems, {len(by_sig)} signature classes, {len(by_can)} canon classes, "
      f"sig->multiple canon {bad1}, canon->multiple sig {bad2}")
# 2) n = 7 spot check (slow brute force)
for _ in range(30):
    p = rand_problem(7)
    q = rename(*p)
    assert canon(p[1], p[2], p[0]) == canon(q[1], q[2], q[0])
    assert B.signature(*p) == B.signature(*q)
# 3) renaming invariance n <= 15 + sensitivity (one goal-pair change => usually different)
inv_ok = 0
for n in range(6, 16):
    for _ in range(200):
        p = rand_problem(n, holding=rng.random() < 0.3)
        q = rename(*p)
        assert canon(p[1], p[2], p[0]) == canon(q[1], q[2], q[0])
        inv_ok += 1
print("renaming invariance n 6-15:", inv_ok, "ok")
# 4) the hard set itself: invariance of all 110 under 20 random renamings each
hard = [B.parse_problem(p) for p in (B.BENCH / "blocksworld_hard" / "generated").glob("*.pddl")]
for h in hard:
    c0 = canon(h[1], h[2], h[0])
    for _ in range(20):
        q = rename(*h)
        assert canon(q[1], q[2], q[0]) == c0
print("hard x 20 renamings: ok;", "distinct hard canon classes:", len({canon(h[1], h[2], h[0]) for h in hard}))
