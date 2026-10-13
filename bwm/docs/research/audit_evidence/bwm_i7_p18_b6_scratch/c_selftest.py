import random, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c_verify import canon, explicit_map

R = random.Random(12345)


def rand_state(blocks, hold_ok=True):
    bl = blocks[:]; R.shuffle(bl)
    hold = None
    if hold_ok and R.random() < 0.4:
        hold = bl.pop()
    towers = []
    for b in bl:
        if towers and R.random() < 0.55:
            R.choice(towers).append(b)
        else:
            towers.append([b])
    f = []
    for t in towers:
        f.append(('ontable', t[0])); f.append(('clear', t[-1]))
        for a, b in zip(t[1:], t[:-1]):
            f.append(('on', a, b))
    f.append(('holding', hold) if hold else ('handempty',))
    return f


def rand_goal(blocks):
    st = rand_state(blocks, hold_ok=False)
    ons = [x for x in st if x[0] == 'on']
    k = R.randint(1, max(1, len(ons))) if ons else 0
    return R.sample(ons, k) if ons else []


def brute_iso(p, q):
    if len(p[0]) != len(q[0]):
        return False
    I2, G2 = set(q[1]), set(q[2])
    for perm in itertools.permutations(q[0]):
        m = dict(zip(p[0], perm))
        if {(f[0],) + tuple(m[x] for x in f[1:]) for f in p[1]} == I2 and {(f[0],) + tuple(m[x] for x in f[1:]) for f in p[2]} == G2:
            return True
    return False


def rename(p, names):
    m = dict(zip(p[0], names))
    return ([m[o] for o in p[0]], [(f[0],) + tuple(m[x] for x in f[1:]) for f in p[1]], [(f[0],) + tuple(m[x] for x in f[1:]) for f in p[2]])


inv_fail = 0; agree = Counter = 0
pool = []
for t in range(4000):
    n = R.randint(2, 6)
    blocks = [chr(97 + i) for i in range(n)]
    p = (blocks, rand_state(blocks), rand_goal(blocks))
    names = R.sample(['x%d' % i for i in range(30)] + list('abcdefghijklmnop'), n)
    q = rename(p, names)
    if canon(*p) != canon(*q) or explicit_map(p, q) is None:
        inv_fail += 1
    pool.append(p)
# equivalence vs brute force on pairs with same n (small n so many collisions)
dis = 0; iso_pairs = 0; tested = 0
small = [p for p in pool if len(p[0]) <= 4]
for i in range(len(small)):
    for j in range(i + 1, min(len(small), i + 40)):
        a, b = small[i], small[j]
        if len(a[0]) != len(b[0]):
            continue
        tested += 1
        bf = brute_iso(a, b)
        cf = canon(*a) == canon(*b)
        iso_pairs += bf
        if bf != cf:
            dis += 1
        if cf and explicit_map(a, b) is None:
            dis += 1
# perturbation test at n=5,6
pert = 0; pert_dis = 0
for p in pool:
    if len(p[0]) < 5:
        continue
    g = list(p[2])
    if not g:
        continue
    q = (p[0], p[1], g[:-1])
    pert += 1
    if (canon(*p) == canon(*q)) != brute_iso(p, q):
        pert_dis += 1
print('invariance failures', inv_fail, 'of', len(pool))
print('brute-force pairs tested', tested, 'isomorphic', iso_pairs, 'disagreements', dis)
print('goal-drop perturbations', pert, 'disagreements', pert_dis)
