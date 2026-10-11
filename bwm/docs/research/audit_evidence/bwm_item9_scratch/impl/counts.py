import sys
sys.path.insert(0, '.')
from pathlib import Path
from bwmn import blocks as B
from bwmn import mystery as M
D = B.BENCH / 'logistics' / 'generated_basic'
res = {}
mism = 0
shapes = {}
for i in range(1, 287):
    p = D / f'instance-{i}.pddl'
    o, init, goal = B.parse_problem(p)
    o2, init2, goal2 = M.parse_problem(p)
    if (o, init, goal) != (o2, init2, goal2):
        mism += 1
    T = lambda t: sorted(f[1] for f in init if f[0] == t and len(f) == 2)
    P, Tr, A, L, C, AP = T('obj'), T('truck'), T('airplane'), T('location'), T('city'), T('airport')
    incity = {f[1]: f[2] for f in init if f[0] == 'in-city'}
    locs_of = {c: [l for l in L if incity[l] == c] for c in C}
    tcity = {f[1]: incity[f[2]] for f in init if f[0] == 'at' and f[1] in Tr}
    nA = 2*len(P)*len(Tr)*len(L) + 2*len(P)*len(A)*len(L) + sum(len(v)**2 for v in locs_of.values())*len(Tr) + len(A)*len(AP)**2
    nB = 2*len(P)*sum(len(locs_of[tcity[t]]) for t in Tr) + 2*len(P)*len(A)*len(AP) + sum(len(locs_of[tcity[t]])**2 for t in Tr) + len(A)*len(AP)**2
    nC = 2*len(P)*len(Tr)*len(L) + 2*len(P)*len(A)*len(L) + len(Tr)*len(L)**2*len(C) + len(A)*len(AP)**2
    nD = 2*len(P)*len(Tr)*len(L) + 2*len(P)*len(A)*len(AP) + sum(len(v)**2 for v in locs_of.values())*len(Tr) + len(A)*len(AP)**2
    res[i] = (nA, nB, nC, nD)
    shapes[i] = (len(C), len(L)//len(C), len(A), len(P))
print('B vs M parse mismatches', mism)
for k, name in enumerate('ABCD'):
    v = [r[k] for i, r in res.items() if i >= 2]
    print(name, min(v), max(v))
print(max(shapes.items(), key=lambda kv: res[kv[0]][1]))
from collections import Counter
print(Counter(shapes.values()).most_common(50))
