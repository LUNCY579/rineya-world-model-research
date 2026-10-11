"""Canonical renaming signature for Logistics states/problems (design check; pure python, reads PDDL text)."""
import itertools, json, sys, collections
from pathlib import Path
sys.path.insert(0, "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import mystery as M
assert not any(m in sys.modules for m in ("torch", "transformers", "peft"))
LG = Path("C:/Users/clclc/AppData/Local/Temp/claude/C--Users-clclc-BusinessProject-WorldModel-Rineya-BWM-001-Research-Bundle/bc6adc92-8f02-442d-b75d-8a2ed62c0851/scratchpad/LLMs-Planning/plan-bench/instances/logistics")

def signature(state, goal):
    st = {f for f in state}
    cities = sorted(f[1] for f in st if f[0] == "city")
    incity = {f[1]: f[2] for f in st if f[0] == "in-city"}
    airports = {f[1] for f in st if f[0] == "airport"}
    trucks = {f[1] for f in st if f[0] == "truck"}; planes = {f[1] for f in st if f[0] == "airplane"}
    pkgs = {f[1] for f in st if f[0] == "obj"}
    at = {f[1]: f[2] for f in st if f[0] == "at"}; inn = {f[1]: f[2] for f in st if f[0] == "in"}
    gl = {f[1]: f[2] for f in goal if f[0] == "at"}
    locs_by_city = {c: sorted(l for l, cc in incity.items() if cc == c) for c in cities}
    best = None
    for cp in itertools.permutations(range(len(cities))):
        per_city = []
        for ci, c in enumerate(cities):
            ap = [l for l in locs_by_city[c] if l in airports]; na = [l for l in locs_by_city[c] if l not in airports]
            per_city.append([(ci, ap, p) for p in itertools.permutations(na)])
        for choice in itertools.product(*per_city):
            lid = {}
            for ci, ap, na in choice:
                for k, l in enumerate(sorted(ap)):  # airports interchangeable only if >1 per city (not in PB)
                    lid[l] = (cp[ci], 0, k)
                for k, l in enumerate(na):
                    lid[l] = (cp[ci], 1, k)
            shape = tuple(sorted((cp[ci], sum(1 for l in locs_by_city[c] if l in airports), len(locs_by_city[c])) for ci, c in enumerate(cities)))
            g = lambda p: lid.get(gl.get(p)) if p in gl else None
            veh = tuple(sorted(((("T" if v in trucks else "A"), lid[at[v]], tuple(sorted(((g(p),) for p, vv in inn.items() if vv == v), key=str))) for v in trucks | planes), key=str))
            free = tuple(sorted(((lid[at[p]], g(p)) for p in pkgs if p in at), key=str))
            key = (shape, veh, free)
            ks = repr(key)
            if best is None or ks < best:
                best = ks
    return best

if __name__ == "__main__":
    S = json.load(open("../inst_stats.json"))["generated_basic"]
    sigs = {}
    for i in range(1, 287):
        o, init, goal = M.parse_problem(LG / "generated_basic" / f"instance-{i}.pddl")
        sigs[i] = signature(init, goal)
    cls = collections.defaultdict(list)
    for i, s in sigs.items(): cls[s].append(i)
    A = [i for i in range(2, 201) if S[str(i)]["packages"] >= 4]
    IN = [i for i in range(2, 287) if S[str(i)]["packages"] <= 3]
    print("classes over 286:", len(cls), "| dup classes:", sum(len(v) > 1 for v in cls.values()))
    print("PB-OUT-A distinct:", len({sigs[i] for i in A}), "of", len(A), "| PB-IN distinct:", len({sigs[i] for i in IN}), "of", len(IN))
    print("dup examples:", [v for v in cls.values() if len(v) > 1][:12])
    # sanity: a renaming (swap c0<->c1 and their locations, and truck names) gives same signature
    o, init, goal = M.parse_problem(LG / "generated_basic" / "instance-150.pddl")
    ren = {"c0": "c1", "c1": "c0", "l0-0": "l1-0", "l1-0": "l0-0", "l0-1": "l1-1", "l1-1": "l0-1", "l0-2": "l1-2", "l1-2": "l0-2", "t0": "t1", "t1": "t0", "p0": "p3", "p3": "p0"}
    R = lambda fs: frozenset(tuple(ren.get(x, x) for x in f) for f in fs)
    print("renaming invariance check:", signature(R(init), R(goal)) == signature(init, goal))
