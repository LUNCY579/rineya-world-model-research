import json, collections
P = "C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/BW_DISTILL_r1/solutions.jsonl"
def step(S, a):
    S = set(S); op = a[0]
    if op == "pick-up":
        x = a[1]
        if not ({("clear", x), ("ontable", x), ("handempty",)} <= S): return None
        S -= {("clear", x), ("ontable", x), ("handempty",)}; S.add(("holding", x))
    elif op == "put-down":
        x = a[1]
        if ("holding", x) not in S: return None
        S.discard(("holding", x)); S |= {("clear", x), ("ontable", x), ("handempty",)}
    elif op == "stack":
        x, y = a[1], a[2]
        if not ({("holding", x), ("clear", y)} <= S) or x == y: return None
        S -= {("holding", x), ("clear", y)}; S |= {("on", x, y), ("clear", x), ("handempty",)}
    elif op == "unstack":
        x, y = a[1], a[2]
        if not ({("on", x, y), ("clear", x), ("handempty",)} <= S) or x == y: return None
        S -= {("on", x, y), ("clear", x), ("handempty",)}; S |= {("holding", x), ("clear", y)}
    else: return None
    return frozenset(S)
c = collections.Counter(); st_mis = 0; st_tot = 0; nb = collections.Counter(); exa = collections.Counter()
lens = []
for l in open(P, encoding="utf-8"):
    r = json.loads(l)
    exa[(r["verified"], r["exact_valid_for_analysis_only"])] += 1
    if not r["verified"]: continue
    nb[r["n"]] += 1
    S = frozenset(tuple(f) for f in r["init"]); ok = True
    for i, a in enumerate(r["plan"]):
        S2 = step(S, a)
        if S2 is None: ok = False; break
        S = S2; st_tot += 1
        if i < len(r["f_states"]) and frozenset(tuple(f) for f in r["f_states"][i]) != S: st_mis += 1
    goal = {tuple(g) for g in r["goal"]}
    c[ok and goal <= S] += 1; lens.append(len(r["plan"]))
print("verified plans exact-valid:", dict(c)); print("F-state mismatches", st_mis, "of", st_tot)
print("by blocks (verified):", sorted(nb.items())); print("stored exact flags:", dict(exa))
print("plan len min/mean/max", min(lens), sum(lens)/len(lens), max(lens), "sum suffix rows", sum(lens))
