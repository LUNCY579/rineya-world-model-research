import json, math, re, sys
from pathlib import Path
R = Path(sys.argv[1])
def load(p):
    d = {}
    for l in open(R/p, encoding="utf-8"):
        if l.strip():
            r = json.loads(l); d[r.get("id") or r.get("instance") or r.get("name")] = r
    return d
def mc(b, c):
    n = b + c; k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k+1)) / 2**n * 2
    return min(1.0, p)
for name, a, b in [("H2", "runs/BW_B3_HARD/h2_beam/plans.jsonl", "runs/BW_TEST_B2/hard_beam/plans.jsonl"),
                   ("H1", "runs/BW_B3_HARD/h1_direct/plans.jsonl", "runs/BW_TEST_B2/hard_direct/plans.jsonl")]:
    A, B = load(a), load(b)
    k0 = next(iter(A.values()))
    print(name, "keys:", sorted(k0.keys())[:30])
    assert set(A) == set(B), "id mismatch"
    def ok(r): return bool(r.get("val", r.get("success")))
    sa = sum(ok(r) for r in A.values()); sb = sum(ok(r) for r in B.values())
    b_ = sum(ok(A[i]) and not ok(B[i]) for i in A); c_ = sum(ok(B[i]) and not ok(A[i]) for i in A)
    disagree = sum(bool(r.get("val")) != bool(r.get("success")) for r in A.values() if "val" in r and "success" in r)
    print(name, len(A), "new", sa, "old", sb, "+%d/-%d" % (b_, c_), "p=%.6g" % mc(b_, c_), "val!=success", disagree)
    nb = lambda r: r.get("n_blocks") or r.get("blocks") or r.get("n")
    if nb(k0) is not None:
        for lo, hi in [(6, 10), (11, 15)]:
            ids = [i for i in A if lo <= nb(A[i]) <= hi]
            bb = sum(ok(A[i]) and not ok(B[i]) for i in ids); cc = sum(ok(B[i]) and not ok(A[i]) for i in ids)
            print("  %d-%d n=%d new=%d old=%d +%d/-%d p=%.3g" % (lo, hi, len(ids), sum(ok(A[i]) for i in ids), sum(ok(B[i]) for i in ids), bb, cc, mc(bb, cc)))
f2 = json.load(open(R/"runs/FREEZE_B3_H2.json")); fb = json.load(open(R/"runs/FREEZE_B2_beam.json"))
a2, ab = f2.get("args", f2), fb.get("args", fb)
print("H2 vs B2beam arg diff:", {k: (ab.get(k), a2.get(k)) for k in set(a2) | set(ab) if a2.get(k) != ab.get(k)})
print("B2beam args:", ab)
