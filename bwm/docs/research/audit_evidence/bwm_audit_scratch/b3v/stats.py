import json, sys
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
assert 'torch' not in sys.modules
raw = open("solutions_copy.jsonl", "rb").read()
print("bytes", len(raw), "ends with newline", raw.endswith(b"\n"), "CRLF count", raw.count(b"\r\n"))
lines = raw.decode("utf-8").splitlines()
sols = [json.loads(l) for l in lines if l.strip()]
print("lines", len(sols), "ids ok", [r["id"] for r in sols] == [f"distill/{i}" for i in range(len(sols))])
ver = [r for r in sols if r["verified"]]
print("verified", len(ver))
byb = Counter(); byv = Counter()
for r in sols:
    byb[r["n"]] += 1; byv[r["n"]] += r["verified"]
print("by blocks verified/total", {k: f"{byv[k]}/{byb[k]}" for k in sorted(byb)})
print("exact among verified", sum(r["exact_valid_for_analysis_only"] for r in ver), "among unverified", sum(r["exact_valid_for_analysis_only"] for r in sols if not r["verified"]))
short = [r["id"] for r in ver if len(r["f_states"]) < len(r["plan"])]
short_m1 = [r["id"] for r in ver if len(r["f_states"]) < len(r["plan"]) - 1]
print("verified with len(f_states)<len(plan):", len(short), "< len(plan)-1:", len(short_m1))
emptyplan = [r["id"] for r in ver if not r["plan"]]
print("verified with empty plan", emptyplan)
diff = 0; nrows = 0; lastmiss = 0; illegal_exact = 0
for r in ver:
    init = frozenset(map(tuple, r["init"])); goal = frozenset(map(tuple, r["goal"]))
    plan = [tuple(x) for x in r["plan"]]
    s = init
    fst = [frozenset(map(tuple, x)) for x in r["f_states"]]
    for t, a in enumerate(plan):
        nrows += 1
        ok = B.applicable(s, a)
        if not ok: illegal_exact += 1; break
        s, _ = B.step(s, a)
        if t < len(fst) and fst[t] != s: diff += 1
    if not fst or not B.goal_met(fst[-1], goal): lastmiss += 1
print("rows", sum(len(r["plan"]) for r in ver), "checked", nrows, "F!=exact", diff, "exact-illegal plans", illegal_exact, "last F state misses goal", lastmiss)
lens = [len(r["plan"]) for r in ver]
print("plan len min/max/mean", min(lens), max(lens), round(sum(lens)/len(lens), 2))
# unverified stats
uv = [r for r in sols if not r["verified"]]
print("unverified plan lens", [len(r["plan"]) for r in uv])
# malformed facts in f_states
bad = 0
for r in ver:
    for st in r["f_states"]:
        for f in st:
            if f[0] not in ("on","ontable","clear","holding","handempty"): bad += 1
            if f[0]=="on" and len(f)!=3: bad+=1
print("malformed facts", bad)
# hand consistency: exactly one of handempty/holding
hb = 0
for r in ver:
    for st in r["f_states"]:
        h = [f for f in st if f[0] in ("handempty","holding")]
        if len(h) != 1: hb += 1
print("states with !=1 hand fact", hb)
