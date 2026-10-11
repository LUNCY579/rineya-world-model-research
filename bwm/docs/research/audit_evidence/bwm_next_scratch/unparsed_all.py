import sys, json, glob, collections
R = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, R)
from bwmn import blocks as B
print("NAMES", len(B.NAMES), B.NAMES[:30])
sets = {"distill240": [R + "/runs/BW_DISTILL_r1/fcache.jsonl"], "select_v1": [R + "/runs/BW_SELECT_v1/fcache_delta.jsonl"]}
for name, files in sets.items():
    seen = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            seen[(frozenset(tuple(x) for x in r["s"]), tuple(r["a"]))] = r["text"]
    c = collections.Counter()
    for (s, a), t in seen.items():
        ns, legal = B.step(s, a); p = B.parse_forward_delta(t, s)
        if p is None: c["unparsed_" + ("legal" if legal else "illegal")] += 1
        elif p != (ns, legal): c["wrong_" + ("legal" if legal else "illegal")] += 1
        c["legal" if legal else "illegal"] += 1
    print(name, dict(c))
