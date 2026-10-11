# Read-only replay of stored F outputs against the exact simulator (no model, no torch).
import sys, json, glob, collections
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import blocks as B
root = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/"
sets = {
  "mirror_beam_B2": glob.glob(root + "BW_TEST_B2/mirror_beam/shard_*/fcache.jsonl"),
  "distill240_K4_v2": [root + "BW_DISTILL_r1/fcache.jsonl"],
  "select_v1": [root + "BW_SELECT_v1/fcache_delta.jsonl"],
}
for name, files in sets.items():
    seen = {}
    for f in files:
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            s = frozenset(tuple(x) for x in r["s"]); a = tuple(r["a"])
            seen[(s, a)] = r["text"]
    by = collections.defaultdict(collections.Counter)
    for (s, a), text in seen.items():
        n = len(B.blocks_of(s))
        ns, legal = B.step(s, a)
        p = B.parse_forward_delta(text, s)
        c = by[n]
        c["n"] += 1
        c["legal_true"] += legal
        if p is None:
            c["unparsed"] += 1; continue
        pns, plegal = p
        if plegal != legal:
            c["false_legal" if plegal else "false_illegal"] += 1
        elif legal and pns != ns:
            c["wrong_state"] += 1
        else:
            c["exact"] += 1
    tot = collections.Counter()
    for n in sorted(by): tot.update(by[n])
    print(name, dict(tot))
    for n in sorted(by):
        c = by[n]; print("   ", n, c["n"], "err", c["n"]-c["exact"], dict((k,v) for k,v in c.items() if k not in ("n","exact","legal_true")))
