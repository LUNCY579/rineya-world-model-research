import sys, json, glob, collections
sys.path.insert(0, r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1")
from bwmn import blocks as B
root = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1/runs/"
files = glob.glob(root + "BW_TEST_B2/mirror_beam/shard_*/fcache.jsonl")
seen = {}
for f in files:
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        seen[(frozenset(tuple(x) for x in r["s"]), tuple(r["a"]))] = r["text"]
c = collections.Counter(); ex = []
for (s, a), text in seen.items():
    ns, legal = B.step(s, a)
    if B.parse_forward_delta(text, s) is None:
        c[(a[0], legal)] += 1
        if len(ex) < 6 and legal: ex.append((a, text, B.forward_target_delta(s, ns, legal)))
print(c)
for e in ex: print(e)
# also legal-action base rates
base = collections.Counter((a[0], B.step(s,a)[1]) for (s,a) in seen)
print(base)
