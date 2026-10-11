import json, sys, glob
from collections import Counter
ROOT = r"C:/Users/clclc/BusinessProject/WorldModel/Rineya_BWM_001_Research_Bundle/native_v1"
sys.path.insert(0, ROOT)
from bwmn import blocks as B
for d in ("runs/BW_TEST_B2/hard_beam", "runs/BW_B3_HARD/h2_beam"):
    c = Counter(); ex = None
    for f in glob.glob(f"{ROOT}/{d}/shard_*/fcache.jsonl"):
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            s = frozenset(tuple(x) for x in r["s"]); a = tuple(r["a"])
            ns, legal = B.step(s, a)
            if B.parse_forward_delta(r["text"], s) is None:
                try: j = json.loads(r["text"]); lab = j.get("legal")
                except Exception: lab = "badjson"
                c[(legal, lab, a[0])] += 1
                if ex is None and lab is True: ex = (sorted(s)[:6], a, r["text"])
    print(d, dict(c)); print("  example", ex)
