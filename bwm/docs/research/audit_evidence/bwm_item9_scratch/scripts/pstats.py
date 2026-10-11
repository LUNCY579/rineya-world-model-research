import json, sys, collections, statistics
p = sys.argv[1]
d = json.load(open(p, encoding="utf-8"))
print("top keys", list(d.keys())[:10])
inst = d.get("instances", d)
print("n instances", len(inst), "prompt_type", d.get("prompt_type"), "domain", d.get("domain"), "task", d.get("task"))
k0 = inst[0]
print("instance keys", list(k0.keys()))
lens = []
for x in inst:
    gt = x.get("ground_truth_plan")
    if gt:
        steps = [s for s in gt.strip().splitlines() if s.strip() and not s.startswith(";")]
        lens.append(len(steps))
if lens:
    print("gt plan len: n", len(lens), "min", min(lens), "max", max(lens), "mean %.2f" % statistics.mean(lens), "median", statistics.median(lens))
    print("dist", dict(sorted(collections.Counter(lens).items())))
ids = [x.get("instance_id") for x in inst]
print("instance_id range", min(ids), max(ids))
q = k0.get("query", "")
print("--- first query head ---"); print(q[:1500])
